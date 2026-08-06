"""Train NARC Next's phase-bucketed threat NNUE on Modal.

Inputs combine the 768 ordinary piece-square features with 768 interaction
features marking occupied pieces currently attacked by the opponent.
"""

import modal

app = modal.App("narc-next-train-threat")
image = modal.Image.debian_slim().pip_install("torch", "numpy", "chess==1.11.2")
volume = modal.Volume.from_name("narc-data")

QA, QB, SCALE = 255, 64, 400
BASE_FEATURES = 768
FEATURES = 1536
PAD = FEATURES
BUCKETS = 8


@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="A10G",
    memory=32768,
    cpu=8,
    timeout=3 * 3600,
)
def train(
    dataset: str = "sf18-gen3-v1",
    output: str = "narc-next-threat-bucket128-sf18-5m-v7.nnue",
    hidden: int = 128,
    epochs: int = 48,
    batch: int = 32768,
    lr: float = 1e-3,
) -> dict:
    import glob
    import time

    import chess
    import numpy as np
    import torch
    import torch.nn as nn

    volume.reload()
    files = sorted(glob.glob(f"/data/{dataset}/*.txt"))
    if not files:
        raise RuntimeError(f"no files found in /data/{dataset}")

    white_indices, black_indices, scores, sides, buckets = [], [], [], [], []
    started = time.time()
    for file_index, path in enumerate(files, start=1):
        with open(path, encoding="utf-8", errors="ignore") as stream:
            for line in stream:
                parts = line.split("|")
                if len(parts) != 3:
                    continue
                try:
                    board = chess.Board(parts[0].strip())
                    score = int(parts[1])
                except (ValueError, IndexError):
                    continue

                wi, bi = [], []
                for square, piece in board.piece_map().items():
                    color = 0 if piece.color == chess.WHITE else 1
                    piece_type = piece.piece_type - 1
                    white_feature = (color * 6 + piece_type) * 64 + square
                    black_feature = ((color ^ 1) * 6 + piece_type) * 64 + (square ^ 56)
                    wi.append(white_feature)
                    bi.append(black_feature)
                    if board.is_attacked_by(not piece.color, square):
                        wi.append(BASE_FEATURES + white_feature)
                        bi.append(BASE_FEATURES + black_feature)

                if len(wi) > 64 or len(wi) < 3:
                    continue
                wi.extend([PAD] * (64 - len(wi)))
                bi.extend([PAD] * (64 - len(bi)))
                white_indices.append(wi)
                black_indices.append(bi)
                scores.append(score)
                sides.append(0 if board.turn == chess.WHITE else 1)
                buckets.append(min(BUCKETS - 1, max(0, (len(board.piece_map()) - 2) * BUCKETS // 31)))
        if file_index % 20 == 0:
            print(f"parsed {file_index}/{len(files)} files, {len(scores):,} positions", flush=True)

    w_idx = np.asarray(white_indices, dtype=np.int16)
    b_idx = np.asarray(black_indices, dtype=np.int16)
    scores = np.asarray(scores, dtype=np.float32)
    sides = np.asarray(sides, dtype=np.int8)
    buckets = np.asarray(buckets, dtype=np.int8)
    count = len(scores)
    print(f"parsed {count:,} positions in {time.time() - started:.1f}s")

    black_to_move = sides.astype(bool)
    score_stm = np.clip(np.where(black_to_move, -scores, scores), -30000, 30000)
    idx_stm = np.where(black_to_move[:, None], b_idx, w_idx).astype(np.int64)
    idx_nstm = np.where(black_to_move[:, None], w_idx, b_idx).astype(np.int64)
    targets = (1.0 / (1.0 + np.exp(-score_stm / SCALE))).astype(np.float32)

    rng = np.random.default_rng(7)
    permutation = rng.permutation(count)
    validation_count = max(20000, count // 50)
    validation = permutation[:validation_count]
    training = permutation[validation_count:]

    device = torch.device("cuda")
    t_stm = torch.from_numpy(idx_stm)
    t_nstm = torch.from_numpy(idx_nstm)
    t_targets = torch.from_numpy(targets)
    t_buckets = torch.from_numpy(buckets.astype(np.int64))

    class ThreatNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = nn.EmbeddingBag(FEATURES + 1, hidden, mode="sum", padding_idx=PAD)
            self.bias = nn.Parameter(torch.zeros(hidden))
            self.output_weight = nn.Parameter(torch.empty(BUCKETS, 2 * hidden))
            self.output_bias = nn.Parameter(torch.zeros(BUCKETS))
            nn.init.uniform_(self.transformer.weight, -0.05, 0.05)
            nn.init.uniform_(self.output_weight, -0.05, 0.05)
            with torch.no_grad():
                self.transformer.weight[PAD].zero_()

        def forward(self, us, them, bucket_ids):
            a = torch.clamp(self.transformer(us) + self.bias, 0, 1)
            b = torch.clamp(self.transformer(them) + self.bias, 0, 1)
            joined = torch.cat((a, b), dim=1)
            return (joined * self.output_weight[bucket_ids]).sum(dim=1) + self.output_bias[bucket_ids]

    net = ThreatNet().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=max(1, epochs // 3), gamma=0.3
    )

    def validation_loss() -> float:
        net.eval()
        total = 0.0
        with torch.no_grad():
            for offset in range(0, len(validation), 65536):
                idx = validation[offset : offset + 65536]
                prediction = net(
                    t_stm[idx].to(device), t_nstm[idx].to(device), t_buckets[idx].to(device)
                )
                total += ((torch.sigmoid(prediction) - t_targets[idx].to(device)) ** 2).sum().item()
        net.train()
        return total / len(validation)

    final_validation = 0.0
    for epoch in range(1, epochs + 1):
        epoch_started = time.time()
        shuffled = rng.permutation(training)
        total = 0.0
        for offset in range(0, len(shuffled), batch):
            idx = shuffled[offset : offset + batch]
            prediction = net(
                t_stm[idx].to(device), t_nstm[idx].to(device), t_buckets[idx].to(device)
            )
            loss = ((torch.sigmoid(prediction) - t_targets[idx].to(device)) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                net.transformer.weight[PAD].zero_()
            total += loss.item() * len(idx)
        scheduler.step()
        final_validation = validation_loss()
        print(
            f"epoch {epoch:2}: train={total / len(training):.6f} "
            f"val={final_validation:.6f} seconds={time.time() - epoch_started:.1f}",
            flush=True,
        )

    with torch.no_grad():
        w0 = net.transformer.weight[:FEATURES].cpu().numpy()
        b0 = net.bias.cpu().numpy()
        w1 = net.output_weight.cpu().numpy()
        b1 = net.output_bias.cpu().numpy()

    w0q = np.clip(np.rint(w0 * QA), -32767, 32767).astype(np.int16)
    b0q = np.clip(np.rint(b0 * QA), -32767, 32767).astype(np.int16)
    w1q = np.clip(np.rint(w1 * QB), -32767, 32767).astype(np.int16)
    b1q = np.rint(b1 * QA * QB).astype(np.int32)
    output_path = f"/data/{output}"
    with open(output_path, "wb") as stream:
        stream.write(b"NARCNET6")
        stream.write(np.int32(hidden).tobytes())
        stream.write(np.int32(BUCKETS).tobytes())
        stream.write(w0q.tobytes())
        stream.write(b0q.tobytes())
        stream.write(w1q.tobytes())
        stream.write(b1q.tobytes())
    volume.commit()
    return {"positions": count, "validation_loss": final_validation, "output": output}


@app.local_entrypoint()
def main(
    dataset: str = "sf18-gen3-v2-5m",
    output: str = "narc-next-threat-bucket128-sf18-5m-v7.nnue",
    hidden: int = 128,
    epochs: int = 48,
    batch: int = 32768,
    lr: float = 1e-3,
):
    result = train.remote(dataset, output, hidden, epochs, batch, lr)
    print(result)
    print(f"modal volume get narc-data {output} .")
