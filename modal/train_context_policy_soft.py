"""Fine-tune NARC's root policy with sparse multi-PV teacher targets."""

import modal


app = modal.App("narc-next-train-context-policy-soft")
image = modal.Image.debian_slim().pip_install("torch", "numpy")
volume = modal.Volume.from_name("narc-data")

FEATURES, HIDDEN, MOVE_LABELS = 768, 256, 4096
QA, POLICY_Q = 255, 256
PIECES = {
    "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
    "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11,
}


@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="A10G",
    cpu=8,
    memory=32768,
    timeout=3 * 3600,
)
def train(
    dataset: str = "sf18-gen3-policy-mpv3-1m",
    network: str = "narc-next-pawn256-sf18-5m-v10.nnue",
    initial: str = "narc-next-context-policy-v1.bin",
    output: str = "narc-next-context-policy-soft-v5.bin",
    epochs: int = 2,
    batch: int = 4096,
    lr: float = 0.0001,
    temperature: float = 120.0,
) -> dict:
    import glob
    import time

    import numpy as np
    import torch
    import torch.nn as nn

    volume.reload()
    files = sorted(glob.glob(f"/data/{dataset}/*.txt"))
    if not files:
        raise RuntimeError(f"no files found in /data/{dataset}")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    pad = FEATURES
    stm_features, nstm_features, labels, scores = [], [], [], []
    skipped = 0
    started = time.time()
    for file_index, path in enumerate(files, 1):
        with open(path, encoding="utf-8", errors="ignore") as stream:
            for line in stream:
                parts = [part.strip() for part in line.split("|")]
                if len(parts) < 3:
                    skipped += 1
                    continue
                fields = parts[0].split()
                if len(fields) < 2:
                    skipped += 1
                    continue
                white, black = [], []
                square, valid = 56, True
                for char in fields[0]:
                    if char == "/":
                        square -= 16
                    elif char.isdigit():
                        square += int(char)
                    elif char in PIECES:
                        piece = PIECES[char]
                        color, piece_type = divmod(piece, 6)
                        white.append((color * 6 + piece_type) * 64 + square)
                        black.append(((color ^ 1) * 6 + piece_type) * 64 + (square ^ 56))
                        square += 1
                    else:
                        valid = False
                        break
                if not valid or not 3 <= len(white) <= 32:
                    skipped += 1
                    continue

                black_to_move = fields[1] == "b"
                row_labels, row_scores = [], []
                for target in parts[2].split(",")[:3]:
                    try:
                        move, score_text = target.rsplit(":", 1)
                        from_sq = (ord(move[0]) - 97) + 8 * (ord(move[1]) - 49)
                        to_sq = (ord(move[2]) - 97) + 8 * (ord(move[3]) - 49)
                        if black_to_move:
                            from_sq ^= 56
                            to_sq ^= 56
                        if not (0 <= from_sq < 64 and 0 <= to_sq < 64):
                            raise ValueError
                        row_labels.append(from_sq * 64 + to_sq)
                        row_scores.append(float(score_text))
                    except (ValueError, IndexError):
                        continue
                if not row_labels:
                    skipped += 1
                    continue
                while len(row_labels) < 3:
                    row_labels.append(row_labels[-1])
                    row_scores.append(-30000.0)
                white.extend([pad] * (32 - len(white)))
                black.extend([pad] * (32 - len(black)))
                stm_features.append(black if black_to_move else white)
                nstm_features.append(white if black_to_move else black)
                labels.append(row_labels)
                scores.append(row_scores)
        if file_index % 20 == 0:
            print(f"parsed {file_index}/{len(files)} files, {len(labels):,} positions", flush=True)

    stm_features = np.asarray(stm_features, dtype=np.int16)
    nstm_features = np.asarray(nstm_features, dtype=np.int16)
    labels = np.asarray(labels, dtype=np.int16)
    scores = np.asarray(scores, dtype=np.float32)
    count = len(labels)
    print(f"parsed {count:,} positions in {time.time() - started:.1f}s", flush=True)

    with open(f"/data/{network}", "rb") as stream:
        if stream.read(8) != b"NARCNET7":
            raise RuntimeError(f"{network} is not a NARCNET7 network")
        hidden, _, _ = np.frombuffer(stream.read(12), dtype=np.int32)
        if hidden != HIDDEN:
            raise RuntimeError(f"policy expects H{HIDDEN}, network is H{hidden}")
        w0q = np.frombuffer(stream.read(FEATURES * HIDDEN * 2), dtype=np.int16).copy()
        b0q = np.frombuffer(stream.read(HIDDEN * 2), dtype=np.int16).copy()

    class ContextPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = nn.EmbeddingBag(FEATURES + 1, HIDDEN, mode="sum", padding_idx=pad)
            self.transformer.weight.requires_grad_(False)
            self.register_buffer("transformer_bias", torch.zeros(HIDDEN))
            self.output = nn.Linear(2 * HIDDEN, MOVE_LABELS)

        def forward(self, us, them):
            a = torch.clamp(self.transformer(us) + self.transformer_bias, 0, 1)
            b = torch.clamp(self.transformer(them) + self.transformer_bias, 0, 1)
            return self.output(torch.cat((a, b), dim=1))

    device = torch.device("cuda")
    net = ContextPolicy().to(device)
    with torch.no_grad():
        net.transformer.weight[:FEATURES].copy_(torch.from_numpy(w0q.reshape(FEATURES, HIDDEN) / QA).to(device))
        net.transformer.weight[pad].zero_()
        net.transformer_bias.copy_(torch.from_numpy(b0q / QA).to(device))
        with open(f"/data/{initial}", "rb") as stream:
            if stream.read(8) != b"NARCPOL2":
                raise RuntimeError(f"{initial} is not a NARCPOL2 policy")
            width, move_labels, policy_q = np.frombuffer(stream.read(12), dtype=np.int32)
            if width != 2 * HIDDEN or move_labels != MOVE_LABELS:
                raise RuntimeError("incompatible initial policy")
            wq = np.frombuffer(stream.read(MOVE_LABELS * width * 2), dtype=np.int16).copy()
            bq = np.frombuffer(stream.read(MOVE_LABELS * 4), dtype=np.int32).copy()
        net.output.weight.copy_(torch.from_numpy(wq.reshape(MOVE_LABELS, width) / policy_q).to(device))
        net.output.bias.copy_(torch.from_numpy(bq / policy_q).to(device))

    t_stm = torch.from_numpy(stm_features)
    t_nstm = torch.from_numpy(nstm_features)
    t_labels = torch.from_numpy(labels.astype(np.int64))
    t_scores = torch.from_numpy(scores)
    rng = np.random.default_rng(29)
    permutation = rng.permutation(count)
    validation_count = max(20000, count // 100)
    validation, training = permutation[:validation_count], permutation[validation_count:]
    optimizer = torch.optim.AdamW(net.output.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    def batch_loss(idx):
        logits = net(t_stm[idx].to(device, dtype=torch.long), t_nstm[idx].to(device, dtype=torch.long))
        candidate_labels = t_labels[idx].to(device)
        candidate_scores = t_scores[idx].to(device)
        target_prob = torch.softmax(candidate_scores / temperature, dim=1)
        selected_log_prob = torch.log_softmax(logits, dim=1).gather(1, candidate_labels)
        return -(target_prob * selected_log_prob).sum(dim=1).mean(), logits, candidate_labels[:, 0]

    final_loss = final_accuracy = 0.0
    for epoch in range(1, epochs + 1):
        net.train()
        epoch_started = time.time()
        shuffled = rng.permutation(training)
        train_sum = 0.0
        for offset in range(0, len(shuffled), batch):
            idx = shuffled[offset:offset + batch]
            loss, _, _ = batch_loss(idx)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_sum += loss.item() * len(idx)
        scheduler.step()
        net.eval()
        loss_sum = correct = 0.0
        with torch.no_grad():
            for offset in range(0, len(validation), batch):
                idx = validation[offset:offset + batch]
                loss, logits, best = batch_loss(idx)
                loss_sum += loss.item() * len(idx)
                correct += (logits.argmax(1) == best).sum().item()
        final_loss = loss_sum / len(validation)
        final_accuracy = correct / len(validation)
        print(
            f"epoch {epoch}: train={train_sum / len(training):.5f} "
            f"val={final_loss:.5f} top1={final_accuracy:.4%} "
            f"seconds={time.time() - epoch_started:.1f}", flush=True,
        )

    with torch.no_grad():
        weight_q = np.clip(np.rint(net.output.weight.cpu().numpy() * POLICY_Q), -32767, 32767).astype(np.int16)
        bias_q = np.rint(net.output.bias.cpu().numpy() * POLICY_Q).astype(np.int32)
    output_path = f"/data/{output}"
    with open(output_path, "wb") as stream:
        stream.write(b"NARCPOL2")
        stream.write(np.int32(2 * HIDDEN).tobytes())
        stream.write(np.int32(MOVE_LABELS).tobytes())
        stream.write(np.int32(POLICY_Q).tobytes())
        stream.write(weight_q.tobytes())
        stream.write(bias_q.tobytes())
    volume.commit()
    return {"positions": count, "skipped": skipped, "validation_loss": final_loss, "top1": final_accuracy, "output": output}


@app.local_entrypoint()
def main(
    dataset: str = "sf18-gen3-policy-mpv3-1m",
    network: str = "narc-next-pawn256-sf18-5m-v10.nnue",
    initial: str = "narc-next-context-policy-v1.bin",
    output: str = "narc-next-context-policy-soft-v5.bin",
    epochs: int = 2,
    batch: int = 4096,
    lr: float = 0.0001,
    temperature: float = 120.0,
):
    print(train.remote(dataset, network, initial, output, epochs, batch, lr, temperature))
    print(f"modal volume get narc-data {output} .")
