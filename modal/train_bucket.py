"""Train NARC Next's phase-bucketed NNUE with pawn-structure features."""

import modal

app = modal.App("narc-next-train-bucket")
image = modal.Image.debian_slim().pip_install("torch", "numpy")
volume = modal.Volume.from_name("narc-data")

QA, QB, QC, SCALE = 255, 64, 64, 400
BASE_FEATURES = 768
OUTPUT_BUCKETS, PAWN_FEATURES = 8, 16
PIECES = {
    "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
    "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11,
}


def pawn_features(pawns, enemy_pawns, king_square, color, occupied):
    """Sixteen bounded counts, each later normalized to 0..1 by dividing by 8."""
    files = [0] * 8
    for square in pawns:
        files[square & 7] += 1
    isolated = doubled = connected = passed = center = blocked = phalanx = shield = 0
    passed_ranks = [0] * 6
    direction = 8 if color == 0 else -8
    for square in pawns:
        file, rank = square & 7, square >> 3
        relative_rank = rank if color == 0 else 7 - rank
        if not any(files[f] for f in (file - 1, file + 1) if 0 <= f < 8):
            isolated += 1
        neighbours = [other for other in pawns if abs((other & 7) - file) == 1]
        if any((other >> 3) == rank for other in neighbours):
            phalanx += 1
        support_rank = rank - 1 if color == 0 else rank + 1
        if any((other >> 3) in (rank, support_rank) for other in neighbours):
            connected += 1
        enemy_ahead = any(
            abs((other & 7) - file) <= 1
            and ((other >> 3) > rank if color == 0 else (other >> 3) < rank)
            for other in enemy_pawns
        )
        if not enemy_ahead:
            passed += 1
            passed_ranks[min(5, max(0, relative_rank - 1))] += 1
        if 2 <= file <= 5 and 2 <= rank <= 5:
            center += 1
        if square + direction in occupied:
            blocked += 1
        if king_square >= 0:
            kf, kr = king_square & 7, king_square >> 3
            if abs(file - kf) <= 1 and (rank - kr if color == 0 else kr - rank) in (1, 2):
                shield += 1
    doubled = sum(max(0, count - 1) for count in files)
    islands = sum(1 for file in range(8) if files[file] and (file == 0 or not files[file - 1]))
    return [
        len(pawns), isolated, doubled, connected, passed,
        *passed_ranks, islands, shield, center, blocked, phalanx,
    ]


@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="A10G",
    memory=32768,
    cpu=8,
    timeout=3 * 3600,
)
def train(
    dataset: str = "sf18-gen3-v2-5m",
    output: str = "narc-next-pawn256-sf18-5m-v10.nnue",
    hidden: int = 256,
    epochs: int = 40,
    batch: int = 32768,
    lr: float = 1e-3,
    initial: str = "",
    king_buckets: int = 1,
    result_blend: float = 0.0,
    extra_dataset: str = "",
    extra_repeat: int = 1,
    head_hidden: int = 0,
    squared_activation: bool = False,
    material_phase: bool = False,
    major_buckets: bool = False,
) -> dict:
    import glob
    import time

    import numpy as np
    import torch
    import torch.nn as nn

    volume.reload()
    if king_buckets not in (1, 4, 16):
        raise ValueError("king_buckets must be 1, 4, or 16")
    if head_hidden not in (0, 32):
        raise ValueError("head_hidden must be 0 (linear) or 32")
    if head_hidden and king_buckets != 1:
        raise ValueError("the NARCNET10 head currently requires king_buckets=1")
    if head_hidden and squared_activation:
        raise ValueError("head_hidden and squared_activation are separate architectures")
    if material_phase and (head_hidden or not squared_activation):
        raise ValueError("material_phase requires squared_activation and a linear head")
    if major_buckets and (material_phase or head_hidden or not squared_activation):
        raise ValueError("major_buckets requires squared activation and excludes other head modes")
    input_features = BASE_FEATURES * king_buckets
    pad = input_features
    files = sorted(glob.glob(f"/data/{dataset}/*.txt"))
    if not files:
        raise RuntimeError(f"no files found in /data/{dataset}")
    base_file_count = len(files)
    if extra_dataset:
        if not 1 <= extra_repeat <= 100:
            raise ValueError("extra_repeat must be between 1 and 100")
        for extra_name in (name.strip() for name in extra_dataset.split(",")):
            if not extra_name:
                continue
            extra_files = sorted(glob.glob(f"/data/{extra_name}/*.txt"))
            if not extra_files:
                raise RuntimeError(f"no files found in /data/{extra_name}")
            files.extend(extra_files * extra_repeat)
            print(
                f"mixing {len(extra_files)} files from {extra_name} x{extra_repeat} "
                f"with {base_file_count} base files",
                flush=True,
            )

    white, black, scores, results, sides, buckets = [], [], [], [], [], []
    white_pawn_features, black_pawn_features = [], []
    started = time.time()
    for file_index, path in enumerate(files, 1):
        with open(path, encoding="utf-8", errors="ignore") as stream:
            for line in stream:
                parts = line.split("|")
                # Policy-labelled corpora may append a fourth best-move field;
                # the first three fields remain the standard evaluator target.
                fields = parts[0].split() if len(parts) >= 3 else []
                if len(fields) < 2:
                    continue
                try:
                    score = int(parts[1])
                    result = float(parts[2])
                except ValueError:
                    continue

                wi, bi, square, valid, phase_score = [], [], 56, True, 0
                queen_count = rook_count = 0
                pawn_squares, king_squares, occupied = [[], []], [-1, -1], set()
                for char in fields[0]:
                    if char == "/":
                        square -= 16
                    elif char.isdigit():
                        square += int(char)
                    elif char in PIECES:
                        piece = PIECES[char]
                        color, piece_type = divmod(piece, 6)
                        phase_score += (0, 1, 1, 2, 4, 0)[piece_type]
                        queen_count += piece_type == 4
                        rook_count += piece_type == 3
                        wi.append((color * 6 + piece_type) * 64 + square)
                        bi.append(((color ^ 1) * 6 + piece_type) * 64 + (square ^ 56))
                        occupied.add(square)
                        if piece_type == 0:
                            pawn_squares[color].append(square)
                        elif piece_type == 5:
                            king_squares[color] = square
                        square += 1
                    else:
                        valid = False
                        break
                if not valid or not 3 <= len(wi) <= 32:
                    continue

                # Bucket 0 is king-only; bucket 7 is the full starting phase.
                if major_buckets:
                    if queen_count == 0:
                        phase = min(2, rook_count)
                    elif queen_count == 1:
                        phase = 3 + min(1, rook_count)
                    else:
                        phase = 5 + (0 if rook_count == 0 else 1 if rook_count <= 2 else 2)
                else:
                    phase = min(
                        OUTPUT_BUCKETS - 1,
                        max(0, (phase_score if material_phase else len(wi) - 2)
                            * OUTPUT_BUCKETS // (25 if material_phase else 31)),
                    )
                wf = pawn_features(pawn_squares[0], pawn_squares[1], king_squares[0], 0, occupied)
                bf = pawn_features(pawn_squares[1], pawn_squares[0], king_squares[1], 1, occupied)
                if king_buckets > 1:
                    def king_bucket(square):
                        file, rank = square & 7, square >> 3
                        if king_buckets == 4:
                            # Two rank zones x outer/inner files. Mirroring the
                            # black perspective above keeps the mapping shared.
                            return (rank // 4) * 2 + (min(file, 7 - file) // 2)
                        return (rank // 2) * 4 + min(file, 7 - file)
                    white_bucket = king_bucket(king_squares[0])
                    black_bucket = king_bucket(king_squares[1] ^ 56)
                    wi = [white_bucket * BASE_FEATURES + feature for feature in wi]
                    bi = [black_bucket * BASE_FEATURES + feature for feature in bi]
                wi.extend([pad] * (32 - len(wi)))
                bi.extend([pad] * (32 - len(bi)))
                white.append(wi)
                black.append(bi)
                scores.append(score)
                results.append(result)
                sides.append(0 if fields[1] == "w" else 1)
                buckets.append(phase)
                white_pawn_features.append(wf)
                black_pawn_features.append(bf)
        if file_index % 20 == 0:
            print(f"parsed {file_index}/{len(files)} files, {len(scores):,} positions", flush=True)

    white = np.asarray(white, dtype=np.int16)
    black = np.asarray(black, dtype=np.int16)
    scores = np.asarray(scores, dtype=np.float32)
    results = np.asarray(results, dtype=np.float32)
    sides = np.asarray(sides, dtype=np.int8)
    buckets = np.asarray(buckets, dtype=np.int8)
    white_pawn_features = np.asarray(white_pawn_features, dtype=np.float32)
    black_pawn_features = np.asarray(black_pawn_features, dtype=np.float32)
    count = len(scores)
    print(f"parsed {count:,} positions in {time.time() - started:.1f}s", flush=True)

    black_to_move = sides.astype(bool)
    score_stm = np.clip(np.where(black_to_move, -scores, scores), -30000, 30000)
    stm_idx = np.where(black_to_move[:, None], black, white).astype(np.int64)
    nstm_idx = np.where(black_to_move[:, None], white, black).astype(np.int64)
    score_targets = (1.0 / (1.0 + np.exp(-score_stm / SCALE))).astype(np.float32)
    result_targets = np.where(black_to_move, 1.0 - results, results).astype(np.float32)
    targets = ((1.0 - result_blend) * score_targets + result_blend * result_targets).astype(np.float32)
    stm_pawns = np.where(black_to_move[:, None], black_pawn_features, white_pawn_features)
    nstm_pawns = np.where(black_to_move[:, None], white_pawn_features, black_pawn_features)
    pawn_inputs = np.concatenate((stm_pawns, nstm_pawns), axis=1) / 8.0

    rng = np.random.default_rng(7)
    permutation = rng.permutation(count)
    # Keep smoke corpora trainable while retaining a 2% validation slice for
    # production datasets. Never consume more than one fifth of the samples.
    validation_count = min(max(20, count // 50), max(1, count // 5))
    validation, training = permutation[:validation_count], permutation[validation_count:]

    t_stm = torch.from_numpy(stm_idx)
    t_nstm = torch.from_numpy(nstm_idx)
    t_targets = torch.from_numpy(targets)
    t_buckets = torch.from_numpy(buckets.astype(np.int64))
    t_pawns = torch.from_numpy(pawn_inputs.astype(np.float32))
    device = torch.device("cuda")

    class BucketNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = nn.EmbeddingBag(input_features + 1, hidden, mode="sum", padding_idx=pad)
            self.bias = nn.Parameter(torch.zeros(hidden))
            output_width = 2 * hidden + 2 * PAWN_FEATURES
            if head_hidden:
                self.head_weight = nn.Parameter(
                    torch.empty(OUTPUT_BUCKETS, head_hidden, output_width)
                )
                self.head_bias = nn.Parameter(torch.zeros(OUTPUT_BUCKETS, head_hidden))
                self.output_weight = nn.Parameter(torch.empty(OUTPUT_BUCKETS, head_hidden))
                self.output_bias = nn.Parameter(torch.zeros(OUTPUT_BUCKETS))
                nn.init.kaiming_uniform_(self.head_weight, a=5 ** 0.5)
            else:
                self.output_weight = nn.Parameter(
                    torch.empty(OUTPUT_BUCKETS, output_width)
                )
                self.output_bias = nn.Parameter(torch.zeros(OUTPUT_BUCKETS))
            nn.init.uniform_(self.transformer.weight, -0.05, 0.05)
            nn.init.uniform_(self.output_weight, -0.05, 0.05)
            with torch.no_grad():
                self.transformer.weight[pad].zero_()

        def forward(self, us, them, pawn_data, bucket_ids):
            a = torch.clamp(self.transformer(us) + self.bias, 0, 1)
            b = torch.clamp(self.transformer(them) + self.bias, 0, 1)
            if squared_activation:
                a = a * a
                b = b * b
            joined = torch.cat((a, b, pawn_data), dim=1)
            if head_hidden:
                first = torch.bmm(
                    self.head_weight[bucket_ids], joined.unsqueeze(2)
                ).squeeze(2) + self.head_bias[bucket_ids]
                joined = torch.clamp(first, 0, 1)
            weights = self.output_weight[bucket_ids]
            return (joined * weights).sum(dim=1) + self.output_bias[bucket_ids]

    net = BucketNet().to(device)
    if initial:
        initial_path = f"/data/{initial}"
        with open(initial_path, "rb") as stream:
            initial_magic = stream.read(8)
            if initial_magic not in (b"NARCNET7", b"NARCNT11", b"NARCNT12", b"NARCNT14"):
                raise RuntimeError(f"{initial} is not a compatible linear-head network")
            if initial_magic == b"NARCNT11" and not squared_activation:
                raise RuntimeError("a NARCNT11 initial network requires squared_activation")
            if initial_magic == b"NARCNT12" and not (material_phase or major_buckets):
                raise RuntimeError("a NARCNT12 initial network requires material_phase")
            if initial_magic == b"NARCNT14" and not major_buckets:
                raise RuntimeError("a NARCNT14 initial network requires major_buckets")
            file_hidden, file_buckets, file_pawns = np.frombuffer(stream.read(12), dtype=np.int32)
            if (file_buckets, file_pawns) != (OUTPUT_BUCKETS, PAWN_FEATURES) or file_hidden > hidden:
                raise RuntimeError(
                    f"incompatible initial network: H{file_hidden}, "
                    f"{file_buckets} buckets, {file_pawns} pawn features"
                )
            base_w0q = np.frombuffer(
                stream.read(BASE_FEATURES * file_hidden * 2), dtype=np.int16
            ).copy()
            b0q = np.frombuffer(stream.read(file_hidden * 2), dtype=np.int16).copy()
            initial_output_width = 2 * file_hidden + 2 * PAWN_FEATURES
            w1q = np.frombuffer(
                stream.read(OUTPUT_BUCKETS * initial_output_width * 2), dtype=np.int16
            ).copy()
            b1q = np.frombuffer(stream.read(OUTPUT_BUCKETS * 4), dtype=np.int32).copy()
        with torch.no_grad():
            # A bucketed model begins as an exact functional copy of the
            # proven unbucketed evaluator. Fine-tuning can then specialize
            # regions without first relearning the shared chess knowledge.
            initial_w0 = base_w0q.reshape(BASE_FEATURES, file_hidden)
            expanded_w0 = net.transformer.weight[:input_features].detach().cpu().numpy().copy()
            for bucket in range(king_buckets):
                start = bucket * BASE_FEATURES
                expanded_w0[start:start + BASE_FEATURES, :file_hidden] = initial_w0 / QA
            net.transformer.weight[:input_features].copy_(torch.from_numpy(expanded_w0).to(device))
            net.transformer.weight[pad].zero_()
            net.bias.zero_()
            net.bias[:file_hidden].copy_(torch.from_numpy(b0q / QA).to(device))
            if not head_hidden:
                # Preserve the smaller evaluator exactly. New neuron outputs
                # start at zero, but their random transformer features let
                # gradients activate the added capacity during fine-tuning.
                old_output = w1q.reshape(OUTPUT_BUCKETS, initial_output_width)
                net.output_weight.zero_()
                net.output_weight[:, :file_hidden].copy_(
                    torch.from_numpy(old_output[:, :file_hidden] / QB).to(device)
                )
                net.output_weight[:, hidden:hidden + file_hidden].copy_(
                    torch.from_numpy(
                        old_output[:, file_hidden:2 * file_hidden] / QB
                    ).to(device)
                )
                net.output_weight[:, 2 * hidden:].copy_(
                    torch.from_numpy(old_output[:, 2 * file_hidden:] / QB).to(device)
                )
                net.output_bias.copy_(torch.from_numpy(b1q / (QA * QB)).to(device))
        print(f"initialized H{hidden} from H{file_hidden} {initial}", flush=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.3)

    def validation_loss() -> float:
        net.eval()
        total = 0.0
        with torch.no_grad():
            for offset in range(0, len(validation), 65536):
                idx = validation[offset:offset + 65536]
                prediction = net(
                    t_stm[idx].to(device), t_nstm[idx].to(device),
                    t_pawns[idx].to(device), t_buckets[idx].to(device)
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
            idx = shuffled[offset:offset + batch]
            prediction = net(
                t_stm[idx].to(device), t_nstm[idx].to(device),
                t_pawns[idx].to(device), t_buckets[idx].to(device)
            )
            loss = ((torch.sigmoid(prediction) - t_targets[idx].to(device)) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                net.transformer.weight[pad].zero_()
            total += loss.item() * len(idx)
        scheduler.step()
        final_validation = validation_loss()
        print(
            f"epoch {epoch:2}: train={total / len(training):.6f} "
            f"val={final_validation:.6f} seconds={time.time() - epoch_started:.1f}",
            flush=True,
        )

    with torch.no_grad():
        w0 = net.transformer.weight[:input_features].cpu().numpy()
        b0 = net.bias.cpu().numpy()
        w1 = net.output_weight.cpu().numpy()
        b1 = net.output_bias.cpu().numpy()
        if head_hidden:
            wh = net.head_weight.cpu().numpy()
            bh = net.head_bias.cpu().numpy()
    w0q = np.clip(np.rint(w0 * QA), -32767, 32767).astype(np.int16)
    b0q = np.clip(np.rint(b0 * QA), -32767, 32767).astype(np.int16)
    if head_hidden:
        whq = np.clip(np.rint(wh * QB), -32767, 32767).astype(np.int16)
        bhq = np.rint(bh * QA * QB).astype(np.int32)
        w1q = np.clip(np.rint(w1 * QC), -32767, 32767).astype(np.int16)
        b1q = np.rint(b1 * QA * QB * QC).astype(np.int32)
    else:
        w1q = np.clip(np.rint(w1 * QB), -32767, 32767).astype(np.int16)
        b1q = np.rint(b1 * QA * QB).astype(np.int32)

    output_path = f"/data/{output}"
    with open(output_path, "wb") as stream:
        stream.write(b"NARCNT10" if head_hidden else b"NARCNT14" if major_buckets else
                     b"NARCNT12" if material_phase else
                     b"NARCNT11" if squared_activation else
                     b"NARCNET7" if king_buckets == 1 else
                     b"NARCNET9" if king_buckets == 4 else b"NARCNET8")
        stream.write(np.int32(hidden).tobytes())
        stream.write(np.int32(OUTPUT_BUCKETS).tobytes())
        stream.write(np.int32(PAWN_FEATURES).tobytes())
        if head_hidden:
            stream.write(np.int32(head_hidden).tobytes())
        if king_buckets > 1:
            stream.write(np.int32(king_buckets).tobytes())
        stream.write(w0q.tobytes())
        stream.write(b0q.tobytes())
        if head_hidden:
            stream.write(whq.tobytes())
            stream.write(bhq.tobytes())
            stream.write(w1q.tobytes())
            stream.write(b1q.tobytes())
        else:
            stream.write(w1q.tobytes())
            stream.write(b1q.tobytes())
    volume.commit()
    return {"positions": count, "validation_loss": final_validation, "output": output}


@app.local_entrypoint()
def main(
    dataset: str = "sf18-gen3-v2-5m",
    output: str = "narc-next-pawn256-sf18-5m-v10.nnue",
    hidden: int = 256,
    epochs: int = 40,
    batch: int = 32768,
    lr: float = 1e-3,
    initial: str = "",
    king_buckets: int = 1,
    result_blend: float = 0.0,
    extra_dataset: str = "",
    extra_repeat: int = 1,
    head_hidden: int = 0,
    squared_activation: bool = False,
    material_phase: bool = False,
    major_buckets: bool = False,
):
    result = train.remote(
        dataset, output, hidden, epochs, batch, lr, initial, king_buckets,
        result_blend, extra_dataset, extra_repeat, head_hidden, squared_activation,
        material_phase, major_buckets
    )
    print(result)
    print(f"modal volume get narc-data {output} .")
