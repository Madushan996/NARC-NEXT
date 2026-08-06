"""Build NARC Next's compact phase-aware quiet-move ordering prior."""

import modal

app = modal.App("narc-next-train-policy")
image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy")
volume = modal.Volume.from_name("narc-data")


@app.function(image=image, volumes={"/data": volume}, cpu=4, memory=4096, timeout=3600)
def train(
    dataset: str = "sf18-gen3-policy-v1",
    output: str = "narc-next-policy-v3.bin",
) -> dict:
    import glob
    import time

    import numpy as np

    volume.reload()
    files = sorted(glob.glob(f"/data/{dataset}/*.txt"))
    if not files:
        raise RuntimeError(f"no files found in /data/{dataset}")

    piece_types = {"p": 0, "n": 1, "b": 2, "r": 3, "q": 4, "k": 5}
    counts = np.zeros((8, 6, 64, 64), dtype=np.uint32)
    exposures = np.zeros((8, 6, 64), dtype=np.uint32)
    positions = skipped = 0
    started = time.time()

    for file_index, path in enumerate(files, 1):
        with open(path, encoding="utf-8", errors="ignore") as stream:
            for line in stream:
                parts = [part.strip() for part in line.split("|")]
                if len(parts) < 4 or len(parts[3]) < 4:
                    skipped += 1
                    continue
                fields = parts[0].split()
                if len(fields) < 2:
                    skipped += 1
                    continue

                board = [""] * 64
                square = 56
                for char in fields[0]:
                    if char == "/":
                        square -= 16
                    elif char.isdigit():
                        square += int(char)
                    else:
                        if 0 <= square < 64:
                            board[square] = char
                        square += 1

                move = parts[3]
                from_sq = (ord(move[0]) - ord("a")) + 8 * (ord(move[1]) - ord("1"))
                to_sq = (ord(move[2]) - ord("a")) + 8 * (ord(move[3]) - ord("1"))
                if not (0 <= from_sq < 64 and 0 <= to_sq < 64) or not board[from_sq]:
                    skipped += 1
                    continue
                piece = piece_types.get(board[from_sq].lower())
                if piece is None:
                    skipped += 1
                    continue

                occupied = sum(bool(value) for value in board)
                phase = min(7, max(0, (occupied - 2) * 8 // 31))
                black_to_move = fields[1] == "b"
                for source, value in enumerate(board):
                    if not value or value.islower() != black_to_move:
                        continue
                    source_piece = piece_types[value.lower()]
                    normalized_source = source ^ 56 if black_to_move else source
                    exposures[phase, source_piece, normalized_source] += 1
                if black_to_move:
                    from_sq ^= 56
                    to_sq ^= 56
                counts[phase, piece, from_sq, to_sq] += 1
                positions += 1
        if file_index % 20 == 0:
            print(f"parsed {file_index}/{len(files)} files, {positions:,} positions", flush=True)

    # Estimate P(best move | this piece occupies this source square).  This
    # removes the severe popularity bias of raw counts toward initial-position
    # pawn moves while retaining a zero-cost table lookup in the engine.
    # Confidence shrinkage keeps a move seen once in a rare configuration from
    # outranking a preference supported by thousands of examples.
    denominator = (exposures[..., None] + 500).astype(np.float64)
    rate = counts.astype(np.float64) / denominator
    scores = np.rint(np.sqrt(rate) * 12000.0)
    scores = np.clip(scores, 0, 12000).astype(np.int16)
    output_path = f"/data/{output}"
    with open(output_path, "wb") as stream:
        stream.write(b"NARCPOL1")
        stream.write(np.int32(8).tobytes())
        stream.write(np.int32(6).tobytes())
        stream.write(scores.tobytes())
    volume.commit()
    return {
        "positions": positions,
        "skipped": skipped,
        "nonzero": int(np.count_nonzero(counts)),
        "max_count": int(counts.max()),
        "max_exposure": int(exposures.max()),
        "max_score": int(scores.max()),
        "seconds": round(time.time() - started, 1),
        "output": output,
    }


@app.local_entrypoint()
def main(
    dataset: str = "sf18-gen3-policy-v1",
    output: str = "narc-next-policy-v3.bin",
):
    result = train.remote(dataset, output)
    print(result)
    print(f"modal volume get narc-data {output} .")
