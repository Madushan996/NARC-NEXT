"""Relabel original NARC self-play positions with Stockfish 18 evaluations.

Stockfish is used only as an external teacher process. NARC Next contains no
Stockfish source or linked code; its inference and search remain independent.
"""

from pathlib import Path

import modal

STOCKFISH_URL = (
    "https://github.com/official-stockfish/Stockfish/releases/download/sf_18/"
    "stockfish-ubuntu-x86-64-avx2.tar"
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "tar")
    .pip_install("python-chess==1.999")
    .run_commands(
        f"curl -L --retry 5 -o /tmp/stockfish.tar '{STOCKFISH_URL}'",
        "mkdir -p /opt/stockfish && tar -xf /tmp/stockfish.tar -C /opt/stockfish",
        "find /opt/stockfish -type f -name 'stockfish*' -exec chmod +x {} +",
    )
)

app = modal.App("narc-next-sf18-relabel")
volume = modal.Volume.from_name("narc-data")


@app.function(volumes={"/data": volume}, timeout=1800)
def list_shards(dataset: str = "gen3") -> list[str]:
    return [str(path.relative_to("/data")) for path in sorted((Path("/data") / dataset).glob("*.txt"))]


@app.function(
    image=image,
    volumes={"/data": volume},
    cpu=2,
    memory=2048,
    timeout=6 * 3600,
    max_containers=60,
)
def relabel_one(
    shard: str,
    output_dataset: str = "sf18-gen3-v1",
    stride: int = 15,
    nodes: int = 1500,
    include_bestmove: bool = False,
) -> dict:
    import glob
    import os
    import time

    import chess
    import chess.engine

    volume.reload()
    source = Path("/data") / shard
    destination_dir = Path("/data") / output_dataset
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source.name

    if destination.exists() and destination.stat().st_size > 0:
        return {"shard": shard, "status": "exists", "positions": 0}

    candidates = [
        path
        for path in glob.glob("/opt/stockfish/**/stockfish*", recursive=True)
        if os.path.isfile(path) and os.access(path, os.X_OK)
    ]
    if not candidates:
        raise RuntimeError("Stockfish executable was not found in the image")
    engine_path = max(candidates, key=os.path.getsize)

    seen = written = skipped = 0
    started = time.time()
    temp = destination.with_suffix(destination.suffix + ".tmp")

    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    engine.configure({"Threads": 1, "Hash": 64})
    try:
        with source.open("r", encoding="utf-8", errors="ignore") as src, temp.open(
            "w", encoding="utf-8"
        ) as dst:
            for line_number, line in enumerate(src):
                if line_number % stride:
                    continue
                parts = line.rstrip().split("|")
                if len(parts) != 3:
                    skipped += 1
                    continue
                fen = parts[0].strip()
                result = parts[2].strip()
                seen += 1
                try:
                    board = chess.Board(fen)
                    info = engine.analyse(board, chess.engine.Limit(nodes=nodes))
                    score = info["score"].white().score(mate_score=30000)
                    pv = info.get("pv", [])
                    if score is None or (include_bestmove and not pv):
                        skipped += 1
                        continue
                except (ValueError, chess.engine.EngineError, chess.engine.EngineTerminatedError):
                    skipped += 1
                    continue
                suffix = f" | {pv[0].uci()}" if include_bestmove else ""
                dst.write(f"{fen} | {int(score)} | {result}{suffix}\n")
                written += 1
    finally:
        engine.quit()

    temp.replace(destination)
    volume.commit()
    return {
        "shard": shard,
        "status": "written",
        "positions": written,
        "skipped": skipped,
        "seconds": round(time.time() - started, 1),
    }


@app.local_entrypoint()
def main(
    dataset: str = "gen3",
    output_dataset: str = "sf18-gen3-v1",
    stride: int = 15,
    nodes: int = 1500,
    max_shards: int = 0,
    include_bestmove: bool = False,
):
    shards = list_shards.remote(dataset)
    if max_shards > 0:
        shards = shards[:max_shards]
    print(f"Relabeling {len(shards)} shards from {dataset} -> {output_dataset}")
    total = completed = 0
    kwargs = {
        "output_dataset": output_dataset,
        "stride": stride,
        "nodes": nodes,
        "include_bestmove": include_bestmove,
    }
    for result in relabel_one.starmap((shard, *kwargs.values()) for shard in shards):
        completed += 1
        total += result.get("positions", 0)
        print(f"[{completed}/{len(shards)}] {result} | total={total:,}", flush=True)
    print(f"Relabel complete: {total:,} positions")
