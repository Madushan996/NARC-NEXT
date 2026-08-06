"""Label NARC self-play positions with official Stockfish 8 move choices.

Stockfish 8 is compiled as an external offline teacher.  NARC remains an
independent engine and contains no Stockfish source or linked implementation.
"""

from pathlib import Path

import modal


SF8_SOURCE = (
    "https://github.com/official-stockfish/Stockfish/"
    "archive/refs/tags/sf_8.tar.gz"
)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential", "curl", "tar")
    .pip_install("python-chess==1.999")
    .run_commands(
        f"curl -L --retry 5 -o /tmp/sf8.tar.gz '{SF8_SOURCE}'",
        "mkdir -p /opt/sf8 && tar -xzf /tmp/sf8.tar.gz -C /opt/sf8 --strip-components=1",
        "make -C /opt/sf8/src build ARCH=x86-64 COMP=gcc -j2",
    )
)

app = modal.App("narc-next-sf8-policy-labels")
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
    output_dataset: str = "sf8-gen3-policy-1m",
    stride: int = 15,
    nodes: int = 750,
) -> dict:
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

    written = skipped = 0
    started = time.time()
    temp = destination.with_suffix(destination.suffix + ".tmp")
    engine = chess.engine.SimpleEngine.popen_uci("/opt/sf8/src/stockfish")
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
                fen, result = parts[0].strip(), parts[2].strip()
                try:
                    board = chess.Board(fen)
                    info = engine.analyse(board, chess.engine.Limit(nodes=nodes))
                    score = info["score"].white().score(mate_score=30000)
                    pv = info.get("pv", [])
                    if score is None or not pv:
                        skipped += 1
                        continue
                except (ValueError, chess.engine.EngineError, chess.engine.EngineTerminatedError):
                    skipped += 1
                    continue
                dst.write(f"{fen} | {int(score)} | {result} | {pv[0].uci()}\n")
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
    output_dataset: str = "sf8-gen3-policy-1m",
    stride: int = 15,
    nodes: int = 750,
    max_shards: int = 0,
):
    shards = list_shards.remote(dataset)
    if max_shards > 0:
        shards = shards[:max_shards]
    print(f"Labeling {len(shards)} shards with Stockfish 8 -> {output_dataset}")
    total = 0
    args = ((shard, output_dataset, stride, nodes) for shard in shards)
    for completed, result in enumerate(relabel_one.starmap(args), 1):
        total += result.get("positions", 0)
        print(f"[{completed}/{len(shards)}] {result} | total={total:,}", flush=True)
    print(f"Labeling complete: {total:,} positions")
