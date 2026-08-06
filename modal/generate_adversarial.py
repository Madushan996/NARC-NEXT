"""Generate NARC-vs-Stockfish-8 trajectory positions on Modal.

Stockfish is an external opponent and opening selector only.  The NARC binary
is compiled exclusively from this repository's original source and embedded
network.  Output lines retain the normal ``fen | score | result`` format so
they can be relabelled by ``relabel_sf18.py``.
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
    .add_local_dir("src", "/narc/src", copy=True)
    .run_commands(
        "g++ -O3 -march=x86-64-v3 -flto -std=c++20 -pthread -DNDEBUG "
        "-o /opt/narc-next /narc/src/main.cpp"
    )
)

app = modal.App("narc-next-adversarial-datagen")
volume = modal.Volume.from_name("narc-data")


@app.function(
    image=image,
    volumes={"/data": volume},
    cpu=2,
    memory=2048,
    timeout=6 * 3600,
    max_containers=50,
)
def generate_shard(
    shard_id: int,
    games: int = 10,
    output_dataset: str = "adversarial-sf8-v1",
    narc_nodes: int = 30000,
    sf8_nodes: int = 38000,
    opening_nodes: int = 3000,
) -> dict:
    import random
    import time

    import chess
    import chess.engine

    volume.reload()
    destination_dir = Path("/data") / output_dataset
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"shard-{shard_id:04d}.txt"
    if destination.exists() and destination.stat().st_size > 0:
        return {"shard": shard_id, "status": "exists", "positions": 0}

    rng = random.Random(0x4E415243 ^ shard_id)
    narc = chess.engine.SimpleEngine.popen_uci("/opt/narc-next")
    sf8 = chess.engine.SimpleEngine.popen_uci("/opt/sf8/src/stockfish")
    narc.configure({"Threads": 1, "Hash": 64})
    sf8.configure({"Threads": 1, "Hash": 64})
    records: list[tuple[str, float]] = []
    outcomes = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
    started = time.time()

    try:
        for game_index in range(games):
            board = chess.Board()

            # Sample plausible but varied openings from SF8's top three moves.
            opening_plies = 8 + rng.randrange(5)
            for _ in range(opening_plies):
                infos = sf8.analyse(
                    board, chess.engine.Limit(nodes=opening_nodes), multipv=3
                )
                candidates = [info["pv"][0] for info in infos if info.get("pv")]
                if not candidates:
                    break
                weights = [6, 3, 1][: len(candidates)]
                board.push(rng.choices(candidates, weights=weights, k=1)[0])

            narc_white = ((shard_id * games + game_index) & 1) == 0
            positions: list[str] = []
            for _ in range(240 - board.ply()):
                if board.is_game_over(claim_draw=True):
                    break
                narc_turn = board.turn == chess.WHITE if narc_white else board.turn == chess.BLACK
                if narc_turn and not board.is_check():
                    positions.append(board.fen())
                engine = narc if narc_turn else sf8
                nodes = narc_nodes if narc_turn else sf8_nodes
                played = engine.play(board, chess.engine.Limit(nodes=nodes))
                if played.move not in board.legal_moves:
                    raise RuntimeError(f"illegal move {played.move} in {board.fen()}")
                board.push(played.move)

            result_text = board.result(claim_draw=True)
            if result_text == "*":
                result_text = "1/2-1/2"
            outcomes[result_text] += 1
            result = 1.0 if result_text == "1-0" else 0.0 if result_text == "0-1" else 0.5
            records.extend((fen, result) for fen in positions)
    finally:
        narc.quit()
        sf8.quit()

    temp = destination.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        for fen, result in records:
            stream.write(f"{fen} | 0 | {result:g}\n")
    temp.replace(destination)
    volume.commit()
    return {
        "shard": shard_id,
        "status": "written",
        "games": games,
        "positions": len(records),
        "outcomes": outcomes,
        "seconds": round(time.time() - started, 1),
    }


@app.local_entrypoint()
def main(
    shards: int = 50,
    games_per_shard: int = 10,
    output_dataset: str = "adversarial-sf8-v1",
    narc_nodes: int = 30000,
    sf8_nodes: int = 38000,
    opening_nodes: int = 3000,
):
    tasks = [
        (index, games_per_shard, output_dataset, narc_nodes, sf8_nodes, opening_nodes)
        for index in range(shards)
    ]
    total_games = total_positions = completed = 0
    aggregate = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
    for result in generate_shard.starmap(tasks):
        completed += 1
        total_games += result.get("games", 0)
        total_positions += result.get("positions", 0)
        for key, value in result.get("outcomes", {}).items():
            aggregate[key] += value
        print(
            f"[{completed}/{shards}] {result} | games={total_games:,} "
            f"positions={total_positions:,}",
            flush=True,
        )
    print(
        f"complete: games={total_games:,} positions={total_positions:,} "
        f"outcomes={aggregate} dataset={output_dataset}"
    )
