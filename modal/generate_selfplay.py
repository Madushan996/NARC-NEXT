"""Generate NARC Next 2.6 self-play trajectories on Modal.

Stockfish 8 is used only to sample varied, credible opening positions.  Every
recorded move and evaluation after the opening comes from the original NARC
binary compiled from this repository.
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

app = modal.App("narc-next-gen4-selfplay")
volume = modal.Volume.from_name("narc-data")


@app.function(
    image=image,
    volumes={"/data": volume},
    cpu=2,
    memory=2048,
    timeout=6 * 3600,
    max_containers=60,
)
def generate_shard(
    shard_id: int,
    games: int = 500,
    output_dataset: str = "gen4-see-fixed",
    nodes: int = 5000,
    opening_nodes: int = 1500,
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

    rng = random.Random(0x4E41524326000000 ^ shard_id)
    white = chess.engine.SimpleEngine.popen_uci("/opt/narc-next")
    black = chess.engine.SimpleEngine.popen_uci("/opt/narc-next")
    opener = chess.engine.SimpleEngine.popen_uci("/opt/sf8/src/stockfish")
    for engine in (white, black, opener):
        engine.configure({"Threads": 1, "Hash": 32})

    records: list[tuple[str, int, float]] = []
    outcomes = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
    started = time.time()
    try:
        for game_index in range(games):
            board = chess.Board()
            opening_plies = 8 + rng.randrange(5)
            for _ in range(opening_plies):
                infos = opener.analyse(
                    board, chess.engine.Limit(nodes=opening_nodes), multipv=3
                )
                candidates = [info["pv"][0] for info in infos if info.get("pv")]
                if not candidates:
                    break
                board.push(rng.choices(candidates, weights=[6, 3, 1][:len(candidates)], k=1)[0])

            samples: list[tuple[str, int]] = []
            decisive = 0
            adjudicated = ""
            game_token = (shard_id, game_index)
            for _ in range(240 - board.ply()):
                if board.is_game_over(claim_draw=True):
                    break
                engine = white if board.turn == chess.WHITE else black
                info = engine.analyse(
                    board, chess.engine.Limit(nodes=nodes), game=game_token
                )
                pv = info.get("pv", [])
                score = info["score"].white().score(mate_score=30000)
                if not pv or score is None or pv[0] not in board.legal_moves:
                    break
                move = pv[0]
                if not board.is_check() and not board.is_capture(move) and not move.promotion:
                    samples.append((board.fen(), int(score)))
                if abs(score) >= 1500:
                    if score > 0:
                        decisive = decisive + 1 if decisive >= 0 else 1
                    else:
                        decisive = decisive - 1 if decisive <= 0 else -1
                    if abs(decisive) >= 4:
                        adjudicated = "1-0" if score > 0 else "0-1"
                        break
                else:
                    decisive = 0
                board.push(move)

            result_text = adjudicated or board.result(claim_draw=True)
            if result_text == "*":
                result_text = "1/2-1/2"
            outcomes[result_text] += 1
            result = 1.0 if result_text == "1-0" else 0.0 if result_text == "0-1" else 0.5
            records.extend((fen, score, result) for fen, score in samples)
    finally:
        white.quit()
        black.quit()
        opener.quit()

    temp = destination.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        for fen, score, result in records:
            stream.write(f"{fen} | {score} | {result:g}\n")
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
    shards: int = 60,
    games_per_shard: int = 500,
    output_dataset: str = "gen4-see-fixed",
    nodes: int = 5000,
    opening_nodes: int = 1500,
):
    tasks = [
        (index, games_per_shard, output_dataset, nodes, opening_nodes)
        for index in range(shards)
    ]
    total_games = total_positions = 0
    aggregate = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
    for completed, result in enumerate(generate_shard.starmap(tasks), 1):
        total_games += result.get("games", 0)
        total_positions += result.get("positions", 0)
        for key, value in result.get("outcomes", {}).items():
            aggregate[key] += value
        print(
            f"[{completed}/{shards}] {result} | games={total_games:,} "
            f"positions={total_positions:,}", flush=True,
        )
    print(
        f"complete: games={total_games:,} positions={total_positions:,} "
        f"outcomes={aggregate} dataset={output_dataset}"
    )
