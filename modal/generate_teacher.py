"""Generate a diverse Stockfish-18 teacher corpus for NARC Next NNUE.

Stockfish runs only as an external data-generation process. NARC Next does not
include, link, or copy Stockfish code. Each shard contains FEN, white-relative
teacher score, and eventual game result in the evaluator trainer's text format.
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

app = modal.App("narc-next-teacher-corpus")
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
    games: int = 400,
    output_dataset: str = "sf18-teacher-v1",
    nodes: int = 5000,
    opening_nodes: int = 1000,
) -> dict:
    import glob
    import os
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

    candidates = [
        path
        for path in glob.glob("/opt/stockfish/**/stockfish*", recursive=True)
        if os.path.isfile(path) and os.access(path, os.X_OK)
    ]
    if not candidates:
        raise RuntimeError("Stockfish executable was not found")
    engine_path = max(candidates, key=os.path.getsize)

    rng = random.Random(0x4E41524318000000 ^ shard_id)
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    engine.configure({"Threads": 1, "Hash": 64})
    records: list[tuple[str, int, float]] = []
    outcomes = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
    started = time.time()

    try:
        for game_index in range(games):
            board = chess.Board()
            game_token = (shard_id, game_index)

            # Create broad but credible openings by sampling among the top
            # teacher choices. The varying length prevents a fixed ply bias.
            opening_plies = 6 + rng.randrange(11)
            for _ in range(opening_plies):
                infos = engine.analyse(
                    board,
                    chess.engine.Limit(nodes=opening_nodes),
                    multipv=4,
                    game=game_token,
                )
                moves = [info["pv"][0] for info in infos if info.get("pv")]
                if not moves:
                    break
                weights = [12, 6, 3, 1][: len(moves)]
                board.push(rng.choices(moves, weights=weights, k=1)[0])

            samples: list[tuple[str, int]] = []
            decisive = 0
            adjudicated = ""
            for ply_index in range(220 - board.ply()):
                if board.is_game_over(claim_draw=True):
                    break
                info = engine.analyse(
                    board, chess.engine.Limit(nodes=nodes), game=game_token
                )
                pv = info.get("pv", [])
                score = info["score"].white().score(mate_score=30000)
                if not pv or score is None or pv[0] not in board.legal_moves:
                    break
                move = pv[0]

                # Quiet positions are the most useful static-evaluation labels;
                # sample half to reduce adjacent-position correlation.
                if (
                    ply_index % 2 == (shard_id & 1)
                    and not board.is_check()
                    and not board.is_capture(move)
                    and not move.promotion
                    and abs(score) < 10000
                ):
                    samples.append((board.fen(), int(score)))

                if abs(score) >= 1800:
                    sign = 1 if score > 0 else -1
                    decisive = decisive + sign if decisive * sign >= 0 else sign
                    if abs(decisive) >= 5:
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
        engine.quit()

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
    shards: int = 120,
    games_per_shard: int = 400,
    output_dataset: str = "sf18-teacher-v1",
    nodes: int = 5000,
    opening_nodes: int = 1000,
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
            f"positions={total_positions:,}",
            flush=True,
        )
    print(
        f"complete: games={total_games:,} positions={total_positions:,} "
        f"outcomes={aggregate} dataset={output_dataset}"
    )
