"""Generate a compact NARC-format opening book from external SF18 analysis."""

from pathlib import Path
import struct

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
app = modal.App("narc-next-opening-book")
volume = modal.Volume.from_name("narc-data")


@app.function(image=image, cpu=2, memory=2048, timeout=3600, max_containers=60)
def analyse_batch(fens: list[str], nodes: int, breadth: int) -> list[tuple[str, str, list[str]]]:
    import glob
    import os
    import chess
    import chess.engine

    candidates = [
        path for path in glob.glob("/opt/stockfish/**/stockfish*", recursive=True)
        if os.path.isfile(path) and os.access(path, os.X_OK)
    ]
    engine = chess.engine.SimpleEngine.popen_uci(max(candidates, key=os.path.getsize))
    engine.configure({"Threads": 1, "Hash": 64})
    output = []
    try:
        for fen in fens:
            board = chess.Board(fen)
            infos = engine.analyse(board, chess.engine.Limit(nodes=nodes), multipv=breadth)
            moves = [info["pv"][0] for info in infos if info.get("pv")]
            children = []
            for move in moves:
                board.push(move)
                children.append(board.fen())
                board.pop()
            if moves:
                output.append((fen, moves[0].uci(), children))
    finally:
        engine.quit()
    return output


def prng_values():
    mask = (1 << 64) - 1
    state = 0x9E3779B97F4A7C15
    while True:
        state ^= state >> 12
        state ^= (state << 25) & mask
        state ^= state >> 27
        state &= mask
        yield (state * 2685821657736338717) & mask


def zobrist_tables():
    values = prng_values()
    psq = [[[next(values) for _ in range(64)] for _ in range(6)] for _ in range(2)]
    castle = [next(values) for _ in range(16)]
    ep = [next(values) for _ in range(8)]
    side = next(values)
    return psq, castle, ep, side


def encode_entry(fen: str, uci: str, tables):
    import chess

    board = chess.Board(fen)
    psq, castle_keys, ep_keys, side_key = tables
    key = 0
    type_map = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5,
    }
    for square, piece in board.piece_map().items():
        color = 0 if piece.color == chess.WHITE else 1
        key ^= psq[color][type_map[piece.piece_type]][square]
    castle = 0
    castle |= 1 if board.has_kingside_castling_rights(chess.WHITE) else 0
    castle |= 2 if board.has_queenside_castling_rights(chess.WHITE) else 0
    castle |= 4 if board.has_kingside_castling_rights(chess.BLACK) else 0
    castle |= 8 if board.has_queenside_castling_rights(chess.BLACK) else 0
    key ^= castle_keys[castle]
    if board.ep_square is not None:
        key ^= ep_keys[chess.square_file(board.ep_square)]
    if board.turn == chess.BLACK:
        key ^= side_key

    move = chess.Move.from_uci(uci)
    flag = 0
    promo = 0
    if move.promotion:
        flag = 1
        promo = move.promotion - chess.KNIGHT
    elif board.is_en_passant(move):
        flag = 2
    elif board.is_castling(move):
        flag = 3
    encoded = move.from_square | (move.to_square << 6) | (promo << 12) | (flag << 14)
    return key, encoded


@app.local_entrypoint()
def main(
    output: str = "narc-next-book-v1.bin",
    nodes: int = 10000,
    breadth: int = 6,
    root_depth: int = 6,
    seed_depth: int = 4,
    batch_size: int = 96,
):
    import chess

    opening_lines = [
        "e2e4 e7e5 g1f3 b8c6", "d2d4 d7d5 c2c4 e7e6", "e2e4 c7c5 g1f3 d7d6",
        "d2d4 g8f6 c2c4 e7e6", "e2e4 e7e6 d2d4 d7d5", "c2c4 e7e5 b1c3 g8f6",
        "g1f3 d7d5 d2d4 g8f6", "e2e4 c7c6 d2d4 d7d5", "d2d4 g8f6 c2c4 g7g6",
        "e2e4 e7e5 g1f3 g8f6", "d2d4 d7d5 g1f3 g8f6 c2c4 c7c6", "e2e4 g7g6 d2d4 f8g7",
        "c2c4 c7c5 g1f3 g8f6", "d2d4 f7f5 g2g3 g8f6", "e2e4 d7d6 d2d4 g8f6",
        "g1f3 g8f6 c2c4 b7b6", "e2e4 e7e5 f1c4 g8f6", "d2d4 e7e6 c2c4 f8b4",
        "b1c3 d7d5 e2e4 d5e4", "g2g3 d7d5 f1g2 c7c6",
    ]
    seeds = [(chess.STARTING_FEN, root_depth)]
    for line in opening_lines:
        board = chess.Board()
        for token in line.split():
            board.push_uci(token)
        seeds.append((board.fen(), seed_depth))

    entries = {}
    for seed_index, (seed, depth_limit) in enumerate(seeds, 1):
        frontier = {seed}
        for depth in range(depth_limit + 1):
            unseen = sorted(fen for fen in frontier if fen not in entries)
            if not unseen:
                break
            batches = [unseen[i:i + batch_size] for i in range(0, len(unseen), batch_size)]
            next_frontier = set()
            args = [(batch, nodes, breadth) for batch in batches]
            for batch_result in analyse_batch.starmap(args):
                for fen, bestmove, children in batch_result:
                    entries[fen] = bestmove
                    next_frontier.update(children)
            frontier = next_frontier
            print(
                f"seed {seed_index}/{len(seeds)} depth {depth}/{depth_limit}: "
                f"entries={len(entries):,} frontier={len(frontier):,}",
                flush=True,
            )

    tables = zobrist_tables()
    encoded = {}
    for fen, move in entries.items():
        key, value = encode_entry(fen, move, tables)
        encoded[key] = value
    path = Path("networks") / output
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(b"NARCBOOK")
        stream.write(struct.pack("<I", len(encoded)))
        for key, move in sorted(encoded.items()):
            stream.write(struct.pack("<QH", key, move))
    with volume.batch_upload(force=True) as batch:
        batch.put_file(str(path), output)
    print({"entries": len(encoded), "bytes": path.stat().st_size, "output": output})
