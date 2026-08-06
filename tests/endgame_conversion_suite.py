"""Deterministic low-time KQK/KRK conversion stress test for UCI engines.

The engine plays both sides from identical sparse winning positions.  This is
not an Elo test; it measures whether a material-winning engine can force mate
before the 50-move rule at tiny per-move budgets.
"""

import random
import subprocess
import sys
import time
import os
import ctypes

import chess


ENGINE = sys.argv[1]
LIMIT_ARG = sys.argv[2] if len(sys.argv) > 2 else "20"
USE_NODES = LIMIT_ARG.startswith("nodes:")
LIMIT = int(LIMIT_ARG.removeprefix("nodes:")) if USE_NODES else int(LIMIT_ARG)
CASES_PER_PIECE = int(sys.argv[3]) if len(sys.argv) > 3 else 32
SEED = 7362026


def send(proc, command):
    proc.stdin.write(command + "\n")
    proc.stdin.flush()


def wait_for(proc, prefix, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("engine terminated")
        if line.startswith(prefix):
            return line.strip()
    raise RuntimeError(f"timeout waiting for {prefix}")


def make_cases(piece_type, count):
    rng = random.Random(SEED + piece_type)
    cases = []
    seen = set()
    while len(cases) < count:
        wk, bk, psq = rng.sample(range(64), 3)
        if chess.square_distance(wk, bk) <= 1:
            continue
        # Avoid trivial positions where the bare king can take an undefended
        # major piece immediately.  White always moves first.
        if chess.square_distance(psq, bk) <= 1 and chess.square_distance(psq, wk) > 1:
            continue
        board = chess.Board(None)
        board.turn = chess.WHITE
        board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(psq, chess.Piece(piece_type, chess.WHITE))
        board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
        board.halfmove_clock = 0
        board.fullmove_number = 1
        if not board.is_valid() or board.is_game_over(claim_draw=True):
            continue
        fen = board.fen()
        if fen not in seen:
            seen.add(fen)
            cases.append(fen)
    return cases


def play_case(proc, fen):
    board = chess.Board(fen)
    moves = []
    send(proc, "ucinewgame")
    send(proc, "isready")
    wait_for(proc, "readyok")
    while not board.is_game_over(claim_draw=True) and len(moves) < 110:
        command = f"position fen {fen}"
        if moves:
            command += " moves " + " ".join(moves)
        send(proc, command)
        send(proc, f"go nodes {LIMIT}" if USE_NODES else f"go movetime {LIMIT}")
        response = wait_for(proc, "bestmove", 30 if USE_NODES else LIMIT / 1000 + 10)
        move_text = response.split()[1]
        move = chess.Move.from_uci(move_text)
        if move not in board.legal_moves:
            return "illegal", len(moves), move_text
        board.push(move)
        moves.append(move_text)
    outcome = board.outcome(claim_draw=True)
    if outcome and outcome.winner == chess.WHITE:
        return "mate", len(moves), ""
    return "draw", len(moves), board.fen()


def main():
    proc = subprocess.Popen(
        [ENGINE], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True, bufsize=1,
    )
    if os.name == "nt" and os.environ.get("NARC_AFFINITY_MASK"):
        mask = int(os.environ["NARC_AFFINITY_MASK"], 0)
        if not ctypes.windll.kernel32.SetProcessAffinityMask(int(proc._handle), mask):
            raise OSError("failed to set engine processor affinity")
    send(proc, "uci")
    wait_for(proc, "uciok")
    send(proc, "setoption name Hash value 64")
    totals = {"mate": 0, "draw": 0, "illegal": 0}
    mate_plies = []
    failures = []
    try:
        for piece_type, label in ((chess.QUEEN, "KQK"), (chess.ROOK, "KRK")):
            local = {"mate": 0, "draw": 0, "illegal": 0}
            for index, fen in enumerate(make_cases(piece_type, CASES_PER_PIECE), 1):
                result, plies, detail = play_case(proc, fen)
                local[result] += 1
                totals[result] += 1
                if result == "mate":
                    mate_plies.append(plies)
                else:
                    failures.append((label, index, result, plies, fen, detail))
            print(f"{label}: {local['mate']}/{CASES_PER_PIECE} mated, "
                  f"draws={local['draw']} illegal={local['illegal']}")
    finally:
        send(proc, "quit")
        proc.wait(timeout=10)
    average = sum(mate_plies) / len(mate_plies) if mate_plies else 0.0
    print(f"TOTAL: {totals['mate']}/{2 * CASES_PER_PIECE} mated, "
          f"average successful conversion={average:.1f} plies")
    for failure in failures[:12]:
        print("FAIL:", " | ".join(map(str, failure)))


if __name__ == "__main__":
    main()
