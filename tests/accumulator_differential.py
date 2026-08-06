"""Differentially verify incremental NNUE accumulators against full refresh.

Usage: python tests/accumulator_differential.py <engine.exe> [positions]

The engine's ``eval`` debug command compares its current incremental result
with a freshly reconstructed accumulator and emits ``ACC MISMATCH`` on error.
"""

from __future__ import annotations

import random
import subprocess
import sys
from pathlib import Path

import chess


ENGINE = str(Path(sys.argv[1]).resolve())
TARGET = int(sys.argv[2]) if len(sys.argv) > 2 else 2500


def send(process: subprocess.Popen[str], command: str) -> None:
    assert process.stdin is not None
    process.stdin.write(command + "\n")
    process.stdin.flush()


def inspect(process: subprocess.Popen[str], position_command: str) -> list[str]:
    send(process, position_command)
    send(process, "eval")
    send(process, "isready")
    lines: list[str] = []
    assert process.stdout is not None
    while True:
        line = process.stdout.readline()
        if not line:
            raise RuntimeError("engine exited during accumulator audit")
        line = line.strip()
        lines.append(line)
        if line == "readyok":
            return lines


def main() -> None:
    process = subprocess.Popen(
        [ENGINE], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        text=True, bufsize=1, cwd=str(Path(ENGINE).parent),
    )
    rng = random.Random(0x4E415243)
    checked = 0
    try:
        send(process, "uci")
        assert process.stdout is not None
        while process.stdout.readline().strip() != "uciok":
            pass

        # Deterministic coverage for castling, en passant, quiet promotion, and
        # capture promotion before the broad random-game audit.
        special = [
            "position startpos moves e2e4 e7e5 g1f3 b8c6 f1e2 g8f6 e1g1",
            "position startpos moves e2e4 a7a6 e4e5 d7d5 e5d6",
            "position fen 8/P6k/8/8/8/8/7K/8 w - - 0 1 moves a7a8q",
            "position fen 1r5k/P7/8/8/8/8/7K/8 w - - 0 1 moves a7b8q",
        ]
        for command in special:
            output = inspect(process, command)
            if any("ACC MISMATCH" in line for line in output):
                raise AssertionError(f"mismatch after: {command}\n" + "\n".join(output))
            checked += 1

        while checked < TARGET:
            board = chess.Board()
            moves: list[str] = []
            for _ in range(160):
                if board.is_game_over(claim_draw=True) or checked >= TARGET:
                    break
                move = rng.choice(list(board.legal_moves))
                board.push(move)
                moves.append(move.uci())
                output = inspect(process, "position startpos moves " + " ".join(moves))
                if any("ACC MISMATCH" in line for line in output):
                    raise AssertionError(
                        f"mismatch after random sequence: {' '.join(moves)}\n"
                        + "\n".join(output)
                    )
                checked += 1
        print(f"accumulator differential: {checked} positions, zero mismatches")
    finally:
        try:
            send(process, "quit")
        except (BrokenPipeError, OSError):
            pass
        process.wait(timeout=10)


if __name__ == "__main__":
    main()
