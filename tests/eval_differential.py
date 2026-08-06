"""Compare exact static evaluations from two UCI engine builds."""

from __future__ import annotations

import random
import subprocess
import sys
from pathlib import Path

import chess


ENGINE_A = str(Path(sys.argv[1]).resolve())
ENGINE_B = str(Path(sys.argv[2]).resolve())
TARGET = int(sys.argv[3]) if len(sys.argv) > 3 else 2000


def start(path: str) -> subprocess.Popen[str]:
    process = subprocess.Popen(
        [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        text=True, bufsize=1, cwd=str(Path(path).parent),
    )
    send(process, "uci")
    wait_for(process, "uciok")
    return process


def send(process: subprocess.Popen[str], command: str) -> None:
    assert process.stdin is not None
    process.stdin.write(command + "\n")
    process.stdin.flush()


def wait_for(process: subprocess.Popen[str], prefix: str) -> str:
    assert process.stdout is not None
    while True:
        line = process.stdout.readline()
        if not line:
            raise RuntimeError("engine terminated")
        if line.strip().startswith(prefix):
            return line.strip()


def evaluate(process: subprocess.Popen[str], position: str) -> int:
    send(process, position)
    send(process, "eval")
    send(process, "isready")
    score = None
    assert process.stdout is not None
    while True:
        line = process.stdout.readline().strip()
        if line.startswith("static eval:"):
            score = int(line.split()[2])
        if line == "readyok":
            if score is None:
                raise RuntimeError("engine did not return a static evaluation")
            return score


def main() -> None:
    engines = [start(ENGINE_A), start(ENGINE_B)]
    rng = random.Random(0x4E415243)
    checked = 0
    try:
        while checked < TARGET:
            board = chess.Board()
            moves: list[str] = []
            for _ in range(120):
                if board.is_game_over(claim_draw=True) or checked >= TARGET:
                    break
                move = rng.choice(list(board.legal_moves))
                board.push(move)
                moves.append(move.uci())
                command = "position startpos moves " + " ".join(moves)
                scores = [evaluate(engine, command) for engine in engines]
                if scores[0] != scores[1]:
                    raise AssertionError(
                        f"evaluation mismatch {scores[0]} != {scores[1]} after {command}"
                    )
                checked += 1
        print(f"evaluation differential: {checked} positions, exact parity")
    finally:
        for engine in engines:
            try:
                send(engine, "quit")
                engine.wait(timeout=10)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                engine.kill()


if __name__ == "__main__":
    main()
