"""Deterministic paired fixed-node SPRT match for UCI engines.

Usage:
  python tests/sprt_match.py engineA engineB [max_pairs] [nodes]
      [elo0] [elo1] [alpha] [beta] [pgn_path] [hash_mb] [pair_offset]

Each opening is played twice with colors reversed.  The sequential likelihood
test treats the two game points as two Bernoulli score observations; draws
contribute half a point.  This is deliberately simpler than a pentanomial
Fishtest model, but is deterministic, paired, and far less noisy than tiny
wall-clock matches on a busy desktop.
"""

from __future__ import annotations

import math
import os
import random
import subprocess
import sys
from dataclasses import dataclass

import chess
import chess.pgn


A_PATH = os.path.abspath(sys.argv[1])
B_PATH = os.path.abspath(sys.argv[2])
MAX_PAIRS = int(sys.argv[3]) if len(sys.argv) > 3 else 500
NODES = int(sys.argv[4]) if len(sys.argv) > 4 else 20_000
ELO0 = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
ELO1 = float(sys.argv[6]) if len(sys.argv) > 6 else 20.0
ALPHA = float(sys.argv[7]) if len(sys.argv) > 7 else 0.05
BETA = float(sys.argv[8]) if len(sys.argv) > 8 else 0.05
PGN_PATH = sys.argv[9] if len(sys.argv) > 9 else ""
HASH_MB = int(sys.argv[10]) if len(sys.argv) > 10 else 128
PAIR_OFFSET = int(sys.argv[11]) if len(sys.argv) > 11 else 0

BASE_OPENINGS = (
    "e2e4 e7e5 g1f3 b8c6", "d2d4 d7d5 c2c4 e7e6",
    "e2e4 c7c5 g1f3 d7d6", "d2d4 g8f6 c2c4 e7e6",
    "e2e4 e7e6 d2d4 d7d5", "c2c4 e7e5 b1c3 g8f6",
    "g1f3 d7d5 d2d4 g8f6", "e2e4 c7c6 d2d4 d7d5",
    "d2d4 g8f6 c2c4 g7g6", "e2e4 e7e5 g1f3 g8f6",
    "d2d4 d7d5 g1f3 g8f6 c2c4 c7c6", "e2e4 g7g6 d2d4 f8g7",
    "c2c4 c7c5 g1f3 g8f6", "d2d4 f7f5 g2g3 g8f6",
    "e2e4 d7d6 d2d4 g8f6", "g1f3 g8f6 c2c4 b7b6",
    "e2e4 e7e5 f1c4 g8f6", "d2d4 e7e6 c2c4 f8b4",
    "b1c3 d7d5 e2e4 d5e4", "g2g3 d7d5 f1g2 c7c6",
)


@dataclass
class Engine:
    process: subprocess.Popen[str]

    @classmethod
    def start(cls, path: str) -> "Engine":
        process = subprocess.Popen(
            [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1, cwd=os.path.dirname(path) or ".",
        )
        engine = cls(process)
        engine.send("uci")
        engine.wait("uciok")
        engine.send("setoption name Threads value 1")
        engine.send(f"setoption name Hash value {HASH_MB}")
        engine.send("isready")
        engine.wait("readyok")
        return engine

    def send(self, command: str) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def wait(self, prefix: str) -> str:
        assert self.process.stdout is not None
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError("engine terminated")
            line = line.strip()
            if line.startswith(prefix):
                return line

    def new_game(self) -> None:
        self.send("ucinewgame")
        self.send("isready")
        self.wait("readyok")

    def bestmove(self, moves: list[str]) -> tuple[str, int]:
        self.send("position startpos" + (" moves " + " ".join(moves) if moves else ""))
        self.send(f"go nodes {NODES}")
        assert self.process.stdout is not None
        score = 0
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError("engine terminated during search")
            line = line.strip()
            if line.startswith("info ") and " score " in line:
                fields = line.split()
                try:
                    index = fields.index("score")
                    if fields[index + 1] == "cp":
                        score = int(fields[index + 2])
                    elif fields[index + 1] == "mate":
                        score = 30_000 if int(fields[index + 2]) > 0 else -30_000
                except (ValueError, IndexError):
                    pass
            if line.startswith("bestmove"):
                return line.split()[1], score

    def close(self) -> None:
        self.send("quit")
        self.process.wait(timeout=10)


def opening_for(pair_index: int) -> list[str]:
    moves = BASE_OPENINGS[pair_index % len(BASE_OPENINGS)].split()
    board = chess.Board()
    for text in moves:
        board.push_uci(text)
    rng = random.Random(0x4E4152435EED + pair_index)
    # Six deterministic book-diversification plies produce hundreds of unique
    # paired starts without allowing an engine under test to choose its opening.
    for _ in range(6):
        legal = list(board.legal_moves)
        quiet = [move for move in legal if not board.is_capture(move) and not move.promotion]
        move = rng.choice(quiet or legal)
        board.push(move)
        moves.append(move.uci())
        if board.is_game_over(claim_draw=True):
            break
    return moves


def play(white: Engine, black: Engine, opening: list[str]) -> tuple[str, list[str]]:
    board = chess.Board()
    moves: list[str] = []
    for text in opening:
        board.push_uci(text)
        moves.append(text)
    winning_side = 0
    winning_streak = draw_streak = 0
    while not board.is_game_over(claim_draw=True) and len(moves) < 400:
        engine = white if board.turn == chess.WHITE else black
        white_to_move = board.turn == chess.WHITE
        text, score = engine.bestmove(moves)
        white_score = score if white_to_move else -score
        move = chess.Move.from_uci(text)
        if move not in board.legal_moves:
            raise RuntimeError(f"illegal move {text} after {' '.join(moves)}")
        board.push(move)
        moves.append(text)
        side = 1 if white_score >= 700 else -1 if white_score <= -700 else 0
        if side and side == winning_side:
            winning_streak += 1
        else:
            winning_side = side
            winning_streak = 1 if side else 0
        draw_streak = draw_streak + 1 if abs(white_score) <= 20 else 0
        if winning_streak >= 8:
            return ("1-0" if winning_side > 0 else "0-1"), moves
        if board.fullmove_number >= 40 and draw_streak >= 16:
            return "1/2-1/2", moves
    return board.result(claim_draw=True), moves


def save_pgn(path: str, moves: list[str], result: str, game_no: int, a_white: bool) -> None:
    game = chess.pgn.Game()
    game.headers["Event"] = "NARC Next fixed-node SPRT"
    game.headers["Round"] = str(game_no)
    game.headers["White"] = "Engine A" if a_white else "Engine B"
    game.headers["Black"] = "Engine B" if a_white else "Engine A"
    game.headers["Result"] = result
    node = game
    for text in moves:
        node = node.add_variation(chess.Move.from_uci(text))
    with open(path, "a", encoding="utf-8") as output:
        print(game, file=output, end="\n\n")


def logistic(elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def elo(score: float, games: int) -> float:
    p = min(max(score / games, 1e-6), 1.0 - 1e-6)
    return 400.0 * math.log10(p / (1.0 - p))


def main() -> None:
    if not (ELO0 < ELO1 and 0 < ALPHA < 1 and 0 < BETA < 1):
        raise ValueError("require elo0 < elo1 and alpha/beta in (0,1)")
    p0, p1 = logistic(ELO0), logistic(ELO1)
    lower = math.log(BETA / (1.0 - ALPHA))
    upper = math.log((1.0 - BETA) / ALPHA)
    llr = score = 0.0
    wins = draws = losses = 0
    if PGN_PATH:
        os.makedirs(os.path.dirname(os.path.abspath(PGN_PATH)), exist_ok=True)
        open(PGN_PATH, "w", encoding="utf-8").close()
    a, b = Engine.start(A_PATH), Engine.start(B_PATH)
    verdict = "INCONCLUSIVE"
    try:
        for pair in range(MAX_PAIRS):
            opening = opening_for(PAIR_OFFSET + pair)
            pair_score = 0.0
            for reverse in (False, True):
                a.new_game()
                b.new_game()
                a_white = not reverse
                white, black = (a, b) if a_white else (b, a)
                result, moves = play(white, black, opening)
                points = 0.5 if result == "1/2-1/2" else float((result == "1-0") == a_white)
                pair_score += points
                score += points
                wins += points == 1.0
                draws += points == 0.5
                losses += points == 0.0
                if PGN_PATH:
                    save_pgn(PGN_PATH, moves, result, 2 * pair + 1 + reverse, a_white)
            llr += pair_score * math.log(p1 / p0)
            llr += (2.0 - pair_score) * math.log((1.0 - p1) / (1.0 - p0))
            games = 2 * (pair + 1)
            print(
                f"pair {pair + 1:4}: {pair_score:.1f}/2  total {score:.1f}/{games} "
                f"(+{wins}-{losses}={draws}) elo {elo(score, games):+.1f} "
                f"LLR {llr:+.3f} [{lower:+.3f},{upper:+.3f}]",
                flush=True,
            )
            if llr >= upper:
                verdict = "ACCEPT_H1"
                break
            if llr <= lower:
                verdict = "ACCEPT_H0"
                break
    finally:
        a.close()
        b.close()
    games = wins + draws + losses
    print(
        f"FINAL {verdict}: {score:.1f}/{games} (+{wins}-{losses}={draws}) "
        f"Elo {elo(score, games):+.1f}, LLR {llr:+.3f} "
        f"for H0={ELO0:g} H1={ELO1:g}",
        flush=True,
    )


if __name__ == "__main__":
    main()
