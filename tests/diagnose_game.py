"""Play paired diagnostic games and locate NARC's largest evaluation losses."""

from pathlib import Path
import argparse
import csv
import chess
import chess.engine
import chess.pgn


OPENINGS = [
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
]


def configure(engine: chess.engine.SimpleEngine):
    supported = engine.options
    options = {}
    if "Threads" in supported:
        options["Threads"] = 1
    if "Hash" in supported:
        options["Hash"] = 128
    if options:
        engine.configure(options)


def play_game(narc, opponent, narc_is_white: bool, move_ms: int,
              opening: list[str], round_name: str) -> chess.pgn.Game:
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["White"] = "NARC Next" if narc_is_white else "Stockfish 8"
    game.headers["Black"] = "Stockfish 8" if narc_is_white else "NARC Next"
    game.headers["Round"] = round_name
    game.headers["OpeningPly"] = str(len(opening))
    node = game
    for uci in opening:
        move = chess.Move.from_uci(uci)
        node = node.add_variation(move)
        board.push(move)

    limit = chess.engine.Limit(time=move_ms / 1000)
    while not board.is_game_over(claim_draw=True) and board.ply() < 240:
        narc_turn = board.turn == chess.WHITE if narc_is_white else board.turn == chess.BLACK
        engine = narc if narc_turn else opponent
        result = engine.play(board, limit)
        if result.move not in board.legal_moves:
            raise RuntimeError(f"illegal move: {result.move} in {board.fen()}")
        node = node.add_variation(result.move)
        board.push(result.move)

    game.headers["Result"] = board.result(claim_draw=True)
    return game


def score_white(engine, board: chess.Board, nodes: int) -> int:
    info = engine.analyse(board, chess.engine.Limit(nodes=nodes))
    score = info["score"].white().score(mate_score=30000)
    return int(score if score is not None else 0)


def diagnose(game: chess.pgn.Game, teacher, teacher_nodes: int) -> list[dict]:
    board = game.board()
    narc_is_white = game.headers["White"] == "NARC Next"
    losses = []
    opening_ply = int(game.headers.get("OpeningPly", "0"))
    for ply, move in enumerate(game.mainline_moves(), start=1):
        narc_turn = board.turn == chess.WHITE if narc_is_white else board.turn == chess.BLACK
        if narc_turn and ply > opening_ply:
            before = score_white(teacher, board, teacher_nodes)
            fen = board.fen()
            san = board.san(move)
            mover_white = board.turn == chess.WHITE
            non_kings = len(board.piece_map()) - 2
            phase = "opening" if ply <= 20 else "endgame" if non_kings <= 10 else "middlegame"
            board.push(move)
            after = score_white(teacher, board, teacher_nodes)
            loss = before - after if mover_white else after - before
            losses.append(
                {"ply": ply, "move": move.uci(), "san": san, "loss": loss,
                 "before": before, "after": after, "phase": phase, "fen": fen}
            )
        else:
            board.push(move)
    return sorted(losses, key=lambda row: row["loss"], reverse=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("narc")
    parser.add_argument("opponent")
    parser.add_argument("teacher")
    parser.add_argument("--move-ms", type=int, default=200)
    parser.add_argument("--teacher-nodes", type=int, default=20000)
    parser.add_argument("--pairs", type=int, default=1)
    parser.add_argument("--opening-offset", type=int, default=0)
    parser.add_argument("--output", default="tests/diagnostic_games.pgn")
    parser.add_argument("--csv-output", default="")
    args = parser.parse_args()

    narc = chess.engine.SimpleEngine.popen_uci(args.narc)
    opponent = chess.engine.SimpleEngine.popen_uci(args.opponent)
    teacher = chess.engine.SimpleEngine.popen_uci(args.teacher)
    for engine in (narc, opponent, teacher):
        configure(engine)
    try:
        games = []
        for pair in range(args.pairs):
            opening_index = (args.opening_offset + pair) % len(OPENINGS)
            opening = OPENINGS[opening_index].split()
            games.append(play_game(
                narc, opponent, True, args.move_ms, opening, f"{pair + 1}.1"
            ))
            games.append(play_game(
                narc, opponent, False, args.move_ms, opening, f"{pair + 1}.2"
            ))
        output = Path(args.output)
        with output.open("w", encoding="utf-8") as stream:
            for game in games:
                print(game, file=stream, end="\n\n")

        all_rows = []
        for index, game in enumerate(games, start=1):
            print(f"GAME {index}: {game.headers['White']} - {game.headers['Black']} "
                  f"{game.headers['Result']}")
            rows = diagnose(game, teacher, args.teacher_nodes)
            for row in rows:
                row.update({"game": index, "result": game.headers["Result"],
                            "narc_color": "white" if game.headers["White"] == "NARC Next" else "black"})
                all_rows.append(row)
            for row in rows[:8]:
                print(
                    f"  ply {row['ply']:3} {row['san']:8} loss={row['loss']:+5} cp "
                    f"eval {row['before']:+5}->{row['after']:+5} | {row['fen']}"
                )

        csv_output = Path(args.csv_output) if args.csv_output else output.with_suffix(".csv")
        fields = ["game", "result", "narc_color", "ply", "phase", "move", "san",
                  "loss", "before", "after", "fen"]
        with csv_output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(sorted(all_rows, key=lambda row: row["loss"], reverse=True))

        severe = [row for row in all_rows if row["loss"] >= 100]
        print(f"SUMMARY: {len(games)} games, {len(all_rows)} NARC moves, "
              f"{len(severe)} losses >=100 cp")
        for phase in ("opening", "middlegame", "endgame"):
            phase_rows = [row for row in severe if row["phase"] == phase]
            print(f"  {phase}: {len(phase_rows)} severe losses")
        print(f"CSV: {csv_output}")
    finally:
        narc.quit()
        opponent.quit()
        teacher.quit()


if __name__ == "__main__":
    main()
