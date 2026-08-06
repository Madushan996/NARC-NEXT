"""Engine A vs Engine B match. python-chess referees every move.

Usage: python engine_match.py <engineA.exe> <engineB.exe> [games] [movetime_ms]
                              [optsA] [optsB]
optsA/optsB: semicolon-separated UCI options, e.g. "Use NNUE=false;Hash=64"
Reports score from A's perspective with an Elo estimate.
"""
import subprocess, sys, time, math, os, ctypes
import chess
import chess.pgn

A_PATH = sys.argv[1]
B_PATH = sys.argv[2]
GAMES = int(sys.argv[3]) if len(sys.argv) > 3 else 40
# arg4: fixed movetime in ms (e.g. "1000"), OR a game clock "tc:BASEms+INCms"
#       e.g. "tc:10000+100" = 10s + 0.1s/move per side (mirrors blitz play)
TC_ARG = sys.argv[4] if len(sys.argv) > 4 else "150"
A_OPTS = sys.argv[5] if len(sys.argv) > 5 else ""
B_OPTS = sys.argv[6] if len(sys.argv) > 6 else ""
OPENING_OFFSET = int(sys.argv[7]) if len(sys.argv) > 7 else 0
PGN_PATH = sys.argv[8] if len(sys.argv) > 8 else ""

USE_CLOCK = TC_ARG.startswith("tc:")
if USE_CLOCK:
    base_inc = TC_ARG[3:]
    TC_BASE = int(base_inc.split("+")[0])
    TC_INC = int(base_inc.split("+")[1]) if "+" in base_inc else 0
else:
    MOVETIME = int(TC_ARG)

OPENINGS = [
    "e2e4 e7e5 g1f3 b8c6", "d2d4 d7d5 c2c4 e7e6", "e2e4 c7c5 g1f3 d7d6",
    "d2d4 g8f6 c2c4 e7e6", "e2e4 e7e6 d2d4 d7d5", "c2c4 e7e5 b1c3 g8f6",
    "g1f3 d7d5 d2d4 g8f6", "e2e4 c7c6 d2d4 d7d5", "d2d4 g8f6 c2c4 g7g6",
    "e2e4 e7e5 g1f3 g8f6", "d2d4 d7d5 g1f3 g8f6 c2c4 c7c6", "e2e4 g7g6 d2d4 f8g7",
    "c2c4 c7c5 g1f3 g8f6", "d2d4 f7f5 g2g3 g8f6", "e2e4 d7d6 d2d4 g8f6",
    "g1f3 g8f6 c2c4 b7b6", "e2e4 e7e5 f1c4 g8f6", "d2d4 e7e6 c2c4 f8b4",
    "b1c3 d7d5 e2e4 d5e4", "g2g3 d7d5 f1g2 c7c6",
]


def start(path, opts=""):
    path = os.path.abspath(path)
    p = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         text=True, bufsize=1, cwd=os.path.dirname(path) or ".")
    if os.name == "nt" and os.environ.get("NARC_MATCH_AFFINITY_MASK"):
        mask = int(os.environ["NARC_MATCH_AFFINITY_MASK"], 0)
        if not ctypes.windll.kernel32.SetProcessAffinityMask(int(p._handle), mask):
            raise OSError("failed to set engine processor affinity")
    send(p, "uci"); wait(p, "uciok")
    send(p, "setoption name Hash value 128")
    for kv in opts.split(";"):
        if "=" in kv:
            k, v = kv.split("=", 1)
            send(p, f"setoption name {k.strip()} value {v.strip()}")
    send(p, "isready"); wait(p, "readyok")
    return p


def send(p, c):
    p.stdin.write(c + "\n"); p.stdin.flush()


def wait(p, token, timeout=20):
    t0 = time.time()
    while time.time() - t0 < timeout:
        line = p.stdout.readline()
        if not line:
            raise RuntimeError("engine died")
        if line.strip().startswith(token):
            return line.strip()
    raise RuntimeError("timeout: " + token)


def bestmove_fixed(p, moves, movetime):
    send(p, "position startpos" + (" moves " + " ".join(moves) if moves else ""))
    send(p, f"go movetime {movetime}")
    return wait(p, "bestmove", movetime / 1000 + 15).split()[1]


def bestmove_clock(p, moves, wt, bt, white_to_move):
    send(p, "position startpos" + (" moves " + " ".join(moves) if moves else ""))
    send(p, f"go wtime {int(wt)} btime {int(bt)} winc {TC_INC} binc {TC_INC}")
    t0 = time.time()
    mv = wait(p, "bestmove", max(wt, bt) / 1000 + 15).split()[1]
    return mv, (time.time() - t0) * 1000.0  # elapsed ms


def play(white, black, opening, a_is_white):
    board = chess.Board()
    moves = list(opening.split())
    for m in moves:
        board.push(chess.Move.from_uci(m))
    # clocks: index by side-to-move colour
    wt = bt = float(TC_BASE) if USE_CLOCK else 0.0
    while not board.is_game_over(claim_draw=True) and len(moves) < 500:
        white_to_move = board.turn == chess.WHITE
        eng = white if white_to_move else black
        if USE_CLOCK:
            mv, used = bestmove_clock(eng, moves, wt, bt, white_to_move)
            if white_to_move:
                before = wt
                wt -= used
                if wt <= 0:
                    return "0-1", f"time used={used:.1f} remaining={before:.1f}", moves   # white flagged
                wt += TC_INC
            else:
                before = bt
                bt -= used
                if bt <= 0:
                    return "1-0", f"time used={used:.1f} remaining={before:.1f}", moves   # black flagged
                bt += TC_INC
        else:
            mv = bestmove_fixed(eng, moves, MOVETIME)
        move = chess.Move.from_uci(mv)
        if move not in board.legal_moves:
            raise RuntimeError(f"illegal move {mv} after {' '.join(moves)}")
        board.push(move)
        moves.append(mv)
    return board.result(claim_draw=True), "", moves


def save_pgn(path, moves, result, game_number, a_is_white, note):
    game = chess.pgn.Game()
    game.headers["Event"] = "NARC Next diagnostic match"
    game.headers["Round"] = str(game_number)
    game.headers["White"] = "Engine A" if a_is_white else "Engine B"
    game.headers["Black"] = "Engine B" if a_is_white else "Engine A"
    game.headers["Result"] = result
    if note:
        game.headers["Termination"] = note
    board = game.board()
    node = game
    for move_text in moves:
        move = chess.Move.from_uci(move_text)
        node = node.add_variation(move)
        board.push(move)
    with open(path, "a", encoding="utf-8") as pgn_file:
        print(game, file=pgn_file, end="\n\n")


def elo_diff(score, n):
    if n == 0:
        return 0.0
    p = min(max(score / n, 1e-4), 1 - 1e-4)
    return -400.0 * math.log10(1.0 / p - 1.0)


engA = start(A_PATH, A_OPTS)
engB = start(B_PATH, B_OPTS)
if PGN_PATH:
    pgn_dir = os.path.dirname(os.path.abspath(PGN_PATH))
    if pgn_dir:
        os.makedirs(pgn_dir, exist_ok=True)
    open(PGN_PATH, "w", encoding="utf-8").close()
score = 0.0
wins = losses = draws = 0

for g in range(GAMES):
    a_is_white = (g % 2 == 0)
    opening = OPENINGS[(OPENING_OFFSET + g // 2) % len(OPENINGS)]
    for e in (engA, engB):
        send(e, "ucinewgame"); send(e, "isready"); wait(e, "readyok")
    w, b = (engA, engB) if a_is_white else (engB, engA)
    r, note, moves = play(w, b, opening, a_is_white)
    if PGN_PATH:
        save_pgn(PGN_PATH, moves, r, g + 1, a_is_white, note)
    if r == "1-0":
        pts = 1.0 if a_is_white else 0.0
    elif r == "0-1":
        pts = 0.0 if a_is_white else 1.0
    else:
        pts = 0.5
    score += pts
    wins += pts == 1.0
    losses += pts == 0.0
    draws += pts == 0.5
    flag = f" [{note}]" if note else ""
    print(f"game {g+1:2}: A as {'W' if a_is_white else 'B'}: {r:7}{flag} -> "
          f"{'W' if pts==1 else 'L' if pts==0 else 'D'}  "
          f"(A: {score}/{g+1}, +{wins}-{losses}={draws}, "
          f"elo {elo_diff(score, g+1):+.0f})", flush=True)

send(engA, "quit"); send(engB, "quit")
print(f"\nFINAL A vs B: {score}/{GAMES}  (+{wins}-{losses}={draws})  "
      f"Elo: {elo_diff(score, GAMES):+.0f}")
