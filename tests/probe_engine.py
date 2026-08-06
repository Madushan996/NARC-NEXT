"""Run one UCI search and print its final info line and best move."""

import os
import subprocess
import sys
import time
import ctypes

path = os.path.abspath(sys.argv[1])
limit = sys.argv[2] if len(sys.argv) > 2 else "1000"
if limit.startswith("nodes:"):
    go_command = f"go nodes {int(limit.removeprefix('nodes:'))}"
elif limit.startswith("depth:"):
    go_command = f"go depth {int(limit.removeprefix('depth:'))}"
else:
    movetime = int(limit)
    go_command = f"go movetime {movetime}"
fen = sys.argv[3] if len(sys.argv) > 3 else "startpos"

engine = subprocess.Popen(
    [path],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    text=True,
    bufsize=1,
    cwd=os.path.dirname(path),
)
if os.name == "nt" and os.environ.get("NARC_AFFINITY_MASK"):
    mask = int(os.environ["NARC_AFFINITY_MASK"], 0)
    if not ctypes.windll.kernel32.SetProcessAffinityMask(int(engine._handle), mask):
        raise OSError("failed to set engine processor affinity")


def send(command: str):
    engine.stdin.write(command + "\n")
    engine.stdin.flush()


send("uci")
while engine.stdout.readline().strip() != "uciok":
    pass
send("setoption name Threads value 1")
send("setoption name Hash value 128")
send("isready")
while engine.stdout.readline().strip() != "readyok":
    pass
send("position " + fen)
started = time.perf_counter()
send(go_command)
last_info = ""
while True:
    line = engine.stdout.readline().strip()
    if line.startswith("info depth"):
        last_info = line
    if line.startswith("bestmove"):
        elapsed = (time.perf_counter() - started) * 1000
        print(last_info)
        print(line)
        print(f"wall time {elapsed:.1f} ms")
        break
send("quit")
engine.wait(timeout=10)
