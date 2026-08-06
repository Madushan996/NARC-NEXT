<div align="center">

<img src="assets/logo.png" alt="NARC Next" width="280">

# NARC Next

**An original, from-scratch C++20 UCI chess engine**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Release](https://img.shields.io/github/v/release/Madushan996/NARC-NEXT)](https://github.com/Madushan996/NARC-NEXT/releases)

</div>

NARC Next is an original, from-scratch C++20 UCI chess engine by Madushan
Dissanayake. It is the successor to NARC Engine and is intended for Arena and
other UCI-compatible chess GUIs.

## Originality and provenance

NARC Next does not copy, link, or derive from Stockfish source code. Its board
representation, move generation, search, and NNUE inference are written from
scratch. The engine builds from a single translation unit (`src/main.cpp`) and
depends on no third-party engine code.

The following external dependencies are disclosed in full:

- **NNUE training data.** The built-in evaluation network is trained by
  knowledge distillation on positions labelled with **Stockfish 18**
  evaluations (see `modal/generate_teacher.py`, `modal/relabel_sf18.py`). This
  is a data dependency only — no Stockfish code is present in this repository
  or in the released binary.
- **Optional root policy.** The embedded context-policy network is trained on
  **Stockfish 8** move choices. It is **disabled by default**
  (`Use Policy=false`) and contributes nothing to play unless explicitly
  enabled by the user.
- **Piece-square tables.** The classical fallback evaluation uses the public
  PeSTO tables by Ronald Friederich, credited in `src/eval.h`.
- **Match opponents.** Stockfish 8 is used as an external opponent for
  strength measurement only.

Search employs techniques from the public computer-chess literature — PVS,
null-move pruning, late move reductions, singular extensions, ProbCut, and
history heuristics. The implementations are original; the underlying ideas are
long-standing community knowledge and are not claimed as novel.

## Current status

NARC Next 4.1 is the current development champion. It combines the phase- and
pawn-aware H256 NNUE evaluator, corrected static-exchange evaluator, and full
draw detection inside quiescence search. Version 2.9 also spends more of its
normal UCI clock budget when a deeper iteration reveals an evaluation drop,
while retaining the same hard reserve. The learned root policy remains
available through UCI but is disabled by default because unassisted search
proved stronger. Version 3.0 preserves more quiet moves at medium depth by
limiting quiet futility pruning to depths four and below.

Version 2.6 fixes a SEE back-propagation error that ignored the final recapture
layer. It scored 32.5/40 against 2.5 at 100 ms/move (+27-2=11, approximately
+255 Elo in that sample), and 7.5/10 at 10s+100ms (+6-1=3, approximately +191
Elo). In the two-offset 40-game fixed-time Stockfish 8 suite, 2.6 scored 3.5
points versus 2.5's 2.5 points.

Version 2.7 disables the learned root policy by default. Against policy-enabled
2.6 it scored 21.5/40 at 100 ms/move and 8.5/10 at 10s+100ms. Against Stockfish
8 it scored 4.5/40 across two opening offsets, compared with 2.6's 3.5/40.
This is a directional external improvement, but **2.7 does not yet beat
Stockfish 8 and is not yet a verified 3500-Elo engine**. Strength development
and external match gating remain active.

Version 2.8 records quiescence positions and terminates repetition,
fifty-move, and insufficient-material draws consistently with main search. It
scored 13/20 against 2.7 at 100 ms/move (+9-3=8, about +107 Elo), and 5/10 at
10s+100ms (+2-2=6). Against Stockfish 8 it scored 4.5/20 on opening offset 0,
up from 2.7's 3/20 on the same offset. **The original engine still does not
beat Stockfish 8 or verify the requested 3500-Elo target.**

Version 2.9 adds evaluation-instability pressure to UCI clock management. It
scored 10.5/20 against 2.8 at 10s+100ms (+5-4=11, about +17 Elo) across two
disjoint opening offsets, with no time forfeits. Fixed `movetime` behavior and
the underlying search/evaluation are unchanged.

Version 3.0 was selected using direct 100-game matches against Stockfish 8,
without a champion self-play gate. At one thread, 128 MB hash, and 100 ms per
move, 3.0 scored 15/100 (+12-82=6, estimated -301 Elo versus Stockfish 8),
while 2.9 scored 7.5/100 (+4-89=7, estimated -436 Elo) under the same settings.
That is a 7.5-point score gain and an estimated +135 Elo improvement over 2.9
through the common opponent. **NARC Next 3.0 still does not beat Stockfish 8
and is not yet a verified 3500-Elo engine.**

Version 3.1 targets low-time endgame conversion. In an eight-position,
10,000-node KQK/KRK suite against Stockfish 8 defense, 3.0 converted 4/8 while
3.1 converted 8/8 and reduced successful conversions from 35.0 to 19.8 plies
on average. Its increment-aware hard clock limit also retains a 20% reserve
while using up to 75% of the increment; from a test KQK position at 1s+1s it
searched 483 ms and found mate in eight, where 3.0 stopped after 181 ms without
seeing mate. In a 100-game candidate-versus-3.0 gate at 0.25s+0.10s, 3.1 scored
52/100 (+25-21=54, about +14 Elo). **It still does not beat Stockfish 8 or
verify the requested 3500-Elo target.**

Version 3.2 replaces the direct-mapped transposition table with four-entry
clusters and depth-aware replacement while preserving the configured hash
memory limit. In a 100-game candidate-versus-3.1 gate at 0.25s+0.10s, it
scored 52/100 (+34-30=36, about +14 Elo). Together with the preceding 3.1
gate, this is about +28 measured Elo over 3.0 toward the +200 checkpoint for
resuming Stockfish gates. **It still does not beat Stockfish 8 or verify the
requested 3500-Elo target.**

Version 3.5 replaces piece-count output buckets with material-phase buckets:
minor pieces count one phase unit, rooks two, and queens four, while pawn
trades no longer select a later evaluator head. The H256 squared network was
retrained on the same 11,117,664-position deep mixed corpus and reduced the
identical holdout loss from 0.003449 to 0.003256 without measurable NPS cost.
In a 100-game candidate-versus-3.4 gate at 0.25s+0.10s, it scored 54/100
(+33-25=42, about +28 Elo). The champion chain is now about +161 measured Elo
over 3.0, leaving about +39 Elo before Stockfish gates resume. **It still does
not beat Stockfish 8 or verify the requested 3500-Elo target.**

Version 3.6 replaces scalar material-phase head routing with eight major-piece
configuration heads, separating queenless minor and rook endings from lone-
queen and full-major middlegames. The H256 squared network was retrained on the
same 11,117,664-position corpus, reducing identical holdout loss from 0.003256
to 0.003143 without increasing model size or inference cost. In a fresh
100-game candidate-versus-3.5 gate at 0.25s+0.10s, it scored 53/100
(+30-24=46, about +21 Elo). The champion chain is now about +182 measured Elo
over 3.0, leaving about +18 Elo before Stockfish gates resume. **It still does
not beat Stockfish 8 or verify the requested 3500-Elo target.**

Version 3.7 strengthens the simplified major-piece conversion gradient while
retaining the v38 major-piece-bucket NNUE. In a deterministic 64-position
KQK/KRK stress suite at only 1 ms per move, successful conversions increased
from 60/64 to 61/64 and averaged 32.8 rather than 34.2 plies. It then scored
55/100 (+31-21=48, about +35 Elo) in the authoritative candidate-versus-3.6
gate at 0.25s+0.10s. The measured champion chain is now about +217 Elo over
3.0, crossing the +200 checkpoint; 100-game Stockfish 8 gates therefore
resume with this version. In the resumed direct gate under the same conditions,
3.7 scored 33/100 (+14-48=38), an estimated -123 Elo versus Stockfish 8. This
substantially narrows the previous external gap but does not close it. **The
final requirement to beat Stockfish 8 and verify roughly 3500 Elo remains
unproven.**

Version 3.8 reduces late-move pruning pressure on quiet conversion moves in
materially decisive positions with ten or fewer pieces. This gives king and
pawn manoeuvres one extra effective ply while keeping the change out of normal
middlegames. It scored 52.5/100 (+25-20=55, about +17 Elo) in the authoritative
candidate-versus-3.7 gate at 0.25s+0.10s. Version 3.7 is the reset baseline for
the next Stockfish checkpoint, so the new accumulation ledger is +17/150 Elo.
Stockfish 8 gating remains deferred until this ledger reaches approximately
+150 Elo. **The final requirement to beat Stockfish 8 and verify roughly 3500
Elo remains unproven.**

Version 3.9 fuses the clamp, squared-clipped activation, and output dot product
inside the linear NNUE head. The evaluation is score-exact to 3.8 but avoids
writing and rereading a large temporary activation buffer, allowing more search
within the same clock time. It scored 64.5/100 (+55-26=19, about +104 Elo) at
20 ms/move and 59/100 (+41-23=36, about +63 Elo) in the confirming 100-game
gate at 50 ms/move. It also converted all 128 KQK/KRK tests at 20 ms/move,
averaging 15.8 plies with no draws or illegal moves. Using the more conservative
confirmation result, the accumulation ledger is now +80/150 Elo. Stockfish 8
gating remains deferred until approximately +150 Elo. **The final requirement
to beat Stockfish 8 and verify roughly 3500 Elo remains unproven.**

Version 4.0 verifies successful null-move cutoffs at depth 12 and above by
searching the real position with another null move disabled. This reduces
false cutoffs in tactically unstable and zugzwang-like positions while leaving
the common shallow path unchanged. It scored 53.5/100 (+30-23=47, about +24
Elo) against 3.9 in the original wall-clock gate after an independent 11.5/20
screening win. A later, more reproducible 100-game audit using 50 diversified
color-reversed opening pairs and 20,000 nodes/move finished exactly 50/100
(+41-41=18). Version 4.0 remains champion for its additional cutoff safety,
but the unconfirmed +24 is excluded from the conservative ledger, which remains
+80/150 Elo. Stockfish 8 gating stays deferred until approximately +150 Elo.
**The final requirement to beat Stockfish 8 and verify roughly 3500 Elo remains
unproven.**

Version 4.1 gives materially decisive sparse endings a larger but bounded UCI
clock budget: with ten or fewer pieces and an absolute evaluation of at least
400 cp, it may use 87.5% of the increment while retaining 10% of the remaining
clock. All other positions keep 4.0's 75% increment use and 20% reserve. It
scored 11/20 (+9-7=4, about +35 Elo) in screening and 53/100 (+32-26=42,
about +21 Elo) in the authoritative 0.25s+0.10s gate against 4.0. The extreme
gate included two candidate time forfeits and none for 4.0, so the change stays
strictly limited to already decisive sparse endings. The conservative
accumulation ledger is now +101/150 Elo. Stockfish 8 gating remains deferred
until approximately +150 Elo. **The final requirement to beat Stockfish 8 and
verify roughly 3500 Elo remains unproven.**

Version 3.3 upgrades the evaluator to squared-clipped NNUE activation and was
trained on 8,462,039 combined Stockfish-18-labelled positions. It preserves
the incremental H256 transformer and compact phase/pawn output while adding
nonlinearity at substantially lower cost than a multilayer head. In a 100-game
candidate-versus-3.2 gate at 0.25s+0.10s, it scored 55/100 (+34-24=42, about
+35 Elo). The champion chain is now about +63 measured Elo over 3.0, leaving
about +137 Elo before Stockfish gates resume. **It still does not beat
Stockfish 8 or verify the requested 3500-Elo target.**

Version 3.4 fine-tunes the squared-clipped evaluator on 11,117,664 positions:
deeper Stockfish-18 relabels of the original corpus, independent teacher games,
and SEE-fixed self-play trajectories. In a 100-game candidate-versus-3.3 gate
at 0.25s+0.10s, it scored 60/100 (+40-20=40, about +70 Elo). The champion
chain is now about +133 measured Elo over 3.0, leaving about +67 Elo before
Stockfish gates resume. **It still does not beat Stockfish 8 or verify the
requested 3500-Elo target.**

## Build from source

The networks are embedded in `src/nnue_net.h` and `src/policy_net.h`, which are
committed to this repository. No network download, Python step, or external
dependency is required to build — a C++20 compiler is sufficient.

Recommended, for modern CPUs (AVX2 / BMI2, Haswell 2013 and later):

```bash
g++ -O3 -march=x86-64-v3 -flto -static -std=c++20 -pthread -DNDEBUG -o NARC-Next src/main.cpp
```

Compatibility build, for older CPUs (SSE4.2 / POPCNT):

```bash
g++ -O3 -march=x86-64-v2 -flto -static -std=c++20 -pthread -DNDEBUG -o NARC-Next src/main.cpp
```

The evaluation contains no hand-written SIMD intrinsics and relies on compiler
auto-vectorisation, so both targets are functionally identical and differ only
in speed. Builds are deterministic: at a fixed depth the engine reports the
same node counts across builds.

### Windows helper script

`build.ps1` reproduces the author's build. It additionally regenerates the two
embedded-network headers from `networks/` using the Python scripts in `tests/`,
so it requires Python and an MSYS2 UCRT64 toolchain at `C:\msys64\ucrt64\bin`.
It compiles with `-march=native`; do not distribute a `-march=native` binary,
as it may not run on other machines.

## Verifying a build

```bash
printf 'uci\nisready\nposition startpos\ngo depth 12\n' | ./NARC-Next
```

Note that the search runs on a background thread. Sending `quit` immediately
after `go` aborts the search before it completes, and the engine will return
its first legal move. Allow the search to finish, as any UCI GUI does.

## Arena

In Arena, choose **Engines > Install New Engine**, select
`dist\NARC-Next.exe`, and choose the UCI protocol. Start with `Threads=1` and
`Hash=256` MB for the tested configuration. Available UCI options are `Hash`,
`Threads`, `Move Overhead`, `EvalFile`, `Use NNUE`, and `Use Policy`.

## License

GNU GPL v3 or later. See `LICENSE`.
