<div align="center">

<img src="assets/logo.png" alt="NARC Next" width="280">

# NARC Next

**C++20 UCI chess engine with an integer NNUE evaluator**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Release](https://img.shields.io/github/v/release/Madushan996/NARC-NEXT)](https://github.com/Madushan996/NARC-NEXT/releases)

</div>

| | |
|---|---|
| **Version** | 4.1 |
| **Author** | Madushan Dissanayake |
| **Protocol** | UCI |
| **Language** | C++20, single translation unit |
| **Platform** | x86-64 (Windows binaries provided; source is portable) |
| **Evaluation** | Embedded NNUE, integer inference |
| **Parallelism** | Lazy SMP, 1–256 threads |
| **License** | GNU GPL v3 or later |

Download: [**Releases**](https://github.com/Madushan996/NARC-NEXT/releases) ·
Development history: [**CHANGELOG.md**](CHANGELOG.md)

---

## Board representation

| Component | Implementation |
|---|---|
| Occupancy | Bitboards — `byColor[2]`, `byPiece[6]`, plus a `board[64]` piece-type array |
| Sliding attacks | Magic bitboards (102,400-entry rook table, 5,248-entry bishop table), magics found at startup |
| Leaper attacks | Precomputed pawn, knight and king tables |
| Move encoding | 16-bit — from (6 bits), to (6), promotion piece (2), flag (2) |
| Move generation | Pseudo-legal, bulk-generated (all moves, or captures and promotions only); legality established by make plus a king-attack test |
| Hashing | Zobrist, seeded from an xorshift\* PRNG |
| Max depth | 128 ply |

`go perft <n>` runs a perft from the current position.

## Search

Principal variation search with iterative deepening and aspiration windows
(initial delta 25, widened by 1.5× on each fail).

### Transposition table

Four-entry clusters with depth-aware replacement. Each entry packs move (16
bits), score (16), static eval (16), depth (8) and bound (8) into a single
64-bit word, stored alongside `key ^ data` — a torn read under a data race
fails the key comparison and is treated as a miss rather than corrupting play.
Size is set by the `Hash` option and is shared across all threads.

### Pruning and reductions

| Technique | Condition |
|---|---|
| Null-move pruning | depth ≥ 3, `R = 3 + depth/3 + min(3, (eval−beta)/200)`; cutoffs at depth ≥ 12 confirmed by a verification search |
| Reverse futility | depth ≤ 8, margin `80 × depth`, relaxed by 60 when improving |
| Razoring | depth ≤ 3, margin `200 + 150 × depth`, verified by quiescence |
| ProbCut | depth ≥ 5, `rBeta = beta + 180`, up to 6 SEE-qualifying captures |
| Internal iterative reduction | depth ≥ 4 with no TT move |
| Singular extensions | depth ≥ 8, `sBeta = ttScore − 2 × depth` at half depth; also yields a multi-cut return |
| Late move reductions | `0.75 + ln(d)·ln(m)/2.25`, adjusted by PV node, improving, killers, history/8192, and sparse decisive endings |
| Late move pruning | depth ≤ 8, limit `4 + d²` when improving, else `2 + d²/2` |
| Quiet futility | depth ≤ 4, margin `150 + 120 × depth` |
| History pruning | depth ≤ 4, threshold `−4000 × depth` |
| Capture SEE pruning | depth ≤ 8, threshold `−150 × depth` |

### Quiescence

Captures and promotions, with full move generation when in check. Filters
losing captures by SEE and applies a 150 cp delta margin. Repetition,
fifty-move and insufficient-material draws are detected here on the same terms
as in the main search.

### Move ordering

TT move → good captures (SEE ≥ 0) → killers → countermove → quiets → bad
captures. Captures are scored by MVV/LVA plus capture history; quiets by
butterfly history plus continuation history at 1, 2 and 4 plies back. All
history tables use gravity updates, `h += bonus − h·|bonus|/16384`, with
`bonus = min(2000, 16·depth²)`.

### Parallel search

Lazy SMP over a shared transposition table. Helper threads on odd indices begin
one iteration deeper to desynchronise their trees; the deepest completed
iteration wins on disagreement. SMP is skipped when the hard time limit is
under 500 ms, since thread startup would dominate the budget.

## Evaluation

A `768 → 256` perspective network, evaluated in integers with an accumulator
updated incrementally by make/unmake.

| Property | Value |
|---|---|
| Input features | 768 — `(colour × 6 + piece type) × 64 + square` |
| Perspective | Both maintained; the black view flips colour and mirrors rank |
| Hidden width | 256 per perspective |
| Activation | Squared clipped ReLU (SCReLU) |
| Output buckets | 8, selected by major-piece configuration (queen and rook counts) |
| Auxiliary inputs | 16 pawn-structure features per side, appended at the output layer |
| Arithmetic | int16 weights, int32 accumulation, `QA = 255`, `QB = 64`, scale 400 |

Pawn features cover count, isolated, doubled, connected, passed (by rank),
islands, king shield, centre occupancy, blocked and phalanx pawns. They are
cached in a 4,096-entry thread-local table keyed on both pawn sets and both
king squares.

Bare-major endings receive a geometric conversion term that drives the weaker
king toward the edge and the stronger king closer, bounded to positions where
the defender has no pieces and at most two pawns.

The network is trained by knowledge distillation on positions labelled with
Stockfish 18 evaluations; the training pipeline is in `modal/`.

### Classical fallback

With `Use NNUE=false`, evaluation falls back to a tapered hand-crafted
function: PeSTO piece-square tables (Ronald Friederich, credited in
`src/eval.h`), per-piece mobility, rook file bonuses, bishop pair, passed,
isolated and doubled pawns, and a tempo term.

### Optional root policy

A `512 → 4096` linear policy head scores root moves from the accumulator, used
as a move-ordering prior. It is **disabled by default** (`Use Policy=false`),
as unassisted search measured stronger.

## Time management

`movetime` is treated as an exact analysis budget with no overhead deduction.
For clock-based control the engine estimates moves remaining as
`clamp(50 − fullmove/2, 20, 40)` and derives a soft limit at ¾ of the per-move
base, with a hard cap of 3× base bounded by a safety reserve. The soft limit
extends when a deeper iteration reveals a score drop and contracts as the best
move stabilises. In materially decisive endings — ten or fewer pieces and an
absolute evaluation of at least 400 cp — the engine may spend 87.5% of the
increment while retaining 10% of the clock; otherwise 75% and 20%.

## UCI options

| Option | Type | Default | Range |
|---|---|---|---|
| `Hash` | spin | 128 | 1–4096 MB |
| `Threads` | spin | 1 | 1–256 |
| `Move Overhead` | spin | 30 | 0–1000 ms |
| `EvalFile` | string | `<embedded>` | path to a `.nnue` file |
| `Use NNUE` | check | `true` | |
| `Use Policy` | check | `false` | |

The engine has no opening book, does not learn between games, and writes no
persistent state. `EvalFile` is the only way to load an external network; the
built-in one is used otherwise, so play is deterministic by default.

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
in speed.

### Windows helper script

`build.ps1` reproduces the author's build. It additionally regenerates the two
embedded-network headers from `networks/` using the Python scripts in `tests/`,
so it requires Python and an MSYS2 UCRT64 toolchain at `C:\msys64\ucrt64\bin`.
It compiles with `-march=native`; do not distribute a `-march=native` binary,
as it may not run on other machines.

## Verifying a build

```bash
printf 'uci\nisready\nposition startpos\ngo depth 13\n' | ./NARC-Next
```

Expected at depth 13 from the start position: `score cp 37`, `nodes 168609`,
`bestmove e2e4`. Node counts are identical across builds, so results are
reproducible.

Note that the search runs on a background thread. Sending `quit` immediately
after `go` aborts it before completion, and the engine returns its first legal
move. Allow the search to finish, as any UCI GUI does.

## Arena

Choose **Engines > Install New Engine**, select the executable, and pick the
UCI protocol. `Threads=1` with `Hash=256` MB matches the tested configuration.

## Acknowledgments

**OpenAI Codex** contributed substantially to this engine's playing strength,
working on search-parameter tuning, evaluation and network-architecture
iteration, and performance optimisation across the version history in
[CHANGELOG.md](CHANGELOG.md). Much of the measured Elo gain from 3.0 onward
came out of that work.

Authorship, design direction, and every gating decision — which candidate
became champion, and on what evidence — remain those of Madushan Dissanayake,
who is named as the engine's sole author accordingly.

## License

GNU GPL v3 or later. See [`LICENSE`](LICENSE).
