// SPDX-License-Identifier: GPL-3.0-or-later
// NARC Engine — Copyright (C) 2026 Madushan Dissanayake. GNU GPL v3+; see LICENSE.
//
// NARC Engine — NNUE parameters and accumulator primitives.
// Included before Position so make/unmake can update accumulators incrementally.
#pragma once
#include "types.h"

namespace nnue {

#ifndef NARC_NNUE_HIDDEN
#define NARC_NNUE_HIDDEN 256
#endif
#ifndef NARC_NNUE_PAWN_FEATURES
#define NARC_NNUE_PAWN_FEATURES 16
#endif

constexpr int H = NARC_NNUE_HIDDEN; // must match the embedded trained net
constexpr int OUTPUT_BUCKETS = 8;
constexpr int PAWN_FEATURES = NARC_NNUE_PAWN_FEATURES;
// Standard C++ does not permit zero-length arrays. Pawn-free H512 candidates
// retain one inert storage element while compiling every pawn-feature loop out.
constexpr int PAWN_FEATURE_STORAGE = PAWN_FEATURES > 0 ? PAWN_FEATURES : 1;
constexpr int MLP_H = 32;
constexpr int QA = 255;
constexpr int QB = 64;
constexpr int QC = 64;
constexpr int SCALE = 400;

inline bool loaded = false;
inline bool enabled = true;
inline int16_t W0[768 * H]; // feature-major: column per (piece, square) feature
inline int16_t b0[H];
inline int16_t W1[OUTPUT_BUCKETS * (2 * H + 2 * PAWN_FEATURES)];
inline int32_t b1[OUTPUT_BUCKETS] = {};
// Optional nonlinear NARCNET10 head. The original NARCNET7 arrays remain so
// Arena users can still load every released evaluator through EvalFile.
inline bool nonlinearHead = false;
inline bool squaredActivation = false;
inline bool materialPhaseBuckets = false;
inline bool majorPieceBuckets = false;
inline int16_t W1M[OUTPUT_BUCKETS * MLP_H * (2 * H + 2 * PAWN_FEATURES)];
inline int32_t b1M[OUTPUT_BUCKETS * MLP_H] = {};
inline int16_t W2M[OUTPUT_BUCKETS * MLP_H];
inline int32_t b2M[OUTPUT_BUCKETS] = {};

inline bool active() { return loaded && enabled; }

// feature index from one side's perspective (black flips color and mirrors rank)
inline int featureIdx(int persp, int color, int pt, int sq) {
    if (persp == BLACK) { color ^= 1; sq ^= 56; }
    return (color * 6 + pt) * 64 + sq;
}

// accumulator ops: every piece add/remove touches both perspectives
inline void accAdd(int16_t (&acc)[2][H], int color, int pt, int sq) {
    const int16_t* cw = &W0[size_t(featureIdx(WHITE, color, pt, sq)) * H];
    const int16_t* cb = &W0[size_t(featureIdx(BLACK, color, pt, sq)) * H];
    for (int i = 0; i < H; i++) acc[WHITE][i] += cw[i];
    for (int i = 0; i < H; i++) acc[BLACK][i] += cb[i];
}
inline void accSub(int16_t (&acc)[2][H], int color, int pt, int sq) {
    const int16_t* cw = &W0[size_t(featureIdx(WHITE, color, pt, sq)) * H];
    const int16_t* cb = &W0[size_t(featureIdx(BLACK, color, pt, sq)) * H];
    for (int i = 0; i < H; i++) acc[WHITE][i] -= cw[i];
    for (int i = 0; i < H; i++) acc[BLACK][i] -= cb[i];
}

} // namespace nnue
