// SPDX-License-Identifier: GPL-3.0-or-later
// NARC Engine — Copyright (C) 2026 Madushan Dissanayake. GNU GPL v3+; see LICENSE.
//
// NARC Engine — NNUE evaluation: (768 -> 128)x2 perspective net, int16 inference.
// The accumulator lives in Position and is updated incrementally by make/unmake
// (see nnue_params.h for the parameter store and accumulator primitives).
//
// score_cp = (sum crelu(acc_stm)*W1[0..H-1] + sum crelu(acc_nstm)*W1[H..2H-1]
//             + b1) * SCALE / (QA * QB)
//
// File format "NARCNET7": i32 hidden, i32 output buckets, i32 pawn features,
// i16 W0[768*H], i16 b0[H], bucketed i16 W1 and i32 b1.
#pragma once
#include <bit>
#include "eval.h"
#include "nnue_net.h"
#include "policy_net.h"
#include <fstream>

namespace nnue {

inline bool loadFromMemory(const unsigned char* d, size_t n) {
    const bool mlp = n >= 8 && memcmp(d, "NARCNT10", 8) == 0;
    const bool squared = n >= 8 && memcmp(d, "NARCNT11", 8) == 0;
    const bool materialPhase = n >= 8 && memcmp(d, "NARCNT12", 8) == 0;
    const bool majorPiecesWithPawns = n >= 8 && memcmp(d, "NARCNT14", 8) == 0;
    const bool majorPiecesPawnFree = n >= 8 && memcmp(d, "NARCNT15", 8) == 0;
    const bool majorPieces = majorPiecesWithPawns || majorPiecesPawnFree;
    const bool linear = n >= 8 && memcmp(d, "NARCNET7", 8) == 0;
    if (!mlp && !linear && !squared && !materialPhase && !majorPieces) {
        loaded = false; return false;
    }
    const size_t need = 8 + 4 + 4 + 4 + (mlp ? 4 : 0) + sizeof(W0) + sizeof(b0)
                      + (mlp ? sizeof(W1M) + sizeof(b1M) + sizeof(W2M) + sizeof(b2M)
                             : sizeof(W1) + sizeof(b1));
    if (n < need) { loaded = false; return false; }
    d += 8;
    int32_t h;
    memcpy(&h, d, 4); d += 4;
    if (h != H) { loaded = false; return false; } // net must match compiled size
    int32_t buckets;
    memcpy(&buckets, d, 4); d += 4;
    if (buckets != OUTPUT_BUCKETS) { loaded = false; return false; }
    int32_t pawnFeatures;
    memcpy(&pawnFeatures, d, 4); d += 4;
    if (pawnFeatures != PAWN_FEATURES) { loaded = false; return false; }
    if (mlp) {
        int32_t mlpHidden;
        memcpy(&mlpHidden, d, 4); d += 4;
        if (mlpHidden != MLP_H) { loaded = false; return false; }
    }
    memcpy(W0, d, sizeof(W0)); d += sizeof(W0);
    memcpy(b0, d, sizeof(b0)); d += sizeof(b0);
    if (mlp) {
        memcpy(W1M, d, sizeof(W1M)); d += sizeof(W1M);
        memcpy(b1M, d, sizeof(b1M)); d += sizeof(b1M);
        memcpy(W2M, d, sizeof(W2M)); d += sizeof(W2M);
        memcpy(b2M, d, sizeof(b2M));
    } else {
        memcpy(W1, d, sizeof(W1)); d += sizeof(W1);
        memcpy(b1, d, sizeof(b1));
    }
    nonlinearHead = mlp;
    squaredActivation = squared || materialPhase || majorPieces;
    materialPhaseBuckets = materialPhase;
    majorPieceBuckets = majorPieces;
    loaded = true;
    return true;
}

inline bool loadEmbedded() {
    if (EMBEDDED_NET_SIZE == 0) return false;
    return loadFromMemory(EMBEDDED_NET, EMBEDDED_NET_SIZE);
}

inline bool load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { loaded = false; return false; }
    std::vector<char> buf((std::istreambuf_iterator<char>(f)),
                          std::istreambuf_iterator<char>());
    return loadFromMemory(reinterpret_cast<const unsigned char*>(buf.data()), buf.size());
}

inline void extractPawnFeatures(const Position& pos, int color,
                                int32_t out[PAWN_FEATURE_STORAGE]) {
    if constexpr (PAWN_FEATURES == 0) {
        (void)pos; (void)color; (void)out;
        return;
    }
    std::fill(out, out + PAWN_FEATURES, 0);
    const u64 pawns = pos.pieces(color, PAWN);
    const u64 enemyPawns = pos.pieces(color ^ 1, PAWN);
    const u64 occ = pos.occupied();
    int fileCounts[8] = {};
    u64 scan = pawns;
    while (scan) ++fileCounts[fileOf(poplsb(scan))];

    int isolated = 0, connected = 0, passed = 0, center = 0;
    int blocked = 0, phalanx = 0, shield = 0;
    int passedRanks[6] = {};
    const int king = pos.kingSq(color);
    scan = pawns;
    while (scan) {
        const int sq = poplsb(scan), file = fileOf(sq), rank = rankOf(sq);
        const int relativeRank = color == WHITE ? rank : 7 - rank;
        bool hasAdjacentFile = (file > 0 && fileCounts[file - 1])
                            || (file < 7 && fileCounts[file + 1]);
        if (!hasAdjacentFile) ++isolated;

        bool hasPhalanx = false, hasConnection = false;
        u64 friendly = pawns;
        while (friendly) {
            const int other = poplsb(friendly);
            if (std::abs(fileOf(other) - file) != 1) continue;
            const int otherRank = rankOf(other);
            if (otherRank == rank) hasPhalanx = true;
            const int supportRank = color == WHITE ? rank - 1 : rank + 1;
            if (otherRank == rank || otherRank == supportRank) hasConnection = true;
        }
        phalanx += hasPhalanx;
        connected += hasConnection;

        bool enemyAhead = false;
        u64 enemies = enemyPawns;
        while (enemies) {
            const int other = poplsb(enemies);
            if (std::abs(fileOf(other) - file) > 1) continue;
            if (color == WHITE ? rankOf(other) > rank : rankOf(other) < rank) {
                enemyAhead = true;
                break;
            }
        }
        if (!enemyAhead) {
            ++passed;
            ++passedRanks[std::clamp(relativeRank - 1, 0, 5)];
        }
        if (file >= 2 && file <= 5 && rank >= 2 && rank <= 5) ++center;
        const int ahead = sq + (color == WHITE ? 8 : -8);
        if (ahead >= 0 && ahead < 64 && (occ & bit(ahead))) ++blocked;
        const int kingFile = fileOf(king), kingRank = rankOf(king);
        const int kingDistance = color == WHITE ? rank - kingRank : kingRank - rank;
        if (std::abs(file - kingFile) <= 1 && (kingDistance == 1 || kingDistance == 2)) ++shield;
    }
    int doubled = 0, islands = 0;
    for (int file = 0; file < 8; ++file) {
        doubled += std::max(0, fileCounts[file] - 1);
        if (fileCounts[file] && (file == 0 || !fileCounts[file - 1])) ++islands;
    }
    out[0] = popcount(pawns); out[1] = isolated; out[2] = doubled;
    out[3] = connected; out[4] = passed;
    for (int i = 0; i < 6; ++i) out[5 + i] = passedRanks[i];
    out[11] = islands; out[12] = shield; out[13] = center;
    out[14] = blocked; out[15] = phalanx;
}

struct PawnFeatureCacheEntry {
    u64 key = 0;
    int8_t values[2][PAWN_FEATURE_STORAGE] = {};
};
inline thread_local PawnFeatureCacheEntry pawnFeatureCache[4096];

inline void cachedPawnFeatures(const Position& pos,
                               int32_t out[2][PAWN_FEATURE_STORAGE]) {
    if constexpr (PAWN_FEATURES == 0) {
        (void)pos; (void)out;
        return;
    }
    const u64 whitePawns = pos.pieces(WHITE, PAWN);
    const u64 blackPawns = pos.pieces(BLACK, PAWN);
    const u64 key = (whitePawns * 0x9E3779B97F4A7C15ULL)
                  ^ std::rotl(blackPawns, 23)
                  ^ (u64(pos.kingSq(WHITE)) << 1)
                  ^ (u64(pos.kingSq(BLACK)) << 8)
                  ^ 0xD1B54A32D192ED03ULL;
    PawnFeatureCacheEntry& entry = pawnFeatureCache[key & 4095];
    if (entry.key != key) {
        int32_t computed[2][PAWN_FEATURE_STORAGE];
        extractPawnFeatures(pos, WHITE, computed[WHITE]);
        extractPawnFeatures(pos, BLACK, computed[BLACK]);
        entry.key = key;
        for (int c = WHITE; c <= BLACK; ++c)
            for (int i = 0; i < PAWN_FEATURES; ++i)
                entry.values[c][i] = int8_t(computed[c][i]);
    }
    for (int c = WHITE; c <= BLACK; ++c)
        for (int i = 0; i < PAWN_FEATURES; ++i)
            out[c][i] = entry.values[c][i];
}

inline int eval(const Position& pos) {
    int stm = pos.stm, nstm = stm ^ 1;
    int phase;
    if (majorPieceBuckets) {
        const int queens = popcount(pos.byPiece[QUEEN]);
        const int rooks = popcount(pos.byPiece[ROOK]);
        if (queens == 0) phase = std::min(2, rooks);
        else if (queens == 1) phase = 3 + std::min(1, rooks);
        else phase = 5 + (rooks == 0 ? 0 : rooks <= 2 ? 1 : 2);
    } else if (materialPhaseBuckets) {
        phase = popcount(pos.byPiece[KNIGHT] | pos.byPiece[BISHOP])
              + 2 * popcount(pos.byPiece[ROOK])
              + 4 * popcount(pos.byPiece[QUEEN]);
        phase = std::clamp(phase * OUTPUT_BUCKETS / 25, 0, OUTPUT_BUCKETS - 1);
    } else {
        const int nonKings = popcount(pos.occupied()) - 2;
        phase = std::clamp(nonKings * OUTPUT_BUCKETS / 31, 0, OUTPUT_BUCKETS - 1);
    }
    const int bucket = phase;
    constexpr int OUTPUT_WIDTH = 2 * H + 2 * PAWN_FEATURES;
    int32_t byColorPawnData[2][PAWN_FEATURE_STORAGE] = {};
    if constexpr (PAWN_FEATURES > 0)
        cachedPawnFeatures(pos, byColorPawnData);

    if (nonlinearHead) {
        int32_t joined[OUTPUT_WIDTH];
        for (int i = 0; i < H; i++)
            joined[i] = std::clamp(int32_t(pos.acc[stm][i]), 0, QA);
        for (int i = 0; i < H; i++)
            joined[H + i] = std::clamp(int32_t(pos.acc[nstm][i]), 0, QA);
        for (int side = 0; side < 2; ++side)
            for (int i = 0; i < PAWN_FEATURES; ++i) {
                const int color = side == 0 ? stm : nstm;
                joined[2 * H + side * PAWN_FEATURES + i]
                    = (byColorPawnData[color][i] * QA + 4) / 8;
            }
        if (squaredActivation)
            for (int i = 0; i < 2 * H; ++i)
                joined[i] = (joined[i] * joined[i] + QA / 2) / QA;

        const int16_t* first = &W1M[bucket * MLP_H * OUTPUT_WIDTH];
        int32_t activation[MLP_H];
        for (int unit = 0; unit < MLP_H; ++unit) {
            int32_t sum = b1M[bucket * MLP_H + unit];
            const int16_t* weights = first + unit * OUTPUT_WIDTH;
            for (int i = 0; i < OUTPUT_WIDTH; ++i)
                sum += joined[i] * int32_t(weights[i]);
            activation[unit] = std::clamp(sum, 0, QA * QB);
        }
        int64_t out = b2M[bucket];
        const int16_t* second = &W2M[bucket * MLP_H];
        for (int unit = 0; unit < MLP_H; ++unit)
            out += int64_t(activation[unit]) * second[unit];
        int score = int(out * SCALE / (int64_t(QA) * QB * QC));
        return std::clamp(score, -MATE_BOUND + 100, MATE_BOUND - 100);
    }

    const int16_t* output = &W1[bucket * OUTPUT_WIDTH];
    int32_t s0 = 0, s1 = 0; // int32 partial sums so the loops vectorize
    if (squaredActivation) {
        // Fuse clamp, SCReLU and the output dot product. This is score-exact
        // but avoids writing and rereading a 2*H int32 scratch buffer.
        for (int i = 0; i < H; ++i) {
            const int32_t value = std::clamp(int32_t(pos.acc[stm][i]), 0, QA);
            const int32_t activated = (value * value + QA / 2) / QA;
            s0 += activated * int32_t(output[i]);
        }
        for (int i = 0; i < H; ++i) {
            const int32_t value = std::clamp(int32_t(pos.acc[nstm][i]), 0, QA);
            const int32_t activated = (value * value + QA / 2) / QA;
            s1 += activated * int32_t(output[H + i]);
        }
    } else {
        for (int i = 0; i < H; ++i) {
            const int32_t value = std::clamp(int32_t(pos.acc[stm][i]), 0, QA);
            s0 += value * int32_t(output[i]);
        }
        for (int i = 0; i < H; ++i) {
            const int32_t value = std::clamp(int32_t(pos.acc[nstm][i]), 0, QA);
            s1 += value * int32_t(output[H + i]);
        }
    }
    int64_t pawnTerm = 0;
    for (int side = 0; side < 2; ++side)
        for (int i = 0; i < PAWN_FEATURES; ++i) {
            const int color = side == 0 ? stm : nstm;
            const int32_t value = (byColorPawnData[color][i] * QA + 4) / 8;
            pawnTerm += int64_t(value)
                      * output[2 * H + side * PAWN_FEATURES + i];
        }
    int64_t out = int64_t(s0) + s1 + pawnTerm + b1[bucket];

    int score = int(out * SCALE / (QA * QB));
    return std::clamp(score, -MATE_BOUND + 100, MATE_BOUND - 100);
}

} // namespace nnue

namespace policy {

constexpr int WIDTH = 2 * nnue::H;
constexpr int LABELS = 64 * 64;
inline const int16_t* weights = nullptr;
inline const int32_t* biases = nullptr;
inline int quant = 1;
inline bool loaded = false;
inline bool enabled = false;

inline bool loadEmbedded() {
    constexpr size_t header = 8 + 3 * sizeof(int32_t);
    constexpr size_t need = header
        + size_t(LABELS) * WIDTH * sizeof(int16_t)
        + size_t(LABELS) * sizeof(int32_t);
    if (EMBEDDED_POLICY_SIZE != need
        || memcmp(EMBEDDED_POLICY, "NARCPOL2", 8) != 0) {
        loaded = false;
        return false;
    }
    const unsigned char* data = EMBEDDED_POLICY + 8;
    int32_t width, labels, fileQuant;
    memcpy(&width, data, 4); data += 4;
    memcpy(&labels, data, 4); data += 4;
    memcpy(&fileQuant, data, 4); data += 4;
    if (width != WIDTH || labels != LABELS || fileQuant <= 0) {
        loaded = false;
        return false;
    }
    weights = reinterpret_cast<const int16_t*>(data);
    data += size_t(LABELS) * WIDTH * sizeof(int16_t);
    biases = reinterpret_cast<const int32_t*>(data);
    quant = fileQuant;
    loaded = true;
    return true;
}

inline int score(const Position& pos, Move move) {
    if (!loaded || !enabled || !nnue::active()) return 0;
    int from = fromSq(move), to = toSq(move);
    if (pos.stm == BLACK) { from ^= 56; to ^= 56; }
    int label = from * 64 + to;
    const int16_t* row = weights + size_t(label) * WIDTH;
    int64_t sum = int64_t(biases[label]) * nnue::QA;
    for (int i = 0; i < nnue::H; ++i)
        sum += int64_t(std::clamp(int(pos.acc[pos.stm][i]), 0, nnue::QA)) * row[i];
    for (int i = 0; i < nnue::H; ++i)
        sum += int64_t(std::clamp(int(pos.acc[pos.stm ^ 1][i]), 0, nnue::QA))
             * row[nnue::H + i];
    return int(sum / nnue::QA);
}

} // namespace policy

// Give simplified major-piece endings a smooth conversion gradient: drive the
// weaker king to the edge and bring the stronger king closer. Restrict the
// term to positions where the defender has no pieces and at most two pawns.
inline int endgameConversionBonus(const Position& pos) {
    constexpr int materialValue[6] = { 100, 320, 330, 500, 900, 0 };
    int material[2] = {};
    for (int c = WHITE; c <= BLACK; ++c)
        for (int pt = PAWN; pt <= QUEEN; ++pt)
            material[c] += popcount(pos.pieces(c, pt)) * materialValue[pt];

    int strong = material[WHITE] >= material[BLACK] ? WHITE : BLACK;
    int weak = strong ^ 1;
    if (material[strong] - material[weak] < 400) return 0;
    if (!(pos.pieces(strong, ROOK) | pos.pieces(strong, QUEEN))) return 0;
    if (pos.pieces(weak, KNIGHT) | pos.pieces(weak, BISHOP)
        | pos.pieces(weak, ROOK) | pos.pieces(weak, QUEEN)) return 0;
    if (popcount(pos.pieces(weak, PAWN)) > 2 || popcount(pos.occupied()) > 6) return 0;

    const int weakKing = pos.kingSq(weak), strongKing = pos.kingSq(strong);
    const int edgeDistance = std::min({fileOf(weakKing), 7 - fileOf(weakKing),
                                       rankOf(weakKing), 7 - rankOf(weakKing)});
    const int kingDistance = std::max(std::abs(fileOf(strongKing) - fileOf(weakKing)),
                                     std::abs(rankOf(strongKing) - rankOf(weakKing)));
    // In bare-major endings this term must dominate small NNUE fluctuations
    // at very shallow depth; otherwise the winning side can repeat harmless
    // moves until the fifty-move draw.  Keep the same smooth geometry while
    // giving edge confinement and king approach a decisive gradient.
    const int bonus = (3 - edgeDistance) * 80 + (7 - kingDistance) * 30;
    const int whiteScore = strong == WHITE ? bonus : -bonus;
    return pos.stm == WHITE ? whiteScore : -whiteScore;
}

// evaluation dispatch: NNUE when a net is loaded and enabled, else classical
inline int evaluate(const Position& pos) {
    const int base = nnue::active() ? nnue::eval(pos) : evaluateHCE(pos);
    return std::clamp(base + endgameConversionBonus(pos),
                      -MATE_BOUND + 100, MATE_BOUND - 100);
}
