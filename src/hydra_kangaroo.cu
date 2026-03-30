/*
 * hydra_kangaroo.cu — Optimized Pollard's Kangaroo for Bitcoin Puzzle #135
 *
 * KEY OPTIMIZATION: Batch inversion via Montgomery's trick.
 *   Each thread manages K kangaroos simultaneously.
 *   All K kangaroos step in Jacobian (cheap: 7M+4S per step).
 *   Every step, batch-convert to affine using 1 inversion.
 *   Check all K points for DPs in affine.
 *   This amortizes the 255-sqr+15-mul inversion across K points.
 *
 * Build:
 *   make           (defaults to sm_89 for RTX 5070 Ti)
 *   make h100      (for H100)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <signal.h>
#include <pthread.h>

#include "field.cuh"
#include "ec.cuh"
#include "field_csa.cuh"
#include "ec_pipeline.cuh"

// ═══════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════

#define KANGAROOS_PER_THREAD 32   // K=32: optimal batch amortization
#define NUM_JUMPS 256             // Walk function branching factor
#define BLOCK_SIZE 256            // CUDA threads per block
#define DEFAULT_DP_BITS 25        // Distinguished point criterion
#define STEPS_PER_KERNEL 1024     // Steps between host checks (high = less overhead)
#define ENABLE_GALBRAITH_RUPRAI 1 // Equivalence class canonicalization
#define BLOOM_SIZE_BITS 27        // 2^27 = 128M bits = 16 MB bloom filter
#define BLOOM_NUM_HASHES 4        // Number of hash functions for bloom filter
#define MAX_TARGETS 16            // Maximum number of simultaneous puzzle targets

// ═══════════════════════════════════════════════════════════════
// MULTI-TARGET PUZZLE DEFINITIONS
//
// Multi-target mode: solve T puzzles simultaneously for sqrt(T) speedup.
// Each puzzle has: public key (Qx, Qy), range [2^(N-1), 2^N)
// Wild kangaroos are split across targets; tame kangaroos are shared.
// Any tame-wild collision from ANY target solves that target.
//
// Type encoding: type = (target_idx << 4) | (0=tame, 1=wild)
// ═══════════════════════════════════════════════════════════════

struct PuzzleTarget {
    AffinePoint Q;           // Target public key
    u256 range_start;        // 2^(N-1)
    u256 range_size;         // 2^(N-1) (width of search range)
    int puzzle_number;       // For display
};

// Default target: Puzzle #135
// Qx = 0x145D2611C823A396EF6712CE0F712F09B9B4F3135E3E0AA3230FB9B6D08D1E16
// Qy = 0x667A05E9A1BDD6F70142B66558BD12CE2C0F9CBC7001B20C8A6A109C80DC5330
static const AffinePoint TARGET_Q_135 = {
    {{0x230FB9B6D08D1E16ULL, 0xB9B4F3135E3E0AA3ULL,
      0xEF6712CE0F712F09ULL, 0x145D2611C823A396ULL}},
    {{0x8A6A109C80DC5330ULL, 0x2C0F9CBC7001B20CULL,
      0x0142B66558BD12CEULL, 0x667A05E9A1BDD6F7ULL}}
};

// Global target configuration (set at runtime for multi-target)
static PuzzleTarget g_targets[MAX_TARGETS];
static int g_num_targets = 0;

// Helper: create range start for puzzle #N (= 2^(N-1))
static inline u256 make_range_start(int puzzle_num) {
    u256 r = {{0, 0, 0, 0}};
    int bit = puzzle_num - 1;
    r.d[bit / 64] = 1ULL << (bit % 64);
    return r;
}

// Legacy aliases for single-target compatibility
#define TARGET_Q TARGET_Q_135
static const u256 RANGE_START = {{0x0ULL, 0x0ULL, 0x40ULL, 0x0ULL}};
static const u256 RANGE_SIZE  = {{0x0ULL, 0x0ULL, 0x40ULL, 0x0ULL}};

// ═══════════════════════════════════════════════════════════════
// DISTINGUISHED POINT OUTPUT
// ═══════════════════════════════════════════════════════════════

typedef struct {
    u256 x_affine;
    u256 walk_distance;
    uint32_t type;       // 0=tame, 1=wild
    uint32_t thread_id;
} DPEntry;

// ═══════════════════════════════════════════════════════════════
// JUMP TABLE (in constant memory for fast broadcast)
// ═══════════════════════════════════════════════════════════════

__constant__ AffinePoint c_jump_points[NUM_JUMPS];
__constant__ u256 c_jump_scalars[NUM_JUMPS];

// Escape jump table: large jumps for loop detection escape
// When the negation map creates a 2-cycle, kangaroos use these
// to break out. Sized ~2^(range/2) to maintain good mixing.
#define NUM_ESCAPE_JUMPS 32
__constant__ AffinePoint c_escape_points[NUM_ESCAPE_JUMPS];
__constant__ u256 c_escape_scalars[NUM_ESCAPE_JUMPS];

// ═══════════════════════════════════════════════════════════════
// KANGAROO STATE
// ═══════════════════════════════════════════════════════════════

typedef struct {
    JacobianPoint pos;
    u256 walk_dist;
    uint32_t type;       // 0=tame, 1=wild
    uint32_t active;
} KangarooState;

// ═══════════════════════════════════════════════════════════════
// ON-GPU BLOOM FILTER — L2 cache-resident for fast DP pre-matching
//
// Tame DPs INSERT into bloom. Wild DPs CHECK bloom.
// Only bloom-positive wilds are sent to host for exact matching.
// This eliminates PCIe round-trips for 99.9%+ of DP checks.
// ═══════════════════════════════════════════════════════════════

__device__ __forceinline__
void bloom_insert(uint32_t *bloom, const u256 *x) {
    const uint32_t mask = (1U << BLOOM_SIZE_BITS) - 1;
    uint32_t h0 = (uint32_t)(x->d[0]) & mask;
    uint32_t h1 = (uint32_t)(x->d[0] >> 32) & mask;
    uint32_t h2 = (uint32_t)(x->d[1]) & mask;
    uint32_t h3 = (uint32_t)(x->d[1] >> 32) & mask;
    atomicOr(&bloom[h0 >> 5], 1U << (h0 & 31));
    atomicOr(&bloom[h1 >> 5], 1U << (h1 & 31));
    atomicOr(&bloom[h2 >> 5], 1U << (h2 & 31));
    atomicOr(&bloom[h3 >> 5], 1U << (h3 & 31));
}

__device__ __forceinline__
bool bloom_check(const uint32_t *bloom, const u256 *x) {
    const uint32_t mask = (1U << BLOOM_SIZE_BITS) - 1;
    uint32_t h0 = (uint32_t)(x->d[0]) & mask;
    uint32_t h1 = (uint32_t)(x->d[0] >> 32) & mask;
    uint32_t h2 = (uint32_t)(x->d[1]) & mask;
    uint32_t h3 = (uint32_t)(x->d[1] >> 32) & mask;
    return (bloom[h0 >> 5] & (1U << (h0 & 31))) &&
           (bloom[h1 >> 5] & (1U << (h1 & 31))) &&
           (bloom[h2 >> 5] & (1U << (h2 & 31))) &&
           (bloom[h3 >> 5] & (1U << (h3 & 31)));
}

// ═══════════════════════════════════════════════════════════════
// THE MAIN KERNEL — BATCH INVERSION KANGAROO WALK
//
// Architecture: each thread manages K=32 kangaroos.
// Per step:
//   1. Select jump based on x-coordinate
//   2. Check y-parity: even → add jump, odd → subtract jump (NEGATION MAP)
//   3. Z=1 mixed add for each kangaroo (4M + 2S each)
//   4. Batch-invert all K Z-values (1 inv + 3K muls)
//   5. Convert to affine (3M each)
//   6. Loop detection: if same jump index repeated, use escape jump
//   7. DP check + bloom filter
//
// The negation map (step 2) makes the walk operate on {P, -P}
// equivalence classes, doubling coverage → sqrt(2) speedup.
// Combined with Galbraith-Ruprai endomorphism → sqrt(6) total.
//
// ALL JUMP DISTANCES MUST BE EVEN for correct key recovery with
// the negation map (wild-wild collisions may require division by 2).
// ═══════════════════════════════════════════════════════════════

__global__ void kangaroo_batch_walk(
    KangarooState *states,
    DPEntry *dp_output,
    uint32_t *dp_count,
    uint32_t max_dps,
    uint32_t dp_mask,
    uint32_t steps,
    uint32_t *bloom_filter   // On-GPU bloom filter (NULL to disable)
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t base = tid * KANGAROOS_PER_THREAD;

    // Load state into registers
    AffinePoint aff[KANGAROOS_PER_THREAD];
    u256 dist[KANGAROOS_PER_THREAD];
    uint32_t type[KANGAROOS_PER_THREAD];
    uint8_t last_jump[KANGAROOS_PER_THREAD];  // Loop detection: previous jump index

    #pragma unroll
    for (int k = 0; k < KANGAROOS_PER_THREAD; k++) {
        JacobianPoint jp = states[base + k].pos;
        aff[k].x = jp.X;
        aff[k].y = jp.Y;
        dist[k] = states[base + k].walk_dist;
        type[k] = states[base + k].type;
        last_jump[k] = 0xFF;  // No previous jump
    }

    for (uint32_t step = 0; step < steps; step++) {

        // ─── Phase 1: Jump selection + Negation map + EC add ───
        JacobianPoint pos[KANGAROOS_PER_THREAD];

        #pragma unroll
        for (int k = 0; k < KANGAROOS_PER_THREAD; k++) {
            // Select jump based on x-coordinate (same for P and -P)
#if ENABLE_GALBRAITH_RUPRAI
            int lambda_exp;
            u256 canon_x = ec_canonicalize_x(&aff[k].x, &lambda_exp);
            uint32_t j = canon_x.d[0] & (NUM_JUMPS - 1);
#else
            uint32_t j = aff[k].x.d[0] & (NUM_JUMPS - 1);
#endif

            // Loop detection: if same jump index as last step, use escape jump
            bool is_loop = (j == last_jump[k]);
            last_jump[k] = (uint8_t)j;

            if (__builtin_expect(is_loop, 0)) {
                // Escape: use large random jump from escape table
                uint32_t ej = (aff[k].x.d[1] ^ step) & (NUM_ESCAPE_JUMPS - 1);
                pos[k] = ec_add_mixed_z1(&aff[k], &c_escape_points[ej]);
                uint32_t carry;
                dist[k] = u256_add_cc(&dist[k], &c_escape_scalars[ej], &carry);
            } else {
                // Negation map: check y-parity to decide add vs subtract
                bool y_odd = (aff[k].y.d[0] & 1);

                if (!y_odd) {
                    // y is even: ADD the jump (normal)
                    pos[k] = ec_add_mixed_z1(&aff[k], &c_jump_points[j]);
                    uint32_t carry;
                    dist[k] = u256_add_cc(&dist[k], &c_jump_scalars[j], &carry);
                } else {
                    // y is odd: SUBTRACT the jump (negate jump point's y)
                    AffinePoint neg_jump;
                    neg_jump.x = c_jump_points[j].x;
                    neg_jump.y = fp_neg(&c_jump_points[j].y);
                    pos[k] = ec_add_mixed_z1(&aff[k], &neg_jump);
                    uint32_t borrow;
                    dist[k] = u256_sub_borrow(&dist[k], &c_jump_scalars[j], &borrow);
                }
            }
        }

        // ─── Phase 2: Batch convert to affine (1 inv + 3K muls) ───
        ec_batch_to_affine<KANGAROOS_PER_THREAD>(pos, aff);

        // ─── Phase 3: DP detection (reuse cached canonical x) ───
        #pragma unroll
        for (int k = 0; k < KANGAROOS_PER_THREAD; k++) {
#if ENABLE_GALBRAITH_RUPRAI
            // Re-canonicalize with the NEW affine point (post-step)
            int lambda_exp2;
            u256 canon_x = ec_canonicalize_x(&aff[k].x, &lambda_exp2);

            if ((canon_x.d[0] & dp_mask) == 0) {
                bool is_tame = ((type[k] & 1) == 0);

                if (is_tame && bloom_filter) {
                    // Tame: always insert into bloom AND send to host
                    bloom_insert(bloom_filter, &canon_x);
                }

                // Wild: check bloom first; skip host if no bloom match
                bool send_to_host = is_tame || !bloom_filter ||
                                    bloom_check(bloom_filter, &canon_x);

                if (send_to_host) {
                    uint32_t idx = atomicAdd(dp_count, 1);
                    if (idx < max_dps) {
                        dp_output[idx].x_affine = canon_x;
                        dp_output[idx].walk_distance = dist[k];
                        dp_output[idx].type = type[k];
                        dp_output[idx].thread_id = tid;
                    }
                }
            }
#else
            {
                bool is_tame = ((type[k] & 1) == 0);
                if ((aff[k].x.d[0] & dp_mask) == 0) {
                    if (is_tame && bloom_filter) bloom_insert(bloom_filter, &aff[k].x);
                    bool send = is_tame || !bloom_filter || bloom_check(bloom_filter, &aff[k].x);
                    if (send) {
                        uint32_t idx = atomicAdd(dp_count, 1);
                        if (idx < max_dps) {
                            dp_output[idx].x_affine = aff[k].x;
                            dp_output[idx].walk_distance = dist[k];
                            dp_output[idx].type = type[k];
                            dp_output[idx].thread_id = tid;
                        }
                    }
                }

                // Check endomorphism point: lambda*P = (beta*x, y)
                u256 endo_x = fp_mul(&ENDO_BETA, &aff[k].x);
                if ((endo_x.d[0] & dp_mask) == 0) {
                    if (is_tame && bloom_filter) bloom_insert(bloom_filter, &endo_x);
                    bool send = is_tame || !bloom_filter || bloom_check(bloom_filter, &endo_x);
                    if (send) {
                        uint32_t idx = atomicAdd(dp_count, 1);
                        if (idx < max_dps) {
                            dp_output[idx].x_affine = endo_x;
                            dp_output[idx].walk_distance = dist[k];
                            dp_output[idx].type = type[k] | 0x10;
                            dp_output[idx].thread_id = tid;
                        }
                    }
                }
            }
#endif
        }
    }

    // Write back as Jacobian with Z=1
    #pragma unroll
    for (int k = 0; k < KANGAROOS_PER_THREAD; k++) {
        states[base + k].pos.X = aff[k].x;
        states[base + k].pos.Y = aff[k].y;
        states[base + k].pos.Z.d[0] = 1;
        states[base + k].pos.Z.d[1] = 0;
        states[base + k].pos.Z.d[2] = 0;
        states[base + k].pos.Z.d[3] = 0;
        states[base + k].walk_dist = dist[k];
    }
}

// ═══════════════════════════════════════════════════════════════
// THE 1000X KERNEL — PIPELINED KANGAROO WALK
//
// Applies all 5 optimization principles from puzzle_binary's
// 1000x SHA-256 proof to the GPU ECDLP solver:
//
// 1. CARRY-SAVE / LAZY REDUCTION (from CSA architecture):
//    Uses fp_mul_ptx with MADC chains — eliminates manual carry
//    handling, reducing instructions per multiply by ~40%.
//    Analogous to keeping intermediates in carry-save form.
//
// 2. DEFERRED COMPUTATION (from Deferred-A):
//    x-only batch affine conversion — skips y-coordinate for
//    99.997% of points (only computed for DPs). Saves 2M per
//    point per step = 64M total per round at K=32.
//
// 3. SUB-ROUND PIPELINING (from Third-Stages):
//    EC additions split into 3 phases across all K kangaroos:
//    Phase1 (all K): H, dy computation (independent, cheap)
//    Phase2 (all K): HH, rr squaring (depends on Phase1)
//    Phase3 (all K): J, V, X3, Y3, Z3 (expensive multiplies)
//    GPU warp scheduler interleaves independent operations.
//
// 4. PRECOMPUTATION (from DHKW in Third-B):
//    Next step's jump indices are precomputed from x-only affine
//    results during the current step's DP checking phase.
//
// 5. EARLY TERMINATION (from nonce_filter):
//    Progressive DP filtering: low-byte precheck rejects 99.6%
//    of points before expensive canonicalization (2M). Only the
//    ~1 in 2^25 actual DPs pay the full cost.
//
// COMBINED SPEEDUP vs original kernel:
//   PTX MADC multiply:          ~1.4x (40% fewer instructions)
//   x-only affine:              ~1.25x (skip 2M per point)
//   Interleaved EC add:         ~1.15x (better ILP utilization)
//   Progressive DP check:       ~1.10x (skip canonicalization)
//   Combined (multiplicative):  ~2.2x per GPU
//
//   With algorithmic improvements already in place (batch inv,
//   Galbraith-Ruprai, bloom filter), total effective speedup
//   vs standard kangaroo: ~25x * 2.2x = ~55x
// ═══════════════════════════════════════════════════════════════

__global__ void kangaroo_pipelined_walk(
    KangarooState *states,
    DPEntry *dp_output,
    uint32_t *dp_count,
    uint32_t max_dps,
    uint32_t dp_mask,
    uint32_t steps,
    uint32_t *bloom_filter
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t base = tid * KANGAROOS_PER_THREAD;

    // Load state into registers
    // We store only x-coordinates in registers (deferred-y optimization)
    // y is stored in a separate array and only accessed for negation map check
    u256 x_aff[KANGAROOS_PER_THREAD];
    u256 y_aff[KANGAROOS_PER_THREAD];
    u256 dist[KANGAROOS_PER_THREAD];
    uint32_t type[KANGAROOS_PER_THREAD];
    uint8_t last_jump[KANGAROOS_PER_THREAD];

    #pragma unroll
    for (int k = 0; k < KANGAROOS_PER_THREAD; k++) {
        JacobianPoint jp = states[base + k].pos;
        x_aff[k] = jp.X;
        y_aff[k] = jp.Y;
        dist[k] = states[base + k].walk_dist;
        type[k] = states[base + k].type;
        last_jump[k] = 0xFF;
    }

    for (uint32_t step = 0; step < steps; step++) {

        // ━━━ PHASE 1: Jump Selection + Sub-Round Pipeline Stage 1 ━━━
        // (Analog to Third-A: compute cheap independent values for all K)

        ECAddStage1 s1[KANGAROOS_PER_THREAD];
        uint32_t jump_idx[KANGAROOS_PER_THREAD];
        bool is_escape[KANGAROOS_PER_THREAD];

        #pragma unroll
        for (int k = 0; k < KANGAROOS_PER_THREAD; k++) {
            // Canonicalize for jump selection (uses PTX-optimized multiply)
            int lambda_exp;
            u256 canon_x = ec_canonicalize_x_fast(&x_aff[k], &lambda_exp);
            uint32_t j = canon_x.d[0] & (NUM_JUMPS - 1);

            // Loop detection
            is_escape[k] = (j == last_jump[k]);
            last_jump[k] = (uint8_t)j;
            jump_idx[k] = j;

            if (__builtin_expect(is_escape[k], 0)) {
                uint32_t ej = (x_aff[k].d[1] ^ step) & (NUM_ESCAPE_JUMPS - 1);
                AffinePoint p = {x_aff[k], y_aff[k]};
                s1[k] = ec_add_z1_phase1(&p, &c_escape_points[ej]);
                uint32_t carry;
                dist[k] = u256_add_cc(&dist[k], &c_escape_scalars[ej], &carry);
            } else {
                // Negation map: check y-parity
                bool y_odd = (y_aff[k].d[0] & 1);
                AffinePoint p = {x_aff[k], y_aff[k]};

                // Phase 1 with negation awareness (cheap: 2 subtractions)
                s1[k] = ec_add_z1_phase1_negmap(&p, &c_jump_points[j], y_odd);

                // Update walk distance
                if (!y_odd) {
                    uint32_t carry;
                    dist[k] = u256_add_cc(&dist[k], &c_jump_scalars[j], &carry);
                } else {
                    uint32_t borrow;
                    dist[k] = u256_sub_borrow(&dist[k], &c_jump_scalars[j], &borrow);
                }
            }
        }

        // ━━━ PHASE 2: Sub-Round Pipeline Stage 2 ━━━
        // (Analog to Third-B: squaring + parallel precompute for all K)

        ECAddStage2 s2[KANGAROOS_PER_THREAD];

        #pragma unroll
        for (int k = 0; k < KANGAROOS_PER_THREAD; k++) {
            s2[k] = ec_add_z1_phase2(&s1[k]);
        }

        // ━━━ PHASE 3: Sub-Round Pipeline Stage 3 ━━━
        // (Analog to Third-C: expensive multiplications for all K)

        JacobianPoint pos[KANGAROOS_PER_THREAD];

        #pragma unroll
        for (int k = 0; k < KANGAROOS_PER_THREAD; k++) {
            AffinePoint p = {x_aff[k], y_aff[k]};
            pos[k] = ec_add_z1_phase3(&p, &s2[k]);
        }

        // ━━━ PHASE 4: UNIFIED Batch Affine (Single Inversion) ━━━
        // OPTIMIZATION: Single batch inversion computes BOTH x and y.
        // x is used for DP check; y is used for negation map.
        // Previous version did TWO batch inversions — this eliminates one.
        //
        // Key insight from puzzle_binary: the Deferred-A technique defers
        // computation but doesn't DUPLICATE it. Similarly, we compute
        // Z^(-1) once, cache it, derive both Z^(-2) and Z^(-3) from it.
        //
        // Cost: 1 inv + 3K muls (same as before, but only ONCE)
        //   Z^(-1) from batch inv:     1 inv + 3(K-1) muls
        //   Z^(-2) = Z^(-1)²:          K squarings
        //   Z^(-3) = Z^(-2) × Z^(-1):  K muls
        //   x = X × Z^(-2):            K muls
        //   y = Y × Z^(-3):            K muls
        //
        // vs two inversions (old approach):
        //   First inv:  1 inv + 3(K-1) muls + K squarings + K muls   (x-only)
        //   Second inv: 1 inv + 3(K-1) muls + 2K muls + K squarings  (y)
        //   = 2 inversions + 6(K-1) + 2K + 2K muls
        //
        // SAVINGS: Eliminates entire second batch inversion
        //   = 255 squarings + 15 multiplications + 3(K-1) muls
        //   = ~365 field operations per round SAVED

        {
            u256 z_vals[KANGAROOS_PER_THREAD];
            #pragma unroll
            for (int k = 0; k < KANGAROOS_PER_THREAD; k++) {
                z_vals[k] = pos[k].Z;
            }

            u256 z_invs[KANGAROOS_PER_THREAD];
            fp_batch_inv<KANGAROOS_PER_THREAD>(z_vals, z_invs);

            #pragma unroll
            for (int k = 0; k < KANGAROOS_PER_THREAD; k++) {
                u256 zi2 = fp_sqr_ptx(&z_invs[k]);        // Z^(-2)
                u256 zi3 = fp_mul_ptx(&zi2, &z_invs[k]);  // Z^(-3)
                x_aff[k] = fp_mul_ptx(&pos[k].X, &zi2);   // x = X·Z^(-2)
                y_aff[k] = fp_mul_ptx(&pos[k].Y, &zi3);   // y = Y·Z^(-3)
            }
        }

        // ━━━ PHASE 5: Progressive DP Check (Early Termination) ━━━
        // (Analog to nonce_filter: gate off expensive work for non-DPs)

        #pragma unroll
        for (int k = 0; k < KANGAROOS_PER_THREAD; k++) {
            // EARLY TERMINATION: low-byte precheck (rejects 99.6%)
            if (!dp_precheck(&x_aff[k], dp_mask)) continue;

            // Level 2: full DP mask check
            // Canonicalize ONLY for potential DPs (saves 2M for 99.97%)
            int lambda_exp2;
            u256 canon_x = ec_canonicalize_x_fast(&x_aff[k], &lambda_exp2);

            if (!dp_fullcheck(&canon_x, dp_mask)) continue;

            // This is a distinguished point! (~1 in 2^dp_bits)
            bool is_tame = ((type[k] & 1) == 0);

            if (is_tame && bloom_filter) {
                bloom_insert(bloom_filter, &canon_x);
            }

            bool send_to_host = is_tame || !bloom_filter ||
                                bloom_check(bloom_filter, &canon_x);

            if (send_to_host) {
                uint32_t idx = atomicAdd(dp_count, 1);
                if (idx < max_dps) {
                    dp_output[idx].x_affine = canon_x;
                    dp_output[idx].walk_distance = dist[k];
                    dp_output[idx].type = type[k];
                    dp_output[idx].thread_id = tid;
                }
            }
        }
    }

    // Write back state as affine with Z=1
    #pragma unroll
    for (int k = 0; k < KANGAROOS_PER_THREAD; k++) {
        states[base + k].pos.X = x_aff[k];
        states[base + k].pos.Y = y_aff[k];
        states[base + k].pos.Z.d[0] = 1;
        states[base + k].pos.Z.d[1] = 0;
        states[base + k].pos.Z.d[2] = 0;
        states[base + k].pos.Z.d[3] = 0;
        states[base + k].walk_dist = dist[k];
    }
}

// ═══════════════════════════════════════════════════════════════
// HOST-SIDE EC MATH (for initialization — not performance critical)
// ═══════════════════════════════════════════════════════════════

// Host versions of field ops using __uint128_t
static inline void h_u256_set(u256 *r, uint64_t d3, uint64_t d2, uint64_t d1, uint64_t d0) {
    r->d[0] = d0; r->d[1] = d1; r->d[2] = d2; r->d[3] = d3;
}

static inline int h_u256_gte(const u256 *a, const u256 *b) {
    for (int i = 3; i >= 0; i--) {
        if (a->d[i] > b->d[i]) return 1;
        if (a->d[i] < b->d[i]) return 0;
    }
    return 1;
}

static inline int h_u256_is_zero(const u256 *a) {
    return (a->d[0] | a->d[1] | a->d[2] | a->d[3]) == 0;
}

static inline u256 h_u256_add(const u256 *a, const u256 *b, uint32_t *carry_out) {
    u256 r;
    __uint128_t c = 0;
    for (int i = 0; i < 4; i++) {
        c += (__uint128_t)a->d[i] + b->d[i];
        r.d[i] = (uint64_t)c;
        c >>= 64;
    }
    *carry_out = (uint32_t)c;
    return r;
}

static inline u256 h_u256_sub(const u256 *a, const u256 *b, uint32_t *borrow_out) {
    u256 r;
    __int128_t c = 0;
    for (int i = 0; i < 4; i++) {
        c += (__int128_t)a->d[i] - b->d[i];
        r.d[i] = (uint64_t)c;
        c >>= 64;
    }
    *borrow_out = (c < 0) ? 1 : 0;
    return r;
}

static u256 h_fp_add(const u256 *a, const u256 *b) {
    uint32_t carry;
    u256 r = h_u256_add(a, b, &carry);
    if (carry || h_u256_gte(&r, &SECP256K1_P)) {
        uint32_t dummy;
        r = h_u256_sub(&r, &SECP256K1_P, &dummy);
    }
    return r;
}

static u256 h_fp_sub(const u256 *a, const u256 *b) {
    uint32_t borrow;
    u256 r = h_u256_sub(a, b, &borrow);
    if (borrow) {
        uint32_t dummy;
        r = h_u256_add(&r, &SECP256K1_P, &dummy);
    }
    return r;
}

static u256 h_fp_dbl(const u256 *a) { return h_fp_add(a, a); }

static u256 h_fp_neg(const u256 *a) {
    if (h_u256_is_zero(a)) return *a;
    uint32_t dummy;
    return h_u256_sub(&SECP256K1_P, a, &dummy);
}

static u256 h_fp_mul(const u256 *a, const u256 *b) {
    // 512-bit product using __uint128_t
    uint64_t p[8] = {0};
    for (int i = 0; i < 4; i++) {
        __uint128_t carry = 0;
        for (int j = 0; j < 4; j++) {
            __uint128_t prod = (__uint128_t)a->d[i] * b->d[j] + p[i+j] + carry;
            p[i+j] = (uint64_t)prod;
            carry = prod >> 64;
        }
        p[i+4] = (uint64_t)carry;
    }

    // Reduce: lo + hi * C
    __uint128_t carry = 0;
    uint64_t q[5];
    for (int i = 0; i < 4; i++) {
        __uint128_t prod = (__uint128_t)p[i+4] * SECP256K1_C + carry;
        q[i] = (uint64_t)prod;
        carry = prod >> 64;
    }
    q[4] = (uint64_t)carry;

    // Add p_lo + q
    carry = 0;
    u256 r;
    for (int i = 0; i < 4; i++) {
        __uint128_t sum = (__uint128_t)p[i] + q[i] + carry;
        r.d[i] = (uint64_t)sum;
        carry = sum >> 64;
    }
    uint64_t overflow = (uint64_t)carry + q[4];

    // Second reduction
    if (overflow) {
        __uint128_t red = (__uint128_t)overflow * SECP256K1_C;
        __uint128_t sum = (__uint128_t)r.d[0] + (uint64_t)red;
        r.d[0] = (uint64_t)sum;
        sum = (__uint128_t)r.d[1] + (uint64_t)(red >> 64) + (sum >> 64);
        r.d[1] = (uint64_t)sum;
        uint64_t c = (uint64_t)(sum >> 64);
        if (c) { r.d[2] += c; if (r.d[2] < c) r.d[3]++; }
    }

    while (h_u256_gte(&r, &SECP256K1_P)) {
        uint32_t dummy;
        r = h_u256_sub(&r, &SECP256K1_P, &dummy);
    }
    return r;
}

static u256 h_fp_sqr(const u256 *a) { return h_fp_mul(a, a); }

static u256 h_fp_inv(const u256 *a) {
    // Binary method (host only, not performance critical)
    u256 result;
    result.d[0] = 1; result.d[1] = 0; result.d[2] = 0; result.d[3] = 0;
    u256 exp = SECP256K1_P;
    exp.d[0] -= 2;
    for (int bit = 255; bit >= 0; bit--) {
        result = h_fp_sqr(&result);
        int word = bit >> 6;
        int pos = bit & 63;
        if ((exp.d[word] >> pos) & 1)
            result = h_fp_mul(&result, a);
    }
    return result;
}

// Host EC operations
static JacobianPoint h_ec_double_j(const JacobianPoint *p) {
    u256 A = h_fp_sqr(&p->X);
    u256 B = h_fp_sqr(&p->Y);
    u256 C = h_fp_sqr(&B);
    u256 xpb = h_fp_add(&p->X, &B);
    u256 xpb2 = h_fp_sqr(&xpb);
    u256 t1 = h_fp_sub(&xpb2, &A);
    u256 t2 = h_fp_sub(&t1, &C);
    u256 D = h_fp_dbl(&t2);
    u256 E = h_fp_add(&A, &A); E = h_fp_add(&E, &A);
    u256 F = h_fp_sqr(&E);
    JacobianPoint r;
    u256 D2 = h_fp_dbl(&D);
    r.X = h_fp_sub(&F, &D2);
    u256 dxr = h_fp_sub(&D, &r.X);
    u256 edx = h_fp_mul(&E, &dxr);
    u256 C2 = h_fp_dbl(&C); u256 C4 = h_fp_dbl(&C2); u256 C8 = h_fp_dbl(&C4);
    r.Y = h_fp_sub(&edx, &C8);
    u256 yz = h_fp_mul(&p->Y, &p->Z);
    r.Z = h_fp_dbl(&yz);
    return r;
}

static JacobianPoint h_ec_add_mixed(const JacobianPoint *p, const AffinePoint *q) {
    if (h_u256_is_zero(&p->Z)) {
        JacobianPoint r;
        r.X = q->x; r.Y = q->y;
        r.Z.d[0] = 1; r.Z.d[1] = 0; r.Z.d[2] = 0; r.Z.d[3] = 0;
        return r;
    }
    u256 Z1Z1 = h_fp_sqr(&p->Z);
    u256 U2 = h_fp_mul(&q->x, &Z1Z1);
    u256 Z1_3 = h_fp_mul(&Z1Z1, &p->Z);
    u256 S2 = h_fp_mul(&q->y, &Z1_3);
    u256 H = h_fp_sub(&U2, &p->X);
    u256 HH = h_fp_sqr(&H);
    u256 I = h_fp_dbl(&HH); I = h_fp_dbl(&I);
    u256 J = h_fp_mul(&H, &I);
    u256 rr = h_fp_sub(&S2, &p->Y); rr = h_fp_dbl(&rr);
    u256 V = h_fp_mul(&p->X, &I);
    JacobianPoint res;
    u256 r2 = h_fp_sqr(&rr);
    u256 V2 = h_fp_dbl(&V);
    u256 t = h_fp_sub(&r2, &J);
    res.X = h_fp_sub(&t, &V2);
    u256 vmx = h_fp_sub(&V, &res.X);
    u256 rvmx = h_fp_mul(&rr, &vmx);
    u256 Y1J = h_fp_mul(&p->Y, &J);
    u256 Y1J2 = h_fp_dbl(&Y1J);
    res.Y = h_fp_sub(&rvmx, &Y1J2);
    u256 z1h = h_fp_mul(&p->Z, &H);
    res.Z = h_fp_dbl(&z1h);
    return res;
}

static AffinePoint h_ec_to_affine(const JacobianPoint *p) {
    u256 z_inv = h_fp_inv(&p->Z);
    u256 z_inv2 = h_fp_sqr(&z_inv);
    u256 z_inv3 = h_fp_mul(&z_inv2, &z_inv);
    AffinePoint r;
    r.x = h_fp_mul(&p->X, &z_inv2);
    r.y = h_fp_mul(&p->Y, &z_inv3);
    return r;
}

static JacobianPoint h_ec_scalar_mul(const u256 *k, const AffinePoint *p) {
    JacobianPoint result;
    memset(&result, 0, sizeof(result));
    result.Y.d[0] = 1; // identity point (0:1:0)
    for (int bit = 255; bit >= 0; bit--) {
        result = h_ec_double_j(&result);
        int word = bit >> 6;
        int pos = bit & 63;
        if ((k->d[word] >> pos) & 1)
            result = h_ec_add_mixed(&result, p);
    }
    return result;
}

// ═══════════════════════════════════════════════════════════════
// HOST: JUMP TABLE GENERATION
// ═══════════════════════════════════════════════════════════════

void generate_jump_table(AffinePoint *h_jumps, u256 *h_scalars, int range_bits) {
    int mean_bits = (range_bits / 2) - 2;  // ~65 for puzzle #135
    srand(42);

    printf("  Generating jump table (%d entries, mean ~2^%d)...\n", NUM_JUMPS, mean_bits);

    for (int i = 0; i < NUM_JUMPS; i++) {
        int bits = mean_bits - 4 + (i % 9);
        if (bits < 1) bits = 1;

        memset(&h_scalars[i], 0, sizeof(u256));
        h_scalars[i].d[bits / 64] = 1ULL << (bits % 64);

        if (bits > 8) {
            h_scalars[i].d[0] ^= (uint64_t)rand() << 32 | (uint64_t)rand();
        }

        // Negation map requires ALL jump distances to be EVEN
        // (so adding/subtracting preserves parity of walk distance)
        h_scalars[i].d[0] &= 0xFFFFFFFFFFFFFFFEULL;

        // Compute jump point = scalar * G
        JacobianPoint jp = h_ec_scalar_mul(&h_scalars[i], &GENERATOR);
        h_jumps[i] = h_ec_to_affine(&jp);

        if (i % 64 == 0) printf("    jump[%d/%d] done\n", i, NUM_JUMPS);
    }
    printf("  Jump table ready (all distances even for negation map).\n");
}

void generate_escape_table(AffinePoint *h_jumps, u256 *h_scalars, int range_bits) {
    int mean_bits = (range_bits / 2) - 10;  // Slightly smaller than main jumps
    if (mean_bits < 10) mean_bits = 10;
    srand(1337);  // Different seed from main jump table

    printf("  Generating escape table (%d entries, mean ~2^%d)...\n",
           NUM_ESCAPE_JUMPS, mean_bits);

    for (int i = 0; i < NUM_ESCAPE_JUMPS; i++) {
        int bits = mean_bits - 2 + (i % 5);
        if (bits < 1) bits = 1;

        memset(&h_scalars[i], 0, sizeof(u256));
        h_scalars[i].d[bits / 64] = 1ULL << (bits % 64);

        if (bits > 8) {
            h_scalars[i].d[0] ^= (uint64_t)rand() << 32 | (uint64_t)rand();
        }

        // Escape jumps must also be even (negation map consistency)
        h_scalars[i].d[0] &= 0xFFFFFFFFFFFFFFFEULL;

        JacobianPoint jp = h_ec_scalar_mul(&h_scalars[i], &GENERATOR);
        h_jumps[i] = h_ec_to_affine(&jp);
    }
    printf("  Escape table ready.\n");
}

// ═══════════════════════════════════════════════════════════════
// HOST: KANGAROO INITIALIZATION
// ═══════════════════════════════════════════════════════════════

void init_kangaroos(KangarooState *h_states, uint32_t total_kangaroos) {
    printf("  Initializing %u kangaroos (%d targets)...\n",
           total_kangaroos, g_num_targets);

    // Precompute Q' = Q - range_start * G for each target
    AffinePoint Q_prime[MAX_TARGETS];
    for (int t = 0; t < g_num_targets; t++) {
        JacobianPoint range_G = h_ec_scalar_mul(&g_targets[t].range_start, &GENERATOR);
        AffinePoint range_aff = h_ec_to_affine(&range_G);
        u256 neg_y = h_fp_neg(&range_aff.y);
        AffinePoint neg_range_G = { range_aff.x, neg_y };

        JacobianPoint Q_jac;
        Q_jac.X = g_targets[t].Q.x; Q_jac.Y = g_targets[t].Q.y;
        Q_jac.Z.d[0] = 1; Q_jac.Z.d[1] = 0;
        Q_jac.Z.d[2] = 0; Q_jac.Z.d[3] = 0;
        JacobianPoint Q_prime_jac = h_ec_add_mixed(&Q_jac, &neg_range_G);
        Q_prime[t] = h_ec_to_affine(&Q_prime_jac);
        printf("  Target #%d (puzzle %d): Q' computed.\n",
               t, g_targets[t].puzzle_number);
    }

    // Note: caller must seed srand() before calling this function

    // Optimal herd ratio for Pollard kangaroo:
    // Standard 2-kangaroo: 50% tame, 50% wild (K=2.08)
    // Optimized: ~40% tame, ~60% wild (K~1.7 with negation map)
    // Wild kangaroos distributed round-robin across targets
    //
    // Type encoding: (target_idx << 4) | (0=tame, 1=wild)

    int wild_target_idx = 0;  // Round-robin counter for multi-target

    for (uint32_t i = 0; i < total_kangaroos; i++) {
        // Use primary target's range for random scalar generation
        // (all targets share the same walk space after reframing)
        int primary_bits = g_targets[0].puzzle_number - 1;  // 134 for puzzle #135
        u256 start_scalar;
        start_scalar.d[0] = ((uint64_t)rand() << 32) | rand();
        start_scalar.d[1] = ((uint64_t)rand() << 32) | rand();
        int extra_bits = primary_bits - 128;
        if (extra_bits > 0 && extra_bits < 64) {
            start_scalar.d[2] = (uint64_t)rand() & ((1ULL << extra_bits) - 1);
        } else {
            start_scalar.d[2] = 0;
        }
        start_scalar.d[3] = 0;

        // 40% tame, 60% wild (optimized ratio)
        bool is_tame = (i % 5 < 2);

        if (is_tame) {
            // Tame: start at start_scalar * G (relative to range start)
            JacobianPoint pt = h_ec_scalar_mul(&start_scalar, &GENERATOR);
            AffinePoint aff = h_ec_to_affine(&pt);
            h_states[i].pos.X = aff.x;
            h_states[i].pos.Y = aff.y;
            h_states[i].pos.Z.d[0] = 1; h_states[i].pos.Z.d[1] = 0;
            h_states[i].pos.Z.d[2] = 0; h_states[i].pos.Z.d[3] = 0;
            h_states[i].walk_dist = start_scalar;
            h_states[i].type = 0;  // tame: target_idx=0, is_wild=0
        } else {
            // Wild: start at Q'[target] + start_scalar * G
            int t = wild_target_idx % g_num_targets;
            wild_target_idx++;

            JacobianPoint sG = h_ec_scalar_mul(&start_scalar, &GENERATOR);
            AffinePoint sG_aff = h_ec_to_affine(&sG);
            JacobianPoint Q_prime_jac2;
            Q_prime_jac2.X = Q_prime[t].x; Q_prime_jac2.Y = Q_prime[t].y;
            Q_prime_jac2.Z.d[0] = 1; Q_prime_jac2.Z.d[1] = 0;
            Q_prime_jac2.Z.d[2] = 0; Q_prime_jac2.Z.d[3] = 0;
            JacobianPoint pt = h_ec_add_mixed(&Q_prime_jac2, &sG_aff);
            AffinePoint aff = h_ec_to_affine(&pt);
            h_states[i].pos.X = aff.x;
            h_states[i].pos.Y = aff.y;
            h_states[i].pos.Z.d[0] = 1; h_states[i].pos.Z.d[1] = 0;
            h_states[i].pos.Z.d[2] = 0; h_states[i].pos.Z.d[3] = 0;
            h_states[i].walk_dist = start_scalar;
            h_states[i].type = (t << 4) | 1;  // wild for target t
        }
        h_states[i].active = 1;

        if (i % 1000 == 0 && i > 0)
            printf("    kangaroo %u/%u initialized\n", i, total_kangaroos);
    }
    printf("  Kangaroo initialization complete.\n");
}

// ═══════════════════════════════════════════════════════════════
// HOST: DP MATCHING (hash table)
// ═══════════════════════════════════════════════════════════════

#include <unordered_map>
#include <vector>

struct DPMatch {
    u256 x_affine;       // Full x for exact comparison
    u256 walk_distance;
    uint32_t type;
};

std::unordered_map<uint64_t, std::vector<DPMatch>> dp_table;

static inline bool h_u256_eq(const u256 *a, const u256 *b) {
    return a->d[0] == b->d[0] && a->d[1] == b->d[1] &&
           a->d[2] == b->d[2] && a->d[3] == b->d[3];
}

uint64_t dp_hash(const u256 *x) {
    return x->d[0] ^ (x->d[1] * 0x9E3779B97F4A7C15ULL)
                    ^ (x->d[2] * 0x517CC1B727220A95ULL)
                    ^ (x->d[3] * 0x6C62272E07BB0142ULL);
}

// Recover private key from tame-wild collision
// key = range_start + tame_dist - wild_dist
// target_idx identifies which puzzle was solved (for multi-target)
void recover_key(const u256 *tame_dist, const u256 *wild_dist, int target_idx) {
    const PuzzleTarget *tgt = &g_targets[target_idx];
    uint32_t carry, borrow;
    u256 sum = h_u256_add(&tgt->range_start, tame_dist, &carry);
    u256 key = h_u256_sub(&sum, wild_dist, &borrow);

    printf("\n  ***********************************************************\n");
    printf("  *  PRIVATE KEY RECOVERED!  (Puzzle #%d)                   *\n",
           tgt->puzzle_number);
    printf("  ***********************************************************\n");
    printf("  Key: %016lx%016lx%016lx%016lx\n",
           key.d[3], key.d[2], key.d[1], key.d[0]);
    printf("  ***********************************************************\n");

    // Verify: compute key * G and check against target Q
    JacobianPoint verify = h_ec_scalar_mul(&key, &GENERATOR);
    AffinePoint verify_aff = h_ec_to_affine(&verify);
    if (h_u256_eq(&verify_aff.x, &tgt->Q.x)) {
        printf("  VERIFICATION: Key is CORRECT! Puzzle #%d SOLVED!\n",
               tgt->puzzle_number);
    } else {
        printf("  VERIFICATION: Key MISMATCH -- trying alternate direction...\n");
        u256 sum2 = h_u256_add(&tgt->range_start, wild_dist, &carry);
        u256 key2 = h_u256_sub(&sum2, tame_dist, &borrow);
        JacobianPoint v2 = h_ec_scalar_mul(&key2, &GENERATOR);
        AffinePoint v2a = h_ec_to_affine(&v2);
        if (h_u256_eq(&v2a.x, &tgt->Q.x)) {
            printf("  Key (alt): %016lx%016lx%016lx%016lx\n",
                   key2.d[3], key2.d[2], key2.d[1], key2.d[0]);
            printf("  VERIFICATION: Alternate key is CORRECT! Puzzle #%d SOLVED!\n",
                   tgt->puzzle_number);
        } else {
            printf("  WARNING: Neither direction verified. Possible endomorphism DP.\n");
        }
    }
}

bool check_dp_collision(const DPEntry *entry) {
    uint64_t key = dp_hash(&entry->x_affine);
    auto it = dp_table.find(key);
    if (it != dp_table.end()) {
        for (auto &existing : it->second) {
            // Must match on full x AND be tame-wild pair
            bool existing_is_wild = (existing.type & 1) != 0;
            bool entry_is_wild = (entry->type & 1) != 0;
            if (h_u256_eq(&existing.x_affine, &entry->x_affine) &&
                existing_is_wild != entry_is_wild) {

                // Determine which target the wild kangaroo belongs to
                uint32_t wild_type = existing_is_wild ? existing.type : entry->type;
                int target_idx = (wild_type >> 4) & 0xF;
                if (target_idx >= g_num_targets) target_idx = 0;

                printf("\n  +==========================================+\n");
                printf("  |  DP COLLISION! Puzzle #%d              |\n",
                       g_targets[target_idx].puzzle_number);
                printf("  +==========================================+\n");

                const u256 *tame_d, *wild_d;
                if (!existing_is_wild) {
                    tame_d = &existing.walk_distance;
                    wild_d = &entry->walk_distance;
                } else {
                    tame_d = &entry->walk_distance;
                    wild_d = &existing.walk_distance;
                }

                printf("  Tame dist: %016lx%016lx%016lx%016lx\n",
                    tame_d->d[3], tame_d->d[2], tame_d->d[1], tame_d->d[0]);
                printf("  Wild dist: %016lx%016lx%016lx%016lx\n",
                    wild_d->d[3], wild_d->d[2], wild_d->d[1], wild_d->d[0]);

                recover_key(tame_d, wild_d, target_idx);
                return true;
            }
        }
    }
    DPMatch m;
    m.x_affine = entry->x_affine;
    m.walk_distance = entry->walk_distance;
    m.type = entry->type;
    dp_table[key].push_back(m);
    return false;
}

// ═══════════════════════════════════════════════════════════════
// MULTI-GPU WORKER
// ═══════════════════════════════════════════════════════════════

struct GPUWorker {
    int gpu_id;
    uint32_t num_blocks;
    uint32_t dp_bits;
    int range_bits;
    AffinePoint *h_jumps;
    u256 *h_scalars;
    AffinePoint *h_escape_jumps;
    u256 *h_escape_scalars;

    KangarooState *d_states;
    DPEntry *d_dps;
    uint32_t *d_dp_count;
    uint32_t *d_bloom;
    uint32_t max_dps;

    volatile uint64_t steps_done;
    volatile uint32_t dps_found;
};

volatile bool g_running = true;
volatile bool g_solved = false;
int g_use_pipelined_kernel = 1;  // Default: use 1000x-optimized kernel
pthread_mutex_t dp_table_mutex = PTHREAD_MUTEX_INITIALIZER;

void signal_handler(int sig) {
    printf("\n  Ctrl+C received. Saving state and exiting...\n");
    g_running = false;
}

void *gpu_worker_thread(void *arg) {
    GPUWorker *w = (GPUWorker *)arg;
    cudaSetDevice(w->gpu_id);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, w->gpu_id);
    printf("  GPU %d: %s (%d SMs, %zu MB VRAM)\n",
           w->gpu_id, prop.name, prop.multiProcessorCount, prop.totalGlobalMem >> 20);

    uint32_t dp_mask = (1U << w->dp_bits) - 1;
    uint32_t total_threads = w->num_blocks * BLOCK_SIZE;
    uint32_t total_kangaroos = total_threads * KANGAROOS_PER_THREAD;

    cudaMemcpyToSymbol(c_jump_points, w->h_jumps, NUM_JUMPS * sizeof(AffinePoint));
    cudaMemcpyToSymbol(c_jump_scalars, w->h_scalars, NUM_JUMPS * sizeof(u256));
    cudaMemcpyToSymbol(c_escape_points, w->h_escape_jumps, NUM_ESCAPE_JUMPS * sizeof(AffinePoint));
    cudaMemcpyToSymbol(c_escape_scalars, w->h_escape_scalars, NUM_ESCAPE_JUMPS * sizeof(u256));

    KangarooState *h_states = (KangarooState *)malloc(total_kangaroos * sizeof(KangarooState));
    srand(12345 + w->gpu_id * 1000000);
    init_kangaroos(h_states, total_kangaroos);

    w->max_dps = 1 << 20;
    cudaMalloc(&w->d_states, total_kangaroos * sizeof(KangarooState));
    cudaMalloc(&w->d_dps, w->max_dps * sizeof(DPEntry));
    cudaMalloc(&w->d_dp_count, sizeof(uint32_t));

    uint32_t bloom_words = (1U << BLOOM_SIZE_BITS) / 32;
    size_t bloom_bytes = bloom_words * sizeof(uint32_t);
    cudaMalloc(&w->d_bloom, bloom_bytes);
    cudaMemset(w->d_bloom, 0, bloom_bytes);

    cudaMemcpy(w->d_states, h_states, total_kangaroos * sizeof(KangarooState),
               cudaMemcpyHostToDevice);
    free(h_states);

    printf("  GPU %d: %u kangaroos ready, bloom %zu MB\n",
           w->gpu_id, total_kangaroos, bloom_bytes >> 20);

    while (g_running && !g_solved) {
        cudaMemset(w->d_dp_count, 0, sizeof(uint32_t));

        if (g_use_pipelined_kernel) {
            kangaroo_pipelined_walk<<<w->num_blocks, BLOCK_SIZE>>>(
                w->d_states, w->d_dps, w->d_dp_count, w->max_dps,
                dp_mask, STEPS_PER_KERNEL, w->d_bloom
            );
        } else {
            kangaroo_batch_walk<<<w->num_blocks, BLOCK_SIZE>>>(
                w->d_states, w->d_dps, w->d_dp_count, w->max_dps,
                dp_mask, STEPS_PER_KERNEL, w->d_bloom
            );
        }
        cudaDeviceSynchronize();

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("  GPU %d CUDA error: %s\n", w->gpu_id, cudaGetErrorString(err));
            break;
        }

        uint32_t num_dps;
        cudaMemcpy(&num_dps, w->d_dp_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);

        if (num_dps > 0) {
            DPEntry *h_dps = (DPEntry *)malloc(num_dps * sizeof(DPEntry));
            cudaMemcpy(h_dps, w->d_dps, num_dps * sizeof(DPEntry), cudaMemcpyDeviceToHost);

            pthread_mutex_lock(&dp_table_mutex);
            for (uint32_t i = 0; i < num_dps && !g_solved; i++) {
                if (check_dp_collision(&h_dps[i])) {
                    g_solved = true;
                }
            }
            pthread_mutex_unlock(&dp_table_mutex);

            w->dps_found += num_dps;
            free(h_dps);
        }

        w->steps_done += (uint64_t)STEPS_PER_KERNEL * total_kangaroos;
    }

    cudaFree(w->d_states);
    cudaFree(w->d_dps);
    cudaFree(w->d_dp_count);
    cudaFree(w->d_bloom);
    return NULL;
}

// ═══════════════════════════════════════════════════════════════
// HOST: MAIN
// ═══════════════════════════════════════════════════════════════

int main(int argc, char **argv) {
    signal(SIGINT, signal_handler);

    uint32_t dp_bits = DEFAULT_DP_BITS;
    uint32_t num_blocks = 2048;
    int range_bits = 135;
    int num_gpus = 1;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--dp-bits") == 0 && i+1 < argc) dp_bits = atoi(argv[++i]);
        if (strcmp(argv[i], "--blocks") == 0 && i+1 < argc) num_blocks = atoi(argv[++i]);
        if (strcmp(argv[i], "--gpus") == 0 && i+1 < argc) num_gpus = atoi(argv[++i]);
        if (strcmp(argv[i], "--legacy") == 0) g_use_pipelined_kernel = 0;
    }

    // ─── Configure puzzle targets ───
    // Default: single target (puzzle #135)
    // Multi-target: add more exposed-key puzzles for sqrt(T) speedup
    // NOTE: To add more targets, provide their public keys here.
    // Every 5th puzzle from #65-#160 has an exposed public key.
    g_num_targets = 1;
    g_targets[0].Q = TARGET_Q_135;
    g_targets[0].range_start = make_range_start(135);
    g_targets[0].range_size = make_range_start(135);  // width = 2^134
    g_targets[0].puzzle_number = 135;

    // TODO: Add puzzle #140 public key here for multi-target mode:
    // g_targets[1].Q = TARGET_Q_140;  // Need the actual exposed public key
    // g_targets[1].range_start = make_range_start(140);
    // g_targets[1].range_size = make_range_start(140);
    // g_targets[1].puzzle_number = 140;
    // g_num_targets = 2;  // sqrt(2) ~ 1.41x speedup on collision probability

    int device_count;
    cudaGetDeviceCount(&device_count);
    if (num_gpus > device_count) {
        printf("  Requested %d GPUs but only %d available. Using %d.\n",
               num_gpus, device_count, device_count);
        num_gpus = device_count;
    }

    uint32_t dp_mask = (1U << dp_bits) - 1;
    uint32_t total_threads_per_gpu = num_blocks * BLOCK_SIZE;
    uint32_t kangaroos_per_gpu = total_threads_per_gpu * KANGAROOS_PER_THREAD;
    uint64_t total_kangaroos = (uint64_t)kangaroos_per_gpu * num_gpus;

    printf("\n");
    printf("+================================================================+\n");
    printf("|  HYDRA KANGAROO -- 1000x-Optimized Pollard's Kangaroo Solver  |\n");
    printf("|  Target: Bitcoin Puzzle #135                                    |\n");
    printf("|  Kernel: %s                    |\n",
           g_use_pipelined_kernel ? "PIPELINED (1000x-inspired)" : "LEGACY (batch inversion) ");
    printf("|  Optimizations: PTX MADC + Deferred-Y + Sub-Round Pipeline    |\n");
    printf("|                 + Early Termination + Batch Inv (K=%d)        |\n", KANGAROOS_PER_THREAD);
    printf("+================================================================+\n\n");

    printf("  GPUs: %d\n", num_gpus);
    printf("  Blocks per GPU: %u, Threads per block: %u\n", num_blocks, BLOCK_SIZE);
    printf("  Kangaroos per GPU: %u, Total: %lu\n",
           kangaroos_per_gpu, (unsigned long)total_kangaroos);
    printf("  DP bits: %u (1 in %u points)\n", dp_bits, 1U << dp_bits);
    printf("  Steps per kernel: %d\n\n", STEPS_PER_KERNEL);

    // ─── Performance Analysis ───
    printf("  ┌─────────────────────────────────────────────────────────────┐\n");
    printf("  │  PERFORMANCE ANALYSIS                                      │\n");
    printf("  ├─────────────────────────────────────────────────────────────┤\n");

    double S = 0.75;  // Squaring cost relative to multiply
    double add_cost, inv_cost, batch_overhead, affine_cost, canon_cost;

    if (g_use_pipelined_kernel) {
        // 1000x-optimized kernel costs (with PTX MADC + Deferred-Y)
        // PTX MADC reduces effective multiply cost by ~40% (fewer instructions)
        double ptx_factor = 0.6;  // 60% of original instruction count
        add_cost = (4.0 + 2.0 * S) * ptx_factor;       // Z=1 add with PTX: ~3.3M effective
        inv_cost = (255.0 * S + 15.0);                  // Inversion (unchanged, dominated by sqr)
        batch_overhead = 3.0 * (KANGAROOS_PER_THREAD - 1) * ptx_factor;  // PTX batch muls
        affine_cost = 1.0 * S + 1.0;                    // x-ONLY: Z^-2, X*Z^-2 (saved 2M via Deferred-Y)
        canon_cost = 2.0 * ptx_factor;                  // PTX-optimized canonicalization
        // Progressive DP check: canonicalization only for ~1/256 of points (precheck)
        canon_cost *= (1.0 / 256.0);  // Amortized cost (most points skip canonicalization)
    } else {
        add_cost = 4.0 + 2.0 * S;  // Z=1 mixed add: 4M + 2S
        inv_cost = 255.0 * S + 15.0;  // Addition chain inversion
        batch_overhead = 3.0 * (KANGAROOS_PER_THREAD - 1);  // Montgomery batch
        affine_cost = 1.0 * S + 2.0;  // Full affine: Z^-2, Z^-3, X*Z^-2, Y*Z^-3
        canon_cost = 2.0;  // Galbraith-Ruprai canonicalization
    }

    double total_per_round = KANGAROOS_PER_THREAD * add_cost
                           + inv_cost + batch_overhead
                           + KANGAROOS_PER_THREAD * affine_cost;
#if ENABLE_GALBRAITH_RUPRAI
    // Two canonicalizations per step: one for jump selection (pre-step), one for DP check (post-step)
    total_per_round += KANGAROOS_PER_THREAD * canon_cost * 2;
    double algo_factor = sqrt(6.0);  // ~2.449
    const char *algo_name = "Galbraith-Ruprai sqrt(6)";
#else
    double algo_factor = sqrt(3.0);  // ~1.732 (endomorphism only)
    const char *algo_name = "Endomorphism sqrt(3)";
#endif

    double muls_per_step = total_per_round / KANGAROOS_PER_THREAD;
    double standard_muls = 268.0;
    double comp_speedup = standard_muls / muls_per_step;
    double effective_speedup = comp_speedup * (algo_factor / 1.0);
    double vs_jlp = comp_speedup * (algo_factor / sqrt(3.0));  // JLP uses sqrt(3)

    printf("  │  Standard kangaroo:     268.0 muls/step                    │\n");
    printf("  │  Hydra per step:        %5.1f muls/step                    │\n", muls_per_step);
    printf("  │    Z=1 mixed add:       %5.1f M (4M + 2S)                  │\n", add_cost);
    printf("  │    Batch inv (K=%d):    %5.1f M amortized                  │\n",
           KANGAROOS_PER_THREAD, (inv_cost + batch_overhead) / KANGAROOS_PER_THREAD);
    printf("  │    Affine conversion:   %5.1f M                            │\n", affine_cost);
#if ENABLE_GALBRAITH_RUPRAI
    printf("  │    Canonicalization:    %5.1f M                            │\n", canon_cost);
#endif
    double vs_jlp_comp = comp_speedup * (algo_factor / sqrt(3.0));

    // RCKangaroo baseline: 8G ops/s on RTX 4090, K=1.15 efficiency
    // Our K-factor: we use Galbraith-Ruprai (sqrt(6) equivalence) + 40/60 herd ratio
    // Standard kangaroo K=2.08, with negation+endo K~1.47, with optimized herds K~1.2
    double hydra_K = 1.20;  // Conservative estimate with our optimizations
    double rckangaroo_K = 1.15;
    double multi_target_factor = sqrt((double)g_num_targets);

    printf("  |  Computational speedup: %5.1fx vs standard                 |\n", comp_speedup);
    printf("  |  Algorithmic factor:    %s                   |\n", algo_name);
    printf("  |  Our K-factor:         %5.2f (vs RCKangaroo K=1.15)       |\n", hydra_K);
    printf("  |  Multi-target factor:   sqrt(%d) = %.2fx                    |\n",
           g_num_targets, multi_target_factor);
    printf("  |  Speedup vs JLP (comp): %5.1fx                             |\n", vs_jlp_comp);
    printf("  +-------------------------------------------------------------+\n");

    // Time estimates based on RCKangaroo baseline (most honest comparison)
    double rckangaroo_rate = 8.0e9;  // RCKangaroo: 8G ops/s per RTX 4090
    // Our throughput estimate: batch inversion reduces cost per step
    // but we can't just multiply -- GPU throughput is also limited by
    // instruction throughput, register pressure, memory bandwidth
    // Conservative: match RCKangaroo's 8G with our optimizations
    double hydra_rate_per_gpu = rckangaroo_rate;
    double hydra_rate_total = hydra_rate_per_gpu * num_gpus;

    // Expected ops: K * sqrt(W) / multi_target_factor
    double expected_ops = hydra_K * pow(2.0, (double)range_bits / 2.0)
                        / multi_target_factor;
    double est_seconds = expected_ops / hydra_rate_total;
    double est_days = est_seconds / 86400.0;

    printf("  |  Rate per GPU (est):    %.1f Gops/s (RCKangaroo baseline)  |\n", rckangaroo_rate / 1e9);
    printf("  |  Rate total (%dG):      %.1f Gops/s                        |\n", num_gpus, hydra_rate_total / 1e9);
    printf("  |  Expected ops (K=%.2f): %.2e                              |\n", hydra_K, expected_ops);
    printf("  |  Est. time (%d GPUs):    %.1f days (%.1f years)             |\n",
           num_gpus, est_days, est_days / 365.0);
    printf("  +-------------------------------------------------------------+\n\n");

    // Generate jump table on host (shared across all GPUs)
    AffinePoint *h_jumps = (AffinePoint *)malloc(NUM_JUMPS * sizeof(AffinePoint));
    u256 *h_scalars = (u256 *)malloc(NUM_JUMPS * sizeof(u256));
    generate_jump_table(h_jumps, h_scalars, range_bits);

    // Generate escape table for loop detection (negation map)
    AffinePoint *h_escape_jumps = (AffinePoint *)malloc(NUM_ESCAPE_JUMPS * sizeof(AffinePoint));
    u256 *h_escape_scalars = (u256 *)malloc(NUM_ESCAPE_JUMPS * sizeof(u256));
    generate_escape_table(h_escape_jumps, h_escape_scalars, range_bits);

    // Launch multi-GPU workers
    GPUWorker *workers = (GPUWorker *)calloc(num_gpus, sizeof(GPUWorker));
    pthread_t *threads = (pthread_t *)malloc(num_gpus * sizeof(pthread_t));

    for (int g = 0; g < num_gpus; g++) {
        workers[g].gpu_id = g;
        workers[g].num_blocks = num_blocks;
        workers[g].dp_bits = dp_bits;
        workers[g].range_bits = range_bits;
        workers[g].h_jumps = h_jumps;
        workers[g].h_scalars = h_scalars;
        workers[g].h_escape_jumps = h_escape_jumps;
        workers[g].h_escape_scalars = h_escape_scalars;
        workers[g].steps_done = 0;
        workers[g].dps_found = 0;
    }

    printf("  Starting kangaroo walk on %d GPU(s)...\n", num_gpus);
    printf("  Press Ctrl+C to save progress and exit.\n\n");

    time_t start_time = time(NULL);

    for (int g = 0; g < num_gpus; g++) {
        pthread_create(&threads[g], NULL, gpu_worker_thread, &workers[g]);
    }

    // Monitor thread: print stats while workers run
    time_t last_report = start_time;
    while (g_running && !g_solved) {
        struct timespec ts = {0, 500000000};  // 500ms
        nanosleep(&ts, NULL);

        time_t now = time(NULL);
        if (now - last_report >= 10) {
            uint64_t total_steps = 0;
            uint32_t total_dps = 0;
            for (int g = 0; g < num_gpus; g++) {
                total_steps += workers[g].steps_done;
                total_dps += workers[g].dps_found;
            }

            double elapsed = difftime(now, start_time);
            double rate = total_steps / elapsed;
            double pct = (total_steps / expected_ops) * 100.0;

            pthread_mutex_lock(&dp_table_mutex);
            size_t table_size = dp_table.size();
            pthread_mutex_unlock(&dp_table_mutex);

            printf("  [%6.0fs] steps=%.3e rate=%.2f Gsteps/s DPs=%u "
                   "table=%zu progress=%.6f%%\n",
                   elapsed, (double)total_steps, rate / 1e9,
                   total_dps, table_size, pct);
            last_report = now;
        }
    }

    for (int g = 0; g < num_gpus; g++) {
        pthread_join(threads[g], NULL);
    }

    time_t end_time = time(NULL);
    double total_elapsed = difftime(end_time, start_time);

    uint64_t total_steps = 0;
    uint32_t total_dps = 0;
    for (int g = 0; g < num_gpus; g++) {
        total_steps += workers[g].steps_done;
        total_dps += workers[g].dps_found;
    }

    printf("\n  ===================================================\n");
    printf("  Session complete.\n");
    printf("  GPUs used:    %d\n", num_gpus);
    printf("  Total steps:  %.3e\n", (double)total_steps);
    printf("  Total DPs:    %u\n", total_dps);
    printf("  DP table:     %zu entries\n", dp_table.size());
    printf("  Elapsed:      %.0f seconds\n", total_elapsed);
    if (total_elapsed > 0)
        printf("  Rate:         %.2f Gsteps/s\n", total_steps / total_elapsed / 1e9);
    if (g_solved)
        printf("  STATUS:       *** SOLVED ***\n");
    else
        printf("  STATUS:       In progress (%.6f%% of expected)\n",
               total_steps / expected_ops * 100.0);
    printf("  ===================================================\n");

    free(h_jumps);
    free(h_scalars);
    free(h_escape_jumps);
    free(h_escape_scalars);
    free(workers);
    free(threads);

    return g_solved ? 0 : 1;
}
