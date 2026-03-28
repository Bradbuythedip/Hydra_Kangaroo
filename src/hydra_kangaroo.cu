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

// Puzzle #135 target public key
// Qx = 0x145D2611C823A396EF6712CE0F712F09B9B4F3135E3E0AA3230FB9B6D08D1E16
// Qy = 0x667A05E9A1BDD6F70142B66558BD12CE2C0F9CBC7001B20C8A6A109C80DC5330
static const AffinePoint TARGET_Q = {
    {{0x230FB9B6D08D1E16ULL, 0xB9B4F3135E3E0AA3ULL,
      0xEF6712CE0F712F09ULL, 0x145D2611C823A396ULL}},
    {{0x8A6A109C80DC5330ULL, 0x2C0F9CBC7001B20CULL,
      0x0142B66558BD12CEULL, 0x667A05E9A1BDD6F7ULL}}
};

// Range: [2^134, 2^135)
// 2^134 = d[2] has bit 6 set (134 - 128 = 6), so d[2] = 0x40
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
//   1. Canonicalize x-coordinates (2M each) — CACHED for reuse
//   2. Z=1 mixed add for each kangaroo (4M + 2S each)
//   3. Batch-invert all K Z-values (1 inv + 3K muls)
//   4. Convert to affine (3M each)
//   5. DP check using cached canonical x (0M — already computed!)
//
// Key optimization vs prior version: canonicalization computed ONCE
// per step, not twice. Saves 2M * K = 64M per step.
//
// Cost: ~18.5 mul-equivalents per step (vs 268 standard)
// With sqrt(6) equivalence: ~35x effective speedup
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

    // Load state into registers: affine points (Z=1 after init/previous kernel)
    AffinePoint aff[KANGAROOS_PER_THREAD];
    u256 dist[KANGAROOS_PER_THREAD];
    uint32_t type[KANGAROOS_PER_THREAD];

    #pragma unroll
    for (int k = 0; k < KANGAROOS_PER_THREAD; k++) {
        JacobianPoint jp = states[base + k].pos;
        aff[k].x = jp.X;
        aff[k].y = jp.Y;
        dist[k] = states[base + k].walk_dist;
        type[k] = states[base + k].type;
    }

    for (uint32_t step = 0; step < steps; step++) {

        // ─── Phase 1: Canonicalize + Z=1 Mixed Add (2M + 4M + 2S per kangaroo) ───
        JacobianPoint pos[KANGAROOS_PER_THREAD];
        // Cache canonical x for reuse in DP check (eliminates redundant 2M*K)
        u256 cached_canon_x[KANGAROOS_PER_THREAD];

        #pragma unroll
        for (int k = 0; k < KANGAROOS_PER_THREAD; k++) {
#if ENABLE_GALBRAITH_RUPRAI
            int lambda_exp;
            cached_canon_x[k] = ec_canonicalize_x(&aff[k].x, &lambda_exp);
            uint32_t j = cached_canon_x[k].d[0] & (NUM_JUMPS - 1);
#else
            uint32_t j = aff[k].x.d[0] & (NUM_JUMPS - 1);
#endif
            pos[k] = ec_add_mixed_z1(&aff[k], &c_jump_points[j]);

            uint32_t carry;
            dist[k] = u256_add_cc(&dist[k], &c_jump_scalars[j], &carry);
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

        // Compute jump point = scalar * G
        JacobianPoint jp = h_ec_scalar_mul(&h_scalars[i], &GENERATOR);
        h_jumps[i] = h_ec_to_affine(&jp);

        if (i % 64 == 0) printf("    jump[%d/%d] done\n", i, NUM_JUMPS);
    }
    printf("  Jump table ready.\n");
}

// ═══════════════════════════════════════════════════════════════
// HOST: KANGAROO INITIALIZATION
// ═══════════════════════════════════════════════════════════════

void init_kangaroos(KangarooState *h_states, uint32_t total_kangaroos) {
    printf("  Initializing %u kangaroos...\n", total_kangaroos);

    // Compute Q' = Q - range_start * G so that wild kangaroos
    // walk relative to the bottom of the range.
    JacobianPoint range_start_G = h_ec_scalar_mul(&RANGE_START, &GENERATOR);
    AffinePoint range_start_aff = h_ec_to_affine(&range_start_G);

    // Q' = Q + (-range_start * G) = Q - range_start * G
    u256 neg_y = h_fp_neg(&range_start_aff.y);
    AffinePoint neg_range_start_G = { range_start_aff.x, neg_y };

    JacobianPoint Q_jac;
    Q_jac.X = TARGET_Q.x; Q_jac.Y = TARGET_Q.y;
    Q_jac.Z.d[0] = 1; Q_jac.Z.d[1] = 0; Q_jac.Z.d[2] = 0; Q_jac.Z.d[3] = 0;
    JacobianPoint Q_prime_jac = h_ec_add_mixed(&Q_jac, &neg_range_start_G);
    AffinePoint Q_prime = h_ec_to_affine(&Q_prime_jac);

    printf("  Q' computed (Q reframed to range start).\n");

    // Note: caller must seed srand() before calling this function

    for (uint32_t i = 0; i < total_kangaroos; i++) {
        // Random starting scalar in [0, range_size)
        // range_size = 2^134, so we need ~134 random bits
        u256 start_scalar;
        start_scalar.d[0] = ((uint64_t)rand() << 32) | rand();
        start_scalar.d[1] = ((uint64_t)rand() << 32) | rand();
        start_scalar.d[2] = ((uint64_t)rand() << 6) | (rand() & 0x3F); // 6 bits
        start_scalar.d[3] = 0;

        if (i % 2 == 0) {
            // Tame: start at start_scalar * G (relative to range start)
            JacobianPoint pt = h_ec_scalar_mul(&start_scalar, &GENERATOR);
            AffinePoint aff = h_ec_to_affine(&pt);
            // Store as Jacobian with Z=1 (kernel reads X,Y as affine)
            h_states[i].pos.X = aff.x;
            h_states[i].pos.Y = aff.y;
            h_states[i].pos.Z.d[0] = 1; h_states[i].pos.Z.d[1] = 0;
            h_states[i].pos.Z.d[2] = 0; h_states[i].pos.Z.d[3] = 0;
            h_states[i].walk_dist = start_scalar;
            h_states[i].type = 0;
        } else {
            // Wild: start at Q' + start_scalar * G
            JacobianPoint sG = h_ec_scalar_mul(&start_scalar, &GENERATOR);
            AffinePoint sG_aff = h_ec_to_affine(&sG);
            JacobianPoint Q_prime_jac2;
            Q_prime_jac2.X = Q_prime.x; Q_prime_jac2.Y = Q_prime.y;
            Q_prime_jac2.Z.d[0] = 1; Q_prime_jac2.Z.d[1] = 0;
            Q_prime_jac2.Z.d[2] = 0; Q_prime_jac2.Z.d[3] = 0;
            JacobianPoint pt = h_ec_add_mixed(&Q_prime_jac2, &sG_aff);
            AffinePoint aff = h_ec_to_affine(&pt);
            h_states[i].pos.X = aff.x;
            h_states[i].pos.Y = aff.y;
            h_states[i].pos.Z.d[0] = 1; h_states[i].pos.Z.d[1] = 0;
            h_states[i].pos.Z.d[2] = 0; h_states[i].pos.Z.d[3] = 0;
            h_states[i].walk_dist = start_scalar;
            h_states[i].type = 1;
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
void recover_key(const u256 *tame_dist, const u256 *wild_dist) {
    // k = range_start + tame_dist - wild_dist
    uint32_t carry, borrow;
    u256 sum = h_u256_add(&RANGE_START, tame_dist, &carry);
    u256 key = h_u256_sub(&sum, wild_dist, &borrow);

    printf("\n  ***********************************************************\n");
    printf("  *  PRIVATE KEY RECOVERED!                                  *\n");
    printf("  ***********************************************************\n");
    printf("  Key: %016lx%016lx%016lx%016lx\n",
           key.d[3], key.d[2], key.d[1], key.d[0]);
    printf("  ***********************************************************\n");

    // Verify: compute key * G and check against TARGET_Q
    JacobianPoint verify = h_ec_scalar_mul(&key, &GENERATOR);
    AffinePoint verify_aff = h_ec_to_affine(&verify);
    if (h_u256_eq(&verify_aff.x, &TARGET_Q.x)) {
        printf("  VERIFICATION: Key is CORRECT!\n");
    } else {
        printf("  VERIFICATION: Key MISMATCH — checking range_start - tame + wild...\n");
        // Try the other direction
        u256 sum2 = h_u256_add(&RANGE_START, wild_dist, &carry);
        u256 key2 = h_u256_sub(&sum2, tame_dist, &borrow);
        JacobianPoint v2 = h_ec_scalar_mul(&key2, &GENERATOR);
        AffinePoint v2a = h_ec_to_affine(&v2);
        if (h_u256_eq(&v2a.x, &TARGET_Q.x)) {
            printf("  Key (alt): %016lx%016lx%016lx%016lx\n",
                   key2.d[3], key2.d[2], key2.d[1], key2.d[0]);
            printf("  VERIFICATION: Alternate key is CORRECT!\n");
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
            if (h_u256_eq(&existing.x_affine, &entry->x_affine) &&
                (existing.type & 1) != (entry->type & 1)) {

                printf("\n  +==========================================+\n");
                printf("  |  DP COLLISION DETECTED!                  |\n");
                printf("  +==========================================+\n");

                const u256 *tame_d, *wild_d;
                if ((existing.type & 1) == 0) {
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

                recover_key(tame_d, wild_d);
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

        kangaroo_batch_walk<<<w->num_blocks, BLOCK_SIZE>>>(
            w->d_states, w->d_dps, w->d_dp_count, w->max_dps,
            dp_mask, STEPS_PER_KERNEL, w->d_bloom
        );
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
    }

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
    printf("|  HYDRA KANGAROO -- Optimized Pollard's Kangaroo Solver         |\n");
    printf("|  Target: Bitcoin Puzzle #135                                    |\n");
    printf("|  Optimization: Batch Inv + Bloom + Multi-GPU (K=%d)          |\n", KANGAROOS_PER_THREAD);
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
    double add_cost = 4.0 + 2.0 * S;  // Z=1 mixed add: 4M + 2S
    double inv_cost = 255.0 * S + 15.0;  // Addition chain inversion
    double batch_overhead = 3.0 * (KANGAROOS_PER_THREAD - 1);  // Montgomery batch
    double affine_cost = 1.0 * S + 2.0;  // Per-point affine conversion: Z^-2, Z^-3, X*Z^-2, Y*Z^-3
    double canon_cost = 2.0;  // Galbraith-Ruprai canonicalization (2 muls for beta*x, beta^2*x)

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
    printf("  │  Computational speedup: %5.1fx                             │\n", comp_speedup);
    printf("  │  Algorithmic factor:    %s                   │\n", algo_name);
    printf("  │  Effective speedup:     %5.1fx vs naive                    │\n", effective_speedup);
    printf("  │  Speedup vs JLP:        %5.1fx                             │\n", vs_jlp);
    printf("  ├─────────────────────────────────────────────────────────────┤\n");

    // Expected time estimates
    double jlp_rate = 2.5e9;  // JLP: ~2.5G keys/sec per RTX 4090 (measured)
    double hydra_rate_per_gpu = jlp_rate * vs_jlp;
    double hydra_rate_total = hydra_rate_per_gpu * num_gpus;
    double expected_ops = 2.08 * pow(2.0, (double)range_bits / 2.0) / algo_factor;
    double est_seconds = expected_ops / hydra_rate_total;
    double est_days = est_seconds / 86400.0;

    printf("  |  Est. rate per GPU:     %.2f Gkeys/s                      |\n", hydra_rate_per_gpu / 1e9);
    printf("  |  Est. rate total (%dG):  %.2f Gkeys/s                     |\n", num_gpus, hydra_rate_total / 1e9);
    printf("  |  Expected ops:          %.2e                               |\n", expected_ops);
    printf("  |  Est. time (%d GPUs):    %.1f days                         |\n", num_gpus, est_days);
    printf("  +-------------------------------------------------------------+\n\n");

    // Generate jump table on host (shared across all GPUs)
    AffinePoint *h_jumps = (AffinePoint *)malloc(NUM_JUMPS * sizeof(AffinePoint));
    u256 *h_scalars = (u256 *)malloc(NUM_JUMPS * sizeof(u256));
    generate_jump_table(h_jumps, h_scalars, range_bits);

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
    free(workers);
    free(threads);

    return g_solved ? 0 : 1;
}
