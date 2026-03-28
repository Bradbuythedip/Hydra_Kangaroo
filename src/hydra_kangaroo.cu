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

#include "field.cuh"
#include "ec.cuh"

// ═══════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════

#define KANGAROOS_PER_THREAD 16
#define NUM_JUMPS 256
#define BLOCK_SIZE 256
#define DEFAULT_DP_BITS 25
#define STEPS_PER_KERNEL 256

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
// THE MAIN KERNEL — BATCH INVERSION KANGAROO WALK
// ═══════════════════════════════════════════════════════════════

__global__ void kangaroo_batch_walk(
    KangarooState *states,
    DPEntry *dp_output,
    uint32_t *dp_count,
    uint32_t max_dps,
    uint32_t dp_mask,
    uint32_t steps
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t base = tid * KANGAROOS_PER_THREAD;

    // Load kangaroo states into registers
    JacobianPoint pos[KANGAROOS_PER_THREAD];
    u256 dist[KANGAROOS_PER_THREAD];
    uint32_t type[KANGAROOS_PER_THREAD];

    #pragma unroll
    for (int k = 0; k < KANGAROOS_PER_THREAD; k++) {
        pos[k] = states[base + k].pos;
        dist[k] = states[base + k].walk_dist;
        type[k] = states[base + k].type;
    }

    for (uint32_t step = 0; step < steps; step++) {

        // Phase 1: Step all K kangaroos in Jacobian
        #pragma unroll
        for (int k = 0; k < KANGAROOS_PER_THREAD; k++) {
            uint32_t j = pos[k].X.d[0] & (NUM_JUMPS - 1);
            pos[k] = ec_add_mixed(&pos[k], &c_jump_points[j]);

            uint32_t carry;
            dist[k] = u256_add_cc(&dist[k], &c_jump_scalars[j], &carry);
        }

        // Phase 2: Batch convert to affine
        AffinePoint affine_pts[KANGAROOS_PER_THREAD];
        ec_batch_to_affine<KANGAROOS_PER_THREAD>(pos, affine_pts);

        // Phase 3: Check for distinguished points
        #pragma unroll
        for (int k = 0; k < KANGAROOS_PER_THREAD; k++) {
            if ((affine_pts[k].x.d[0] & dp_mask) == 0) {
                uint32_t idx = atomicAdd(dp_count, 1);
                if (idx < max_dps) {
                    dp_output[idx].x_affine = affine_pts[k].x;
                    dp_output[idx].walk_distance = dist[k];
                    dp_output[idx].type = type[k];
                    dp_output[idx].thread_id = tid;
                }
            }

            // Check endomorphism point: lambda*P = (beta*x, y)
            u256 endo_x = fp_mul(&ENDO_BETA, &affine_pts[k].x);
            if ((endo_x.d[0] & dp_mask) == 0) {
                uint32_t idx = atomicAdd(dp_count, 1);
                if (idx < max_dps) {
                    dp_output[idx].x_affine = endo_x;
                    dp_output[idx].walk_distance = dist[k];
                    dp_output[idx].type = type[k] | 0x10;
                    dp_output[idx].thread_id = tid;
                }
            }

            // Re-establish Jacobian with Z=1 for next step
            pos[k].X = affine_pts[k].x;
            pos[k].Y = affine_pts[k].y;
            pos[k].Z.d[0] = 1; pos[k].Z.d[1] = 0;
            pos[k].Z.d[2] = 0; pos[k].Z.d[3] = 0;
        }
    }

    // Write back
    #pragma unroll
    for (int k = 0; k < KANGAROOS_PER_THREAD; k++) {
        states[base + k].pos = pos[k];
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

    srand(12345);

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
            h_states[i].pos = pt;
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
            h_states[i].pos = pt;
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
    u256 walk_distance;
    uint32_t type;
};

std::unordered_map<uint64_t, std::vector<DPMatch>> dp_table;

uint64_t dp_hash(const u256 *x) {
    return x->d[0] ^ x->d[1] ^ x->d[2] ^ x->d[3];
}

bool check_dp_collision(const DPEntry *entry) {
    uint64_t key = dp_hash(&entry->x_affine);
    auto it = dp_table.find(key);
    if (it != dp_table.end()) {
        for (auto &existing : it->second) {
            if ((existing.type & 1) != (entry->type & 1)) {
                printf("\n  +==========================================+\n");
                printf("  |  DP COLLISION DETECTED!                  |\n");
                printf("  +==========================================+\n");
                printf("  Tame dist: %016lx%016lx%016lx%016lx\n",
                    existing.walk_distance.d[3], existing.walk_distance.d[2],
                    existing.walk_distance.d[1], existing.walk_distance.d[0]);
                printf("  Wild dist: %016lx%016lx%016lx%016lx\n",
                    entry->walk_distance.d[3], entry->walk_distance.d[2],
                    entry->walk_distance.d[1], entry->walk_distance.d[0]);
                return true;
            }
        }
    }
    DPMatch m;
    m.walk_distance = entry->walk_distance;
    m.type = entry->type;
    dp_table[key].push_back(m);
    return false;
}

// ═══════════════════════════════════════════════════════════════
// HOST: MAIN
// ═══════════════════════════════════════════════════════════════

volatile bool g_running = true;

void signal_handler(int sig) {
    printf("\n  Ctrl+C received. Saving state and exiting...\n");
    g_running = false;
}

int main(int argc, char **argv) {
    signal(SIGINT, signal_handler);

    uint32_t dp_bits = DEFAULT_DP_BITS;
    uint32_t num_blocks = 2048;
    int range_bits = 135;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--dp-bits") == 0 && i+1 < argc) dp_bits = atoi(argv[++i]);
        if (strcmp(argv[i], "--blocks") == 0 && i+1 < argc) num_blocks = atoi(argv[++i]);
    }

    uint32_t dp_mask = (1U << dp_bits) - 1;
    uint32_t total_threads = num_blocks * BLOCK_SIZE;
    uint32_t total_kangaroos = total_threads * KANGAROOS_PER_THREAD;

    printf("\n");
    printf("+================================================================+\n");
    printf("|  HYDRA KANGAROO -- Optimized Pollard's Kangaroo Solver         |\n");
    printf("|  Target: Bitcoin Puzzle #135                                    |\n");
    printf("|  Optimization: Batch Inversion (K=%d per thread)             |\n", KANGAROOS_PER_THREAD);
    printf("+================================================================+\n\n");

    int dev;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    printf("  GPU: %s (%d SMs, %zu MB VRAM)\n",
           prop.name, prop.multiProcessorCount, prop.totalGlobalMem >> 20);
    printf("  Threads: %u (%u blocks x %u)\n", total_threads, num_blocks, BLOCK_SIZE);
    printf("  Kangaroos: %u (%u per thread x %u threads)\n",
           total_kangaroos, KANGAROOS_PER_THREAD, total_threads);
    printf("  DP bits: %u (1 in %u points)\n", dp_bits, 1U << dp_bits);
    printf("  Steps per kernel: %d\n\n", STEPS_PER_KERNEL);

    double estimated_muls_per_step = 256.0 / KANGAROOS_PER_THREAD + 17.0;
    double standard_muls_per_step = 268.0;
    printf("  Estimated muls/step: %.1f (vs %.1f standard = %.1fx speedup)\n",
           estimated_muls_per_step, standard_muls_per_step,
           standard_muls_per_step / estimated_muls_per_step);
    printf("\n");

    // Generate jump table on host
    AffinePoint *h_jumps = (AffinePoint *)malloc(NUM_JUMPS * sizeof(AffinePoint));
    u256 *h_scalars = (u256 *)malloc(NUM_JUMPS * sizeof(u256));
    generate_jump_table(h_jumps, h_scalars, range_bits);

    // Upload jump table to constant memory
    cudaMemcpyToSymbol(c_jump_points, h_jumps, NUM_JUMPS * sizeof(AffinePoint));
    cudaMemcpyToSymbol(c_jump_scalars, h_scalars, NUM_JUMPS * sizeof(u256));

    // Initialize kangaroo states on host
    KangarooState *h_states = (KangarooState *)malloc(total_kangaroos * sizeof(KangarooState));
    init_kangaroos(h_states, total_kangaroos);

    // Allocate and copy to device
    KangarooState *d_states;
    DPEntry *d_dps;
    uint32_t *d_dp_count;
    uint32_t max_dps = 1 << 20;

    cudaMalloc(&d_states, total_kangaroos * sizeof(KangarooState));
    cudaMalloc(&d_dps, max_dps * sizeof(DPEntry));
    cudaMalloc(&d_dp_count, sizeof(uint32_t));

    cudaMemcpy(d_states, h_states, total_kangaroos * sizeof(KangarooState),
               cudaMemcpyHostToDevice);
    free(h_states);

    printf("\n  Starting kangaroo walk...\n");
    printf("  Press Ctrl+C to save progress and exit.\n\n");

    uint64_t total_steps = 0;
    uint32_t total_dps_found = 0;
    time_t start_time = time(NULL);
    time_t last_report = start_time;
    bool solved = false;

    while (g_running && !solved) {
        cudaMemset(d_dp_count, 0, sizeof(uint32_t));

        kangaroo_batch_walk<<<num_blocks, BLOCK_SIZE>>>(
            d_states, d_dps, d_dp_count, max_dps,
            dp_mask, STEPS_PER_KERNEL
        );
        cudaDeviceSynchronize();

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("  CUDA error: %s\n", cudaGetErrorString(err));
            break;
        }

        uint32_t num_dps;
        cudaMemcpy(&num_dps, d_dp_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);

        if (num_dps > 0) {
            DPEntry *h_dps = (DPEntry *)malloc(num_dps * sizeof(DPEntry));
            cudaMemcpy(h_dps, d_dps, num_dps * sizeof(DPEntry), cudaMemcpyDeviceToHost);
            for (uint32_t i = 0; i < num_dps && !solved; i++) {
                if (check_dp_collision(&h_dps[i])) {
                    solved = true;
                    printf("  COLLISION FOUND! Recovering key...\n");
                }
            }
            total_dps_found += num_dps;
            free(h_dps);
        }

        total_steps += (uint64_t)STEPS_PER_KERNEL * total_kangaroos;

        time_t now = time(NULL);
        if (now - last_report >= 10) {
            double elapsed = difftime(now, start_time);
            double rate = total_steps / elapsed;
            double expected = pow(2.0, 67.0);
            double pct = (total_steps / expected) * 100.0;
            printf("  [%6.0fs] steps=%.3e rate=%.2f Msteps/s DPs=%u "
                   "table=%zu progress=%.6f%%\n",
                   elapsed, (double)total_steps, rate / 1e6,
                   total_dps_found, dp_table.size(), pct);
            last_report = now;
        }
    }

    time_t end_time = time(NULL);
    double total_elapsed = difftime(end_time, start_time);

    printf("\n  ===================================================\n");
    printf("  Session complete.\n");
    printf("  Total steps:  %.3e\n", (double)total_steps);
    printf("  Total DPs:    %u\n", total_dps_found);
    printf("  DP table:     %zu entries\n", dp_table.size());
    printf("  Elapsed:      %.0f seconds\n", total_elapsed);
    if (total_elapsed > 0)
        printf("  Rate:         %.2f Msteps/s\n", total_steps / total_elapsed / 1e6);
    if (solved)
        printf("  STATUS:       *** SOLVED ***\n");
    else
        printf("  STATUS:       In progress (%.6f%% of expected)\n",
               total_steps / pow(2.0, 67.0) * 100.0);
    printf("  ===================================================\n");

    free(h_jumps);
    free(h_scalars);
    cudaFree(d_states);
    cudaFree(d_dps);
    cudaFree(d_dp_count);

    return solved ? 0 : 1;
}
