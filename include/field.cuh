#pragma once
/*
 * field.cuh — secp256k1 field arithmetic (mod P)
 *
 * P = 2^256 - 2^32 - 977 = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
 *
 * Uses 4×uint64 limb representation.
 * Reduction exploits P's special form: 2^256 ≡ 2^32 + 977 (mod P)
 * All operations stay in [0, P).
 */

#include <stdint.h>

// ═══════════════════════════════════════════════════════════════
// 256-BIT TYPE
// ═══════════════════════════════════════════════════════════════

typedef struct __align__(32) {
    uint64_t d[4]; // d[0] = least significant
} u256;

static const u256 SECP256K1_P = {{
    0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL
}};

static const u256 SECP256K1_N = {{
    0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL,
    0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL
}};

// 2^32 + 977 — the reduction constant for secp256k1
#define SECP256K1_C 0x1000003D1ULL

// ═══════════════════════════════════════════════════════════════
// COMPARISON AND BASIC OPS
// ═══════════════════════════════════════════════════════════════

__device__ __forceinline__
int u256_is_zero(const u256 *a) {
    return (a->d[0] | a->d[1] | a->d[2] | a->d[3]) == 0;
}

__device__ __forceinline__
int u256_gte(const u256 *a, const u256 *b) {
    #pragma unroll
    for (int i = 3; i >= 0; i--) {
        if (a->d[i] > b->d[i]) return 1;
        if (a->d[i] < b->d[i]) return 0;
    }
    return 1;
}

__device__ __forceinline__
int u256_eq(const u256 *a, const u256 *b) {
    return (a->d[0] == b->d[0]) && (a->d[1] == b->d[1]) &&
           (a->d[2] == b->d[2]) && (a->d[3] == b->d[3]);
}

// ═══════════════════════════════════════════════════════════════
// ADDITION / SUBTRACTION (with carry/borrow via PTX intrinsics)
// ═══════════════════════════════════════════════════════════════

__device__ __forceinline__
u256 u256_add_cc(const u256 *a, const u256 *b, uint32_t *carry_out) {
    u256 r;
    uint32_t cc = 0;
    asm("add.cc.u64  %0, %1, %2;" : "=l"(r.d[0]) : "l"(a->d[0]), "l"(b->d[0]));
    asm("addc.cc.u64 %0, %1, %2;" : "=l"(r.d[1]) : "l"(a->d[1]), "l"(b->d[1]));
    asm("addc.cc.u64 %0, %1, %2;" : "=l"(r.d[2]) : "l"(a->d[2]), "l"(b->d[2]));
    asm("addc.cc.u64 %0, %1, %2;" : "=l"(r.d[3]) : "l"(a->d[3]), "l"(b->d[3]));
    asm("addc.u32    %0,  0,  0;" : "=r"(cc));
    *carry_out = cc;
    return r;
}

__device__ __forceinline__
u256 u256_sub_borrow(const u256 *a, const u256 *b, uint32_t *borrow_out) {
    u256 r;
    uint32_t bb = 0;
    asm("sub.cc.u64  %0, %1, %2;" : "=l"(r.d[0]) : "l"(a->d[0]), "l"(b->d[0]));
    asm("subc.cc.u64 %0, %1, %2;" : "=l"(r.d[1]) : "l"(a->d[1]), "l"(b->d[1]));
    asm("subc.cc.u64 %0, %1, %2;" : "=l"(r.d[2]) : "l"(a->d[2]), "l"(b->d[2]));
    asm("subc.cc.u64 %0, %1, %2;" : "=l"(r.d[3]) : "l"(a->d[3]), "l"(b->d[3]));
    asm("subc.u32    %0,  0,  0;" : "=r"(bb));
    *borrow_out = bb;
    return r;
}

// ═══════════════════════════════════════════════════════════════
// FIELD OPERATIONS (mod P)
// ═══════════════════════════════════════════════════════════════

__device__ __forceinline__
u256 fp_add(const u256 *a, const u256 *b) {
    uint32_t carry;
    u256 r = u256_add_cc(a, b, &carry);
    if (carry || u256_gte(&r, &SECP256K1_P)) {
        uint32_t dummy;
        r = u256_sub_borrow(&r, &SECP256K1_P, &dummy);
    }
    return r;
}

__device__ __forceinline__
u256 fp_sub(const u256 *a, const u256 *b) {
    uint32_t borrow;
    u256 r = u256_sub_borrow(a, b, &borrow);
    if (borrow) {
        uint32_t dummy;
        r = u256_add_cc(&r, &SECP256K1_P, &dummy);
    }
    return r;
}

__device__ __forceinline__
u256 fp_dbl(const u256 *a) {
    return fp_add(a, a);
}

__device__ __forceinline__
u256 fp_neg(const u256 *a) {
    if (u256_is_zero(a)) return *a;
    uint32_t dummy;
    return u256_sub_borrow(&SECP256K1_P, a, &dummy);
}

// ═══════════════════════════════════════════════════════════════
// MULTIPLICATION (schoolbook + secp256k1 fast reduction)
//
// Uses the identity: 2^256 ≡ 0x1000003D1 (mod P)
// So for a 512-bit product [hi:lo]:
//   result ≡ lo + hi * 0x1000003D1  (mod P)
// ═══════════════════════════════════════════════════════════════

__device__
u256 fp_mul(const u256 *a, const u256 *b) {
    // 512-bit schoolbook multiply into p[0..7]
    uint64_t p[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            uint64_t lo = a->d[i] * b->d[j];
            uint64_t hi = __umul64hi(a->d[i], b->d[j]);

            // Add lo to accumulator
            uint64_t sum = p[i+j] + lo;
            uint64_t c1 = (sum < p[i+j]) ? 1ULL : 0ULL;
            // Add carry from previous iteration
            uint64_t sum2 = sum + carry;
            uint64_t c2 = (sum2 < sum) ? 1ULL : 0ULL;
            p[i+j] = sum2;
            carry = hi + c1 + c2;
        }
        p[i+4] = carry;
    }

    // Reduce: result = p_lo + p_hi * C mod P
    // p_hi = [p[7], p[6], p[5], p[4]], p_lo = [p[3], p[2], p[1], p[0]]

    // Multiply p_hi by C = 0x1000003D1
    uint64_t carry = 0;
    uint64_t q[5];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint64_t lo = p[i+4] * SECP256K1_C;
        uint64_t hi = __umul64hi(p[i+4], SECP256K1_C);
        uint64_t sum = lo + carry;
        uint64_t c = (sum < lo) ? 1ULL : 0ULL;
        carry = hi + c;
        q[i] = sum;
    }
    q[4] = carry;

    // Add p_lo + q using PTX carry chain
    u256 r;
    uint64_t c;

    // Limb 0
    uint64_t s = p[0] + q[0];
    c = (s < p[0]) ? 1ULL : 0ULL;
    r.d[0] = s;

    // Limb 1
    s = p[1] + q[1];
    uint64_t c1 = (s < p[1]) ? 1ULL : 0ULL;
    uint64_t s2 = s + c;
    uint64_t c2 = (s2 < s) ? 1ULL : 0ULL;
    r.d[1] = s2;
    c = c1 + c2;

    // Limb 2
    s = p[2] + q[2];
    c1 = (s < p[2]) ? 1ULL : 0ULL;
    s2 = s + c;
    c2 = (s2 < s) ? 1ULL : 0ULL;
    r.d[2] = s2;
    c = c1 + c2;

    // Limb 3
    s = p[3] + q[3];
    c1 = (s < p[3]) ? 1ULL : 0ULL;
    s2 = s + c;
    c2 = (s2 < s) ? 1ULL : 0ULL;
    r.d[3] = s2;
    c = c1 + c2;

    // Second reduction: overflow = carry_out + q[4]
    // overflow * C can be up to ~2^67, need full 128-bit multiply
    uint64_t overflow = c + q[4];
    if (overflow) {
        uint64_t red_lo = overflow * SECP256K1_C;
        uint64_t red_hi = __umul64hi(overflow, SECP256K1_C);

        // Add red_lo to r.d[0]
        s = r.d[0] + red_lo;
        c = (s < r.d[0]) ? 1ULL : 0ULL;
        r.d[0] = s;

        // Add red_hi + carry to r.d[1]
        uint64_t add1 = red_hi + c;
        s = r.d[1] + add1;
        c = (s < r.d[1]) ? 1ULL : 0ULL;
        r.d[1] = s;

        // Propagate carry through r.d[2], r.d[3]
        if (c) {
            r.d[2]++;
            if (r.d[2] == 0) {
                r.d[3]++;
                // If r.d[3] overflows, need one more reduction (extremely rare)
                if (r.d[3] == 0) {
                    // Value wrapped past 2^256, reduce by adding C
                    s = r.d[0] + SECP256K1_C;
                    c = (s < r.d[0]) ? 1ULL : 0ULL;
                    r.d[0] = s;
                    if (c) { r.d[1]++; if (r.d[1] == 0) { r.d[2]++; if (r.d[2] == 0) r.d[3]++; } }
                }
            }
        }
    }

    // Final conditional subtraction (at most once)
    if (u256_gte(&r, &SECP256K1_P)) {
        uint32_t dummy;
        r = u256_sub_borrow(&r, &SECP256K1_P, &dummy);
    }

    return r;
}

// ═══════════════════════════════════════════════════════════════
// OPTIMIZED SQUARING
//
// Exploits symmetry: a[i]*a[j] = a[j]*a[i], so cross-terms
// are computed once and doubled. 6 muls + 4 sqrs + shifts
// vs 16 muls for generic multiplication.
// ═══════════════════════════════════════════════════════════════

__device__
u256 fp_sqr(const u256 *a) {
    uint64_t p[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    // Cross terms (i < j): compute once, will double later
    // (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    uint64_t cross[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    // a[0]*a[1] at position 1
    {
        uint64_t lo = a->d[0] * a->d[1];
        uint64_t hi = __umul64hi(a->d[0], a->d[1]);
        cross[1] += lo;
        cross[2] += hi + ((cross[1] < lo) ? 1ULL : 0ULL);
    }
    // a[0]*a[2] at position 2
    {
        uint64_t lo = a->d[0] * a->d[2];
        uint64_t hi = __umul64hi(a->d[0], a->d[2]);
        uint64_t old = cross[2];
        cross[2] += lo;
        uint64_t c = (cross[2] < old) ? 1ULL : 0ULL;
        cross[3] += hi + c;
    }
    // a[0]*a[3] at position 3
    {
        uint64_t lo = a->d[0] * a->d[3];
        uint64_t hi = __umul64hi(a->d[0], a->d[3]);
        uint64_t old = cross[3];
        cross[3] += lo;
        uint64_t c = (cross[3] < old) ? 1ULL : 0ULL;
        cross[4] += hi + c;
    }
    // a[1]*a[2] at position 3
    {
        uint64_t lo = a->d[1] * a->d[2];
        uint64_t hi = __umul64hi(a->d[1], a->d[2]);
        uint64_t old = cross[3];
        cross[3] += lo;
        uint64_t c = (cross[3] < old) ? 1ULL : 0ULL;
        old = cross[4];
        cross[4] += hi + c;
        c = (cross[4] < old) ? 1ULL : 0ULL;
        cross[5] += c;
    }
    // a[1]*a[3] at position 4
    {
        uint64_t lo = a->d[1] * a->d[3];
        uint64_t hi = __umul64hi(a->d[1], a->d[3]);
        uint64_t old = cross[4];
        cross[4] += lo;
        uint64_t c = (cross[4] < old) ? 1ULL : 0ULL;
        old = cross[5];
        cross[5] += hi + c;
        c = (cross[5] < old) ? 1ULL : 0ULL;
        cross[6] += c;
    }
    // a[2]*a[3] at position 5
    {
        uint64_t lo = a->d[2] * a->d[3];
        uint64_t hi = __umul64hi(a->d[2], a->d[3]);
        uint64_t old = cross[5];
        cross[5] += lo;
        uint64_t c = (cross[5] < old) ? 1ULL : 0ULL;
        cross[6] += hi + c;
    }

    // Double the cross terms (shift left by 1 bit)
    cross[7] = cross[6] >> 63;
    cross[6] = (cross[6] << 1) | (cross[5] >> 63);
    cross[5] = (cross[5] << 1) | (cross[4] >> 63);
    cross[4] = (cross[4] << 1) | (cross[3] >> 63);
    cross[3] = (cross[3] << 1) | (cross[2] >> 63);
    cross[2] = (cross[2] << 1) | (cross[1] >> 63);
    cross[1] = cross[1] << 1;
    // cross[0] is always 0 (no cross term at position 0)

    // Add diagonal terms (a[i]^2) and combine with doubled cross terms
    // a[0]^2 at position 0,1
    {
        uint64_t lo = a->d[0] * a->d[0];
        uint64_t hi = __umul64hi(a->d[0], a->d[0]);
        p[0] = lo;
        uint64_t s = cross[1] + hi;
        uint64_t c = (s < cross[1]) ? 1ULL : 0ULL;
        p[1] = s;
        // carry c propagates into position 2
        uint64_t old = cross[2];
        cross[2] += c;
        if (cross[2] < old) { cross[3]++; if (cross[3] == 0) { cross[4]++; if (cross[4] == 0) { cross[5]++; if (cross[5] == 0) { cross[6]++; if (cross[6] == 0) cross[7]++; }}}}
    }
    // a[1]^2 at position 2,3
    {
        uint64_t lo = a->d[1] * a->d[1];
        uint64_t hi = __umul64hi(a->d[1], a->d[1]);
        uint64_t s = cross[2] + lo;
        uint64_t c = (s < cross[2]) ? 1ULL : 0ULL;
        p[2] = s;
        s = cross[3] + hi;
        uint64_t c2 = (s < cross[3]) ? 1ULL : 0ULL;
        s += c;
        c2 += (s < c) ? 1ULL : 0ULL;
        p[3] = s;
        if (c2) { cross[4] += c2; if (cross[4] < c2) { cross[5]++; if (cross[5] == 0) { cross[6]++; if (cross[6] == 0) cross[7]++; }}}
    }
    // a[2]^2 at position 4,5
    {
        uint64_t lo = a->d[2] * a->d[2];
        uint64_t hi = __umul64hi(a->d[2], a->d[2]);
        uint64_t s = cross[4] + lo;
        uint64_t c = (s < cross[4]) ? 1ULL : 0ULL;
        p[4] = s;
        s = cross[5] + hi;
        uint64_t c2 = (s < cross[5]) ? 1ULL : 0ULL;
        s += c;
        c2 += (s < c) ? 1ULL : 0ULL;
        p[5] = s;
        if (c2) { cross[6] += c2; if (cross[6] < c2) cross[7]++; }
    }
    // a[3]^2 at position 6,7
    {
        uint64_t lo = a->d[3] * a->d[3];
        uint64_t hi = __umul64hi(a->d[3], a->d[3]);
        uint64_t s = cross[6] + lo;
        uint64_t c = (s < cross[6]) ? 1ULL : 0ULL;
        p[6] = s;
        s = cross[7] + hi + c;
        p[7] = s;
    }

    // Now reduce p[0..7] mod P, same as in fp_mul
    uint64_t carry = 0;
    uint64_t q[5];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint64_t lo = p[i+4] * SECP256K1_C;
        uint64_t hi = __umul64hi(p[i+4], SECP256K1_C);
        uint64_t sum = lo + carry;
        uint64_t cc = (sum < lo) ? 1ULL : 0ULL;
        carry = hi + cc;
        q[i] = sum;
    }
    q[4] = carry;

    u256 r;
    uint64_t c;

    uint64_t ss = p[0] + q[0];
    c = (ss < p[0]) ? 1ULL : 0ULL;
    r.d[0] = ss;

    ss = p[1] + q[1];
    uint64_t cc1 = (ss < p[1]) ? 1ULL : 0ULL;
    uint64_t ss2 = ss + c;
    uint64_t cc2 = (ss2 < ss) ? 1ULL : 0ULL;
    r.d[1] = ss2;
    c = cc1 + cc2;

    ss = p[2] + q[2];
    cc1 = (ss < p[2]) ? 1ULL : 0ULL;
    ss2 = ss + c;
    cc2 = (ss2 < ss) ? 1ULL : 0ULL;
    r.d[2] = ss2;
    c = cc1 + cc2;

    ss = p[3] + q[3];
    cc1 = (ss < p[3]) ? 1ULL : 0ULL;
    ss2 = ss + c;
    cc2 = (ss2 < ss) ? 1ULL : 0ULL;
    r.d[3] = ss2;
    c = cc1 + cc2;

    uint64_t overflow = c + q[4];
    if (overflow) {
        uint64_t red_lo = overflow * SECP256K1_C;
        uint64_t red_hi = __umul64hi(overflow, SECP256K1_C);

        ss = r.d[0] + red_lo;
        c = (ss < r.d[0]) ? 1ULL : 0ULL;
        r.d[0] = ss;

        uint64_t add1 = red_hi + c;
        ss = r.d[1] + add1;
        c = (ss < r.d[1]) ? 1ULL : 0ULL;
        r.d[1] = ss;

        if (c) {
            r.d[2]++;
            if (r.d[2] == 0) {
                r.d[3]++;
                if (r.d[3] == 0) {
                    ss = r.d[0] + SECP256K1_C;
                    c = (ss < r.d[0]) ? 1ULL : 0ULL;
                    r.d[0] = ss;
                    if (c) { r.d[1]++; if (r.d[1] == 0) { r.d[2]++; if (r.d[2] == 0) r.d[3]++; } }
                }
            }
        }
    }

    if (u256_gte(&r, &SECP256K1_P)) {
        uint32_t dummy;
        r = u256_sub_borrow(&r, &SECP256K1_P, &dummy);
    }

    return r;
}

// ═══════════════════════════════════════════════════════════════
// FIELD INVERSION: Optimized addition chain for a^(P-2) mod P
//
// P-2 = 2^256 - 2^32 - 979
//
// Uses the secp256k1-specific chain from libsecp256k1:
//   Build x2=a^3, x3=a^7, ..., x223=a^(2^223-1)
//   Then: 23 sqr, *x22, 5 sqr, *a, 3 sqr, *x2, 2 sqr, *a
//
// Total: 255 squarings + 15 multiplications
// ═══════════════════════════════════════════════════════════════

__device__
u256 fp_inv(const u256 *a) {
    u256 x2, x3, x6, x9, x11, x22, x44, x88, x176, x220, x223, t;
    int i;

    // x2 = a^(2^2 - 1) = a^3
    x2 = fp_sqr(a);
    x2 = fp_mul(&x2, a);

    // x3 = a^(2^3 - 1) = a^7
    x3 = fp_sqr(&x2);
    x3 = fp_mul(&x3, a);

    // x6 = a^(2^6 - 1)
    x6 = x3;
    for (i = 0; i < 3; i++) x6 = fp_sqr(&x6);
    x6 = fp_mul(&x6, &x3);

    // x9 = a^(2^9 - 1)
    x9 = x6;
    for (i = 0; i < 3; i++) x9 = fp_sqr(&x9);
    x9 = fp_mul(&x9, &x3);

    // x11 = a^(2^11 - 1)
    x11 = x9;
    for (i = 0; i < 2; i++) x11 = fp_sqr(&x11);
    x11 = fp_mul(&x11, &x2);

    // x22 = a^(2^22 - 1)
    x22 = x11;
    for (i = 0; i < 11; i++) x22 = fp_sqr(&x22);
    x22 = fp_mul(&x22, &x11);

    // x44 = a^(2^44 - 1)
    x44 = x22;
    for (i = 0; i < 22; i++) x44 = fp_sqr(&x44);
    x44 = fp_mul(&x44, &x22);

    // x88 = a^(2^88 - 1)
    x88 = x44;
    for (i = 0; i < 44; i++) x88 = fp_sqr(&x88);
    x88 = fp_mul(&x88, &x44);

    // x176 = a^(2^176 - 1)
    x176 = x88;
    for (i = 0; i < 88; i++) x176 = fp_sqr(&x176);
    x176 = fp_mul(&x176, &x88);

    // x220 = a^(2^220 - 1)
    x220 = x176;
    for (i = 0; i < 44; i++) x220 = fp_sqr(&x220);
    x220 = fp_mul(&x220, &x44);

    // x223 = a^(2^223 - 1)
    x223 = x220;
    for (i = 0; i < 3; i++) x223 = fp_sqr(&x223);
    x223 = fp_mul(&x223, &x3);

    // Final assembly: P-2 = (2^223-1)*2^33 + 2^32 - 2^10 + 45
    // = (2^223-1)*2^33 + 0xFFFFFC2D

    // Square 23 times, multiply by x22
    t = x223;
    for (i = 0; i < 23; i++) t = fp_sqr(&t);
    t = fp_mul(&t, &x22);

    // Square 5 times, multiply by a
    for (i = 0; i < 5; i++) t = fp_sqr(&t);
    t = fp_mul(&t, a);

    // Square 3 times, multiply by x2
    for (i = 0; i < 3; i++) t = fp_sqr(&t);
    t = fp_mul(&t, &x2);

    // Square 2 times, multiply by a
    for (i = 0; i < 2; i++) t = fp_sqr(&t);
    t = fp_mul(&t, a);

    return t;
}

// ═══════════════════════════════════════════════════════════════
// BATCH INVERSION — Montgomery's trick
//
// Invert K values using 1 inversion + 3(K-1) multiplications.
// This is THE key optimization: 256 muls → 256/K + 3 muls per element.
// ═══════════════════════════════════════════════════════════════

template<int K>
__device__
void fp_batch_inv(u256 *values, u256 *results) {
    // Step 1: Build product tree (forward pass)
    u256 partials[K];
    partials[0] = values[0];
    #pragma unroll
    for (int i = 1; i < K; i++) {
        partials[i] = fp_mul(&partials[i-1], &values[i]);
    }

    // Step 2: Single inversion of the total product
    u256 inv_total = fp_inv(&partials[K-1]);

    // Step 3: Peel back (backward pass)
    #pragma unroll
    for (int i = K-1; i > 0; i--) {
        results[i] = fp_mul(&inv_total, &partials[i-1]);
        inv_total = fp_mul(&inv_total, &values[i]);
    }
    results[0] = inv_total;
}
