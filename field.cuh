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
    // If carry or r >= P, subtract P
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
    // 512-bit schoolbook multiply
    // Using __umul64hi and regular multiply for full 128-bit products
    uint64_t p[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            // p[i+j] += a[i] * b[j] + carry
            // Need 128-bit multiply: use __umul64hi
            uint64_t lo = a->d[i] * b->d[j];
            uint64_t hi = __umul64hi(a->d[i], b->d[j]);
            
            // Add to accumulator with carry chain
            uint64_t sum = p[i+j] + lo;
            uint64_t c1 = (sum < p[i+j]) ? 1ULL : 0ULL;
            sum += carry;
            uint64_t c2 = (sum < carry) ? 1ULL : 0ULL;
            p[i+j] = sum;
            carry = hi + c1 + c2;
        }
        p[i+4] = carry;
    }
    
    // Reduce: result = p_lo + p_hi * 0x1000003D1 mod P
    // p_hi = [p[7], p[6], p[5], p[4]]
    // p_lo = [p[3], p[2], p[1], p[0]]
    
    // Multiply p_hi by C = 0x1000003D1
    uint64_t carry = 0;
    uint64_t q[5];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint64_t lo = p[i+4] * SECP256K1_C;
        uint64_t hi = __umul64hi(p[i+4], SECP256K1_C);
        uint64_t sum = lo + carry;
        carry = hi + ((sum < lo) ? 1ULL : 0ULL);
        q[i] = sum;
    }
    q[4] = carry;
    
    // Add p_lo + q
    u256 r;
    carry = 0;
    uint64_t sum;
    
    sum = p[0] + q[0]; carry = (sum < p[0]) ? 1ULL : 0ULL; r.d[0] = sum;
    sum = p[1] + q[1] + carry; carry = (sum < p[1] || (carry && sum == p[1])) ? 1ULL : 0ULL; r.d[1] = sum;
    sum = p[2] + q[2] + carry; carry = (sum < p[2] || (carry && sum == p[2])) ? 1ULL : 0ULL; r.d[2] = sum;
    sum = p[3] + q[3] + carry; carry = (sum < p[3] || (carry && sum == p[3])) ? 1ULL : 0ULL; r.d[3] = sum;
    
    // Remaining carry + q[4] → multiply by C again
    uint64_t overflow = carry + q[4];
    if (overflow) {
        uint64_t red = overflow * SECP256K1_C;
        sum = r.d[0] + red;
        carry = (sum < r.d[0]) ? 1ULL : 0ULL;
        r.d[0] = sum;
        if (carry) {
            r.d[1]++;
            if (r.d[1] == 0) { r.d[2]++; if (r.d[2] == 0) r.d[3]++; }
        }
    }
    
    // Final conditional subtraction
    while (u256_gte(&r, &SECP256K1_P)) {
        uint32_t dummy;
        r = u256_sub_borrow(&r, &SECP256K1_P, &dummy);
    }
    
    return r;
}

__device__ __forceinline__
u256 fp_sqr(const u256 *a) {
    // TODO: Specialized squaring (saves ~25% muls vs generic multiply)
    // For now, use generic multiply
    return fp_mul(a, a);
}

// ═══════════════════════════════════════════════════════════════
// FIELD INVERSION (Fermat: a^(P-2) mod P)
//
// Uses an addition chain optimized for secp256k1's P-2.
// ~256 squarings + ~40 multiplications.
// THIS IS THE EXPENSIVE OPERATION that batch inversion amortizes.
// ═══════════════════════════════════════════════════════════════

__device__
u256 fp_inv(const u256 *a) {
    // Addition chain for P-2 (simplified; production should use optimal chain)
    u256 x2, x3, x6, x9, x11, x22, x44, x88, x176, x220, x223, t;
    
    x2 = fp_sqr(a);
    x2 = fp_mul(&x2, a);          // a^3
    x3 = fp_sqr(&x2);
    x3 = fp_mul(&x3, a);          // a^7
    // ... (simplified — full chain in production)
    
    // Fallback: binary method (correct but slower)
    u256 result;
    result.d[0] = 1; result.d[1] = 0; result.d[2] = 0; result.d[3] = 0;
    u256 base = *a;
    
    // P-2 in binary
    u256 exp = SECP256K1_P;
    exp.d[0] -= 2;
    
    for (int bit = 255; bit >= 0; bit--) {
        result = fp_sqr(&result);
        int word = bit >> 6;
        int pos = bit & 63;
        if ((exp.d[word] >> pos) & 1) {
            result = fp_mul(&result, a);
        }
    }
    
    return result;
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
        // results[i] = inv_total * partials[i-1]
        results[i] = fp_mul(&inv_total, &partials[i-1]);
        // inv_total = inv_total * values[i]
        inv_total = fp_mul(&inv_total, &values[i]);
    }
    results[0] = inv_total;
}
