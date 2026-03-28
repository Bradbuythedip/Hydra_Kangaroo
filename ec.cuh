#pragma once
/*
 * ec.cuh — secp256k1 elliptic curve point arithmetic
 *
 * Uses Jacobian projective coordinates: (X, Y, Z) = affine (X/Z², Y/Z³)
 * This avoids inversions in the hot loop.
 * 
 * Key operations:
 *   ec_double_j:       Jacobian doubling (4M + 4S)
 *   ec_add_mixed:      Jacobian + Affine → Jacobian (8M + 3S)  ← THE HOT PATH
 *   ec_to_affine:      Jacobian → Affine (1 inversion + 3M)
 *   ec_to_affine_batch: Batch Jacobian → Affine (1 inv + 3K muls)
 *
 * Formulas from: https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html
 * (a=0 for secp256k1)
 */

#include "field.cuh"

// ═══════════════════════════════════════════════════════════════
// POINT TYPES
// ═══════════════════════════════════════════════════════════════

// Affine point: (x, y)
typedef struct {
    u256 x;
    u256 y;
} AffinePoint;

// Jacobian point: (X, Y, Z) represents affine (X/Z², Y/Z³)
typedef struct {
    u256 X;
    u256 Y;
    u256 Z;
} JacobianPoint;

// secp256k1 generator point
static const AffinePoint GENERATOR = {
    {{0x59F2815B16F81798ULL, 0x029BFCDB2DCE28D9ULL,
      0x55A06295CE870B07ULL, 0x79BE667EF9DCBBACULL}},
    {{0x9C47D08FFB10D4B8ULL, 0xFD17B448A6855419ULL,
      0x5DA4FBFC0E1108A8ULL, 0x483ADA7726A3C465ULL}}
};

// Endomorphism: β such that (βx, y) = λ·(x, y) on the curve
static const u256 ENDO_BETA = {{
    0xC1396C28719501EEULL, 0x9CF0497512F58995ULL,
    0x6E64479EAC3434E9ULL, 0x7AE96A2B657C0710ULL
}};

// ═══════════════════════════════════════════════════════════════
// JACOBIAN POINT DOUBLING
// dbl-2009-l: 1M + 5S + 1*a + 4add + 2*2 + 1*3 + 1*8
// Since a=0 for secp256k1: 1M + 5S + 4add + 2*2 + 1*3 + 1*8
// ═══════════════════════════════════════════════════════════════

__device__
JacobianPoint ec_double_j(const JacobianPoint *p) {
    u256 A = fp_sqr(&p->X);           // A = X²
    u256 B = fp_sqr(&p->Y);           // B = Y²
    u256 C = fp_sqr(&B);              // C = B² = Y⁴
    
    // D = 2*((X+B)²-A-C)
    u256 xpb = fp_add(&p->X, &B);
    u256 xpb2 = fp_sqr(&xpb);
    u256 t1 = fp_sub(&xpb2, &A);
    u256 t2 = fp_sub(&t1, &C);
    u256 D = fp_dbl(&t2);
    
    // E = 3*A (since a=0, no a*Z^4 term)
    u256 E = fp_add(&A, &A);
    E = fp_add(&E, &A);
    
    u256 F = fp_sqr(&E);              // F = E²
    
    JacobianPoint r;
    
    // X3 = F - 2*D
    u256 D2 = fp_dbl(&D);
    r.X = fp_sub(&F, &D2);
    
    // Y3 = E*(D-X3) - 8*C
    u256 dxr = fp_sub(&D, &r.X);
    u256 edx = fp_mul(&E, &dxr);
    u256 C2 = fp_dbl(&C);
    u256 C4 = fp_dbl(&C2);
    u256 C8 = fp_dbl(&C4);
    r.Y = fp_sub(&edx, &C8);
    
    // Z3 = 2*Y*Z
    u256 yz = fp_mul(&p->Y, &p->Z);
    r.Z = fp_dbl(&yz);
    
    return r;
}

// ═══════════════════════════════════════════════════════════════
// MIXED ADDITION: Jacobian + Affine → Jacobian
// madd-2008-g: 7M + 4S + 9add + 3*2 + 1*4
// This is THE hot path — called once per kangaroo step.
// ═══════════════════════════════════════════════════════════════

__device__
JacobianPoint ec_add_mixed(const JacobianPoint *p, const AffinePoint *q) {
    // Special case: if P is identity
    if (u256_is_zero(&p->Z)) {
        JacobianPoint r;
        r.X = q->x;
        r.Y = q->y;
        r.Z.d[0] = 1; r.Z.d[1] = 0; r.Z.d[2] = 0; r.Z.d[3] = 0;
        return r;
    }
    
    u256 Z1Z1 = fp_sqr(&p->Z);               // Z1² 
    u256 U2 = fp_mul(&q->x, &Z1Z1);          // U2 = X2·Z1²
    u256 Z1_3 = fp_mul(&Z1Z1, &p->Z);        // Z1³
    u256 S2 = fp_mul(&q->y, &Z1_3);          // S2 = Y2·Z1³
    
    u256 H = fp_sub(&U2, &p->X);             // H = U2 - X1
    u256 HH = fp_sqr(&H);                    // HH = H²
    u256 I = fp_dbl(&HH);
    I = fp_dbl(&I);                           // I = 4*H²
    u256 J = fp_mul(&H, &I);                 // J = H·I
    
    u256 rr = fp_sub(&S2, &p->Y);
    rr = fp_dbl(&rr);                         // r = 2*(S2 - Y1)
    
    u256 V = fp_mul(&p->X, &I);              // V = X1·I
    
    JacobianPoint res;
    
    // X3 = r² - J - 2*V
    u256 r2 = fp_sqr(&rr);
    u256 V2 = fp_dbl(&V);
    u256 t = fp_sub(&r2, &J);
    res.X = fp_sub(&t, &V2);
    
    // Y3 = r*(V - X3) - 2*Y1*J
    u256 vmx = fp_sub(&V, &res.X);
    u256 rvmx = fp_mul(&rr, &vmx);
    u256 Y1J = fp_mul(&p->Y, &J);
    u256 Y1J2 = fp_dbl(&Y1J);
    res.Y = fp_sub(&rvmx, &Y1J2);
    
    // Z3 = (Z1+H)² - Z1Z1 - HH  =  2*Z1*H (equivalent, cheaper)
    u256 z1h = fp_mul(&p->Z, &H);
    res.Z = fp_dbl(&z1h);
    
    return res;
}

// ═══════════════════════════════════════════════════════════════
// JACOBIAN → AFFINE (single point)
// Cost: 1 inversion (256 muls) + 2 multiplications
// THIS IS WHAT BATCH INVERSION REPLACES
// ═══════════════════════════════════════════════════════════════

__device__
AffinePoint ec_to_affine(const JacobianPoint *p) {
    u256 z_inv = fp_inv(&p->Z);
    u256 z_inv2 = fp_sqr(&z_inv);
    u256 z_inv3 = fp_mul(&z_inv2, &z_inv);
    
    AffinePoint r;
    r.x = fp_mul(&p->X, &z_inv2);
    r.y = fp_mul(&p->Y, &z_inv3);
    return r;
}

// ═══════════════════════════════════════════════════════════════
// BATCH JACOBIAN → AFFINE (K points, 1 inversion total)
//
// THE KEY OPTIMIZATION: amortize 1 inversion across K points.
// Cost: 1 inversion + ~3K multiplications
//       vs K inversions = 256K multiplications
// Speedup: 256K / (256 + 3K) → ~10x at K=32
// ═══════════════════════════════════════════════════════════════

template<int K>
__device__
void ec_batch_to_affine(const JacobianPoint *points, AffinePoint *out) {
    // Extract Z values
    u256 z_vals[K];
    #pragma unroll
    for (int i = 0; i < K; i++) {
        z_vals[i] = points[i].Z;
    }
    
    // Batch invert all Z values
    u256 z_invs[K];
    fp_batch_inv<K>(z_vals, z_invs);
    
    // Convert each point using its Z inverse
    #pragma unroll
    for (int i = 0; i < K; i++) {
        u256 zi2 = fp_sqr(&z_invs[i]);
        u256 zi3 = fp_mul(&zi2, &z_invs[i]);
        out[i].x = fp_mul(&points[i].X, &zi2);
        out[i].y = fp_mul(&points[i].Y, &zi3);
    }
}

// ═══════════════════════════════════════════════════════════════
// ENDOMORPHISM: λ·P = (β·x, y) where β³ ≡ 1 (mod p)
// Cost: 1 field multiplication (β·x)
// Gets us 3 points for 1 mul: P, λP, λ²P
// ═══════════════════════════════════════════════════════════════

__device__ __forceinline__
AffinePoint ec_endomorphism(const AffinePoint *p) {
    AffinePoint r;
    r.x = fp_mul(&ENDO_BETA, &p->x);
    r.y = p->y;
    return r;
}

__device__ __forceinline__
AffinePoint ec_endomorphism2(const AffinePoint *p) {
    AffinePoint t = ec_endomorphism(p);
    return ec_endomorphism(&t);
}

// ═══════════════════════════════════════════════════════════════
// SCALAR MULTIPLICATION (for initialization only, not hot path)
// Double-and-add. Not optimized — only called at kernel start.
// ═══════════════════════════════════════════════════════════════

__device__
JacobianPoint ec_scalar_mul(const u256 *k, const AffinePoint *p) {
    JacobianPoint result;
    result.X.d[0] = 0; result.X.d[1] = 0; result.X.d[2] = 0; result.X.d[3] = 0;
    result.Y.d[0] = 1; result.Y.d[1] = 0; result.Y.d[2] = 0; result.Y.d[3] = 0;
    result.Z.d[0] = 0; result.Z.d[1] = 0; result.Z.d[2] = 0; result.Z.d[3] = 0;
    
    for (int bit = 255; bit >= 0; bit--) {
        result = ec_double_j(&result);
        int word = bit >> 6;
        int pos = bit & 63;
        if ((k->d[word] >> pos) & 1) {
            result = ec_add_mixed(&result, p);
        }
    }
    
    return result;
}
