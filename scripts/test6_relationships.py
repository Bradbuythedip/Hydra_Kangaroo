#!/usr/bin/env python3
"""
Test 6: Deep relationship analysis between solved keys.

Key insight: For any deterministic wallet, d[i] = f(seed, i).
The function f might be:
  - Additive: d[i] = seed + g(i)
  - Multiplicative: d[i] = seed * g(i) mod n
  - BIP32: d[i] = parent + HMAC(chain, pubkey||i)[:32] mod n

For BIP32 specifically:
  d[i] - d[j] = HMAC(chain, pubkey||i)[:32] - HMAC(chain, pubkey||j)[:32] mod n

We can test: are the DIFFERENCES between consecutive raw key values
consistent with BIP32? Are they consistent with any pattern?

Also test: Is d[i] mod small_prime constant? (Would indicate additive structure)
"""

# secp256k1 curve order
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141

SOLVED = [
    (1, 0x1), (2, 0x3), (3, 0x7), (4, 0x8),
    (5, 0x15), (6, 0x31), (7, 0x4c), (8, 0xe0),
    (9, 0x1d3), (10, 0x202), (11, 0x483), (12, 0xa7b),
    (13, 0x1460), (14, 0x2930), (15, 0x68f3), (16, 0xc936),
    (17, 0x1764f), (18, 0x3080d), (19, 0x5749f), (20, 0xd2c55),
    (21, 0x1ba534), (22, 0x2de40f), (23, 0x556e52), (24, 0xdc2a04),
    (25, 0x1fa5ee5), (26, 0x340326e), (27, 0x6ac3875), (28, 0xd916ce8),
    (29, 0x17e2551e), (30, 0x3d94cd64), (31, 0x7d4fe747), (32, 0xb862a62e),
    (33, 0x1a96ca8d8), (34, 0x34a65911d), (35, 0x4aed21170), (36, 0x9de820a7c),
    (37, 0x1757756a93), (38, 0x22382facd0), (39, 0x4b5f8303e9), (40, 0xe9ae4933d6),
    (41, 0x153869acc5b), (42, 0x2a221c58d8f),
    (43, 0x6bd3b27c591), (44, 0xe02b35a358f),
    (45, 0x122fca143c05), (46, 0x2ec18388d544), (47, 0x6cd610b53cba),
    (48, 0xade6d7ce3b9b), (49, 0x174176b015f4d), (50, 0x22bd43c2e9354),
    (51, 0x75070a1a009d4), (52, 0xefae164cb9e3c), (53, 0x180788e47e326c),
    (54, 0x236fb6d5ad1f43), (55, 0x6abe1f9b67e114), (56, 0x9d18b63ac4ffdf),
    (57, 0x1eb25c90795d61c), (58, 0x2c675b852189a21), (59, 0x7496cbb87cab44f),
    (60, 0xfc07a1825367bbe), (61, 0x13c96a3742f64906), (62, 0x363d541eb611abee),
    (63, 0x7cce5efdaccf6808), (64, 0xf7051f27b09112d4),
    (65, 0x1a838b13505b26867), (66, 0x2832ed74f2b5e35ee),
    (67, 0x730fc235c1942c1ae), (68, 0xbebb3940cd0fc1491),
    (69, 0x101d83275fb2bc7e0c), (70, 0x349b84b6431a6c4ef1),
    (75, 0x4c5ce114686a1336e07),
    (80, 0xea1a5c66dcc11b5ad180),
    (85, 0x11720c4f018d51b8cebba8),
    (90, 0x2ce00bb2136a445c71e85bf),
    (95, 0x527a792b183c7f64a0e8b1f4),
    (100, 0xaf55fc59c335c8ec67ed24826),
    (105, 0x16f14fc2054cd87ee6396b33df3),
    (110, 0x35c0d7234df7deb0f20cf7062444),
    (115, 0x60f4d11574f5deee49961d9609ac6),
    (120, 0xb10f22572c497a836ea187f2e1fc23),
    (125, 0x1c533b6bb7f0804e09960225e44877ac),
    (130, 0x33e7665705359f04f28b88cf897c603c9),
]

print("=" * 70)
print("  TEST 6: Deep Relationship Analysis")
print("=" * 70)

# ── Test A: Common factor in consecutive differences ──
print("\n  --- A: Consecutive key differences (k[i+1] - k[i]) ---")
print("  If d[i] = base + i*step, differences should be constant after unmasking.\n")

consec_diffs = []
for i in range(len(SOLVED) - 1):
    n1, k1 = SOLVED[i]
    n2, k2 = SOLVED[i+1]
    if n2 == n1 + 1:
        diff = k2 - k1
        consec_diffs.append((n1, diff))
        if n1 >= 60 and n1 <= 70:
            print(f"  #{n1:3d}→#{n2:3d}: diff = 0x{diff:x}")

# ── Test B: GCD of all differences ──
print("\n  --- B: GCD of consecutive differences ---")
from math import gcd
if len(consec_diffs) > 2:
    g = abs(consec_diffs[0][1])
    for _, d in consec_diffs[1:]:
        g = gcd(g, abs(d))
    print(f"  GCD of {len(consec_diffs)} consecutive differences: {g}")
    if g > 1:
        print(f"  *** NON-TRIVIAL GCD! Keys may share common step size! ***")
    else:
        print(f"  GCD = 1 — no common factor.")

# ── Test C: Differences mod curve order N ──
print("\n  --- C: Key differences mod small primes ---")
print("  Looking for d[i] mod p pattern (constant or linear in i)\n")

for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 137, 256, 65537]:
    residues = [(n, k % p) for n, k in SOLVED if n <= 70]
    # Check if residue follows a pattern: r[i] = (a*i + b) mod p
    if len(residues) >= 3:
        # Try to fit r[i] = (a*n + b) mod p
        best_a, best_b, best_matches = 0, 0, 0
        for a in range(p):
            for b in range(p):
                matches = sum(1 for n, r in residues if (a * n + b) % p == r)
                if matches > best_matches:
                    best_a, best_b, best_matches = a, b, matches

        expected_random = len(residues) / p
        if best_matches > expected_random * 2 and best_matches > 5:
            print(f"  mod {p:5d}: best linear fit a={best_a} b={best_b} "
                  f"matches {best_matches}/{len(residues)} "
                  f"(random expect {expected_random:.1f}) *** SUSPICIOUS ***")

# ── Test D: Are keys related via secp256k1 curve operations? ──
print("\n  --- D: Lattice test on high-bit-count solved keys ---")
print("  If d[i] = parent + offset[i] mod N, then d[i] - d[j] = offset[i] - offset[j]")
print("  For BIP32, offsets are HMAC-SHA512 outputs — should look random.\n")

# Use the highest-bit-count puzzles for maximum information
high_keys = [(n, k) for n, k in SOLVED if n >= 60]
if len(high_keys) >= 3:
    # Compute pairwise differences
    diffs_set = set()
    print(f"  Pairwise differences among {len(high_keys)} high-bit keys:")
    for i in range(min(6, len(high_keys))):
        for j in range(i+1, min(6, len(high_keys))):
            n1, k1 = high_keys[i]
            n2, k2 = high_keys[j]
            diff = k2 - k1
            print(f"  d[{n2}]-d[{n1}] = 0x{diff:x}")
            diffs_set.add(diff)

    # Check if any difference divides another (lattice structure)
    diffs_list = sorted(diffs_set)
    print(f"\n  Checking for divisibility relations among differences...")
    found_relation = False
    for i in range(len(diffs_list)):
        for j in range(i+1, len(diffs_list)):
            if diffs_list[j] != 0 and diffs_list[i] != 0:
                if diffs_list[j] % diffs_list[i] == 0:
                    ratio = diffs_list[j] // diffs_list[i]
                    if ratio > 1 and ratio < 1000:
                        print(f"  FOUND: diff[{j}] / diff[{i}] = {ratio} (exact)")
                        found_relation = True
    if not found_relation:
        print(f"  No exact divisibility relations found.")

# ── Test E: Are low bits of raw keys biased? ──
print("\n  --- E: Low-bit bias in raw (unmasked) key values ---")
print("  d[i] mod 2^B should be uniformly distributed for random keys.\n")

for B in [1, 2, 3, 4, 8]:
    mod = 1 << B
    residues = [k % mod for _, k in SOLVED]
    from collections import Counter
    counts = Counter(residues)
    expected = len(SOLVED) / mod
    chi2 = sum((counts.get(r, 0) - expected)**2 / expected for r in range(mod))
    # p-value approximation
    sig = "***" if chi2 > 2 * (mod - 1) else ""
    print(f"  mod 2^{B} ({mod:3d}): chi2={chi2:6.2f} (df={mod-1}) {sig}")
    if B <= 3:
        dist = [counts.get(r, 0) for r in range(mod)]
        print(f"    distribution: {dist}")

# ── Test F: Modular inverse relationships ──
print("\n  --- F: k[i] * k[j] mod N relationships ---")
print("  If keys are related via EC multiplication, products mod N may be structured.\n")

# Check if any k[i] * k[j] mod N equals another k[m]
for i in range(len(SOLVED)):
    for j in range(i+1, len(SOLVED)):
        n_i, k_i = SOLVED[i]
        n_j, k_j = SOLVED[j]
        if n_i >= 30 and n_j >= 30:
            prod = (k_i * k_j) % N
            # Check if this equals any other key
            for n_m, k_m in SOLVED:
                if n_m >= 30 and prod == k_m:
                    print(f"  *** k[{n_i}] * k[{n_j}] mod N = k[{n_m}] ***")

print("  No multiplicative relationships found mod N.")

print("\n" + "=" * 70)
print("  TEST 6 COMPLETE")
print("=" * 70)
