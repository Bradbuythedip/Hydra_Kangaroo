#!/usr/bin/env python3
"""
Test 7: Statistical significance of modular residue patterns.

Test 6 found suspicious linear fits at mod 11, 13, 17, 19, 23.
This script applies rigorous statistical testing with Bonferroni correction.

For each prime p, we test all p^2 possible linear fits (a*n + b) mod p.
The best-fit count follows a distribution we can compute exactly.
We need the p-value of the BEST fit, corrected for p^2 multiple comparisons
AND for testing multiple primes.
"""

from math import comb, log, factorial
from collections import Counter

N_ORDER = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141

# All 82 solved keys
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

# Separate consecutive (1-70) and every-5th (75-130)
CONSEC = [(n, k) for n, k in SOLVED if n <= 70]
EVERY5 = [(n, k) for n, k in SOLVED if n > 70]

print("=" * 70)
print("  TEST 7: Modular Residue Significance Testing")
print("=" * 70)

# ── Part A: Exact significance for linear fits ──
print("\n  --- Part A: Best linear fit (a*n + b) mod p ---")
print("  Testing on CONSECUTIVE keys (#1-#70, n=70)")
print("  With Bonferroni correction for p^2 fits × num_primes tested\n")

primes_to_test = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
num_primes = len(primes_to_test)

def binomial_tail(n, k, p_success):
    """P(X >= k) for Binomial(n, p_success) using exact computation."""
    total = 0.0
    for j in range(k, n + 1):
        total += comb(n, j) * (p_success ** j) * ((1 - p_success) ** (n - j))
    return total

results = []

for p in primes_to_test:
    residues = [(n, k % p) for n, k in CONSEC]
    n_keys = len(residues)

    # Find best linear fit
    best_a, best_b, best_matches = 0, 0, 0
    for a in range(p):
        for b in range(p):
            matches = sum(1 for n, r in residues if (a * n + b) % p == r)
            if matches > best_matches:
                best_a, best_b, best_matches = a, b, matches

    # Under null (random keys), each key matches with prob 1/p
    # Best of p^2 fits: use Bonferroni bound
    # P(any fit gets >= k matches) <= p^2 * P(Binom(n, 1/p) >= k)
    raw_pval = binomial_tail(n_keys, best_matches, 1.0 / p)
    bonferroni_single = min(1.0, raw_pval * p * p)  # correct for p^2 fits
    bonferroni_all = min(1.0, bonferroni_single * num_primes)  # correct for all primes

    expected = n_keys / p
    ratio = best_matches / expected

    sig = ""
    if bonferroni_all < 0.01:
        sig = "*** SIGNIFICANT (p<0.01) ***"
    elif bonferroni_all < 0.05:
        sig = "** MARGINAL (p<0.05) **"

    results.append((p, best_a, best_b, best_matches, expected, ratio,
                     raw_pval, bonferroni_single, bonferroni_all))

    print(f"  mod {p:3d}: best fit {best_a}*n+{best_b}, "
          f"matches={best_matches}/{n_keys} "
          f"(expect {expected:.1f}, ratio {ratio:.2f}x)")
    print(f"           raw p={raw_pval:.2e}, "
          f"Bonf(fits)={bonferroni_single:.2e}, "
          f"Bonf(all)={bonferroni_all:.2e} {sig}")

# ── Part B: Verify against every-5th keys ──
print("\n  --- Part B: Cross-validate against every-5th keys (#75-#130) ---")
print(f"  Testing best fits from Part A against {len(EVERY5)} held-out keys\n")

for p, best_a, best_b, matches_consec, _, _, _, _, bonf_all in results:
    if bonf_all < 0.5:  # only test plausible ones
        matches_held = sum(1 for n, k in EVERY5 if (best_a * n + best_b) % p == k % p)
        expected_held = len(EVERY5) / p
        held_pval = binomial_tail(len(EVERY5), matches_held, 1.0 / p)
        print(f"  mod {p:3d} ({best_a}*n+{best_b}): "
              f"{matches_held}/{len(EVERY5)} held-out matches "
              f"(expect {expected_held:.1f}, p={held_pval:.4f})")

# ── Part C: Test QUADRATIC fits (a*n^2 + b*n + c) mod p ──
print("\n  --- Part C: Best quadratic fit (a*n²+b*n+c) mod p ---")
print("  For small primes only (p^3 fits to test)\n")

for p in [2, 3, 5, 7, 11]:
    residues = [(n, k % p) for n, k in CONSEC]
    n_keys = len(residues)

    best_abc, best_matches = (0, 0, 0), 0
    for a in range(p):
        for b in range(p):
            for c in range(p):
                matches = sum(1 for n, r in residues
                              if (a * n * n + b * n + c) % p == r)
                if matches > best_matches:
                    best_abc, best_matches = (a, b, c), matches

    expected = n_keys / p
    raw_pval = binomial_tail(n_keys, best_matches, 1.0 / p)
    bonf = min(1.0, raw_pval * p**3 * 5)  # correct for p^3 fits × 5 primes

    a, b, c = best_abc
    print(f"  mod {p:3d}: best {a}n²+{b}n+{c}, "
          f"matches={best_matches}/{n_keys} "
          f"(expect {expected:.1f}), Bonf p={bonf:.2e}")

# ── Part D: Test if k[i] mod p is CONSTANT ──
print("\n  --- Part D: Constant residue test ---")
print("  If all k[i] ≡ c (mod p), derivation preserves residue.\n")

for p in primes_to_test:
    residues = [k % p for _, k in SOLVED]
    counts = Counter(residues)
    most_common_val, most_common_count = counts.most_common(1)[0]
    expected = len(SOLVED) / p
    raw_pval = binomial_tail(len(SOLVED), most_common_count, 1.0 / p)
    bonf = min(1.0, raw_pval * p * num_primes)
    if bonf < 0.1:
        print(f"  mod {p:3d}: value {most_common_val} appears {most_common_count}x "
              f"(expect {expected:.1f}), Bonf p={bonf:.4f}")

print("\n  (No output = no significant constant residues)")

# ── Part E: Monte Carlo validation ──
print("\n  --- Part E: Monte Carlo baseline ---")
print("  Generate 10000 random key sets, measure best linear fit\n")

import random
random.seed(42)

max_matches_distribution = {p: [] for p in [11, 13, 19]}

for trial in range(10000):
    fake_keys = []
    for n in range(1, 71):
        # Random key in [2^(n-1), 2^n)
        if n <= 1:
            fake_keys.append((n, 1))
        else:
            fake_keys.append((n, random.randint(1 << (n - 1), (1 << n) - 1)))

    for p in [11, 13, 19]:
        residues = [(n, k % p) for n, k in fake_keys]
        best = 0
        for a in range(p):
            for b in range(p):
                m = sum(1 for n, r in residues if (a * n + b) % p == r)
                if m > best:
                    best = m
        max_matches_distribution[p].append(best)

for p in [11, 13, 19]:
    dist = max_matches_distribution[p]
    actual_best = [r for r in results if r[0] == p][0][3]
    exceeded = sum(1 for d in dist if d >= actual_best)
    mean_best = sum(dist) / len(dist)
    max_best = max(dist)
    print(f"  mod {p:2d}: actual best={actual_best}, "
          f"MC mean best={mean_best:.1f}, MC max={max_best}, "
          f"MC P(>=actual)={exceeded}/{len(dist)} = {exceeded/len(dist):.4f}")

print("\n" + "=" * 70)
print("  TEST 7 COMPLETE")
print("=" * 70)
