#!/usr/bin/env python3
"""
Test 12: Golden ratio / phi-based patterns in puzzle keys.

For each puzzle #n, the key k[n] sits in [2^(n-1), 2^n).
Its normalized position is: pos[n] = (k[n] - 2^(n-1)) / 2^(n-1)  ∈ [0, 1)

Test hypotheses:
1. pos[n] ≈ φ - 1 = 0.6180339... (inverse golden ratio)
2. pos[n] ≈ 1/φ^n or φ^n mod 1 (Weyl/golden angle sequence)
3. pos[n] ≈ {n * φ} (fractional part — equidistributed golden sequence)
4. Ratios between consecutive keys relate to φ
5. Key bits follow Fibonacci-based construction
"""

PHI = (1 + 5**0.5) / 2       # 1.6180339887...
INV_PHI = PHI - 1              # 0.6180339887... = 1/φ
INV_PHI2 = 1 / PHI             # same thing

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
print("  TEST 12: Golden Ratio / Phi Patterns")
print("=" * 70)

# ── Part A: Normalized position of each key in its range ──
print("\n  --- Part A: Key position within range [2^(n-1), 2^n) ---")
print(f"  φ = {PHI:.10f}")
print(f"  1/φ = {INV_PHI:.10f}")
print()

positions = []
for n, k in SOLVED:
    range_start = 1 << (n - 1)
    range_size = 1 << (n - 1)  # = 2^(n-1)
    pos = (k - range_start) / range_size  # in [0, 1)
    positions.append((n, pos))
    if n <= 20 or n in [32, 64, 130]:
        phi_dist = abs(pos - INV_PHI)
        print(f"  #{n:3d}: pos = {pos:.10f}  |pos - 1/φ| = {phi_dist:.6f}")

# ── Part B: Check if positions cluster near phi ──
print("\n  --- Part B: Statistical test — do positions cluster near 1/φ? ---")

phi_dists = [abs(pos - INV_PHI) for n, pos in positions]
mean_phi_dist = sum(phi_dists) / len(phi_dists)

# Under uniform distribution, E[|X - 0.618|] depends on the value:
# For uniform [0,1]: E[|X-c|] = c^2/2 + (1-c)^2/2 = (c^2 + (1-c)^2)/2
c = INV_PHI
expected_uniform_dist = (c**2 + (1 - c)**2) / 2

print(f"  Mean |pos - 1/φ|:    {mean_phi_dist:.6f}")
print(f"  Expected (uniform):  {expected_uniform_dist:.6f}")
print(f"  Ratio:               {mean_phi_dist / expected_uniform_dist:.4f}")
if mean_phi_dist < expected_uniform_dist * 0.7:
    print(f"  *** SUSPICIOUS: positions cluster near 1/φ! ***")
else:
    print(f"  No clustering near 1/φ (ratio ≈ 1 = random)")

# ── Part C: Golden angle / Weyl sequence ──
print("\n  --- Part C: Weyl sequence — does pos[n] ≈ {n·φ}? ---")
print(f"  If keys were generated as k[n] = floor({{n·φ}} × 2^(n-1)) + 2^(n-1)")
print()

weyl_matches = 0
for n, pos in positions:
    weyl = (n * PHI) % 1  # fractional part of n*φ
    diff = abs(pos - weyl)
    if diff > 0.5:
        diff = 1 - diff  # wrap-around distance
    if n <= 20 or n in [32, 64, 130]:
        match = "✓" if diff < 0.05 else " "
        print(f"  #{n:3d}: pos={pos:.6f}  {{n·φ}}={weyl:.6f}  Δ={diff:.6f} {match}")
    if diff < 0.05:
        weyl_matches += 1

expected_matches = len(positions) * 0.10  # 10% chance within ±0.05
print(f"\n  Matches within ±0.05: {weyl_matches}/{len(positions)} "
      f"(expected {expected_matches:.1f} by chance)")
if weyl_matches > expected_matches * 2:
    print(f"  *** SUSPICIOUS: too many Weyl matches! ***")

# ── Part D: Test various irrational/constant-based sequences ──
print("\n  --- Part D: Test multiple constants — {n·c} for various c ---")

import math
constants = {
    "φ (golden)": PHI,
    "1/φ": INV_PHI,
    "√2": math.sqrt(2),
    "√3": math.sqrt(3),
    "√5": math.sqrt(5),
    "π": math.pi,
    "e": math.e,
    "ln(2)": math.log(2),
    "π/4": math.pi / 4,
    "φ²": PHI**2,
    "2φ": 2 * PHI,
    "1/π": 1 / math.pi,
    "1/e": 1 / math.e,
    "√φ": math.sqrt(PHI),
}

print(f"  Testing {{n·c}} mod 1 against actual positions:\n")
best_const = None
best_score = 0

for name, c in constants.items():
    matches = 0
    total_diff = 0
    for n, pos in positions:
        weyl = (n * c) % 1
        diff = min(abs(pos - weyl), 1 - abs(pos - weyl))
        total_diff += diff
        if diff < 0.05:
            matches += 1
    mean_diff = total_diff / len(positions)
    if matches > best_score:
        best_score = matches
        best_const = name
    if matches >= 10 or name in ["φ (golden)", "1/φ", "π", "e"]:
        print(f"  {name:12s}: {matches}/{len(positions)} matches, mean Δ = {mean_diff:.4f}")

print(f"\n  Best: {best_const} with {best_score} matches "
      f"(expected ~{len(positions)*0.10:.1f} by chance)")

# ── Part E: Ratio between consecutive keys ──
print("\n  --- Part E: Ratio k[n+1] / k[n] — does it approach φ? ---")
print(f"  For truly random keys: E[k[n+1]/k[n]] ≈ 2 × (uniform ratio)\n")

ratios = []
for i in range(len(SOLVED) - 1):
    n1, k1 = SOLVED[i]
    n2, k2 = SOLVED[i + 1]
    if n2 == n1 + 1:
        ratio = k2 / k1
        # Normalized: each key roughly doubles, so ratio ≈ 2
        norm_ratio = ratio / 2  # should be ~uniform in [0.5, 1.5]
        ratios.append((n1, ratio, norm_ratio))
        if n1 <= 15 or n1 in [32, 63, 64]:
            phi_r = abs(norm_ratio - INV_PHI)
            print(f"  k[{n2}]/k[{n1}] = {ratio:.6f}, /2 = {norm_ratio:.6f}, |./2 - 1/φ| = {phi_r:.4f}")

if ratios:
    norm_ratios = [nr for _, _, nr in ratios]
    mean_nr = sum(norm_ratios) / len(norm_ratios)
    phi_match = abs(mean_nr - INV_PHI)
    print(f"\n  Mean(k[n+1]/k[n]/2) = {mean_nr:.6f}")
    print(f"  |mean - 1/φ| = {phi_match:.6f}")
    print(f"  Expected (uniform [0.5,1]): mean ≈ 0.75")

# ── Part F: Fibonacci-indexed keys ──
print("\n  --- Part F: Do Fibonacci-indexed puzzles have special properties? ---")

fibs = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]  # Fibonacci numbers
fib_keys = [(n, k) for n, k in SOLVED if n in fibs]

print(f"  Fibonacci puzzle numbers: {[n for n, _ in fib_keys]}")
if fib_keys:
    fib_positions = []
    for n, k in fib_keys:
        range_start = 1 << (n - 1)
        range_size = 1 << (n - 1)
        pos = (k - range_start) / range_size
        fib_positions.append(pos)
        print(f"  #{n:3d}: pos = {pos:.6f}")

    mean_fib_pos = sum(fib_positions) / len(fib_positions)
    print(f"  Mean position: {mean_fib_pos:.6f} (expected 0.5 for random)")
    print(f"  |mean - 1/φ| = {abs(mean_fib_pos - INV_PHI):.6f}")

# ── Part G: Predict #135 using each hypothesis ──
print("\n  --- Part G: Predictions for puzzle #135 ---")

range_start_135 = 1 << 134
range_size_135 = 1 << 134

# Hypothesis 1: pos = 1/φ
pred1 = range_start_135 + int(INV_PHI * range_size_135)
print(f"  If pos=1/φ:       k[135] ≈ 0x{pred1:034x}")

# Hypothesis 2: pos = {135·φ} mod 1
weyl_135 = (135 * PHI) % 1
pred2 = range_start_135 + int(weyl_135 * range_size_135)
print(f"  If pos={{135·φ}}: k[135] ≈ 0x{pred2:034x}  (pos={weyl_135:.6f})")

# Hypothesis 3: pos = {135/φ} mod 1
inv_weyl = (135 * INV_PHI) % 1
pred3 = range_start_135 + int(inv_weyl * range_size_135)
print(f"  If pos={{135/φ}}: k[135] ≈ 0x{pred3:034x}  (pos={inv_weyl:.6f})")

# Reality check
print(f"\n  ⚠ These are predictions based on UNVERIFIED hypotheses.")
print(f"  The creator said 'no pattern' — keys are from a deterministic")
print(f"  wallet with a cryptographic hash, so positions should be uniform.")
print(f"  φ-based patterns would only exist if the seed or derivation")
print(f"  function has a hidden relationship to the golden ratio.")

print("\n" + "=" * 70)
print("  TEST 12 COMPLETE")
print("=" * 70)
