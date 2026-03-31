#!/usr/bin/env python3
"""
Test 1: Sequential Key Hypothesis
If d[i] = seed + (i-1), then k[i] = (d[i] mod 2^(i-1)) | 2^(i-1)
From any solved key k[i], seed mod 2^(i-1) = k[i] - 2^(i-1) - (i-1)
If consistent across ALL solved keys, we found the seed.
"""

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
print("  TEST 1: Sequential Key Hypothesis (d[i] = seed + i - 1)")
print("=" * 70)

# For each solved key, compute: seed_candidate = k[i] - 2^(i-1) - (i-1)
# Then check: does seed_candidate mod 2^(j-1) match for ALL other solved keys?

# Use puzzle #130 for the tightest constraint (129 known bits of seed)
n_ref = 130
k_ref = 0x33e7665705359f04f28b88cf897c603c9
seed_mod_129 = (k_ref - (1 << 129) - (n_ref - 1)) % (1 << 129)

print(f"\n  Reference: puzzle #{n_ref}")
print(f"  k[{n_ref}] = 0x{k_ref:x}")
print(f"  seed mod 2^129 = 0x{seed_mod_129:x}")

# Now verify against ALL other solved keys
matches = 0
mismatches = 0
for n, k in SOLVED:
    if n == n_ref:
        continue
    mod_bits = n - 1
    expected_low = (seed_mod_129 + (n - 1)) % (1 << mod_bits)
    actual_low = k - (1 << (n - 1))  # strip the high bit

    if expected_low == actual_low:
        matches += 1
        if n <= 10 or n % 10 == 0:
            print(f"  #{n:3d}: MATCH ✓")
    else:
        mismatches += 1
        if mismatches <= 20:
            print(f"  #{n:3d}: MISMATCH ✗  expected=0x{expected_low:x} actual=0x{actual_low:x}")

print(f"\n  Result: {matches} matches, {mismatches} mismatches out of {matches+mismatches}")
if mismatches == 0:
    print("  *** SEQUENTIAL HYPOTHESIS CONFIRMED! Seed found! ***")
else:
    print("  Sequential hypothesis REJECTED.")
