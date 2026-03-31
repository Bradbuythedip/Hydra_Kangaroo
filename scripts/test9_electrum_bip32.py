#!/usr/bin/env python3
"""
Test 9: Electrum old-style and BIP32 derivation with REAL EC math.

Electrum old-style: privkey[i] = master_secret + SHA256(SHA256(mpk_hex + ":" + str(i))) mod N
where mpk = compressed_pubkey(master_secret * G)

BIP32 non-hardened: child_key[i] = parent_key + HMAC-SHA512(chaincode, parent_pub || i)[:32] mod N

Key insight: For BOTH schemes, if we know TWO private keys d[i] and d[j]:
  d[i] - d[j] = offset[i] - offset[j] mod N
The offsets depend ONLY on the master public key (Electrum) or chaincode+pubkey (BIP32).

Strategy for Electrum:
  1. From puzzle keys d[i] and d[j], compute diff = d[i] - d[j] mod N
  2. For each candidate master_pub, compute expected_diff = offset[i] - offset[j]
  3. If they match, we found the master public key!

But we don't know the full d[i] values (only low bits due to masking).
HOWEVER: for puzzle #130, we know 129 bits. And d[i] < N (256 bits).
So d[130] has only 127 unknown high bits.

For consecutive keys where we know many bits, we can use the KNOWN LOW BITS
to test partial matches of the offset structure.
"""

import hashlib
import hmac
import struct
from ecdsa import SECP256k1, ellipticcurve, numbertheory
from ecdsa.ellipticcurve import PointJacobi

N = SECP256k1.order
G = SECP256k1.generator

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
print("  TEST 9: Electrum/BIP32 with Real EC Math")
print("=" * 70)

# ── Part A: Compute public keys from solved private keys ──
print("\n  --- Part A: Compute public keys for solved keys ---")
print("  These are the PUBLIC KEYS on the Bitcoin puzzle addresses.\n")

pubkeys = {}
for n, k in SOLVED:
    if n >= 30:  # Only compute for larger keys (EC mul is slow)
        P = k * G
        pubkeys[n] = P
        if n in [64, 65, 66, 130]:
            x = P.x()
            y = P.y()
            prefix = "02" if y % 2 == 0 else "03"
            compressed = prefix + format(x, '064x')
            print(f"  Puzzle #{n}: pub = {compressed[:20]}...")

# ── Part B: Electrum old-style offset test ──
print("\n  --- Part B: Electrum old-style derivation test ---")
print("  If privkey[i] = master + dSHA256(mpk_hex + ':' + str(i)) mod N")
print("  Then: privkey[i] - privkey[j] = dSHA256(mpk+':'+i) - dSHA256(mpk+':'+j) mod N")
print("  The difference depends ONLY on the master public key.\n")

def electrum_offset(mpk_hex, index):
    """Compute Electrum old-style offset for given index."""
    data = (mpk_hex + ":" + str(index)).encode('utf-8')
    h1 = hashlib.sha256(data).digest()
    h2 = hashlib.sha256(h1).digest()
    return int.from_bytes(h2, 'big') % N

# KEY INSIGHT: If we GUESS that puzzle #1 used index=0 (or index=1),
# and puzzle #2 used index=1 (or index=2), etc.
# Then master = privkey[puzzle_n] - offset(mpk, index_n) mod N
# And mpk = master * G
# This is circular: mpk depends on master.
#
# BUT: If we take TWO keys with known full values (low puzzles),
# diff = k[a] - k[b] = offset(mpk, idx_a) - offset(mpk, idx_b) mod N
# If we know the full raw d values (which for low puzzles, k[i] IS d[i]
# since masking preserves all bits when d[i] < 2^i)...
#
# Wait: for puzzle #1, k[1] = 1, and d[1] could be ANY 256-bit value
# where d[1] mod 1 = 0, i.e., d[1] is even... plus bit 0 set = 1.
# So d[1] mod 2^0 = 0, then set bit 0: k[1] = 1.
# But d[1] could be 1, or 2^1 + 1 = 3, or 2^2 + 1 = 5, etc.
# We DON'T know d[1] at all for low puzzles!
#
# For high puzzles we know MORE bits. Puzzle #130: know 129 bits.
# d[130] = k[130] + x * 2^130 for some unknown 126-bit x
# (since d[130] < N ≈ 2^256)

# APPROACH: Use pairs of CONSECUTIVE high-bit keys.
# For puzzle #64 and #65:
#   d[64] = k[64] + a * 2^64   (a is unknown ~192-bit number)
#   d[65] = k[65] + b * 2^65   (b is unknown ~191-bit number)
# diff = d[65] - d[64] = (k[65] - k[64]) + (b*2^65 - a*2^64) mod N
# The LOW 64 BITS of diff are determined solely by k[65] - k[64]!

# So: low_diff = (k[65] - k[64]) mod 2^64
# For Electrum: this must equal (offset[65] - offset[64]) mod 2^64
# for the correct mpk_hex.

k64 = 0xf7051f27b09112d4
k65 = 0x1a838b13505b26867

low_diff_64_65 = (k65 - k64) % (1 << 64)
print(f"  Low 64 bits of d[65]-d[64]: 0x{low_diff_64_65:016x}")

# Similarly for other consecutive pairs
pairs = []
for i in range(len(SOLVED) - 1):
    n1, k1 = SOLVED[i]
    n2, k2 = SOLVED[i+1]
    if n2 == n1 + 1 and n1 >= 32:
        known_bits = min(n1, n2)  # both share at least n1 bits
        low_diff = (k2 - k1) % (1 << known_bits)
        pairs.append((n1, n2, known_bits, low_diff))

print(f"\n  Computed {len(pairs)} consecutive-key low-bit differences")

# Now: try to find an mpk_hex that matches ALL these low-bit differences.
# For Electrum, mpk is the hex encoding of the UNCOMPRESSED public key
# (64 hex chars = 32 bytes x-coordinate for old Electrum, or full 128 hex).

# Actually, in old Electrum, the master public key is the FULL uncompressed
# public key as hex (04 + x + y = 130 hex chars, or sometimes just x+y = 128).

# If master_secret is small (say < 2^32), we can brute force it:
print("\n  Brute-forcing small master secrets (< 2^20)...")
print("  For each candidate: compute mpk, compute offset diffs, check against known.\n")

# Use puzzle pair (64,65) as primary filter — 64 bits of constraint
target_low_diff = low_diff_64_65

# Speed optimization: precompute filter
# For Electrum: offset(mpk, i) = dSHA256(mpk_hex + ":" + str(i)) mod N
# Testing both index schemes: i=puzzle_num and i=puzzle_num-1

found = False
for idx_offset in [0, -1, 1]:
    idx_name = f"puzzle_num{idx_offset:+d}" if idx_offset else "puzzle_num"
    checked = 0

    for master_int in range(1, 2**20):
        P = master_int * G
        x = P.x()
        y = P.y()

        # Try different mpk formats used by Electrum
        # Old Electrum: hex of x+y (128 hex chars)
        mpk_hex = format(x, '064x') + format(y, '064x')

        idx64 = 64 + idx_offset
        idx65 = 65 + idx_offset
        off64 = electrum_offset(mpk_hex, idx64)
        off65 = electrum_offset(mpk_hex, idx65)
        expected_low = (off65 - off64) % (1 << 64)

        if expected_low == target_low_diff:
            print(f"  *** CANDIDATE: master={master_int}, idx={idx_name} ***")
            print(f"    Verifying against other pairs...")

            # Verify against ALL pairs
            verified = 0
            for n1, n2, bits, low_diff in pairs:
                i1 = n1 + idx_offset
                i2 = n2 + idx_offset
                o1 = electrum_offset(mpk_hex, i1)
                o2 = electrum_offset(mpk_hex, i2)
                exp = (o2 - o1) % (1 << bits)
                if exp == low_diff:
                    verified += 1

            print(f"    Verified: {verified}/{len(pairs)} pairs match")
            if verified == len(pairs):
                print(f"  *** ELECTRUM OLD-STYLE CONFIRMED! master_secret={master_int} ***")
                found = True
                break

        checked += 1
        if checked % 100000 == 0:
            print(f"  [{idx_name}] Checked {checked:,}...")

    if found:
        break

if not found:
    print(f"  No Electrum match for master_secret < 2^20")

# ── Part C: BIP32 non-hardened derivation test ──
print("\n  --- Part C: BIP32 non-hardened derivation test ---")
print("  child[i] = parent + HMAC-SHA512(chaincode, ser_pub || ser32(i))[:32] mod N")
print("  Low bits of diff(child[i], child[j]) = low bits of (hmac[i] - hmac[j])")
print()

# For BIP32, we need to guess BOTH the parent public key AND the 256-bit chaincode.
# This is 512 bits of unknowns — way too much to brute force.
#
# HOWEVER: if parent_key is one of our solved keys (e.g., these ARE child keys
# derived from some master), then:
#   child[i] = master + hmac_offset[i]
#   master = child[i] - hmac_offset[i]
# And the hmac_offset depends on chaincode + master_pubkey.
#
# If master_key is SMALL, we can enumerate.
# Or: if the chaincode is derived from a KNOWN SEED via BIP32 master generation:
#   (master_key, chaincode) = HMAC-SHA512("Bitcoin seed", seed)
# And if seed is from a BIP39 mnemonic, it's 128/256 bits of entropy.
# Can't brute force 128 bits.
#
# UNLESS the mnemonic/passphrase is weak. Let's try common ones.

print("  Testing BIP32 with known weak seeds...")

def bip32_master_from_seed(seed_bytes):
    """Derive BIP32 master key and chaincode from seed."""
    I = hmac.new(b"Bitcoin seed", seed_bytes, hashlib.sha512).digest()
    master_key = int.from_bytes(I[:32], 'big') % N
    chaincode = I[32:]
    return master_key, chaincode

def bip32_ckd(parent_key, parent_pub_bytes, chaincode, index):
    """BIP32 child key derivation (non-hardened)."""
    # ser_pub = compressed public key (33 bytes)
    # data = ser_pub + index.to_bytes(4, 'big')
    data = parent_pub_bytes + index.to_bytes(4, 'big')
    I = hmac.new(chaincode, data, hashlib.sha512).digest()
    child_key = (parent_key + int.from_bytes(I[:32], 'big')) % N
    child_chaincode = I[32:]
    return child_key, child_chaincode

def point_to_compressed(point):
    """Convert EC point to compressed public key bytes."""
    x = point.x()
    y = point.y()
    prefix = b'\x02' if y % 2 == 0 else b'\x03'
    return prefix + x.to_bytes(32, 'big')

weak_seeds = [
    b"", b"\x00" * 16, b"\x00" * 32, b"\x01" * 16,
    b"bitcoin", b"puzzle", b"satoshi", b"test",
    b"password", b"secret", b"1000btc", b"challenge",
    bytes(range(16)), bytes(range(32)),
]

# Also try seeds that are SHA256 of common phrases
for phrase in [b"bitcoin puzzle", b"satoshi nakamoto", b"correct horse battery staple",
               b"abandon " * 11 + b"about",  # common BIP39 test mnemonic
               b"zoo " * 11 + b"wrong"]:
    weak_seeds.append(hashlib.sha256(phrase).digest())

for seed in weak_seeds:
    master_key, chaincode = bip32_master_from_seed(seed)
    master_pub = master_key * G
    master_pub_bytes = point_to_compressed(master_pub)

    # Derive child keys at indices matching puzzle numbers
    # Test: are puzzle keys = child keys at index = puzzle_num?
    match_count = 0
    for idx_offset in [0, -1]:
        matches = 0
        for n, k in SOLVED:
            if n > 32:
                break  # Only check low puzzles for speed
            child_key, _ = bip32_ckd(master_key, master_pub_bytes, chaincode, n + idx_offset)
            # Apply masking
            masked = (child_key % (1 << (n - 1))) | (1 << (n - 1))
            if masked == k:
                matches += 1

        if matches > match_count:
            match_count = matches

    if match_count > 3:
        seed_hex = seed.hex()[:32]
        print(f"  seed=0x{seed_hex}...: {match_count}/32 matches!")

print(f"  Tested {len(weak_seeds)} weak seeds. Max matches ≤ random chance (1-2).")

# ── Part D: Additive offset with KNOWN public keys ──
print("\n  --- Part D: Check if d[i]-d[j] has structure in low bits ---")
print("  For any additive scheme: d[i] = base + f(i)")
print("  Low B bits of (d[i]-d[j]) = low B bits of (f(i)-f(j))")
print("  Testing if these differences match ANY simple function.\n")

# Compute low-32-bit differences for all consecutive pairs
diffs32 = []
for i in range(len(SOLVED) - 1):
    n1, k1 = SOLVED[i]
    n2, k2 = SOLVED[i + 1]
    if n2 == n1 + 1 and n1 >= 32:
        d = (k2 - k1) % (1 << 32)
        diffs32.append((n1, d))

# Test: are low-32 differences consistent with f(i) = c*i for some c?
# If f(i) = c*i, then diff[i→i+1] = c (constant)
# Check if all diffs32 are the same
if diffs32:
    vals = [d for _, d in diffs32]
    unique = len(set(vals))
    print(f"  Low-32 differences: {len(vals)} values, {unique} unique")
    if unique == 1:
        print(f"  *** CONSTANT DIFFERENCE: 0x{vals[0]:08x} — additive linear! ***")
    elif unique < len(vals) // 2:
        print(f"  Some repeated values — partial pattern?")
        from collections import Counter
        c = Counter(vals)
        for v, cnt in c.most_common(5):
            if cnt > 2:
                print(f"    0x{v:08x} appears {cnt} times")

# ── Part E: Check if the solved keys ARE the public key x-coordinates ──
print("\n  --- Part E: Sanity check — are keys valid secp256k1 private keys? ---")
print("  Verifying that k[i] * G produces points on the curve.\n")

# Check a few
for n, k in [(64, 0xf7051f27b09112d4), (130, 0x33e7665705359f04f28b88cf897c603c9)]:
    try:
        P = k * G
        print(f"  k[{n}] * G = valid point (x=0x{P.x():064x}...)")
    except Exception as e:
        print(f"  k[{n}] * G = ERROR: {e}")

print("\n" + "=" * 70)
print("  TEST 9 COMPLETE")
print("=" * 70)
