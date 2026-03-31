#!/usr/bin/env python3
"""
Test 2: Hash-based key derivation
Tests if d[i] = H(something || i) for various hash functions and formats.

Strategy: We know k[i] for 82 puzzles. For low puzzles (small bit width),
the masking operation k[i] = d[i] mod 2^(i-1) | 2^(i-1) loses most of d[i].
But for puzzle #130, we know 129 of 256 bits of d[130].

We test multiple derivation formats by trying them with known seeds (empty,
common strings) and checking if they match the known bits.
"""
import hashlib
import struct

SOLVED = {
    1: 0x1, 2: 0x3, 3: 0x7, 4: 0x8,
    5: 0x15, 6: 0x31, 7: 0x4c, 8: 0xe0,
    9: 0x1d3, 10: 0x202, 11: 0x483, 12: 0xa7b,
    13: 0x1460, 14: 0x2930, 15: 0x68f3, 16: 0xc936,
    17: 0x1764f, 18: 0x3080d, 19: 0x5749f, 20: 0xd2c55,
    21: 0x1ba534, 22: 0x2de40f, 23: 0x556e52, 24: 0xdc2a04,
    25: 0x1fa5ee5, 26: 0x340326e, 27: 0x6ac3875, 28: 0xd916ce8,
    29: 0x17e2551e, 30: 0x3d94cd64, 31: 0x7d4fe747, 32: 0xb862a62e,
    33: 0x1a96ca8d8, 34: 0x34a65911d, 35: 0x4aed21170, 36: 0x9de820a7c,
    37: 0x1757756a93, 38: 0x22382facd0, 39: 0x4b5f8303e9, 40: 0xe9ae4933d6,
    41: 0x153869acc5b, 42: 0x2a221c58d8f,
    43: 0x6bd3b27c591, 44: 0xe02b35a358f,
    45: 0x122fca143c05, 46: 0x2ec18388d544, 47: 0x6cd610b53cba,
    48: 0xade6d7ce3b9b, 49: 0x174176b015f4d, 50: 0x22bd43c2e9354,
    51: 0x75070a1a009d4, 52: 0xefae164cb9e3c, 53: 0x180788e47e326c,
    54: 0x236fb6d5ad1f43, 55: 0x6abe1f9b67e114, 56: 0x9d18b63ac4ffdf,
    57: 0x1eb25c90795d61c, 58: 0x2c675b852189a21, 59: 0x7496cbb87cab44f,
    60: 0xfc07a1825367bbe, 61: 0x13c96a3742f64906, 62: 0x363d541eb611abee,
    63: 0x7cce5efdaccf6808, 64: 0xf7051f27b09112d4,
    65: 0x1a838b13505b26867, 66: 0x2832ed74f2b5e35ee,
    67: 0x730fc235c1942c1ae, 68: 0xbebb3940cd0fc1491,
    69: 0x101d83275fb2bc7e0c, 70: 0x349b84b6431a6c4ef1,
    75: 0x4c5ce114686a1336e07,
    80: 0xea1a5c66dcc11b5ad180,
    85: 0x11720c4f018d51b8cebba8,
    90: 0x2ce00bb2136a445c71e85bf,
    95: 0x527a792b183c7f64a0e8b1f4,
    100: 0xaf55fc59c335c8ec67ed24826,
    105: 0x16f14fc2054cd87ee6396b33df3,
    110: 0x35c0d7234df7deb0f20cf7062444,
    115: 0x60f4d11574f5deee49961d9609ac6,
    120: 0xb10f22572c497a836ea187f2e1fc23,
    125: 0x1c533b6bb7f0804e09960225e44877ac,
    130: 0x33e7665705359f04f28b88cf897c603c9,
}

def mask_key(raw_256bit, puzzle_num):
    """Apply puzzle masking: take low (n-1) bits, set bit (n-1)"""
    n = puzzle_num
    low_bits = raw_256bit & ((1 << (n - 1)) - 1)
    return low_bits | (1 << (n - 1))

def check_derivation(derive_func, name, verbose=False):
    """Test a derivation function against all solved keys."""
    matches = 0
    total = len(SOLVED)
    first_fail = None
    for puzzle_num in sorted(SOLVED.keys()):
        raw = derive_func(puzzle_num)
        masked = mask_key(raw, puzzle_num)
        if masked == SOLVED[puzzle_num]:
            matches += 1
        elif first_fail is None:
            first_fail = puzzle_num

    status = "MATCH ALL" if matches == total else f"{matches}/{total}"
    if verbose or matches > 5:
        print(f"  {name:<55s} {status}")
        if first_fail and matches > 3:
            print(f"    First fail at puzzle #{first_fail}")
    return matches == total

print("=" * 70)
print("  TEST 2: Hash-Based Key Derivation")
print("  Testing: d[i] = H(seed || i) with various H and seed formats")
print("=" * 70)

# ── Test common seeds with SHA256(seed || i) ──
print("\n  --- SHA256(seed + str(i)) ---")

common_seeds = [
    b"", b"bitcoin", b"puzzle", b"satoshi", b"1000btc",
    b"challenge", b"secret", b"key", b"test", b"password",
    b"brainwallet", b"correct horse battery staple",
    b"The Times 03/Jan/2009", b"saatoshi_rising",
    b"bitcoin puzzle", b"private key", b"master",
    b"\x00", b"\x01", b"\xff",
    b"1", b"0", b"seed",
]

for seed in common_seeds:
    # Format 1: SHA256(seed + str(i).encode())
    def derive_sha256_str(i, s=seed):
        h = hashlib.sha256(s + str(i).encode()).digest()
        return int.from_bytes(h, 'big')
    check_derivation(derive_sha256_str, f"SHA256('{seed.decode('utf-8', errors='replace')}' + str(i))", verbose=True)

    # Format 2: SHA256(seed + i as 4-byte big-endian)
    def derive_sha256_be4(i, s=seed):
        h = hashlib.sha256(s + struct.pack('>I', i)).digest()
        return int.from_bytes(h, 'big')
    check_derivation(derive_sha256_be4, f"SHA256('{seed.decode('utf-8', errors='replace')}' + be4(i))", verbose=True)

    # Format 3: SHA256(seed + i as 1 byte)
    if len(seed) < 10:
        def derive_sha256_byte(i, s=seed):
            h = hashlib.sha256(s + bytes([i & 0xFF])).digest()
            return int.from_bytes(h, 'big')
        check_derivation(derive_sha256_byte, f"SHA256('{seed.decode('utf-8', errors='replace')}' + byte(i))", verbose=True)

# ── Double SHA256 ──
print("\n  --- Double SHA256 (Bitcoin standard) ---")
for seed in common_seeds[:10]:
    def derive_dsha256(i, s=seed):
        h1 = hashlib.sha256(s + str(i).encode()).digest()
        h2 = hashlib.sha256(h1).digest()
        return int.from_bytes(h2, 'big')
    check_derivation(derive_dsha256, f"SHA256(SHA256('{seed.decode('utf-8', errors='replace')}' + str(i)))", verbose=True)

# ── HMAC-SHA256 ──
print("\n  --- HMAC-SHA256(seed, i) ---")
import hmac
for seed in common_seeds[:10]:
    def derive_hmac(i, s=seed):
        h = hmac.new(s if s else b"\x00", str(i).encode(), hashlib.sha256).digest()
        return int.from_bytes(h, 'big')
    check_derivation(derive_hmac, f"HMAC-SHA256('{seed.decode('utf-8', errors='replace')}', str(i))", verbose=True)

# ── SHA512 (BIP32 uses HMAC-SHA512) ──
print("\n  --- SHA512(seed + str(i)) ---")
for seed in common_seeds[:10]:
    def derive_sha512(i, s=seed):
        h = hashlib.sha512(s + str(i).encode()).digest()
        return int.from_bytes(h[:32], 'big')  # Take first 32 bytes
    check_derivation(derive_sha512, f"SHA512('{seed.decode('utf-8', errors='replace')}' + str(i))[:32]", verbose=True)

# ── Chained hash: d[i] = SHA256(d[i-1]) ──
print("\n  --- Chained: d[i] = SHA256(d[i-1]) ---")
for start_val in [0, 1, 42, 0xdeadbeef, 0x7fffffff]:
    def derive_chain(target_i, start=start_val):
        val = start.to_bytes(32, 'big')
        for j in range(1, target_i + 1):
            val = hashlib.sha256(val).digest()
        return int.from_bytes(val, 'big')
    check_derivation(derive_chain, f"SHA256^i(start=0x{start_val:x})", verbose=True)

# ── Index-based: d[i] = SHA256(i) ──
print("\n  --- Pure index: SHA256(i) ---")
for fmt in ['str', 'be4', 'le4', 'be32', 'le32']:
    def derive_pure(i, f=fmt):
        if f == 'str': data = str(i).encode()
        elif f == 'be4': data = struct.pack('>I', i)
        elif f == 'le4': data = struct.pack('<I', i)
        elif f == 'be32': data = i.to_bytes(32, 'big')
        elif f == 'le32': data = i.to_bytes(32, 'little')
        return int.from_bytes(hashlib.sha256(data).digest(), 'big')
    check_derivation(derive_pure, f"SHA256({fmt}(i))", verbose=True)

print("\n" + "=" * 70)
print("  TEST 2 COMPLETE")
print("=" * 70)
