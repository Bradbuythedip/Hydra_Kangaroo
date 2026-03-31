#!/usr/bin/env python3
"""
Test 5: BIP32 / Electrum derivation test

For BIP32 non-hardened derivation:
  d[i] = parent_key + HMAC-SHA512(chaincode, parent_pubkey || i)[:32] mod N

Key property: d[i] - d[j] = HMAC(chain, pub||i)[:32] - HMAC(chain, pub||j)[:32] mod N
This difference depends ONLY on chaincode and parent_pubkey, not on parent_key.

If we find that d[i] - d[j] for multiple (i,j) pairs are consistent with the
same (chaincode, pubkey), we've confirmed BIP32 and can potentially recover
the parent key.

For Electrum old-style:
  d[i] = master_key + SHA256(SHA256(master_pub_hex + ":" + str(i)))

The offset is: SHA256(SHA256(mpk_hex + ":" + str(i)))
master_key = d[i] - offset[i]

If we knew master_pub, we could compute all offsets.
But master_pub = master_key * G. So it's circular.

HOWEVER: If we guess master_key, we can derive master_pub, compute offsets,
and verify against all solved keys.

For puzzle #1: k[1] = 1, so d[1] mod 1 = 0 (0 bits known). Useless.
For puzzle #2: k[2] = 3, so d[2] mod 2 = 1. We know 1 bit.
For puzzle #64: k[64] = 0xf7051f27b09112d4, so d[64] mod 2^63 = 0x77051f27b09112d4.

APPROACH: For Electrum-style, if we know d[64] fully, we could compute
master_key = d[64] - SHA256(SHA256(mpk_hex + ":64")). But mpk depends on
master_key. This is solvable by iteration:

  1. Guess master_key = k[64] (assuming d[64] = k[64] for a moment)
  2. Compute mpk = master_key * G
  3. Compute offset[64] = SHA256(SHA256(mpk_hex + ":64"))
  4. Compute new_master = k[64] - offset[64] mod N
  5. Repeat until convergence

Actually this won't converge because the function is pseudorandom.
But we can ENUMERATE: for each possible high-bit extension of d[64]:
  d[64] = known_63_bits + x * 2^63, where x is 0..2^193

This is still too large. We need a different approach.

BETTER APPROACH: Use the STRUCTURE of BIP32.
For BIP32 non-hardened, knowing any child private key + chaincode = parent private key.
The key question is: what's the chaincode?
"""

import hashlib
import hmac
import struct

N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141

SOLVED = {
    64: 0xf7051f27b09112d4,
    65: 0x1a838b13505b26867,
    66: 0x2832ed74f2b5e35ee,
    67: 0x730fc235c1942c1ae,
    68: 0xbebb3940cd0fc1491,
    69: 0x101d83275fb2bc7e0c,
    70: 0x349b84b6431a6c4ef1,
}

print("=" * 70)
print("  TEST 5: BIP32 / Electrum Derivation Analysis")
print("=" * 70)

# ── Method 1: Test if differences are HMAC-SHA512 outputs ──
print("\n  --- Method 1: Difference structure ---")
print("  For BIP32: d[i+1] - d[i] = HMAC(chain, pub||i+1)[:32] - HMAC(chain, pub||i)[:32]")
print("  These differences should look random (256-bit) but be deterministic.\n")

# Compute consecutive differences for high-bit keys
for n in sorted(SOLVED.keys()):
    if n + 1 in SOLVED:
        diff = SOLVED[n+1] - SOLVED[n]
        # For BIP32, this diff depends on the unknown chaincode and pubkey
        # But the diff should be approximately 2^(n-1) in magnitude if keys
        # are "random" within their ranges
        expected_mag = 1 << (n - 1)
        ratio = diff / expected_mag
        print(f"  d[{n+1}] - d[{n}] = 0x{diff:x}")
        print(f"    magnitude ratio (vs 2^{n-1}): {ratio:.4f}")

# ── Method 2: Consistency check ──
print("\n  --- Method 2: Fixed-point iteration for Electrum-style ---")
print("  Testing: does any iteration converge?")
print("  master = k[64]; mpk = master*G; offset = DSHA256(mpk+':64'); new_master = k[64]-offset\n")

# We can't do EC multiplication without a library, so let's check
# a simpler model: what if d[i] = SHA256(master + str(i)) and master is short?

print("  --- Method 3: Short master key brute force ---")
print("  Testing: d[i] = SHA256(master_bytes + str(i).encode())")
print("  For master_bytes in range [0, 2^32)\n")

# For speed, only check puzzle #64 first (63 known bits = very selective filter)
k64_low = SOLVED[64] & ((1 << 63) - 1)  # low 63 bits

checked = 0
found = False
for master_int in range(0, 2**24):  # Test first 16M values (feasible)
    master_bytes = master_int.to_bytes(4, 'big').lstrip(b'\x00') or b'\x00'

    h = hashlib.sha256(master_bytes + b'64').digest()
    d64 = int.from_bytes(h, 'big')
    if (d64 & ((1 << 63) - 1)) == k64_low:
        print(f"  *** CANDIDATE: master=0x{master_int:x} matches puzzle #64! ***")
        # Verify against other puzzles
        matches = 0
        for n, k in SOLVED.items():
            h = hashlib.sha256(master_bytes + str(n).encode()).digest()
            d = int.from_bytes(h, 'big')
            masked = (d & ((1 << (n-1)) - 1)) | (1 << (n-1))
            if masked == k:
                matches += 1
        print(f"    Verified: {matches}/{len(SOLVED)} puzzles match")
        if matches == len(SOLVED):
            print(f"  *** FULL MATCH! SEED = 0x{master_int:x} ***")
            found = True
            break

    checked += 1
    if checked % 4000000 == 0:
        print(f"  Checked {checked:,} values...")

if not found:
    print(f"  No match in first {checked:,} values.")
    print(f"  (63-bit filter: expect ~1 false positive per 2^63 ≈ 9.2×10^18 tries)")

# Also try with different encodings
print("\n  --- Method 4: d[i] = SHA256(i.to_bytes(4) + master_bytes) ---")
for master_int in range(0, 2**20):
    master_bytes = master_int.to_bytes(4, 'big')
    h = hashlib.sha256(struct.pack('>I', 64) + master_bytes).digest()
    d64 = int.from_bytes(h, 'big')
    if (d64 & ((1 << 63) - 1)) == k64_low:
        print(f"  *** CANDIDATE: master=0x{master_int:x} ***")
        found = True
        break

if not found:
    print(f"  No match in 2^20 values.")

print("\n" + "=" * 70)
print("  TEST 5 COMPLETE")
print("=" * 70)
