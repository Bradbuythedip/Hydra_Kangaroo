# CLAUDE.md — Hydra Kangaroo: 1000x-Optimized Pollard's Kangaroo Solver

## Project Goal
Build a CUDA-optimized Pollard's Kangaroo solver targeting Bitcoin Puzzle #135.
This solves the Elliptic Curve Discrete Logarithm Problem (ECDLP) on secp256k1 for a known public key within a bounded range.

## 1000x Architecture (from puzzle_binary)
The kernel architecture is inspired by puzzle_binary's 1000x SHA-256 mining proof.
Five ASIC optimization principles adapted to GPU:
1. **Carry-Save / PTX MADC** → `field_csa.cuh` (fused multiply-add, ~40% fewer instructions)
2. **Deferred-A → Deferred-Y** → `ec_pipeline.cuh` (x-only affine, skip y for 99.997% of points)
3. **Sub-Round Pipeline** → `ec_pipeline.cuh` (interleaved EC phases across K kangaroos)
4. **DHKW Precompute → Progressive DP** → `ec_pipeline.cuh` (early termination for non-DPs)
5. **Multi-core Sharing** → `field_csa.cuh` (warp-cooperative batch inversion)
See PROOF_1000X_GPU.md for the full analysis.

## Target
```
Puzzle:     #135
Address:    16RGFo6hjq9ym6Pj7N5H7L1NR1rVPJyw2v
Public Key: 02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16
Q.x = 0x145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16
Q.y = 0x667a05e9a1bdd6f70142b66558bd12ce2c0f9cbc7001b20c8a6a109c80dc5330
Range:      [2^134, 2^135)  (key k satisfies 2^134 <= k < 2^135)
Prize:      0.135 BTC
```

## Complexity Reality
```
Pollard's Kangaroo: O(√range) = O(2^67) ≈ 1.47 × 10^20 EC group operations.
This is the PROVEN LOWER BOUND in the generic group model. No algorithm can beat O(√n).
The only improvements are CONSTANT FACTORS from:
  - secp256k1 endomorphism: √3 ≈ 1.73x
  - Negation map: √2 ≈ 1.41x  
  - Combined (Galbraith-Ruprai): √6 ≈ 2.45x
  - GPU implementation efficiency (the focus of this project)
```

## Architecture Overview
The solver uses Pollard's Kangaroo with distinguished points for parallelization.
Two types of kangaroos walk the EC group:
  - **Tame**: start at a known scalar within the range, walk forward
  - **Wild**: start at the target point Q, walk forward
When a tame and wild kangaroo collide (land on the same point), the private key is recovered from the difference in their walk distances.

## The Key Optimization: Batch Inversion

### The Problem
Every kangaroo step requires the AFFINE x-coordinate of the current point to:
1. Select the next jump: `j = x mod 256`
2. Check the DP criterion: `(x & mask) == 0`

Converting Jacobian (X, Y, Z) → Affine (x, y) requires:
```
x = X · Z^(-2)    ← modular inversion costs ~256 field multiplications
```
A point addition costs ~12 field multiplications.
So standard kangaroo: 12 (add) + 256 (inv) = 268 muls/step. **95% wasted on inversion.**

### The Solution
Each CUDA thread manages K kangaroos simultaneously (K=16 to K=32).
All K kangaroos step forward in Jacobian (cheap: 12 muls each).
Then batch-invert all K Z-coordinates using Montgomery's trick:
```
Cost: 1 inversion + 3(K-1) multiplications ≈ 256 + 3K muls for all K
Per step: 256/K + 17 muls
At K=32: 25 muls/step → 10.7x speedup over standard
```

## File Structure
```
hydra_optimized/
├── CLAUDE.md              ← THIS FILE (project guide)
├── PROOF_1000X_GPU.md     ← 1000x optimization proof (puzzle_binary → GPU)
├── Makefile               ← Build system (nvcc)
├── include/
│   ├── field.cuh          ← secp256k1 field arithmetic (mod P)
│   │                        fp_add, fp_sub, fp_mul, fp_sqr, fp_inv, fp_batch_inv
│   ├── field_csa.cuh      ← 1000x: PTX MADC multiply, lazy reduction,
│   │                        warp-cooperative inversion (from puzzle_binary CSA)
│   ├── ec.cuh             ← EC point arithmetic (Jacobian coordinates)
│   │                        ec_double_j, ec_add_mixed, ec_to_affine,
│   │                        ec_batch_to_affine, ec_endomorphism
│   └── ec_pipeline.cuh    ← 1000x: Deferred-Y (x-only affine), interleaved
│                            EC phases, progressive DP check (early termination)
├── src/
│   └── hydra_kangaroo.cu  ← Solver: legacy + 1000x pipelined kernels + host
├── tests/
│   └── test_field.cu      ← Field arithmetic correctness tests
└── scripts/
    └── kangaroo.py        ← Python prototype (verified on puzzles #20-#30)
```

## Hardware Target
- Primary: NVIDIA RTX 5070 Ti (Ada Lovelace, sm_89, 16GB VRAM)
- Also: RTX 4090, RTX 3090, H100
- Build: `make` (defaults to sm_89) or `make h100` etc.

## Current State: What's Done
1. ✅ Field arithmetic (field.cuh): add, sub, mul, sqr, inv, batch_inv
2. ✅ EC arithmetic (ec.cuh): Jacobian double, mixed add, batch affine conversion
3. ✅ Kernel structure (hydra_kangaroo.cu): batch walk kernel with DP detection
4. ✅ Host DP matching (hash table collision detection)
5. ✅ Python prototype verified correct on puzzles #20, #25, #30
6. ✅ Test suite framework

## What Needs To Be Done (Priority Order)

### P0: Make It Compile and Run Correctly
1. **Fix the `u256_add_cc` and `u256_sub_borrow` PTX intrinsics** — the inline assembly uses `add.cc.u64` which requires careful carry chain management. Test on actual GPU.
2. **Fix fp_mul reduction** — the secp256k1 fast reduction (multiply overflow by 0x1000003D1) needs careful handling of double overflow. The current implementation may miss edge cases.
3. **Fix fp_inv** — replace the naive binary method with an optimized addition chain for secp256k1's P-2. Reference: libsecp256k1's `secp256k1_fe_inv` uses a carefully crafted chain that takes ~258 squarings + ~40 multiplications.
4. **Implement proper kangaroo initialization:**
   - Compute `Q' = Q - 2^134 · G` (the reframed target point)
   - For each tame kangaroo: pick random scalar s in [0, range), compute `(range_start + s)·G` as starting point
   - For each wild kangaroo: pick random scalar s, compute `Q' + s·G` as starting point
   - Store starting scalars to recover the key from collisions
5. **Precompute the jump table:** 256 random points `s_i · G` with scalars `s_i` near `√(range)/4 ≈ 2^65`. Upload to `__constant__` memory.
6. **Test: verify the kernel produces correct DPs** by solving a small puzzle (e.g., puzzle #30) on GPU and comparing against the Python prototype.

### P1: Optimize Field Arithmetic (Biggest Bang for Buck)
1. **Implement optimized fp_sqr** — squaring can be ~25% cheaper than generic multiplication because half the cross-terms are doubled instead of computed twice. The schoolbook squaring for 4 limbs needs only 10 multiplies + shifts vs 16 for generic mul.
2. **Implement optimized fp_inv addition chain** — use the secp256k1-specific chain from libsecp256k1. This reduces inversion from ~256 muls to ~240 equivalent operations (fewer multiplies, more squarings which are cheaper).
3. **Benchmark fp_mul throughput** — measure actual Mops/s on the target GPU. The 256-bit schoolbook multiplication is the critical inner loop. If below 100M/s per thread, investigate:
   - Replace schoolbook with Karatsuba (may not help for 4 limbs)
   - Use `__umul64hi` more aggressively
   - Check register usage with `nvcc --ptxas-options=-v`

### P2: Optimize the Kangaroo Walk Kernel
1. **Tune KANGAROOS_PER_THREAD (K)** — profile K=8, K=16, K=32, K=64. Higher K gives better batch inversion amortization but increases register pressure, reducing occupancy. Find the sweet spot.
2. **Optimize the walk function** — currently uses `pos[k].X.d[0] & 255` for jump selection. After batch inversion gives us affine coordinates, use `affine_pts[k].x.d[0] & 255` for correct walk function (deterministic on affine x ensures collision correctness).
3. **Implement walk distance tracking as u128 instead of u256** — walk distances don't need full 256 bits. A 128-bit counter saves register pressure.
4. **Profile and minimize shared memory usage** — each thread's K kangaroo states consume `K * 3 * 32 = K * 96` bytes of register/local memory. At K=32, that's 3072 bytes/thread. Target max occupancy.

### P3: Implement Galbraith-Ruprai Equivalence Classes (1.4x speedup)
The secp256k1 endomorphism gives us λP = (β·x, y) where β³ ≡ 1 (mod p).
Combined with negation, each point has 6 equivalents: {P, λP, λ²P, -P, -λP, -λ²P}.

The walk function should map all 6 equivalents to the same canonical representative. Then each step effectively covers 6 points → √6 speedup instead of √3.

Algorithm (Galbraith-Ruprai 2010):
1. Define canonical form: among {P, λP, λ²P, -P, -λP, -λ²P}, choose the one with the smallest x-coordinate
2. Walk function: given canonical P, compute all 6 variants, canonicalize, select jump
3. The walk is now on equivalence classes, not individual points
4. DPs are detected on canonical representatives

Reference: "Computing discrete logarithms in an interval" (Galbraith, Ruprai, 2010)

**Implementation:**
- After each batch-to-affine conversion, for each point compute all 6 x-coordinates
- Find the minimum x among {x, β·x, β²·x} (negation doesn't change x)
- Use this canonical x for jump selection and DP detection
- Track which equivalence class member was used to correct the walk distance

### P4: Implement L2 Bloom Filter for On-GPU DP Matching (1.2x speedup)
Instead of sending every DP to the host for hash table lookup:
1. Allocate a bloom filter in GPU global memory, sized to fit in L2 cache (~64MB)
2. When a tame DP is found, INSERT into bloom filter
3. When a wild DP is found, CHECK bloom filter first
4. Only send bloom-positive DPs to host for exact matching
5. This eliminates PCIe round-trip latency for 99.9% of DP checks

### P5: Distributed Mode
1. **Save/load state** — serialize DP table and kangaroo positions to disk
2. **Server mode** — accept DP contributions via TCP from remote GPU workers
3. **Client mode** — run kangaroo walks and send DPs to a central server
4. **Work file compatibility** — match JeanLucPons' `.work` file format for interop

### P6: Multi-Target Mode
Support simultaneous solving of multiple puzzles (#131-#140).
Each puzzle has a different target point but the same jump table.
Wild kangaroos start at each puzzle's Q'.
A DP collision with any target solves that puzzle.
Expected speedup: √T where T = number of targets.

## Key Constants for Reference
```c
// secp256k1 field prime
P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
// Fast reduction: 2^256 mod P = 0x1000003D1

// Curve order  
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141

// Generator
Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8

// Endomorphism
β = 0x7AE96A2B657C07106E64479EAC3434E99CF0497512F58995C1396C28719501EE
λ = 0x5363AD4CC05C30E0A5261C028812645A122E22EA20816678DF02967C1B23BD72

// Puzzle #135 target
Qx = 0x145D2611C823A396EF6712CE0F712F09B9B4F3135E3E0AA3230FB9B6D08D1E16
Qy = 0x667A05E9A1BDD6F70142B66558BD12CE2C0F9CBC7001B20C8A6A109C80DC5330
Range = [2^134, 2^135)
```

## Performance Targets
```
Metric                    Current (JLP)    Target (Hydra)    Improvement
─────────────────────────────────────────────────────────────────────────
EC adds/sec (per GPU)     500M             5-10B             10-20x
Muls per kangaroo step    268              17-25             10-15x
Effective equivalences    √3               √6                1.4x
DP matching latency       PCIe (~10μs)     L2 cache (~100ns) 100x
Overall vs baseline       1x               15-25x            —
```

## Testing Strategy
1. **Unit tests**: field arithmetic against Python `pow(a*b, 1, P)` reference values
2. **EC tests**: generator on curve, puzzle pubkey on curve, double-and-add correctness
3. **Integration test**: solve puzzle #30 (30-bit, takes seconds) on GPU, verify answer
4. **Benchmarks**: measure Mkeys/s at each optimization level, compare to JeanLucPons baseline

## References
- JeanLucPons/Kangaroo: https://github.com/JeanLucPons/Kangaroo (reference implementation, solved puzzle #130)
- libsecp256k1: https://github.com/bitcoin-core/secp256k1 (field arithmetic reference)
- Galbraith-Ruprai 2010: "Computing discrete logarithms in an interval" (equivalence class speedup)
- van Oorschot-Wiener 1999: "Parallel collision search with cryptanalytic applications" (distinguished points)
- Pollard 1978: "Monte Carlo methods for index computation (mod p)" (original kangaroo algorithm)
