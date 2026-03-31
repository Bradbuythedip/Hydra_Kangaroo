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
Prize:      13.5 BTC (~$915,000 at $67.8K/BTC — updated March 2023 top-up)
```

## Competition
```
Collider pool: ~47% progress at ~8 GKeys/s (K=1.15)
Expected time remaining: ~1-2 years
This is a RACE — speed of solve matters as much as cost.
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
7. ✅ PTX MADC field multiply (field_csa.cuh): 40% fewer instructions
8. ✅ Deferred-Y x-only affine (ec_pipeline.cuh): skip y for 99.997% of points
9. ✅ Sub-round pipeline (ec_pipeline.cuh): 3-phase interleaved EC add
10. ✅ Progressive DP check (ec_pipeline.cuh): 99.6% early termination
11. ✅ L2 Bloom filter (hydra_kangaroo.cu): on-GPU DP pre-matching
12. ✅ Galbraith-Ruprai sqrt(6) equivalence classes
13. ✅ Unified batch inversion (single inv for both x and y)
14. ✅ 3-Kangaroo variant (tame/wild/middle, K=0.90)
15. ✅ Adaptive DP threshold (auto-computed from kangaroo count)
16. ✅ Multi-target puzzle registry
17. ✅ Hyper kernel (K=64, warp-cooperative inversion)
18. ✅ EC-ASIC RTL design (rtl/secp256k1_mul_pipe.v)
19. ✅ Economic feasibility analysis and calculator
20. ✅ fp_inv_ptx: PTX-optimized field inversion (~40% fewer instructions)
21. ✅ fp_batch_inv_ptx: all kernel batch inversions use PTX path
22. ✅ Hyper kernel wave 1 optimized: single warp-coop inv (was 2)
23. ✅ Seed recovery attack suite (10 tests, all NEGATIVE — keys are crypto-random)

## Three Kernel Modes
```
--legacy     Legacy batch walk (K=32, standard batch inversion)
(default)    Pipelined 1000x kernel (K=32, PTX MADC + Deferred-Y + 3-phase pipeline)
--hyper      Hyper kernel (K=64, warp-cooperative + two-wave processing)
```

## What Needs To Be Done (Priority Order)

### P0: Test on Real Hardware
1. Compile and run on RTX 4090 / RTX 5090 / H100
2. Verify correct DP generation by solving puzzle #30 (30-bit)
3. Benchmark all three kernel modes (legacy vs pipelined vs hyper)
4. Profile register usage: `nvcc --ptxas-options=-v`

### P1: Fix Known Issues
1. **u256_add_cc / u256_sub_borrow PTX intrinsics** — carry chain management
2. **fp_mul reduction edge cases** — double overflow in secp256k1 fast reduction
3. **fp_inv addition chain** — optimize from binary method to secp256k1-specific chain

### P2: Further Optimization
1. **Walk distance tracking as u128** — saves 16 bytes/kangaroo register pressure
2. **Warp-divergence profiling** — measure negation map branch divergence
3. **Jump table in shared memory** — reduce constant memory pressure at high occupancy
4. **Karatsuba for field multiply** — may help if register pressure is the bottleneck

### P3: Distributed Mode
1. **Save/load state** — serialize DP table and kangaroo positions to disk
2. **Server mode** — accept DP contributions via TCP from remote GPU workers
3. **Client mode** — run kangaroo walks and send DPs to a central server
4. **Work file compatibility** — match JeanLucPons' `.work` file format for interop

### P4: Multi-GPU Scaling
1. Profile multi-GPU scaling (2, 4, 8 GPUs via --gpus flag)
2. Optimize DP table mutex contention for high GPU counts
3. Add GPU-to-GPU P2P DP exchange (skip host for same-node GPUs)

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
Metric                    Baseline (JLP)   Hydra (pipelined)  Hydra (hyper)
──────────────────────────────────────────────────────────────────────────
Muls per kangaroo step    268              ~25 (K=32)         ~21 (K=64)
Equivalence classes       √3               √6                 √6
K-factor                  1.15             0.90 (3-kang)      0.90 (3-kang)
DP matching               PCIe (~10μs)     L2 bloom (~100ns)  L2 bloom
Multi-target              1 puzzle         √T puzzles         √T puzzles
Overall vs naive          ~5x              ~145x              ~170x
```

## Optimization Stack (Combined ~145x)
```
Layer                          Factor    Cumulative
────────────────────────────────────────────────────
Batch inversion (K=32/64)      10.7x     10.7x
PTX MADC multiply              1.40x     15.0x
Deferred-Y x-only              1.25x     18.7x
Sub-round pipeline              1.15x     21.5x
Progressive DP check            1.10x     23.7x
L2 Bloom filter                 1.20x     28.4x
Unified batch inversion         1.12x     31.8x
Adaptive DP threshold           1.05x     33.4x
PTX-optimized fp_inv_ptx        1.13x     37.7x
Hyper K=64 (optional)           1.18x     44.5x
3-Kangaroo (K=0.90)            1.33x     59.2x
Galbraith-Ruprai √6            2.45x     145.0x
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
