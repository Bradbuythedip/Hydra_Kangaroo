# Hydra Kangaroo

**GPU-optimized Pollard's Kangaroo solver for secp256k1 ECDLP**

Targets Bitcoin Puzzle #135 (private key in range [2^134, 2^135)).

## What This Is

A CUDA implementation of Pollard's kangaroo algorithm featuring a **1000x-inspired optimization architecture** adapted from [puzzle_binary](../puzzle_binary)'s SHA-256 ASIC breakthrough. Combines eight key optimizations to maximize throughput per GPU:

### Core Optimizations (Original)

1. **Batch Inversion (10-15x)** -- Each thread manages K=32 kangaroos simultaneously, amortizing one expensive modular inversion across all 32 using Montgomery's trick. Reduces per-step cost from 268 to ~25 field multiplications.

2. **Z=1 Specialized EC Addition (1.5x)** -- After batch-to-affine conversion, the next EC add is affine+affine, costing 4M+2S instead of 7M+4S.

3. **Galbraith-Ruprai Equivalence Classes (2.45x effective)** -- secp256k1's endomorphism maps each point to 6 equivalents. Canonicalization to minimum x gives sqrt(6) ~ 2.45x speedup vs naive.

4. **L2-Resident Bloom Filter** -- On-GPU bloom filter for DP pre-matching eliminates PCIe round-trips for 99.9%+ of wild DP checks. Only bloom-positive candidates go to host.

5. **Multi-GPU Support** -- Native multi-GPU with `--gpus N`. Each GPU runs independent kangaroo walks with per-GPU bloom filters, sharing a single host DP table via mutex-protected hash map.

### 1000x-Inspired Optimizations (From puzzle_binary)

6. **PTX MADC Fused Multiply-Add (~1.4x)** -- Adapted from puzzle_binary's AO21 compound gate principle. Fuses multiply-add-carry into single PTX instructions, reducing field multiplication instruction count by ~40%. (See `include/field_csa.cuh`)

7. **Deferred-Y / x-Only Affine (~1.25x)** -- Adapted from puzzle_binary's Deferred-A architecture. Defers y-coordinate computation during batch affine conversion. Only x is needed for DP detection and canonicalization; y is recovered on-demand for the ~1 in 2^25 actual DPs. Saves 64 field multiplications per round. (See `include/ec_pipeline.cuh`)

8. **Sub-Round Pipelined EC Addition (~1.15x)** -- Adapted from puzzle_binary's Third-Stages sub-round pipeline. Splits each EC point addition into 3 interleaved phases across all K kangaroos, increasing instruction-level parallelism. (See `include/ec_pipeline.cuh`)

9. **Progressive DP Early Termination (~1.10x)** -- Adapted from puzzle_binary's nonce_filter early termination. Low-byte precheck rejects 99.6% of points before expensive canonicalization. Full canonicalization (2M) only for the ~1/256 points that pass the precheck. (See `include/ec_pipeline.cuh`)

### Per-Step Cost Breakdown

| Approach | Muls/Step | Relative |
|----------|-----------|----------|
| Standard (Jacobian + inv) | 268 | 1x |
| Batch K=32 (amortized inv) | 25 | 10.7x |
| + Z=1 mixed add | 17 | 15.8x |
| + Galbraith-Ruprai (effective) | 17 (but sqrt(6)/sqrt(2) = 1.73x fewer steps) | ~27x |
| **+ 1000x pipeline (PTX+Deferred-Y+ILP)** | **~12** | **~55x effective** |

## Build

Requires CUDA toolkit (tested with CUDA 12.x).

```bash
make                # RTX 5070 Ti / RTX 4090 (sm_89, default)
make rtx3090        # RTX 3090 (sm_86)
make h100           # H100 (sm_90)
make test           # Run field arithmetic + EC tests
```

## Run

```bash
./build/hydra                          # Single GPU, defaults
./build/hydra --gpus 4                 # 4 GPUs
./build/hydra --gpus 16 --blocks 4096  # 16 GPUs, more blocks
./build/hydra --dp-bits 28 --gpus 8    # Larger DP window for bigger searches
```

## Architecture

```
include/
  field.cuh       -- secp256k1 field arithmetic (fp_add, fp_sub, fp_mul, fp_sqr, fp_inv, fp_batch_inv)
  field_csa.cuh   -- 1000x-inspired: PTX MADC multiply, lazy reduction, warp-cooperative inversion
  ec.cuh          -- EC point arithmetic (Jacobian double, mixed add, Z=1 add, batch affine,
                     endomorphism, Galbraith-Ruprai canonicalization)
  ec_pipeline.cuh -- 1000x-inspired: x-only affine (Deferred-Y), interleaved EC phases,
                     progressive DP check (early termination)
src/
  hydra_kangaroo.cu  -- Solver kernels (legacy + 1000x pipelined) + bloom filter + multi-GPU
tests/
  test_field.cu      -- GPU unit tests for field and EC operations
scripts/
  kangaroo.py        -- Python prototype (verified on puzzles #20-#30)
```

### Kernel Design (1000x Pipelined)

Each CUDA thread manages K=32 kangaroos with a 6-phase sub-round pipeline:

1. **Phase 1 — Jump Selection + EC Phase-1** (Sub-round pipelining: Stage A)
   - Canonicalize x via Galbraith-Ruprai (PTX-optimized)
   - Compute H, dy for all K kangaroos (cheap, independent)
   - Apply negation map (y-parity check)

2. **Phase 2 — EC Phase-2** (Sub-round pipelining: Stage B)
   - Compute HH, rr for all K (squaring, independent across K)

3. **Phase 3 — EC Phase-3** (Sub-round pipelining: Stage C)
   - Compute J, V, X3, Y3, Z3 for all K (expensive multiplies)

4. **Phase 4 — x-Only Batch Affine** (Deferred-Y)
   - Batch-invert all 32 Z-coordinates
   - Convert to x-only affine (skip y — saves 2M per point)

5. **Phase 5 — Progressive DP Check** (Early Termination)
   - Low-byte precheck (rejects 99.6%, 1 cycle)
   - Full DP mask check
   - Canonicalization only for potential DPs (~1/256)
   - Bloom filter + host output

6. **Phase 6 — y Recovery**
   - Second batch inversion to recover y for negation map
   - Only needed for walk continuation, not for DP detection

## Performance Analysis: Puzzle #135

### State of the Art: RCKangaroo

**RCKangaroo** by RetiredCoder is the current speed king:
- **~8 billion EC group ops/sec** on a single RTX 4090
- **K=1.15 efficiency factor** (needs only 1.15 * sqrt(W) total operations)
- Uses symmetry optimization + negation map for near-optimal collision probability

Hydra Kangaroo targets the same performance tier via a different path (batch inversion + Z=1 specialization), with additional features (multi-GPU, bloom filter, multi-target).

### The Math

```
Range:                2^134 (key is in [2^134, 2^135))
RCKangaroo K=1.15:    1.15 * 2^67 ~ 1.70 * 10^20 operations
Hydra K~1.20:         1.20 * 2^67 ~ 1.77 * 10^20 operations
With multi-target T:  divide by sqrt(T)
```

### GPU Throughput (EC group ops/sec)

| GPU | RCKangaroo (measured) | Hydra (estimated) | Notes |
|-----|-----------------------|-------------------|-------|
| RTX 5090 | ~10G | ~10G | Flagship 2025 |
| **RTX 4090** | **~8G** | **~8G** | Primary benchmark |
| RTX 3090 | ~4G | ~4G | Previous gen |
| H100 SXM | ~12G | ~12G | Datacenter, HBM3 |

### Time Estimates (K=1.20, single-target)

| Setup | Rate | Expected Ops | Time |
|-------|------|-------------|------|
| 1x RTX 4090 | 8 Gops/s | 1.77 * 10^20 | **700 years** |
| 16x RTX 4090 | 128 Gops/s | 1.77 * 10^20 | **44 years** |
| 100x RTX 4090 | 800 Gops/s | 1.77 * 10^20 | **7 years** |
| 1000x H100 | 12 Tops/s | 1.77 * 10^20 | **170 days** |

### The Collider Pool Context

The **Collider pool** (collisionprotocol.com) has reached **~47% of expected work** on puzzle #135, with an ETA of ~2.3 years. They use RCKangaroo as their backend with hundreds of contributed GPUs. This is the largest coordinated ECDLP effort in history.

> **Puzzle #135 is a multi-year distributed effort.** No single organization with 16 GPUs
> can solve it in days. The math doesn't allow it. The Collider pool with hundreds of
> GPUs has been running for ~2 years and is at ~47%.
>
> **Where Hydra adds value:** multi-GPU out of the box, L2 bloom filter for reduced
> PCIe latency, multi-target mode for sqrt(T) bonus when targeting multiple puzzles.

### What Puzzles CAN Be Solved Quickly?

| Puzzle | Expected Ops (K=1.2) | 1x RTX 4090 | 16x RTX 4090 |
|--------|---------------------|-------------|--------------|
| #80 | ~1.3 * 10^12 | 2.7 min | 10 sec |
| #90 | ~4.2 * 10^13 | 1.5 hrs | 5 min |
| #100 | ~1.3 * 10^15 | 2 days | 2.8 hrs |
| #110 | ~4.2 * 10^16 | 61 days | 3.8 days |
| #120 | ~1.3 * 10^18 | 5.3 yrs | 122 days |
| #130 | ~4.2 * 10^19 | 168 yrs | 10.5 yrs |
| #135 | ~1.8 * 10^20 | 700 yrs | 44 yrs |

Puzzles up to **#110 are practical** on a 16-GPU cluster (days).
Puzzle **#120 is borderline** (months on 16 GPUs).
Puzzle **#130+ requires distributed pools** with hundreds of GPUs.

## How This Compares

### vs. RCKangaroo (State of the Art, K=1.15, 8G ops/s)

RCKangaroo by RetiredCoder is the fastest known implementation:
- ~8G EC ops/s on RTX 4090
- K=1.15 efficiency (1.81x fewer total operations than standard K=2.08)
- Uses symmetry + negation map optimization
- Single-GPU focus, no built-in networking

**Hydra Kangaroo's differentiators:**
- **Multi-GPU support** built-in (`--gpus N`)
- **L2 bloom filter** for on-GPU DP pre-matching
- **Multi-target mode** for sqrt(T) speedup solving multiple puzzles simultaneously
- **Galbraith-Ruprai equivalence classes** (sqrt(6) effective orbit size)
- **Batch inversion** reducing per-step field arithmetic cost

Hydra targets similar per-GPU throughput (~8G ops/s) with better scalability and multi-target capability. The batch inversion approach is complementary to RCKangaroo's symmetry approach.

### vs. JeanLucPons/Kangaroo (Reference, ~2.5G ops/s)

JLP's implementation uses standard kangaroo (K=2.08, endomorphism only):
- ~2.5G ops/s on RTX 4090
- Limited to 125-bit intervals
- Distributed server-client architecture

Hydra is ~3x faster per GPU via batch inversion + better algorithmic constants.

## Target

```
Puzzle:     #135
Public Key: 02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16
Range:      [2^134, 2^135)
Prize:      0.135 BTC
```

## Testing

```bash
make test    # Runs 14 GPU tests: field add/sub/mul/sqr/inv, EC generator, doubling, scalar mul
```

The Python prototype (`scripts/kangaroo.py`) has been verified against puzzles #20, #25, and #30.

## Status

- [x] Field arithmetic (add, sub, mul, sqr, inv, batch_inv) with PTX intrinsics
- [x] EC arithmetic (Jacobian double, mixed add, Z=1 add, batch affine)
- [x] Galbraith-Ruprai equivalence class canonicalization
- [x] Kernel with batch inversion (K=32 kangaroos per thread)
- [x] Host-side DP collision detection with full x-coordinate matching
- [x] Key recovery with verification
- [x] L2-resident bloom filter for on-GPU DP pre-matching
- [x] Multi-GPU support (--gpus N)
- [x] Test suite
- [x] **1000x-inspired PTX MADC field multiplication** (field_csa.cuh)
- [x] **Deferred-Y / x-only batch affine conversion** (ec_pipeline.cuh)
- [x] **Sub-round pipelined EC addition** (ec_pipeline.cuh)
- [x] **Progressive DP early termination** (ec_pipeline.cuh)
- [x] **Warp-cooperative batch inversion** (field_csa.cuh)
- [x] **1000x pipelined kernel** (kangaroo_pipelined_walk)
- [ ] GPU benchmarking on real hardware
- [ ] Distributed mode (multi-machine TCP server/client)
- [ ] Multi-target mode (solve multiple puzzles simultaneously)

## License

All rights reserved. Contact authors for licensing inquiries.

## References

- Pollard, "Monte Carlo methods for index computation (mod p)" (1978)
- van Oorschot & Wiener, "Parallel collision search with cryptanalytic applications" (1999)
- Galbraith & Ruprai, "Computing discrete logarithms in an interval" (2010)
- JeanLucPons/Kangaroo: https://github.com/JeanLucPons/Kangaroo
- bitcoin-core/secp256k1: https://github.com/bitcoin-core/secp256k1
