# 1000x GPU Mining Optimization — From SHA-256 ASIC to ECDLP GPU

## Executive Summary

This document describes how the **1000x J/TH improvement** proven in [puzzle_binary](../puzzle_binary) for SHA-256 Bitcoin mining ASICs has been adapted to GPU-accelerated elliptic curve discrete logarithm solving (ECDLP) in Hydra Kangaroo.

The puzzle_binary project achieves its 1000x improvement through 5 architectural layers. Each layer has a direct analog in GPU ECDLP optimization:

| puzzle_binary Layer | ASIC Factor | GPU Analog | GPU Factor |
|---|---|---|---|
| Sub-round pipelining (41→7 gate levels) | 5.85x | Interleaved EC addition phases | ~1.15x ILP |
| Compound gates (AO21) | 1.50x | PTX MADC fused multiply-add | ~1.40x |
| Voltage scaling (V² reduction) | 5.24x | *(hardware-level, not applicable)* | — |
| Process scaling (28nm→5nm) | 13.3x | *(hardware-level, not applicable)* | — |
| Architecture (midstate, early termination) | 2.53x | Deferred-Y + progressive DP check | ~1.35x |
| **Combined (ASIC)** | **1,230x** | **Combined (GPU software)** | **~2.2x** |

**Note:** The full 1000x in puzzle_binary comes primarily from voltage scaling (V²) and process technology, which are hardware-level improvements. The software-transferable architectural principles yield ~2.2x on GPU, which when combined with Hydra Kangaroo's existing ~25x speedup from batch inversion, produces **~55x effective speedup** over standard kangaroo implementations.

---

## Layer 1: Carry-Save Accumulation → PTX MADC Multiplication

### puzzle_binary (ASIC)

SHA-256 intermediate values are stored in **carry-save form** (sum + carry) rather than fully resolved. This eliminates carry propagation from the critical path, reducing the gate depth from 41 to 20 levels (2.05x improvement).

**Key file:** `puzzle_binary/rtl/sha256_csa_pipe_stage_v3.v`

```verilog
// Intermediate A stored as carry-save pair (no completion adder)
// Resolution deferred until needed for Sigma0/Maj computation
assign a_s_out = a_s_new;  // sum component
assign a_c_out = a_c_new;  // carry component
```

### Hydra Kangaroo (GPU)

The GPU analog is **PTX MADC (Multiply-Add with Carry) chains**. Instead of computing multiplications with separate multiply, add, and carry-check instructions (6 instructions per partial product), MADC fuses these into 2 instructions:

**Key file:** `include/field_csa.cuh`

```cuda
// Standard approach: 6 instructions per partial product
uint64_t lo = a * b;                    // mul
uint64_t hi = __umul64hi(a, b);         // mul.hi
uint64_t sum = accumulator + lo;        // add
uint64_t carry = (sum < accumulator);   // compare (branch)
...

// MADC approach: 2 instructions per partial product
asm("mad.lo.cc.u64  %1, %8, %13, %1;"  // fused multiply-add + carry
    "madc.hi.u64    %2, %8, %13,  0;"); // fused multiply-add-carry
```

This is exactly analogous to puzzle_binary's AO21 compound gates: multiple logic operations fused into a single hardware instruction. The ~40% instruction reduction directly translates to ~1.4x throughput improvement for field multiplication, which is the innermost loop of the ECDLP solver.

**Implementation:** `fp_mul_ptx()` and `fp_sqr_ptx()` in `include/field_csa.cuh`

---

## Layer 2: Deferred-A Architecture → Deferred-Y (x-Only Affine)

### puzzle_binary (ASIC)

The **Deferred-A** technique eliminates the serial dependency between A and E computation paths in SHA-256:

```
Standard:  A_new = T1 + T2  (T1 depends on computed E → serial)
Deferred:  A_new = E_in + ~D_prev + Sigma0(B) + Maj(B,C,D)
                   ↑ all from registers — no dependency on E computation
```

**Key insight:** A doesn't need to be fully computed until it's actually used. By deferring its resolution, A and E can compute **in parallel**.

### Hydra Kangaroo (GPU)

The **Deferred-Y** technique applies the same principle to elliptic curve point arithmetic. When converting from Jacobian to affine coordinates after batch inversion, the standard approach computes both x and y:

```
Standard batch-to-affine per point:
  Z^(-1)              (from batch inversion)
  Z^(-2) = Z^(-1)²   (1 squaring)
  Z^(-3) = Z^(-2) × Z^(-1)  (1 multiply)
  x = X × Z^(-2)     (1 multiply)
  y = Y × Z^(-3)     (1 multiply)
  Total: 1S + 3M per point
```

But for **DP detection** and **Galbraith-Ruprai canonicalization**, only the x-coordinate is needed. The y-coordinate is only required for the negation map (parity check) and for DP output. Since ~99.997% of points are NOT distinguished points, we defer y computation:

```
Deferred-Y batch-to-affine per point:
  Z^(-1)              (from batch inversion)
  Z^(-2) = Z^(-1)²   (1 squaring)
  x = X × Z^(-2)     (1 multiply)
  Total: 1S + 1M per point

  On-demand (only for DPs, ~1 in 2^25):
  Z^(-3) = Z^(-2) × Z^(-1)  (1 multiply)
  y = Y × Z^(-3)     (1 multiply)
```

**Savings:** 2M per point × K=32 = **64 field multiplications per round** eliminated.

**Implementation:** `ec_batch_to_xonly_ptx()` in `include/ec_pipeline.cuh`

---

## Layer 3: Sub-Round Pipelining → Interleaved EC Addition

### puzzle_binary (ASIC)

Each SHA-256 round is split into **3 sub-stages** (Third-A, Third-B, Third-C) with register boundaries between them. This increases instruction-level parallelism:

```
Third-A (6 levels): Sigma/Ch/Maj + CSA tree
Third-B (6 levels): KS prefix L0-L2 + DHKW precompute (parallel)
Third-C (7 levels): KS prefix L3-L5 + final sum
```

Independent operations from different stages can execute simultaneously in the hardware pipeline.

### Hydra Kangaroo (GPU)

Each Z=1 mixed EC addition is split into **3 interleaved phases** across all K kangaroos:

```
Phase 1 (all K): H = qx - px, dy = qy - py     (2 subtractions, independent)
Phase 2 (all K): HH = H², rr = 2*dy             (1 squaring, independent)
Phase 3 (all K): J, V, X3, Y3, Z3               (4M + 1S, depends on Phase 2)
```

Instead of computing the full EC addition sequentially for each kangaroo, all K Phase-1 operations execute first (cheap, independent), then all Phase-2 operations, then all Phase-3 operations. The GPU's warp scheduler can interleave independent multiply instructions from different kangaroos, hiding multiply latency.

**Implementation:** `ec_add_z1_phase1()`, `ec_add_z1_phase2()`, `ec_add_z1_phase3()` in `include/ec_pipeline.cuh`

---

## Layer 4: DHKW Precomputation → Progressive DP Check

### puzzle_binary (ASIC)

DHKW (D+H+K+W) is precomputed during Third-B, running **in parallel** with the KS prefix computation. This "fills the bubble" — using otherwise-idle hardware to prepare values needed in the next stage.

### Hydra Kangaroo (GPU)

**Progressive DP check** applies the same principle: avoid expensive computation when simpler checks can reject most candidates:

```
Level 1: Check low byte of x against DP mask    (~1 cycle, rejects 99.6%)
Level 2: Check full dp_mask bits                 (~1 cycle, rejects remaining)
Level 3: Galbraith-Ruprai canonicalization       (2M, only for potential DPs)
Level 4: Bloom filter check                      (L2 cache access)
Level 5: On-demand y recovery                    (2M, only for actual DPs)
```

The 2M canonicalization cost is paid for only ~1/256 of points (those passing the byte precheck), and the 2M y-recovery cost for only ~1 in 2^25 points (actual DPs). This is exactly analogous to puzzle_binary's early termination where downstream logic is clock-gated for the 99.999% of hashes that fail.

**Implementation:** `dp_precheck()` and `dp_fullcheck()` in `include/ec_pipeline.cuh`

---

## Layer 5: Warp-Cooperative Batch Inversion

### puzzle_binary (ASIC)

Multi-core designs share W expansion and DHKW precompute logic across multiple pipeline copies, reducing per-core overhead.

### Hydra Kangaroo (GPU)

**Warp-cooperative inversion** uses warp shuffle primitives to share the single expensive modular inversion (255 squarings + 15 multiplications) across all 32 threads in a warp.

Standard per-thread: 1 inversion = 255S + 15M per thread
Warp-cooperative: 1 shared inversion = (255S + 15M)/32 ≈ 8S + 0.5M per thread

The product tree is built using butterfly reduction across warp lanes, with `__shfl_xor_sync` exchanging partial products between threads.

**Implementation:** `fp_warp_inv()` in `include/field_csa.cuh`

---

## Combined Speedup Analysis

### Per-Step Cost Comparison

| Operation | Legacy Kernel | 1000x Kernel | Savings |
|---|---|---|---|
| EC Z=1 mixed add | 5.5M (4M+2S) | 3.3M (PTX MADC) | 40% |
| Batch inv amortized | 11.4M | 6.8M (PTX) | 40% |
| Affine conversion | 2.75M (1S+2M) | 1.75M (x-only) | 36% |
| Canonicalization | 2.0M (every step) | 0.008M (amortized) | 99.6% |
| **Total per step** | **~21.7M** | **~11.9M** | **45%** |

### Effective Throughput Multiplier

| Factor | Improvement | Source |
|---|---|---|
| PTX MADC chains | 1.40x | `fp_mul_ptx` (fewer instructions per mul) |
| Deferred-Y (x-only) | 1.25x | Skip 2M per point for y-coordinate |
| Sub-round pipelining | 1.15x | Better ILP from interleaved phases |
| Progressive DP check | 1.10x | Skip canonicalization for non-DPs |
| **Combined** | **~2.2x** | Multiplicative |

### End-to-End Performance

| Configuration | Standard Kangaroo | Hydra Legacy | Hydra 1000x |
|---|---|---|---|
| Muls per step | 268 | ~22 | ~12 |
| Effective speedup | 1x | ~12x | ~22x |
| With Galbraith-Ruprai | 1x | ~30x | ~55x |
| RTX 4090 (est.) | 2.5G ops/s | 8G ops/s | 18G ops/s |

### Time Estimates: Puzzle #135

| Setup | Hydra Legacy | Hydra 1000x |
|---|---|---|
| 1x RTX 4090 | 700 years | ~320 years |
| 16x RTX 4090 | 44 years | ~20 years |
| 100x RTX 4090 | 7 years | ~3.2 years |
| 1000x H100 | 170 days | ~77 days |

---

## Architectural Principle Mapping

| puzzle_binary Principle | GPU Adaptation | File |
|---|---|---|
| Carry-Save Accumulation | Lazy reduction, redundant representation | `field_csa.cuh` |
| Deferred-A (parallel A/E) | Deferred-Y (x-only affine) | `ec_pipeline.cuh` |
| Sub-round pipelining | Interleaved EC phases across K kangaroos | `ec_pipeline.cuh` |
| DHKW precompute | Progressive DP filtering | `ec_pipeline.cuh` |
| Early termination | dp_precheck() low-byte reject | `ec_pipeline.cuh` |
| Compound gates (AO21) | PTX MADC fused multiply-add | `field_csa.cuh` |
| Pipelined Kogge-Stone | PTX carry chain for 256-bit addition | `field_csa.cuh` |
| Multi-core sharing | Warp-cooperative batch inversion | `field_csa.cuh` |
| Clock gating | Bloom filter (skip PCIe for non-matches) | `hydra_kangaroo.cu` |

---

## What's Transferable and What's Not

### Transferable (software optimization)
- Deferred computation (defer expensive work until proven necessary)
- Carry-save / lazy reduction (skip normalization on intermediate values)
- Operation pipelining (increase ILP by splitting into independent phases)
- Precomputation (fill idle cycles with useful work)
- Early termination (reject fast, verify slow)

### Not Transferable (hardware-level)
- Voltage scaling (V² improvement) — requires ASIC/FPGA control
- Process technology scaling (28nm→5nm) — hardware manufacturing
- Literal carry-save register storage — GPU uses fixed 64-bit ALUs
- Gate-level compound operations — GPU ISA is fixed

The 1000x in puzzle_binary is ~1,230x total. Of that:
- ~14x comes from software-transferable architecture (5.85x pipeline × 2.53x midstate/gating)
- ~88x comes from hardware-level optimizations (voltage × process)

For GPU, we capture the architectural principles (~2.2x improvement), which when combined with Hydra's existing batch inversion framework (~12x over standard), delivers ~55x total effective speedup over naive Pollard's kangaroo implementations.
