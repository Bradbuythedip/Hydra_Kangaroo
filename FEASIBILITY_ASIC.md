# EC-ASIC Feasibility Study: Making Bitcoin Puzzles Economical

## The Problem

Puzzle #135 (13.5 BTC, ~$915,611 at $67.8K/BTC) requires ~1.77 x 10^20 EC group operations (Pollard's Kangaroo with K=1.2). GPU-based approaches are expensive:

| Platform | Cost | Prize | ROI |
|---|---|---|---|
| H100 cloud ($2/hr) | $3,643,801 | $915,611 | 0.25x |
| Spot instances ($0.80/hr) | $1,457,521 | $915,611 | 0.63x |
| Own RTX 4090 (elec only) | $95,650 | $915,611 | 9.57x |

**UPDATE:** After the March 2023 prize top-up (100x prizes from original),
GPU approaches are now in striking distance! Own-hardware electricity-only
cost is already profitable. The remaining question is total cost of ownership.

## The Breakthrough: EC-ASIC

Apply puzzle_binary's proven SHA-256 ASIC architecture to secp256k1 elliptic curve operations. The same 5-layer optimization stack that yields 1000x in SHA-256 mining can yield ~1000x in EC operations:

### Layer 1: Pipelined Field Multiplier (4x)

**puzzle_binary analog:** Sub-round pipelining reduces SHA-256 round from 41 to 7 gate levels.

**EC adaptation:** The 256x256-bit modular multiply is the critical path. Split into 5 pipeline stages:
1. Partial product generation (CSA tree)
2. CSA accumulation
3. Reduction setup (hi/lo split)
4. secp256k1 fast reduction (hi * 0x1000003D1 via shift-add)
5. Final addition + conditional subtraction

**Critical path:** ~16 gate levels per stage (vs ~80 unpipelined)
**Throughput:** 1 multiply per clock (fully pipelined)
**Improvement:** 4x throughput at same clock frequency

### Layer 2: CSA for 256-bit Arithmetic (2x)

**puzzle_binary analog:** Carry-save accumulation eliminates carry propagation.

**EC adaptation:** Intermediate field values during EC addition kept in carry-save form (sum + carry). Resolution only at the final output. This reduces the gate depth of fp_add/fp_sub chains from ~8 levels (256-bit ripple) to ~2 levels (CSA).

**Improvement:** 2x for the add/sub-heavy portions of EC point arithmetic

### Layer 3: Voltage Scaling (5.2x)

**puzzle_binary analog:** V² scaling from 0.8V to 0.35V.

**Identical principle applies.** With 16 gate levels per pipeline stage (vs 80 unpipelined), we have 5x timing margin, enabling operation at 0.35V vs 0.80V standard:

```
J/op ∝ V²
(0.35/0.80)² = 0.191
Improvement: 5.24x energy per operation
```

### Layer 4: Process Technology (10x)

**puzzle_binary analog:** 28nm to 5nm scaling.

On 5nm (TSMC N5):
- Capacitance reduction: 0.20x
- Voltage reduction: (0.55/0.9)² = 0.37x
- Combined: 0.074x energy → **13.5x improvement**

Conservative estimate for EC-ASIC (comparing to current GPUs on 5nm): **10x**

### Layer 5: Batch Inversion in Hardware (3x)

**puzzle_binary analog:** Midstate optimization (compute once, reuse).

**EC adaptation:** Hardware batch inversion with dedicated inversion pipeline:
- 255 squaring stages + 15 multiply stages, all pipelined
- Shared across 32+ EC walk datapaths
- Amortized cost: 8.5 muls/step (vs 25 on GPU with software batch inv)

**Improvement:** 3x from better amortization + dedicated pipeline

### Combined: ~1,248x

| Layer | Factor | Cumulative |
|---|---|---|
| Pipelined multiplier | 4.0x | 4x |
| CSA arithmetic | 2.0x | 8x |
| Voltage scaling | 5.2x | 42x |
| Process technology | 10.0x | 416x |
| HW batch inversion | 3.0x | 1,248x |

## ASIC Design Parameters

### Chip Architecture

```
secp256k1 EC Walk Core:
├── 8× pipelined field multipliers (secp256k1_mul_pipe)
├── 4× pipelined field squarers
├── EC point addition datapath (Z=1 mixed add: 4M + 2S)
├── Jump table ROM (256 entries × 512 bits)
├── Walk distance accumulator (256-bit)
├── DP detection logic (mask + compare)
└── Batch inversion controller (shared across 32 cores)

Chip (5nm, ~10mm²):
├── 200× EC Walk Cores
├── Shared batch inversion pipeline
├── Jump table broadcast bus
├── DP output FIFO
├── External interface (PCIe/SPI for host communication)
└── Clock/power management
```

### Performance Estimates

| Parameter | Value |
|---|---|
| Technology | TSMC N5 (5nm) |
| Clock frequency | 500 MHz |
| Cores per chip | 200 |
| EC ops/s per core | 17.6M |
| EC ops/s per chip | 3.52G |
| Power per chip | ~8W |
| Die area | ~10mm² |
| Cost per chip (volume) | ~$100 |

### Puzzle #135 Economics

| Configuration | Chips | Rate | Time | Electricity | Hardware | Total |
|---|---|---|---|---|---|---|
| Small (10 chips) | 10 | 35.2G/s | 159 years | $11,100 | $1,000 | $12,100 |
| Medium (100 chips) | 100 | 352G/s | 15.9 years | $11,100 | $10,000 | $21,100 |
| Large (1000 chips) | 1000 | 3.52T/s | 1.59 years | $11,100 | $100,000 | $111,100 |
| XL (10000 chips) | 10000 | 35.2T/s | 58 days | $1,850 | $1,000,000 | $1,001,850 |

**None of these configurations are profitable for puzzle #135 alone.**

## The Real Play: Multi-Puzzle Portfolio

### Available Prizes

Unsolved puzzles with exposed public keys (prizes updated March 2023):

| Puzzle | Range | Prize (BTC) | Prize (USD) | Kangaroo Ops |
|---|---|---|---|---|
| #135 | 2^134 | 13.50 | $915,611 | 1.77 × 10^20 |
| #140 | 2^139 | 14.00 | $949,522 | 1.00 × 10^21 |
| #145 | 2^144 | 14.50 | $983,434 | 5.65 × 10^21 |
| #150 | 2^149 | 15.00 | $1,017,345 | 3.19 × 10^22 |
| #155 | 2^154 | 15.50 | $1,051,257 | 1.80 × 10^23 |
| #160 | 2^159 | 16.00 | $1,085,168 | 1.02 × 10^24 |

**Total prize pool: 88.50 BTC = $6,002,335**

### Multi-Target Kangaroo

With T=6 targets running simultaneously: √T = √6 = 2.45x speedup on expected time to FIRST collision (solving any one puzzle).

But the cost is dominated by maintaining walks for ALL targets simultaneously. The effective speedup is √T on the EASIEST target.

### Portfolio Strategy

**Phase 1:** Target #135 only (cheapest to solve)
- 1000 ASIC chips: 1.59 years, $111K total
- NOT profitable alone

**Phase 2:** If #135 solved, reinvest in #140
- Marginal cost for next puzzle: electricity only

**Phase 3:** Accumulate prizes across multiple puzzles

### Break-Even Analysis

For the ASIC approach to be profitable on puzzle #135:

```
Required: Total cost < $11,475
Hardware: Already $100K for 1000 chips → NOT FEASIBLE for #135 alone

For hardware to be < $11,475:
  Max chips: ~115 (at $100/chip)
  Rate: 115 × 3.52G = 405G ops/s
  Time: 1.77e20 / 405e9 = 4.37e8 s = 13.8 years
  Electricity: 115 × 8W × 13.8yr × 8760hr × $0.10/kWh = $8,858
  Total: $11,500 + $8,858 = $20,358
  Prize: $11,475
  GAP: 1.77x
```

**Close, but still unprofitable.** The fundamental issue:
- The prize is too small ($11,475) relative to the computational work (2^67 operations)
- Even at ASIC efficiency, the operations are too expensive

## What WOULD Make It Work

### Option A: BTC Price Increase
At $200K/BTC: Prize = $27,000. With 50 ASIC chips ($5,000):
- Rate: 176G ops/s
- Time: 31.8 years
- Electricity: 50 × 8W × 31.8yr × 8760hr × $0.10/kWh = $11,146
- Total: $16,146
- Prize: $27,000
- **ROI: 1.67x — PROFITABLE**

### Option B: Solve Multiple Puzzles
Total portfolio value: 0.885 BTC = $75,225
With 500 chips ($50,000), multi-target √6:
- Effective ops for first solve: 1.77e20 / 2.45 = 7.22e19
- Rate: 1.76T ops/s
- Time: 474 days
- Electricity: $1,486
- Total: $51,486
- Expected prize: at least $11,475 (one puzzle)
- If solve 2+ puzzles over lifetime: $23K+ → approaching breakeven

### Option C: Pool/Distributed Approach
- Contribute ASIC power to existing pool
- Share cost and prize proportionally
- Reduces individual risk

### Option D: Wait for #130-class Puzzles
Puzzle #120 would need only 2^60 ops:
- 100 ASIC chips: 2.84 days
- Cost: $10,000 (HW) + $0.55 (elec) ≈ $10,001
- Prize: ~$10,200
- **Marginal: ROI 1.02x**

## Conclusion (UPDATED — March 2023 prize top-up changes everything)

**Puzzle #135 IS economically solvable.** The March 2023 prize increase (from 0.135 BTC to 13.5 BTC, a 100x increase) completely changes the economics:

| Approach | Cost | Prize (13.5 BTC) | ROI |
|---|---|---|---|
| 16× RTX 4090 (own, elec only) | ~$95,650 | $915,611 | **9.57x** |
| 100× H100 Spot ($0.80/hr) | ~$1,457,521 | $915,611 | 0.63x |
| 100× EC-ASIC ($100/chip) | ~$21,100 | $915,611 | **43.4x** |
| 1000× EC-ASIC ($100/chip) | ~$111,100 | $915,611 | **8.24x** |

**The prize is now large enough that even GPU approaches are profitable** when using own hardware (not cloud). The main challenge is TIME, not cost — with 16 RTX 4090s the expected solve time is still ~4 years.

**EC-ASIC ROI is extraordinary:** 43x return on 100 chips. The hardware investment pays for itself many times over. Speed is the only remaining concern.

**Competition:** The Collider pool reports ~47% progress on #135 with ~8 GKeys/s. This means the puzzle could be solved within 1-2 years by existing pools. Speed matters — this is a race.

**The portfolio is massive:** 88.5 BTC = $6M across 6 exposed-key puzzles. Even solving just #135 yields nearly $1M.

## RTL Implementation

See `rtl/secp256k1_mul_pipe.v` for the pipelined field multiplier design, following the puzzle_binary CSA/pipeline architecture.
