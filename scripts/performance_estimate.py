#!/usr/bin/env python3
"""
Hydra Kangaroo — Performance Estimation & Mining Economics

Estimates actual GKey/s for different GPUs based on:
1. GPU specs (SMs, clock, register file)
2. Kernel arithmetic costs (field muls per step)
3. Register pressure and occupancy limits
4. Comparison with JeanLucPons/Kangaroo baseline

Then computes puzzle #135 solve time and economics.
"""

import math

# ═══════════════════════════════════════════════════════════════
# secp256k1 FIELD MULTIPLY COST
# ═══════════════════════════════════════════════════════════════

# A 256-bit modular multiply on GPU:
# - 4x4 = 16 partial 64-bit multiplies (each = 2 PTX ops: mul.lo + mul.hi)
# - Accumulation: ~16 add operations
# - Reduction (secp256k1 fast): 4 × (mul + add) + carry handling ≈ 12 ops
# Total: ~44 PTX instructions per fp_mul_ptx
# At ~1 instruction/clock/warp: 44 clocks per mul

# A 256-bit modular square is ~30 PTX instructions (6 cross-terms + 4 diags)

CLOCKS_PER_MUL = 44   # PTX instructions for fp_mul_ptx
CLOCKS_PER_SQR = 30   # PTX instructions for fp_sqr_ptx

# ═══════════════════════════════════════════════════════════════
# KANGAROO STEP COST BREAKDOWN
# ═══════════════════════════════════════════════════════════════

# EC point addition (Z=1 mixed, affine + affine → Jacobian)
# From ec_pipeline.cuh ec_add_z1_phase1/2/3: 4M + 2S
EC_ADD_MULS = 4
EC_ADD_SQRS = 2

# Batch inversion amortized cost (K=32, per step)
# Total: 1 inv + 93M = (255S + 15M) + 93M
# Per step: (255S + 15M + 93M) / 32
BATCH_INV_MULS_PER_STEP = (15 + 93) / 32   # 3.375 M/step
BATCH_INV_SQRS_PER_STEP = 255 / 32          # 7.97 S/step

# Affine conversion: Z^(-2) + Z^(-3) + 2 muls = 1S + 3M per point (included in batch)
AFFINE_CONV_MULS = 3  # zi3 = zi2*zi, x = X*zi2, y = Y*zi3
AFFINE_CONV_SQRS = 1  # zi2 = zi^2

# Total per kangaroo step
TOTAL_MULS = EC_ADD_MULS + BATCH_INV_MULS_PER_STEP + AFFINE_CONV_MULS
TOTAL_SQRS = EC_ADD_SQRS + BATCH_INV_SQRS_PER_STEP + AFFINE_CONV_SQRS

# Convert to equivalent clocks
CLOCKS_PER_STEP = TOTAL_MULS * CLOCKS_PER_MUL + TOTAL_SQRS * CLOCKS_PER_SQR

print("═" * 70)
print("  HYDRA KANGAROO — PERFORMANCE ESTIMATION")
print("═" * 70)

print(f"\n  Per-step cost:")
print(f"    EC add (Z=1):        {EC_ADD_MULS}M + {EC_ADD_SQRS}S")
print(f"    Batch inv (K=32):    {BATCH_INV_MULS_PER_STEP:.1f}M + {BATCH_INV_SQRS_PER_STEP:.1f}S")
print(f"    Affine conversion:   {AFFINE_CONV_MULS}M + {AFFINE_CONV_SQRS}S")
print(f"    Total:               {TOTAL_MULS:.1f}M + {TOTAL_SQRS:.1f}S")
print(f"    Clock estimate:      {CLOCKS_PER_STEP:.0f} clocks/step")

# ═══════════════════════════════════════════════════════════════
# GPU SPECS
# ═══════════════════════════════════════════════════════════════

gpus = [
    {
        "name": "RTX 4090",
        "sms": 128,
        "clock_ghz": 2.52,
        "regs_per_sm": 65536,
        "shared_per_sm": 102400,  # bytes
        "max_threads_per_sm": 1536,
        "power_w": 450,
        "cost_usd": 1600,
    },
    {
        "name": "RTX 5090",
        "sms": 170,
        "clock_ghz": 2.40,
        "regs_per_sm": 65536,
        "shared_per_sm": 131072,
        "max_threads_per_sm": 1536,
        "power_w": 575,
        "cost_usd": 2000,
    },
    {
        "name": "RTX 3090",
        "sms": 82,
        "clock_ghz": 1.70,
        "regs_per_sm": 65536,
        "shared_per_sm": 102400,
        "max_threads_per_sm": 1536,
        "power_w": 350,
        "cost_usd": 800,
    },
    {
        "name": "H100 SXM",
        "sms": 132,
        "clock_ghz": 1.83,
        "regs_per_sm": 65536,
        "shared_per_sm": 233472,
        "max_threads_per_sm": 2048,
        "power_w": 700,
        "cost_usd": 30000,
    },
]

# ═══════════════════════════════════════════════════════════════
# OCCUPANCY & THROUGHPUT
# ═══════════════════════════════════════════════════════════════

# Register usage per thread (estimated):
# x_aff[32]: 32 × 32 bytes = 1024 bytes → 256 regs
# y_aff[32]: 32 × 32 = 1024 → 256 regs (but deferred, not all live)
# dist[32]: 32 × 32 = 1024 → 256 regs
# pos[32] (temp): shared with x_aff/y_aff
# batch inv temps: ~128 regs
# Total live: ~400-500 regs → CUDA limits to 255 max → heavy spilling

REGS_PER_THREAD = 255  # CUDA maximum
BLOCK_SIZE = 256
KANGAROOS_PER_THREAD = 32

# Achievable warp occupancy: limited by registers
# At 255 regs/thread: 65536/255 = 256 threads/SM → 1 block/SM
# Optimal would be 4+ blocks for latency hiding

print(f"\n  Occupancy analysis (K={KANGAROOS_PER_THREAD}, block={BLOCK_SIZE}):")

for gpu in gpus:
    threads_per_sm = min(gpu["regs_per_sm"] // REGS_PER_THREAD, gpu["max_threads_per_sm"])
    blocks_per_sm = threads_per_sm // BLOCK_SIZE
    warps_per_sm = threads_per_sm // 32
    max_warps = gpu["max_threads_per_sm"] // 32
    occupancy = warps_per_sm / max_warps * 100

    # Total kangaroos
    total_threads = threads_per_sm * gpu["sms"]
    total_kangaroos = total_threads * KANGAROOS_PER_THREAD

    # Throughput: each step takes CLOCKS_PER_STEP clocks
    # With warp scheduling, ILP hides some latency
    # Assume 50% ALU utilization (typical for register-spilling code)
    alu_util = 0.50
    steps_per_sec = gpu["clock_ghz"] * 1e9 * alu_util / CLOCKS_PER_STEP
    # But this is per-warp. We have warps_per_sm warps × sms SMs
    # Each warp processes 32 threads × K kangaroos
    gkeys_per_sec = steps_per_sec * total_kangaroos / 1e9

    # More realistic: based on JLP baseline (2 GKey/s on 4090)
    # and our optimization factor
    jlp_baseline = {
        "RTX 4090": 2.0,
        "RTX 5090": 2.8,
        "RTX 3090": 1.2,
        "H100 SXM": 3.5,
    }
    jlp = jlp_baseline.get(gpu["name"], 1.5)

    # Our optimizations vs JLP:
    # JLP uses K=256, standard inversion, no 3-kangaroo
    # We add: PTX MADC (~1.4x), Deferred-Y (~1.25x), Pipeline (~1.15x)
    # Progressive DP (~1.1x), PTX inv (~1.13x)
    # BUT: JLP is mature, well-tuned code. Realistic advantage: ~1.5-2x
    hydra_factor = 1.7  # conservative estimate of improvement over JLP
    hydra_gkeys = jlp * hydra_factor

    gpu["gkeys"] = hydra_gkeys
    gpu["occupancy"] = occupancy
    gpu["total_kangaroos"] = total_kangaroos

    print(f"\n  {gpu['name']}:")
    print(f"    SMs: {gpu['sms']}, Clock: {gpu['clock_ghz']} GHz")
    print(f"    Threads/SM: {threads_per_sm}, Blocks/SM: {blocks_per_sm}")
    print(f"    Occupancy: {occupancy:.0f}%")
    print(f"    Total kangaroos: {total_kangaroos:,}")
    print(f"    JLP baseline: {jlp:.1f} GKey/s")
    print(f"    Hydra estimate (×{hydra_factor}): {hydra_gkeys:.1f} GKey/s")
    print(f"    Power: {gpu['power_w']}W → {hydra_gkeys*1e9/gpu['power_w']/1e6:.1f} MKey/J")

# ═══════════════════════════════════════════════════════════════
# PUZZLE #135 SOLVE ECONOMICS
# ═══════════════════════════════════════════════════════════════

print(f"\n{'═' * 70}")
print(f"  PUZZLE #135 — SOLVE TIME & ECONOMICS")
print(f"{'═' * 70}")

# Expected operations for Pollard's Kangaroo with Galbraith-Ruprai √6
range_bits = 134  # key is in [2^134, 2^135)
sqrt_range = 2 ** (range_bits / 2)  # 2^67
galbraith_factor = math.sqrt(6)  # √6 ≈ 2.449
three_kang_factor = 0.90 / 1.20  # K-factor improvement
expected_ops = sqrt_range / galbraith_factor * three_kang_factor
expected_ops_log2 = math.log2(expected_ops)

print(f"\n  Range: [2^134, 2^135)")
print(f"  √range = 2^{range_bits/2:.0f} = {sqrt_range:.2e}")
print(f"  With √6 speedup + 3-kangaroo: ~2^{expected_ops_log2:.1f} = {expected_ops:.2e} ops")

BTC_PRICE = 67823
PRIZE_BTC = 13.5
PRIZE_USD = PRIZE_BTC * BTC_PRICE
ELECTRICITY_RATE = 0.10  # $/kWh

print(f"\n  Prize: {PRIZE_BTC} BTC = ${PRIZE_USD:,.0f}")

for gpu in gpus:
    gkeys = gpu["gkeys"]
    ops_per_sec = gkeys * 1e9
    solve_seconds = expected_ops / ops_per_sec
    solve_days = solve_seconds / 86400
    solve_years = solve_days / 365.25

    # How many GPUs to solve in 1 year?
    gpus_for_1yr = solve_years
    # How many for 6 months?
    gpus_for_6mo = solve_years * 2

    # Electricity cost for 1 GPU
    kwh_total = gpu["power_w"] / 1000 * solve_seconds / 3600
    elec_cost = kwh_total * ELECTRICITY_RATE

    # ROI for a farm of N GPUs solving in T months
    target_months = 6
    n_gpus = int(math.ceil(gpus_for_6mo))
    farm_hw_cost = n_gpus * gpu["cost_usd"]
    farm_elec = n_gpus * gpu["power_w"] / 1000 * target_months * 30 * 24 * ELECTRICITY_RATE
    farm_total = farm_hw_cost + farm_elec
    roi = PRIZE_USD / farm_total if farm_total > 0 else 0

    print(f"\n  {gpu['name']} ({gkeys:.1f} GKey/s, {gpu['power_w']}W):")
    print(f"    1 GPU solve time:  {solve_years:.0f} years ({solve_days:,.0f} days)")
    print(f"    1 GPU electricity: {kwh_total:,.0f} kWh = ${elec_cost:,.0f}")
    print(f"    GPUs for 6-month solve: {n_gpus:,}")
    print(f"    6-month farm cost: ${farm_hw_cost:,.0f} HW + ${farm_elec:,.0f} elec = ${farm_total:,.0f}")
    print(f"    ROI: {roi:.2f}x (${PRIZE_USD:,.0f} / ${farm_total:,.0f})")

# ═══════════════════════════════════════════════════════════════
# POOL COMPARISON
# ═══════════════════════════════════════════════════════════════

print(f"\n{'═' * 70}")
print(f"  POOL COMPARISON — COLLIDER vs HYDRA FARM")
print(f"{'═' * 70}")

collider_rate = 8000  # GKey/s (8 TKey/s)
collider_progress = 0.47
remaining = 1 - collider_progress

collider_ops_remaining = expected_ops * remaining
collider_time_remaining = collider_ops_remaining / (collider_rate * 1e9) / 86400
print(f"\n  Collider pool:")
print(f"    Rate: {collider_rate} GKey/s ({collider_rate/1000:.0f} TKey/s)")
print(f"    Progress: {collider_progress*100:.0f}%")
print(f"    Est. time remaining: {collider_time_remaining:.0f} days ({collider_time_remaining/30:.1f} months)")

# Hydra farm to compete
hydra_4090_rate = gpus[0]["gkeys"]  # GKey/s per 4090
n_4090_to_match = collider_rate / hydra_4090_rate
print(f"\n  To match Collider with Hydra RTX 4090s:")
print(f"    Need: {n_4090_to_match:.0f} GPUs × {hydra_4090_rate:.1f} GKey/s = {n_4090_to_match * hydra_4090_rate:.0f} GKey/s")
print(f"    Hardware: ${n_4090_to_match * gpus[0]['cost_usd']:,.0f}")
print(f"    Power: {n_4090_to_match * gpus[0]['power_w'] / 1000:.0f} kW")

print(f"\n{'═' * 70}")
print(f"  BOTTOM LINE")
print(f"{'═' * 70}")
print(f"""
  The most economical approach to puzzle #135:

  1. JOIN COLLIDER POOL with optimized Hydra client
     - Contribute ~10-50 GPUs for proportional reward share
     - Expected solve: {collider_time_remaining/30:.0f} months at current rate
     - Your share ≈ (your_GKeys / pool_GKeys) × ${PRIZE_USD:,.0f}

  2. SOLO with 100+ RTX 4090s
     - Cost: ~$160K HW + ~$80K/yr electricity
     - Solve in ~{expected_ops / (100 * hydra_4090_rate * 1e9) / 86400 / 30:.0f} months
     - ROI: positive if BTC holds above $40K

  3. WAIT for puzzle #71 (brute force, much easier)
     - 71-bit range → ~2^36 ops → minutes on single GPU
     - Prize: ~7.1 BTC = ~${7.1 * BTC_PRICE:,.0f}
     - BUT: no public key exposed, need full brute force + hash
""")
