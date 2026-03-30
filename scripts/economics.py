#!/usr/bin/env python3
"""
Bitcoin Puzzle Economic Calculator

Calculates profitability for solving Bitcoin puzzles using various
hardware configurations. Implements the breakthrough analysis from
puzzle_binary's 1000x architecture applied to ECDLP.

Usage:
    python3 economics.py [--btc-price PRICE] [--puzzle NUM]
"""

import math
import sys

# ═══════════════════════════════════════════════════════════════
# PUZZLE DATABASE
# ═══════════════════════════════════════════════════════════════

# Unsolved puzzles with known public keys (exposed via 2019 transactions)
# Prizes were increased ~10x in March 2023 top-up transaction.
# Format: (prize_btc, has_pubkey)
# Prize = puzzle_number / 10 BTC (e.g., #135 = 13.5 BTC)
PUZZLES = {
    135: (13.50, True),
    140: (14.00, True),
    145: (14.50, True),
    150: (15.00, True),
    155: (15.50, True),
    160: (16.00, True),
}

# ═══════════════════════════════════════════════════════════════
# HARDWARE PROFILES
# ═══════════════════════════════════════════════════════════════

HARDWARE = {
    'RTX 4090': {
        'ops_per_sec': 8e9,        # EC group ops/s (legacy kernel)
        'watts': 350,
        'buy_price': 1599,
        'cloud_per_hr': None,      # Not available as cloud
    },
    'RTX 4090 (1000x kernel)': {
        'ops_per_sec': 18e9,       # With PTX MADC + pipelining
        'watts': 350,
        'buy_price': 1599,
        'cloud_per_hr': None,
    },
    'H100 SXM': {
        'ops_per_sec': 12e9,
        'watts': 700,
        'buy_price': 25000,
        'cloud_per_hr': 2.00,
    },
    'H100 (1000x kernel)': {
        'ops_per_sec': 27e9,
        'watts': 700,
        'buy_price': 25000,
        'cloud_per_hr': 2.00,
    },
    'H100 Spot': {
        'ops_per_sec': 27e9,
        'watts': 700,
        'buy_price': 25000,
        'cloud_per_hr': 0.80,
    },
    'EC-ASIC (projected)': {
        'ops_per_sec': 3.52e9,     # Per chip, 200 cores
        'watts': 8,
        'buy_price': 100,
        'cloud_per_hr': None,
    },
}

# ═══════════════════════════════════════════════════════════════
# CORE CALCULATIONS
# ═══════════════════════════════════════════════════════════════

def kangaroo_ops(puzzle_num, k_factor=1.20, galbraith_ruprai=True):
    """Expected EC group operations for Pollard's Kangaroo."""
    range_bits = puzzle_num  # Key in [2^(n-1), 2^n)
    sqrt_range = 2 ** (range_bits / 2.0)
    ops = k_factor * sqrt_range
    if galbraith_ruprai:
        ops /= math.sqrt(6) / math.sqrt(2)  # sqrt(6)/sqrt(2) = sqrt(3) effective
    return ops

def solve_economics(puzzle_num, hw_name, num_units, btc_price,
                    elec_rate=0.10, multi_target_t=1):
    """Calculate full economics for solving a puzzle."""
    hw = HARDWARE[hw_name]
    prize_btc = PUZZLES.get(puzzle_num, (puzzle_num * 0.001, True))[0]
    prize_usd = prize_btc * btc_price

    ops = kangaroo_ops(puzzle_num) / math.sqrt(multi_target_t)
    total_rate = hw['ops_per_sec'] * num_units
    time_s = ops / total_rate
    time_days = time_s / 86400
    time_years = time_days / 365.25

    # Cost calculation
    if hw['cloud_per_hr'] is not None:
        # Cloud: pay per hour
        total_gpu_hrs = (ops / hw['ops_per_sec']) / 3600
        cost = total_gpu_hrs * hw['cloud_per_hr']
        hw_cost = 0
        elec_cost = 0  # Included in cloud price
    else:
        # Own hardware
        hw_cost = num_units * hw['buy_price']
        elec_kwh = (num_units * hw['watts'] / 1000) * (time_s / 3600)
        elec_cost = elec_kwh * elec_rate
        cost = hw_cost + elec_cost

    roi = prize_usd / cost if cost > 0 else float('inf')

    return {
        'puzzle': puzzle_num,
        'hardware': hw_name,
        'units': num_units,
        'ops': ops,
        'time_days': time_days,
        'time_years': time_years,
        'hw_cost': hw_cost,
        'elec_cost': elec_cost,
        'total_cost': cost,
        'prize_usd': prize_usd,
        'roi': roi,
        'profitable': roi > 1.0,
    }

def find_breakeven_btc(puzzle_num, hw_name, num_units, elec_rate=0.10):
    """Find BTC price that makes solving break-even."""
    prize_btc = PUZZLES.get(puzzle_num, (puzzle_num * 0.001, True))[0]
    result = solve_economics(puzzle_num, hw_name, num_units, 1.0, elec_rate)
    # At what BTC price does prize = cost?
    # prize_btc * btc_price = total_cost (which doesn't depend on BTC price for own hw)
    if result['total_cost'] > 0:
        return result['total_cost'] / prize_btc
    return 0

def main():
    btc_price = 67823  # Approximate BTC price as of March 2026
    target_puzzle = 135

    # Parse args
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == '--btc-price' and i + 1 < len(args):
            btc_price = float(args[i + 1])
            i += 2
        elif args[i] == '--puzzle' and i + 1 < len(args):
            target_puzzle = int(args[i + 1])
            i += 2
        else:
            i += 1

    print(f"{'='*72}")
    print(f"  BITCOIN PUZZLE ECONOMIC CALCULATOR")
    print(f"  BTC Price: ${btc_price:,.0f}  |  Target: Puzzle #{target_puzzle}")
    print(f"{'='*72}\n")

    prize_btc = PUZZLES.get(target_puzzle, (target_puzzle * 0.001, True))[0]
    ops = kangaroo_ops(target_puzzle)
    print(f"  Prize: {prize_btc} BTC = ${prize_btc * btc_price:,.0f}")
    print(f"  Expected ops: {ops:.3e}")
    print(f"  Galbraith-Ruprai enabled (sqrt(6) equivalence)\n")

    # Test configurations
    configs = [
        ('RTX 4090', 1),
        ('RTX 4090', 16),
        ('RTX 4090 (1000x kernel)', 16),
        ('RTX 4090 (1000x kernel)', 100),
        ('H100 (1000x kernel)', 100),
        ('H100 Spot', 100),
        ('EC-ASIC (projected)', 100),
        ('EC-ASIC (projected)', 1000),
        ('EC-ASIC (projected)', 10000),
    ]

    print(f"  {'Config':<45s} {'Time':>10s} {'Cost':>12s} {'ROI':>8s}")
    print(f"  {'-'*45} {'-'*10} {'-'*12} {'-'*8}")

    for hw_name, units in configs:
        r = solve_economics(target_puzzle, hw_name, units, btc_price)
        if r['time_years'] > 1:
            time_str = f"{r['time_years']:.1f} yr"
        else:
            time_str = f"{r['time_days']:.0f} d"
        status = " <--" if r['profitable'] else ""
        print(f"  {units:>5}x {hw_name:<38s} {time_str:>10s} ${r['total_cost']:>10,.0f} {r['roi']:>6.3f}x{status}")

    print(f"\n{'='*72}")
    print(f"  BREAK-EVEN BTC PRICES")
    print(f"{'='*72}\n")

    breakeven_configs = [
        ('RTX 4090 (1000x kernel)', 100),
        ('H100 Spot', 100),
        ('EC-ASIC (projected)', 100),
        ('EC-ASIC (projected)', 500),
    ]

    for hw_name, units in breakeven_configs:
        btc_be = find_breakeven_btc(target_puzzle, hw_name, units)
        print(f"  {units:>5}x {hw_name:<38s} BTC must reach ${btc_be:>12,.0f}")

    # Multi-puzzle portfolio
    print(f"\n{'='*72}")
    print(f"  MULTI-PUZZLE PORTFOLIO")
    print(f"{'='*72}\n")

    total_prize = sum(p * btc_price for p, _ in PUZZLES.values())
    easiest = min(PUZZLES.keys())
    hardest = max(PUZZLES.keys())
    T = len(PUZZLES)

    print(f"  Puzzles: {sorted(PUZZLES.keys())}")
    print(f"  Total prize: {sum(p for p, _ in PUZZLES.values()):.3f} BTC = ${total_prize:,.0f}")
    print(f"  Multi-target speedup: sqrt({T}) = {math.sqrt(T):.2f}x")
    print(f"  Easiest: #{easiest} ({kangaroo_ops(easiest):.2e} ops)")
    print(f"  Hardest: #{hardest} ({kangaroo_ops(hardest):.2e} ops)\n")

    # Portfolio economics
    r_portfolio = solve_economics(
        easiest, 'EC-ASIC (projected)', 1000, btc_price,
        multi_target_t=T
    )
    print(f"  1000 EC-ASICs targeting easiest (#{easiest}) with sqrt({T}) bonus:")
    print(f"    Time to first solve: {r_portfolio['time_days']:.0f} days")
    print(f"    Cost: ${r_portfolio['total_cost']:,.0f}")
    print(f"    Min prize (one solve): ${r_portfolio['prize_usd']:,.0f}")
    print(f"    ROI (one solve): {r_portfolio['roi']:.2f}x")
    print(f"    ROI (all solves): {total_prize / r_portfolio['total_cost']:.2f}x")

    print(f"\n{'='*72}")
    print(f"  VERDICT")
    print(f"{'='*72}\n")

    r_best_gpu = solve_economics(target_puzzle, 'RTX 4090 (1000x kernel)', 100, btc_price)
    r_best_asic = solve_economics(target_puzzle, 'EC-ASIC (projected)', 1000, btc_price)

    print(f"  Best GPU:  ${r_best_gpu['total_cost']:>12,.0f} cost for ${r_best_gpu['prize_usd']:>8,.0f} prize")
    print(f"  Best ASIC: ${r_best_asic['total_cost']:>12,.0f} cost for ${r_best_asic['prize_usd']:>8,.0f} prize")
    print(f"  Portfolio: ${r_portfolio['total_cost']:>12,.0f} cost for ${total_prize:>8,.0f} prize pool")
    print()

    if r_portfolio['roi'] > 1:
        print(f"  ** PORTFOLIO APPROACH IS PROFITABLE (ROI {r_portfolio['roi']:.2f}x) **")
    elif r_best_asic['roi'] > 0.5:
        print(f"  ** ASIC APPROACH NEARLY PROFITABLE (ROI {r_best_asic['roi']:.2f}x) **")
        print(f"  ** Profitable at BTC = ${find_breakeven_btc(target_puzzle, 'EC-ASIC (projected)', 1000):,.0f} **")
    else:
        print(f"  ** NOT ECONOMICAL AT CURRENT PRICES **")
        print(f"  ** Need BTC > ${find_breakeven_btc(target_puzzle, 'EC-ASIC (projected)', 1000):,.0f} OR ECDLP breakthrough **")

if __name__ == '__main__':
    main()
