#!/usr/bin/env python3
"""
Deep analysis of the φ-sequence signal and other unconventional patterns.
"""

import math
import random

PHI = (1 + math.sqrt(5)) / 2

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

def get_frac(n, k):
    lo = 1 << (n - 1)
    return (k - lo) / lo

def corr(a, b):
    n = min(len(a), len(b))
    if n < 3: return 0
    a, b = a[:n], b[:n]
    ma = sum(a)/n
    mb = sum(b)/n
    cov = sum((a[i]-ma)*(b[i]-mb) for i in range(n))/n
    va = sum((x-ma)**2 for x in a)/n
    vb = sum((x-mb)**2 for x in b)/n
    if va > 0 and vb > 0:
        return cov / math.sqrt(va*vb)
    return 0

def main():
    consec = [(n, k) for n, k in SOLVED if n <= 70]
    fracs = [get_frac(n, k) for n, k in consec]

    print("=" * 78)
    print("  DEEP PATTERN ANALYSIS")
    print("=" * 78)

    # ─── 1. VALIDATE φ-SIGNAL: Is r=0.34 significant? ───
    print("\n[1] φ-SEQUENCE SIGNAL VALIDATION (Monte Carlo)")
    print("-" * 78)
    print("  Testing if r=0.34 is within noise for random uniform keys...")

    # The flagged signal was: f[n] vs ((n+12) * φ * 2) mod 1
    test_fracs = [((n + 12) * PHI * 2) % 1.0 for n, _ in consec]
    observed_corr = corr(fracs, test_fracs)
    print(f"  Observed correlation (offset=12, scale=2): {observed_corr:.4f}")

    # Monte Carlo: generate random uniform keys, find best correlation
    # across same parameter grid, count how often best > observed
    N_MC = 10000
    exceed_count = 0
    for trial in range(N_MC):
        # Random uniform fractional positions
        rand_fracs = [random.random() for _ in consec]
        # Search same parameter grid as original test
        best = 0
        for offset in range(0, 20):
            for scale_num in range(1, 10):
                scale = scale_num / 3.0
                tf = [((n + offset) * PHI * scale) % 1.0 for n, _ in consec]
                c = abs(corr(rand_fracs, tf))
                if c > best:
                    best = c
        if best >= abs(observed_corr):
            exceed_count += 1

    p_value = exceed_count / N_MC
    print(f"  Monte Carlo p-value: {p_value:.4f} ({exceed_count}/{N_MC} trials exceeded)")
    if p_value > 0.05:
        print(f"  >>> φ-signal is NOT significant. Noise floor artifact. <<<")
    else:
        print(f"  >>> φ-signal IS significant! p = {p_value:.4f} <<<")

    # ─── 2. DEEPER RATIO ANALYSIS ───
    print(f"\n[2] RATIO PATTERNS BETWEEN NON-CONSECUTIVE KEYS")
    print("-" * 78)

    # Check ratios between keys separated by 5 (the puzzle spacing for exposed keys)
    print("  Ratios k[n+5] / k[n] (every-5th puzzle pattern):")
    for i in range(len(SOLVED) - 1):
        for j in range(i + 1, len(SOLVED)):
            n1, k1 = SOLVED[i]
            n2, k2 = SOLVED[j]
            if n2 - n1 == 5 and n1 <= 70:
                r = k2 / k1
                expected = 2**5  # = 32
                frac_r = r / expected
                print(f"    #{n1:3d}→#{n2:3d}: ratio = {r:.2f}  /32 = {frac_r:.4f}")

    # ─── 3. KEY AS FUNCTION OF PUZZLE NUMBER ───
    print(f"\n[3] KEY = f(puzzle_number) REGRESSION")
    print("-" * 78)
    # log2(key) should ≈ puzzle_number if uniform in range
    # Check if there's a better fit than pure linear
    print("  n     log2(k)     n-log2(k)  residual")
    residuals = []
    for n, k in SOLVED:
        if k > 0:
            l2k = math.log2(k)
            resid = n - l2k  # should be ~0.5 on average (since key in [2^(n-1), 2^n))
            residuals.append((n, resid))
            if n <= 40 or n % 10 == 0:
                print(f"  {n:3d}   {l2k:8.4f}   {resid:+.4f}")

    r_vals = [r for _, r in residuals]
    mean_r = sum(r_vals) / len(r_vals)
    std_r = math.sqrt(sum((r - mean_r)**2 for r in r_vals) / len(r_vals))
    print(f"\n  Mean residual: {mean_r:.4f} (expect ~0.585 for uniform)")
    print(f"  Std residual:  {std_r:.4f} (expect ~0.416 for uniform)")

    # ─── 4. SPECTRAL ANALYSIS of fractional positions ───
    print(f"\n[4] SPECTRAL ANALYSIS (DFT of fractional positions)")
    print("-" * 78)
    # Only use consecutive puzzles 1-70
    consec_fracs_only = []
    for n in range(1, 71):
        found = False
        for nn, k in SOLVED:
            if nn == n:
                consec_fracs_only.append(get_frac(n, k))
                found = True
                break
        if not found:
            consec_fracs_only.append(0.5)  # placeholder for missing

    N = len(consec_fracs_only)
    # Compute DFT magnitudes
    print("  Freq  Magnitude  (looking for dominant frequencies)")
    magnitudes = []
    for freq in range(1, N // 2):
        re = sum(consec_fracs_only[t] * math.cos(2 * math.pi * freq * t / N) for t in range(N))
        im = sum(consec_fracs_only[t] * math.sin(2 * math.pi * freq * t / N) for t in range(N))
        mag = math.sqrt(re**2 + im**2) / N
        magnitudes.append((freq, mag))

    magnitudes.sort(key=lambda x: -x[1])
    for freq, mag in magnitudes[:15]:
        period = N / freq
        bar = "#" * int(mag * 100)
        print(f"  f={freq:2d} (period={period:5.1f}):  {mag:.4f}  {bar}")

    # ─── 5. LOOK FOR RECURSIVE/FIBONACCI STRUCTURE ───
    print(f"\n[5] FIBONACCI/RECURSIVE STRUCTURE")
    print("-" * 78)
    # Check: k[n] ≈ k[n-1] + k[n-2]  (Fibonacci-like)?
    print("  k[n] vs k[n-1] + k[n-2]:")
    for i in range(2, len(SOLVED)):
        if SOLVED[i][0] == SOLVED[i-1][0] + 1 == SOLVED[i-2][0] + 2:
            n, k = SOLVED[i]
            k1 = SOLVED[i-1][1]
            k2 = SOLVED[i-2][1]
            fib_pred = k1 + k2
            ratio = k / fib_pred if fib_pred > 0 else 0
            if n <= 40 or n % 10 == 0:
                print(f"    #{n:3d}: k={k}, k[-1]+k[-2]={fib_pred}, ratio={ratio:.4f}")

    # ─── 6. KNOWN PRNG TESTS ───
    print(f"\n[6] PRNG REVERSE ENGINEERING")
    print("-" * 78)

    # Test: are fractional positions consistent with rejection-sampled uniform?
    # (i.e., generate random 256-bit number, keep if in range)
    # This is the most likely generation method
    print("  Hypothesis: keys = random 256-bit values masked to n-bit range")
    print("  This would produce uniform fractional positions (already tested)")

    # Test: are keys related by XOR with a constant?
    print("\n  XOR pattern test (k[n] XOR k[n+1]):")
    xor_results = []
    for i in range(len(SOLVED) - 1):
        n1, k1 = SOLVED[i]
        n2, k2 = SOLVED[i + 1]
        if n2 == n1 + 1:
            x = k1 ^ k2
            xor_results.append(x)
            if n1 <= 20:
                print(f"    #{n1:3d} XOR #{n2:3d} = {x:#x}")

    # ─── 7. DISTANCE FROM SPECIAL VALUES ───
    print(f"\n[7] DISTANCE FROM SPECIAL MATHEMATICAL VALUES")
    print("-" * 78)

    for n, k in SOLVED:
        lo = 1 << (n - 1)
        hi = 1 << n
        mid = (lo + hi) // 2

        # Distance from phi * range
        phi_point = int(lo + (hi - lo) * (PHI - 1))  # φ-1 ≈ 0.618
        phi_dist = abs(k - phi_point) / (hi - lo)

        # Distance from 1/phi * range
        inv_phi_point = int(lo + (hi - lo) / PHI)  # 1/φ ≈ 0.618
        inv_phi_dist = abs(k - inv_phi_point) / (hi - lo)

        # Is key close to φ-point?
        if phi_dist < 0.02 and n >= 10:
            print(f"  #{n:3d}: KEY IS WITHIN 2% OF φ-POINT! dist={phi_dist:.4f}")

    # Check how many keys are closer to φ-point than expected
    phi_close_count = 0
    for n, k in SOLVED:
        if n < 10: continue
        lo = 1 << (n - 1)
        hi = 1 << n
        phi_point = int(lo + (hi - lo) * (PHI - 1))
        phi_dist = abs(k - phi_point) / (hi - lo)
        if phi_dist < 0.1:
            phi_close_count += 1

    n_eligible = sum(1 for n, _ in SOLVED if n >= 10)
    expected_close = n_eligible * 0.2  # 20% of range is within 0.1
    print(f"\n  Keys within 10% of φ-point: {phi_close_count}/{n_eligible}")
    print(f"  Expected by chance: {expected_close:.1f}")
    if phi_close_count > expected_close * 1.5:
        print(f"  >>> SIGNIFICANT φ-bias! <<<")

    # ─── 8. NIBBLE (4-bit) FREQUENCY ANALYSIS ───
    print(f"\n[8] NIBBLE FREQUENCY (looking for non-random hex digit distribution)")
    print("-" * 78)
    from collections import Counter
    all_nibbles = Counter()
    for n, k in SOLVED:
        if n >= 20:
            hex_str = format(k, 'x')
            for h in hex_str[1:]:  # skip leading nibble (biased by range)
                all_nibbles[h] += 1

    total_nibbles = sum(all_nibbles.values())
    expected_per = total_nibbles / 16
    print(f"  Total nibbles: {total_nibbles}, expected per hex digit: {expected_per:.1f}")
    chi2 = 0
    for digit in "0123456789abcdef":
        obs = all_nibbles.get(digit, 0)
        chi2 += (obs - expected_per)**2 / expected_per
        bar = "#" * int(obs / expected_per * 20)
        print(f"    '{digit}': {obs:4d}  ({obs/total_nibbles*100:.1f}%)  {bar}")
    print(f"  χ² = {chi2:.2f}  (critical at α=0.05, df=15: 25.0)")
    if chi2 > 25:
        print(f"  >>> NON-UNIFORM hex digit distribution! <<<")

    # ─── 9. PREDICTION SECTION ───
    print(f"\n{'=' * 78}")
    print(f"  PREDICTION FOR PUZZLE #135")
    print(f"{'=' * 78}")

    lo_135 = 1 << 134
    hi_135 = 1 << 135

    # Method 1: If fractional position follows Weyl sequence
    f_pred_weyl = ((135 + 12) * PHI * 2) % 1.0
    k_pred_weyl = int(lo_135 + f_pred_weyl * lo_135)
    print(f"\n  Weyl/φ prediction (f = {f_pred_weyl:.6f}):")
    print(f"    {k_pred_weyl:#0{34}x}")

    # Method 2: Mean fractional position
    all_fracs = [get_frac(n, k) for n, k in SOLVED]
    mean_f = sum(all_fracs) / len(all_fracs)
    k_pred_mean = int(lo_135 + mean_f * lo_135)
    print(f"\n  Mean position prediction (f = {mean_f:.6f}):")
    print(f"    {k_pred_mean:#0{34}x}")

    # Method 3: Log-linear regression
    # log2(k) ≈ a*n + b
    ns = [n for n, k in SOLVED if k > 0 and n >= 10]
    l2ks = [math.log2(k) for n, k in SOLVED if k > 0 and n >= 10]
    n_pts = len(ns)
    mean_n = sum(ns) / n_pts
    mean_l = sum(l2ks) / n_pts
    cov_nl = sum((ns[i] - mean_n) * (l2ks[i] - mean_l) for i in range(n_pts)) / n_pts
    var_n = sum((n - mean_n)**2 for n in ns) / n_pts
    a = cov_nl / var_n
    b = mean_l - a * mean_n
    l2k_pred = a * 135 + b
    k_pred_reg = int(2**l2k_pred)
    print(f"\n  Log-linear regression (log2(k) = {a:.6f}*n + {b:.4f}):")
    print(f"    Predicted log2(k) = {l2k_pred:.4f}")
    print(f"    {k_pred_reg:#0{34}x}")

    print(f"\n  Range: [{lo_135:#0{34}x}, {hi_135:#0{34}x})")
    print(f"  Range size: 2^134 = {lo_135:.3e}")

    # ─── 10. ACTIONABLE CONCLUSION ───
    print(f"\n{'=' * 78}")
    print(f"  ACTIONABLE CONCLUSION")
    print(f"{'=' * 78}")
    print(f"""
  The φ-sequence correlation (r=0.34) at offset=12, scale=2.0 needs Monte Carlo
  validation. If p-value > 0.05, it's noise from searching 180 parameter combos.

  Monte Carlo result: p = {p_value:.4f}
  """)

    if p_value > 0.05:
        print(f"""  CONCLUSION: Keys appear genuinely random. No exploitable PRNG pattern found.

  The puzzle creator likely used a CSPRNG (e.g., /dev/urandom) or hardware RNG.
  With 82 solved keys showing no detectable structure, the probability of a
  hidden deterministic pattern is very low.

  IMPLICATION FOR SPEED:
  - Cannot shortcut via key prediction
  - Must rely on algorithmic/hardware optimization
  - Multi-target (√T) remains the best mathematical speedup
  - GPU throughput optimization remains the best engineering speedup
  """)
    else:
        print(f"""  POSSIBLE STRUCTURE DETECTED! p = {p_value:.4f}

  The φ-based Weyl sequence shows statistically significant correlation.
  This could narrow the search space for puzzle #135.

  Predicted fractional position: {f_pred_weyl:.6f}
  This narrows the search to ~10% of the range if the pattern holds,
  giving a ~10x speedup.
  """)

if __name__ == "__main__":
    random.seed(42)
    main()
