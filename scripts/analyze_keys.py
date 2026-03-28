#!/usr/bin/env python3
"""
Analyze solved Bitcoin puzzle private keys for patterns.
If the creator used a PRNG or deterministic process, patterns here
could predict unsolved keys — bypassing kangaroo entirely.
"""

import math
import sys
from collections import Counter

# Solved puzzle keys: (puzzle_number, private_key_hex)
SOLVED = [
    (1, "1"), (2, "3"), (3, "7"), (4, "8"),
    (5, "15"), (6, "31"), (7, "4c"), (8, "e0"),
    (9, "1d3"), (10, "202"), (11, "483"), (12, "a7b"),
    (13, "1460"), (14, "2930"), (15, "68f3"), (16, "c936"),
    (17, "1764f"), (18, "3080d"), (19, "5749f"), (20, "d2c55"),
    (21, "1ba534"), (22, "2de40f"), (23, "556e52"), (24, "dc2a04"),
    (25, "1fa5ee5"), (26, "340326e"), (27, "6ac3875"), (28, "d916ce8"),
    (29, "17e2551e"), (30, "3d94cd64"), (31, "7d4fe747"), (32, "b862a62e"),
    (33, "1a96ca8d8"), (34, "34a65911d"), (35, "4aed21170"), (36, "9de820a7c"),
    (37, "1757756a93"), (38, "22382facd0"), (39, "4b5f8303e9"), (40, "e9ae4933d6"),
    (41, "153869acc5b"), (42, "2a221c58d8f"),  # corrected from table
    (43, "6bd3b27c591"), (44, "e02b35a358f"),
    (45, "122fca143c05"), (46, "2ec18388d544"), (47, "6cd610b53cba"),
    (48, "ade6d7ce3b9b"), (49, "174176b015f4d"), (50, "22bd43c2e9354"),
    (51, "75070a1a009d4"), (52, "efae164cb9e3c"), (53, "180788e47e326c"),
    (54, "236fb6d5ad1f43"), (55, "6abe1f9b67e114"), (56, "9d18b63ac4ffdf"),
    (57, "1eb25c90795d61c"), (58, "2c675b852189a21"), (59, "7496cbb87cab44f"),
    (60, "fc07a1825367bbe"), (61, "13c96a3742f64906"), (62, "363d541eb611abee"),
    (63, "7cce5efdaccf6808"), (64, "f7051f27b09112d4"),
    (65, "1a838b13505b26867"), (66, "2832ed74f2b5e35ee"),
    (67, "730fc235c1942c1ae"), (68, "bebb3940cd0fc1491"),
    (69, "101d83275fb2bc7e0c"), (70, "349b84b6431a6c4ef1"),
    (75, "4c5ce114686a1336e07"),
    (80, "ea1a5c66dcc11b5ad180"),
    (85, "11720c4f018d51b8cebba8"),
    (90, "2ce00bb2136a445c71e85bf"),
    (95, "527a792b183c7f64a0e8b1f4"),
    (100, "af55fc59c335c8ec67ed24826"),
    (105, "16f14fc2054cd87ee6396b33df3"),
    (110, "35c0d7234df7deb0f20cf7062444"),
    (115, "60f4d11574f5deee49961d9609ac6"),
    (120, "b10f22572c497a836ea187f2e1fc23"),
    (125, "1c533b6bb7f0804e09960225e44877ac"),
    (130, "33e7665705359f04f28b88cf897c603c9"),
]

PHI = (1 + math.sqrt(5)) / 2  # Golden ratio

def analyze():
    keys = [(n, int(h, 16)) for n, h in SOLVED]

    print("=" * 78)
    print("  BITCOIN PUZZLE KEY ANALYSIS — Searching for Generator Patterns")
    print("=" * 78)

    # ─── 1. Fractional Position Within Range ───
    print("\n[1] FRACTIONAL POSITION (key position within [2^(n-1), 2^n))")
    print("-" * 78)
    fracs = []
    for n, k in keys:
        lo = 1 << (n - 1)
        hi = 1 << n
        f = (k - lo) / lo  # fraction in [0, 1)
        fracs.append(f)
        if n <= 70 or n % 5 == 0:
            print(f"  #{n:3d}: {f:.6f}  {'*' * int(f * 40)}")

    mean_f = sum(fracs) / len(fracs)
    var_f = sum((f - mean_f)**2 for f in fracs) / len(fracs)
    print(f"\n  Mean:     {mean_f:.4f}  (expected 0.5000 for uniform)")
    print(f"  Std dev:  {math.sqrt(var_f):.4f}  (expected 0.2887 for uniform)")
    print(f"  Min:      {min(fracs):.6f}")
    print(f"  Max:      {max(fracs):.6f}")

    # Kolmogorov-Smirnov test (manual)
    sorted_fracs = sorted(fracs)
    n_f = len(sorted_fracs)
    ks_stat = max(max(abs((i+1)/n_f - sorted_fracs[i]), abs(sorted_fracs[i] - i/n_f))
                  for i in range(n_f))
    # Critical value at α=0.05: 1.36/√n
    ks_crit = 1.36 / math.sqrt(n_f)
    print(f"\n  KS statistic: {ks_stat:.4f}  (critical α=0.05: {ks_crit:.4f})")
    if ks_stat > ks_crit:
        print(f"  >>> REJECT uniformity at 5% level! Possible structure! <<<")
    else:
        print(f"  Cannot reject uniformity (looks random)")

    # ─── 2. Consecutive Key Ratios ───
    print(f"\n[2] CONSECUTIVE KEY RATIOS (k[n+1] / k[n])")
    print("-" * 78)
    ratios = []
    phi_hits = 0
    for i in range(len(keys) - 1):
        n1, k1 = keys[i]
        n2, k2 = keys[i + 1]
        if n2 == n1 + 1:  # consecutive puzzles only
            r = k2 / k1
            ratios.append((n1, n2, r))
            phi_dev = abs(r - PHI) / PHI
            marker = ""
            if phi_dev < 0.05:
                marker = f" *** φ ± {phi_dev*100:.1f}% ***"
                phi_hits += 1
            elif phi_dev < 0.10:
                marker = f" *  φ ± {phi_dev*100:.1f}%"
            if n1 <= 70:
                print(f"  #{n1:3d}→#{n2:3d}: ratio = {r:.6f}{marker}")

    if ratios:
        r_vals = [r for _, _, r in ratios]
        print(f"\n  Mean ratio:   {sum(r_vals)/len(r_vals):.4f}  (expected ~2.0 for uniform)")
        print(f"  Median ratio: {sorted(r_vals)[len(r_vals)//2]:.4f}")
        print(f"  φ-close hits: {phi_hits}/{len(ratios)} within 5% of φ={PHI:.6f}")

    # ─── 3. Golden Ratio Test: k[n] ≈ c · φ^n ? ───
    print(f"\n[3] GOLDEN RATIO HYPOTHESIS: k[n] = c · φ^n")
    print("-" * 78)
    # If k = c·φ^n, then log(k) = log(c) + n·log(φ)
    # Compute implied c for each key
    log_phi = math.log(PHI)
    for n, k in keys:
        if k > 0:
            c_implied = k / (PHI ** n)
            log_c = math.log(k) - n * log_phi
            if n <= 30 or n % 5 == 0:
                print(f"  #{n:3d}: c_implied = {c_implied:.6e}  log(c) = {log_c:.4f}")

    # Check if c_implied is constant (would indicate φ^n pattern)
    c_vals = [k / (PHI ** n) for n, k in keys if k > 0]
    c_mean = sum(c_vals) / len(c_vals)
    c_cv = math.sqrt(sum((c - c_mean)**2 for c in c_vals) / len(c_vals)) / c_mean
    print(f"\n  Coefficient of variation: {c_cv:.4f}")
    print(f"  (CV ≈ 0 would confirm φ^n pattern, CV >> 1 rejects it)")

    # ─── 4. k[n] ≈ c · 2^n (trivially true, but check constant) ───
    print(f"\n[4] POWER-OF-2 FIT: k[n] = c · 2^n")
    print("-" * 78)
    c2_vals = [k / (2**n) for n, k in keys]
    c2_mean = sum(c2_vals) / len(c2_vals)
    c2_cv = math.sqrt(sum((c - c2_mean)**2 for c in c2_vals) / len(c2_vals)) / c2_mean
    print(f"  Mean c:  {c2_mean:.6f}")
    print(f"  CV:      {c2_cv:.4f}  (CV ≈ 0.577 expected for uniform)")

    # ─── 5. Bit Pattern Analysis ───
    print(f"\n[5] BIT PATTERN ANALYSIS")
    print("-" * 78)

    # For each puzzle, look at the bit density (Hamming weight / n)
    print(f"  Puzzle  Bits  HW  Density  Expected")
    hw_devs = []
    for n, k in keys:
        hw = bin(k).count('1')
        density = hw / n
        expected = 0.5
        dev = (hw - n*0.5) / math.sqrt(n * 0.25)  # z-score
        hw_devs.append(dev)
        if n <= 40 or n % 5 == 0:
            bar = "+" if dev > 1.5 else ("-" if dev < -1.5 else " ")
            print(f"  #{n:3d}    {n:3d}   {hw:3d}   {density:.3f}    {expected:.3f}  z={dev:+.2f} {bar}")

    mean_z = sum(hw_devs) / len(hw_devs)
    print(f"\n  Mean z-score: {mean_z:+.4f}  (expect 0.0 for random)")
    print(f"  (Positive = more 1-bits than expected)")

    # ─── 6. Modular Residues ───
    print(f"\n[6] MODULAR RESIDUE ANALYSIS")
    print("-" * 78)
    for mod in [3, 5, 7, 11, 13, 17, 137]:
        residues = [k % mod for _, k in keys]
        counts = Counter(residues)
        expected = len(keys) / mod
        chi2 = sum((counts.get(r, 0) - expected)**2 / expected for r in range(mod))
        # Chi-squared critical value at α=0.05, df=mod-1
        # Approximate: 2*(mod-1) for rough check
        sig = "**" if chi2 > 2*(mod-1) else ""
        dist_str = " ".join(f"{counts.get(r,0):2d}" for r in range(min(mod, 17)))
        print(f"  mod {mod:3d}: χ²={chi2:6.2f}  dist=[{dist_str}] {sig}")

    # ─── 7. Consecutive Differences ───
    print(f"\n[7] CONSECUTIVE KEY DIFFERENCES (k[n+1] - 2*k[n])")
    print("-" * 78)
    print("  (If keys are related, k[n+1] - 2*k[n] should show structure)")
    for i in range(len(keys) - 1):
        n1, k1 = keys[i]
        n2, k2 = keys[i + 1]
        if n2 == n1 + 1:
            diff = k2 - 2 * k1
            # Normalize by range size
            norm_diff = diff / (1 << n1)
            if n1 <= 35 or n1 % 5 == 0:
                print(f"  #{n1:3d}→#{n2:3d}: diff = {diff:+d}  norm = {norm_diff:+.6f}")

    # ─── 8. XOR Consecutive (look for fixed bits) ───
    print(f"\n[8] XOR ANALYSIS (consecutive keys)")
    print("-" * 78)
    print("  Looking for bit positions that are consistently 0 or 1 across keys...")

    # For puzzles with same bit width, check if any bit positions are biased
    bit_counts = {}  # bit_position -> [0_count, 1_count]
    for n, k in keys:
        if n >= 20 and n <= 70:
            for b in range(n):
                if b not in bit_counts:
                    bit_counts[b] = [0, 0]
                bit_counts[b][(k >> b) & 1] += 1

    biased_bits = []
    for b in sorted(bit_counts.keys()):
        c0, c1 = bit_counts[b]
        total = c0 + c1
        if total >= 10:
            p = c1 / total
            if p < 0.2 or p > 0.8:
                biased_bits.append((b, p, total))
                print(f"  Bit {b:2d}: P(1) = {p:.3f}  ({c1}/{total}) {'<<< BIASED' if p < 0.15 or p > 0.85 else ''}")

    if not biased_bits:
        print("  No significantly biased bit positions found.")

    # ─── 9. PRNG Fingerprinting ───
    print(f"\n[9] PRNG FINGERPRINTING")
    print("-" * 78)

    # Check if keys could be LCG: k[n+1] = (a*k[n] + c) mod m
    print("  Testing Linear Congruential Generator hypothesis...")
    # For consecutive puzzles, check if k[n+1] = (a*k[n] + c) mod 2^n
    lcg_consistent = True
    for i in range(2, min(len(keys)-1, 20)):
        if keys[i+1][0] == keys[i][0] + 1:
            n1, k1 = keys[i]
            n2, k2 = keys[i+1]
            # k2 mod k1 should be consistent if LCG
            if k1 > 0:
                r = k2 % k1

    # Test Mersenne Twister characteristic: MT19937 output has specific bit correlations
    # Lower bits of MT have shorter period
    low_bits = [k & 0x1F for _, k in keys if _ >= 10]
    low_unique = len(set(low_bits))
    print(f"  Low 5-bit diversity: {low_unique}/{len(low_bits)} unique values (expect ~{min(32, len(low_bits))})")

    # ─── 10. Autocorrelation of fractional positions ───
    print(f"\n[10] AUTOCORRELATION OF FRACTIONAL POSITIONS")
    print("-" * 78)
    # Only consecutive puzzles
    consec_fracs = []
    for n, k in keys:
        lo = 1 << (n - 1)
        f = (k - lo) / lo
        consec_fracs.append((n, f))

    # Lag-1 autocorrelation for consecutive puzzles
    pairs = []
    for i in range(len(consec_fracs) - 1):
        if consec_fracs[i+1][0] == consec_fracs[i][0] + 1:
            pairs.append((consec_fracs[i][1], consec_fracs[i+1][1]))

    if len(pairs) > 2:
        mean_x = sum(p[0] for p in pairs) / len(pairs)
        mean_y = sum(p[1] for p in pairs) / len(pairs)
        cov = sum((p[0]-mean_x)*(p[1]-mean_y) for p in pairs) / len(pairs)
        var_x = sum((p[0]-mean_x)**2 for p in pairs) / len(pairs)
        var_y = sum((p[1]-mean_y)**2 for p in pairs) / len(pairs)
        if var_x > 0 and var_y > 0:
            autocorr = cov / math.sqrt(var_x * var_y)
            print(f"  Lag-1 autocorrelation: {autocorr:+.4f}  (expect ~0 for random)")
            if abs(autocorr) > 0.3:
                print(f"  >>> SIGNIFICANT autocorrelation! Keys may be sequentially generated! <<<")
            else:
                print(f"  No significant autocorrelation.")

    # ─── 11. Test: k[n] related to mathematical constants? ───
    print(f"\n[11] MATHEMATICAL CONSTANT CORRELATION")
    print("-" * 78)

    # Test: is fractional position correlated with digits of π, e, φ, √2?
    import decimal
    decimal.getcontext().prec = 200

    # Get digits of pi (hardcoded first 200)
    pi_digits = "14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798214808651328230664709384460955058223172535940812848111745028410270193852110555964462294895493038196"
    e_digits  = "71828182845904523536028747135266249775724709369995957496696762772407663035354759457138217852516642742746639193200305992181741359662904357290033429526059563073813232862794349076323382988075319525101901"

    print("  Testing if fractional positions match digit sequences of π, e...")
    # Create sequences of 2-digit chunks from constants
    pi_fracs = [int(pi_digits[i:i+2]) / 100.0 for i in range(0, 140, 2)]
    e_fracs = [int(e_digits[i:i+2]) / 100.0 for i in range(0, 140, 2)]

    # Compare first N fractional positions with constant digits
    consec_only = [(n, f) for n, f in consec_fracs if n <= 70]

    def corr(a, b):
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        ma = sum(a)/n
        mb = sum(b)/n
        cov = sum((a[i]-ma)*(b[i]-mb) for i in range(n))/n
        va = sum((x-ma)**2 for x in a)/n
        vb = sum((x-mb)**2 for x in b)/n
        if va > 0 and vb > 0:
            return cov / math.sqrt(va*vb)
        return 0

    key_fracs = [f for _, f in consec_only]
    pi_corr = corr(key_fracs, pi_fracs)
    e_corr = corr(key_fracs, e_fracs)
    print(f"  Correlation with π digits: {pi_corr:+.4f}")
    print(f"  Correlation with e digits: {e_corr:+.4f}")
    if abs(pi_corr) > 0.3 or abs(e_corr) > 0.3:
        print(f"  >>> SIGNIFICANT correlation with mathematical constant! <<<")

    # ─── 12. Golden Ratio Spiral Test ───
    print(f"\n[12] GOLDEN RATIO SPIRAL: f[n] ≈ (n·φ) mod 1 ?")
    print("-" * 78)
    print("  (Weyl/golden-ratio low-discrepancy sequence)")
    # The golden ratio sequence s[n] = (n * φ) mod 1 is equidistributed
    # If the puzzle creator used this, we'd see f[n] ≈ frac(n * φ)
    phi_fracs = [(n * PHI) % 1.0 for n in range(1, 200)]

    best_offset = 0
    best_scale = 1.0
    best_corr = 0

    for offset in range(0, 20):
        for scale_num in range(1, 10):
            scale = scale_num / 3.0
            test_fracs = [((n + offset) * PHI * scale) % 1.0 for n, _ in consec_only]
            c = corr(key_fracs, test_fracs)
            if abs(c) > abs(best_corr):
                best_corr = c
                best_offset = offset
                best_scale = scale

    print(f"  Best correlation with φ-sequence: {best_corr:+.4f}")
    print(f"    (offset={best_offset}, scale={best_scale:.4f})")
    if abs(best_corr) > 0.3:
        print(f"  >>> POSSIBLE golden ratio pattern! <<<")
    else:
        print(f"  No golden ratio sequence pattern found.")

    # ─── 13. Summary ───
    print(f"\n{'=' * 78}")
    print(f"  SUMMARY")
    print(f"{'=' * 78}")
    print(f"""
  Keys analyzed:    {len(keys)}
  KS uniformity:   {"REJECTED" if ks_stat > ks_crit else "PASSED"} (stat={ks_stat:.4f})
  Autocorrelation:  {"SIGNIFICANT" if len(pairs) > 2 and abs(autocorr) > 0.3 else "None detected"}
  Biased bits:      {len(biased_bits)} found
  φ-sequence:       {"POSSIBLE" if abs(best_corr) > 0.3 else "Not detected"}
  π/e correlation:  {"POSSIBLE" if abs(pi_corr) > 0.3 or abs(e_corr) > 0.3 else "Not detected"}

  VERDICT: {"STRUCTURE DETECTED — investigate further!" if (ks_stat > ks_crit or (len(pairs) > 2 and abs(autocorr) > 0.3) or len(biased_bits) > 0 or abs(best_corr) > 0.3) else "Keys appear genuinely random. No exploitable pattern found."}
""")

    # ─── 14. Prediction (if any pattern found) ───
    if ks_stat > ks_crit or (len(pairs) > 2 and abs(autocorr) > 0.3):
        print(f"\n[PREDICTION] Based on detected patterns:")
        # Use mean fractional position to predict #135
        print(f"  Mean fractional position: {mean_f:.4f}")
        predicted_key = int((1 << 134) * (1 + mean_f))
        print(f"  Predicted #135 key: {predicted_key:#066x}")
        print(f"  (This narrows search by the detected bias)")

    # ─── 15. Deep dive: ratio to phi powers ───
    print(f"\n[15] RATIO ANALYSIS: k[n] / φ^n and k[n] / 2^(n·log2(φ))")
    print("-" * 78)
    for n, k in keys:
        r_phi = k / (PHI ** n)
        r_2 = k / (2 ** (n * math.log2(PHI)))
        if n <= 20 or n % 10 == 0:
            print(f"  #{n:3d}: k/φ^n = {r_phi:.6e}   k/2^(n·0.694) = {r_2:.6e}")

if __name__ == "__main__":
    analyze()
