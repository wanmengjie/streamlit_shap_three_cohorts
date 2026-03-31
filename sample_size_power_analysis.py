# -*- coding: utf-8 -*-
"""
样本量与效能分析
用于论文 Methods 或 Supplementary：说明二次分析中样本量的考量，
以及事后效能（在给定效应量下，现有样本能否达到足够检验效能）。
"""
import numpy as np
from scipy import stats
import pandas as pd
import os

def sample_size_two_proportions(p0, p1, alpha=0.05, power=0.80, ratio=1.0):
    """
    两独立组比例比较所需样本量（双侧检验）
    p0: 对照组发生率
    p1: 干预组发生率（p1-p0 为风险差）
    ratio: n1/n0，通常 1 表示 1:1 分配
    返回: (n0, n1) 每组所需样本量
    """
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    p_bar = (p0 + ratio * p1) / (1 + ratio)
    n0 = (z_alpha + z_beta)**2 * (p0*(1-p0)/ratio + p1*(1-p1)) / (p1 - p0)**2
    n1 = n0 * ratio
    return int(np.ceil(n0)), int(np.ceil(n1))


def posthoc_power_two_proportions(n0, n1, p0, p1, alpha=0.05):
    """
    事后效能：在给定样本量和效应量下，检验效能
    """
    p_pooled = (p0 * n0 + p1 * n1) / (n0 + n1)
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n0 + 1/n1))
    z_obs = (p1 - p0) / se if se > 0 else 0
    z_crit = stats.norm.ppf(1 - alpha/2)
    power = 1 - stats.norm.cdf(z_crit - z_obs) + stats.norm.cdf(-z_crit - z_obs)
    return float(np.clip(power, 0, 1))


def main():
    # 实际数据（来自 table1）
    # First-ever incident cohort (updated 2026-03)
    axis_info = {
        'Cohort_A': {'n': 8828, 'incidence': 0.041, 'events': 366},
        'Cohort_B': {'n': 3123, 'incidence': 0.136, 'events': 426},
        'Cohort_C': {'n': 2435, 'incidence': 0.169, 'events': 411},
    }
    # 运动暴露率约 50%（来自 table1：Regular exercise 约 90%+ 在有数据者中，但 43% 缺失按 0 计后约 50%）
    exercise_rate = 0.5  # 简化假设 1:1

    print("="*70)
    print("Sample Size & Power Analysis")
    print("="*70)

    # 1. A priori sample size
    print("\n[1] A priori sample size (if prospective design)")
    print("Assumptions: alpha=0.05 (2-sided), power=80%, 1:1 allocation")
    print("-"*50)
    for axis, info in axis_info.items():
        p0 = info['incidence']
        for rd in [0.03, 0.05, 0.07]:  # 风险差 3%, 5%, 7%
            p1 = p0 - rd  # 干预降低风险
            p1 = max(0.01, min(0.99, p1))
            n0, n1 = sample_size_two_proportions(p0, p1)
            total_req = n0 + n1
            actual = info['n']
            status = "[OK]" if actual >= total_req else "[insufficient]"
            print(f"  {axis}: detect RD={rd:.0%} need n~{total_req:,} | actual n={actual:,} {status}")

    # 2. Post-hoc power
    print("\n[2] Post-hoc power (given observed effect size)")
    print("B axis ATE~-0.04, C axis ATE~+0.03; assume p0=incidence, p1=p0+ATE")
    print("-"*50)
    for axis, info in axis_info.items():
        if axis == 'Cohort_A':
            continue
        n = info['n']
        p0 = info['incidence']
        ate = -0.04 if axis == 'Cohort_B' else 0.03
        p1 = p0 + ate
        p1 = max(0.01, min(0.99, p1))
        n0 = n1 = n // 2  # 简化
        pw = posthoc_power_two_proportions(n0, n1, p0, p1)
        print(f"  {axis}: if true RD={ate:+.2f}, n={n:,} -> power~{pw:.1%}")

    # 3. Min detectable RD at 80% power
    print("\n[3] Min detectable |RD| at 80% power given current n")
    print("-"*50)
    for axis, info in axis_info.items():
        if axis == 'Cohort_A':
            continue
        n = info['n']
        p0 = info['incidence']
        n0 = n1 = n // 2
        # 二分搜索最小可检测 RD
        for rd in np.arange(0.02, 0.25, 0.01):
            p1 = p0 - rd
            if p1 <= 0.01:
                continue
            pw = posthoc_power_two_proportions(n0, n1, p0, p1)
            if pw >= 0.80:
                print(f"  {axis}: n={n:,} can detect |RD|>={rd:.0%} (power>=80%)")
                break

    # 4. Output table for paper Supplementary
    out_dir = 'LIU_JUE_STRATEGIC_SUMMARY'
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for axis, info in axis_info.items():
        p0 = info['incidence']
        for rd in [0.03, 0.05, 0.07]:
            p1 = max(0.01, p0 - rd)
            n0, n1 = sample_size_two_proportions(p0, p1)
            rows.append({'axis': axis, 'baseline_rate': p0, 'target_RD': rd, 'n_required': n0+n1, 'n_actual': info['n']})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, 'sample_size_power_table.csv'), index=False, encoding='utf-8-sig')
    print(f"\nTable saved: {out_dir}/sample_size_power_table.csv")


if __name__ == '__main__':
    main()
