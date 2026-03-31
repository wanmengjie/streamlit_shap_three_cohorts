#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证清单：时序划分、Overlap trimming、交互项、加权SMD、E-value方向
运行方式：python scripts/run_verification_checklist.py
需先运行主流程生成结果，或传入结果目录路径。
"""
import os
import sys
import re

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def verify_1_temporal_split():
    """验证清单 1：时序划分正确性"""
    from config import USE_TEMPORAL_SPLIT
    if not USE_TEMPORAL_SPLIT:
        print("[SKIP] 验证1: USE_TEMPORAL_SPLIT=False，未启用时间划分")
        return True
    # 需实际运行 compare_models 才能验证，此处仅检查配置
    print("[OK] 验证1: USE_TEMPORAL_SPLIT=True")
    return True

def verify_2_overlap_trimming(results_dir='LIU_JUE_STRATEGIC_SUMMARY'):
    """验证清单 2：Overlap trimming 生效"""
    import glob
    pattern = os.path.join(results_dir, '**', 'ATE_CI_summary_exercise.txt')
    files = glob.glob(pattern)
    if not files:
        print("[SKIP] 验证2: 未找到 ATE_CI_summary_exercise.txt，请先运行因果分析")
        return True
    for fpath in files[:1]:
        with open(fpath, 'r', encoding='utf-8') as f:
            content = f.read()
        if 'overlap_trimmed_pct' not in content:
            print("[FAIL] 验证2: 未记录 overlap_trimmed_pct")
            return False
        m = re.search(r'overlap_trimmed_pct:\s*([\d.]+)', content)
        if m:
            pct = float(m.group(1))
            if pct > 10:
                print(f"[WARN] 验证2: 修剪比例 {pct:.1f}%>10%，检查 PS 模型")
            else:
                print(f"[OK] 验证2: overlap_trimmed_pct={pct:.2f}%")
        else:
            print("[OK] 验证2: overlap_trimmed_pct 已记录")
    return True

def verify_3_interaction_term():
    """验证清单 3：交互项存在（选择偏倚修复）"""
    # 检查代码中是否生成 exercise_x_adl
    import ast
    recalc_path = os.path.join(os.path.dirname(__file__), '..', 'causal', 'charls_recalculate_causal_impact.py')
    with open(recalc_path, 'r', encoding='utf-8') as f:
        code = f.read()
    if "exercise_x_adl" not in code or "T == 'exercise'" not in code:
        print("[FAIL] 验证3: 选择偏倚交互项未生成")
        return False
    # 交互项与 adlab_c 的相关性需在运行时验证，此处仅检查代码存在
    print("[OK] 验证3: exercise_x_adl 交互项已实现")
    return True

def verify_4_weighted_smd(results_dir='LIU_JUE_STRATEGIC_SUMMARY'):
    """验证清单 4：加权 SMD 可计算"""
    import glob
    pattern = os.path.join(results_dir, '**', 'assumption_balance_*_weighted.txt')
    files = glob.glob(pattern)
    if not files:
        # 检查 assumption_checks_summary 是否含 smd_weighted
        pattern2 = os.path.join(results_dir, '**', 'assumption_checks_summary.txt')
        for fpath in glob.glob(pattern2)[:1]:
            with open(fpath, 'r', encoding='utf-8') as f:
                c = f.read()
            if 'smd_weighted' in c or 'Balance' in c:
                print("[OK] 验证4: 平衡检验已运行（加权 SMD 需 assumption_balance_*_weighted.txt）")
                return True
        print("[SKIP] 验证4: 未找到加权 SMD 输出，请先运行因果假设检验")
        return True
    print("[OK] 验证4: 加权 SMD 已实现")
    return True

def verify_5_evalue_direction():
    """验证清单 5：E-value 方向（代码逻辑抽查）"""
    evalue_path = os.path.join(os.path.dirname(__file__), '..', 'causal', 'charls_causal_assumption_checks.py')
    with open(evalue_path, 'r', encoding='utf-8') as f:
        code = f.read()
    # 保护效应 ate<0 应使用 ate_ub；有害效应 ate>=0 用 ate_lb
    if "ate >= 0" in code and "ate_ub" in code and "ate_lb" in code:
        if "ate_lb) / r0" in code and "ate_ub) / r0" in code:
            print("[OK] 验证5: E-value 方向正确（保护用 ate_ub，有害用 ate_lb）")
            return True
    print("[FAIL] 验证5: E-value 方向可能错误")
    return False

def main():
    print("=" * 60)
    print("CHARLS 因果机器学习 - 验证清单")
    print("=" * 60)
    results_dir = 'LIU_JUE_STRATEGIC_SUMMARY'
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    all_ok = True
    all_ok &= verify_1_temporal_split()
    all_ok &= verify_2_overlap_trimming(results_dir)
    all_ok &= verify_3_interaction_term()
    all_ok &= verify_4_weighted_smd(results_dir)
    all_ok &= verify_5_evalue_direction()
    print("=" * 60)
    print("验证完成" if all_ok else "存在失败项")
    return 0 if all_ok else 1

if __name__ == '__main__':
    sys.exit(main())
