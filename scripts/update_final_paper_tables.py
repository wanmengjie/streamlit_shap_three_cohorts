#!/usr/bin/env python3
"""更新 FINAL_PAPER_TABLES：从 Cohort_*、results、LIU_JUE_STRATEGIC_SUMMARY 收集所有论文表格。"""

import os
import sys
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT, "FINAL_PAPER_TABLES")


def main():
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 运行 collect_final_tables_v2
    print("=" * 60)
    print(">>> 步骤 1: 运行 collect_final_tables_v2...")
    print("=" * 60)
    try:
        from archive.collect_final_tables_v2 import collect_tables
        collect_tables()
    except Exception as e:
        print(f"collect_final_tables_v2 出错: {e}")

    # 2. 运行 generate_table_s4_full
    print("\n" + "=" * 60)
    print(">>> 步骤 2: 生成 Table S4 (Exploratory Causal)...")
    print("=" * 60)
    try:
        from archive.generate_table_s4_full import generate_table_s4_full
        generate_table_s4_full()
    except Exception as e:
        print(f"generate_table_s4_full 出错: {e}")

    # 3. 补充 Table S3 截断值敏感性 (sensitivity_summary)
    print("\n" + "=" * 60)
    print(">>> 步骤 3: 补充 Table S3 截断值敏感性...")
    print("=" * 60)
    sens_paths = [
        os.path.join(ROOT, "LIU_JUE_STRATEGIC_SUMMARY", "sensitivity_summary_formatted.csv"),
        os.path.join(ROOT, "LIU_JUE_STRATEGIC_SUMMARY", "sensitivity_summary.csv"),
    ]
    for p in sens_paths:
        if os.path.exists(p):
            dst = os.path.join(OUTPUT_DIR, "Table_S3_Sensitivity_Thresholds.csv")
            shutil.copy(p, dst)
            print(f"Table S3 (截断值敏感性) 已复制: {p}")
            break
    else:
        print("Table S3 (截断值敏感性) 未找到 sensitivity_summary.csv")

    # 4. 从 results/tables 补充可能缺失的表格
    print("\n" + "=" * 60)
    print(">>> 步骤 4: 从 results/tables 补充...")
    print("=" * 60)
    results_tables = os.path.join(ROOT, "results", "tables")
    if os.path.exists(results_tables):
        copies = [
            ("table4_ate_summary.csv", "Table_4_ATE_Summary.csv"),
            ("table6_external_validation_axisA.csv", "Table_S5_External_Validation_A.csv"),
            ("table6_external_validation_axisB.csv", "Table_S5_External_Validation_B.csv"),
            ("table6_external_validation_axisC.csv", "Table_S5_External_Validation_C.csv"),
        ]
        for src_name, dst_name in copies:
            src = os.path.join(results_tables, src_name)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(OUTPUT_DIR, dst_name))
                print(f"  已复制: {dst_name}")

    # 5. 从 subgroup_and_joint_causal 补充 Table 3（若 06_subgroup 不存在）
    print("\n" + "=" * 60)
    print(">>> 步骤 5: 检查 Table 3 亚组 CATE 备选...")
    print("=" * 60)
    alt_t3 = os.path.join(ROOT, "subgroup_and_joint_causal", "subgroup_ate_exercise.csv")
    if os.path.exists(alt_t3):
        dst = os.path.join(OUTPUT_DIR, "Table_3_Subgroup_CATE_Alt.csv")
        shutil.copy(alt_t3, dst)
        print(f"  Table 3 备选已复制: subgroup_ate_exercise.csv")

    print("\n" + "=" * 60)
    print(f">>> 更新完成！输出目录: {OUTPUT_DIR}")
    print("=" * 60)
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith((".csv", ".xlsx")):
            print(f"  - {f}")


if __name__ == "__main__":
    main()
