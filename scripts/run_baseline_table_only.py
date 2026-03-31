# -*- coding: utf-8 -*-
"""
仅生成基线表（无需跑完整 run_all_charls_analyses）
用于快速更新 table1_baseline_characteristics.csv，包含最新 TABLE1_CONFIG 变量。
"""
import os
import shutil
from data.charls_complete_preprocessing import preprocess_charls_data
from data.charls_table1_stats import generate_baseline_table

if __name__ == '__main__':
    df = preprocess_charls_data('CHARLS.csv', age_min=60, write_output=False)
    if df is not None:
        os.makedirs('LIU_JUE_STRATEGIC_SUMMARY', exist_ok=True)
        generate_baseline_table(df, output_dir='LIU_JUE_STRATEGIC_SUMMARY')
        # 复制到 results/tables
        os.makedirs('results/tables', exist_ok=True)
        shutil.copy('LIU_JUE_STRATEGIC_SUMMARY/table1_baseline_characteristics.csv',
                    'results/tables/table1_baseline_characteristics.csv')
        print("✅ 基线表已生成并复制至 results/tables/table1_baseline_characteristics.csv")
    else:
        print("❌ 预处理失败，请检查 CHARLS.csv")
