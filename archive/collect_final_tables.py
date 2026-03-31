import os
import shutil
import pandas as pd
import glob

# Define output directory
OUTPUT_DIR = "FINAL_PAPER_TABLES"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def collect_tables():
    print(">>> 开始收集论文表格 (FINAL_PAPER_TABLES)...")

    # ---------------------------------------------------------
    # Table 1: Baseline Characteristics
    # ---------------------------------------------------------
    print("正在收集 Table 1 (Baseline Characteristics)...")
    # Try multiple possible locations
    t1_paths = [
        r'table1_baseline_characteristics.csv',
        r'results/table1_baseline_characteristics.csv',
        r'evaluation_results/table1_baseline_characteristics.csv'
    ]
    found_t1 = False
    for p in t1_paths:
        if os.path.exists(p):
            shutil.copy(p, os.path.join(OUTPUT_DIR, 'Table_1_Baseline_Characteristics.csv'))
            print(f"Table 1 收集成功: {p}")
            found_t1 = True
            break
    if not found_t1:
        print("Table 1 未找到。可能需要运行 run_baseline_table_only.py 或检查主流程输出。")

    # ---------------------------------------------------------
    # Table 2: Prediction Performance (Aggregated)
    # ---------------------------------------------------------
    print("正在收集 Table 2 (Prediction Performance)...")
    try:
        perf_data = []
        for cohort, folder in [('A', 'Cohort_A_Healthy_Prospective'), 
                               ('B', 'Cohort_B_Depression_to_Comorbidity'), 
                               ('C', 'Cohort_C_Cognition_to_Comorbidity')]:
            p_path = os.path.join(folder, '01_prediction', 'model_performance_full_is_comorbidity_next.csv')
            if os.path.exists(p_path):
                df = pd.read_csv(p_path)
                # Assume first row is best model
                best_row = df.iloc[0]
                perf_data.append({
                    'Cohort': cohort,
                    'Best Model': best_row['Model'],
                    'AUC (95% CI)': best_row.get('AUC', 'N/A'),
                    'Brier': best_row.get('Brier', 'N/A')
                })
        
        if perf_data:
            df_t2 = pd.DataFrame(perf_data)
            df_t2.to_csv(os.path.join(OUTPUT_DIR, 'Table_2_Prediction_Performance.csv'), index=False, encoding='utf-8-sig')
            print("Table 2 生成成功。")
        else:
            print("Table 2 生成失败 (未找到模型性能文件)。")
    except Exception as e:
        print(f"Table 2 收集出错: {e}")

    # ---------------------------------------------------------
    # Table 3: Subgroup CATE (Aggregated)
    # ---------------------------------------------------------
    print("正在收集 Table 3 (Subgroup CATE)...")
    try:
        sub_data = []
        # We need B and C
        for cohort, folder in [('B', 'Cohort_B_Depression_to_Comorbidity'), 
                               ('C', 'Cohort_C_Cognition_to_Comorbidity')]:
            s_path = os.path.join(folder, '06_subgroup', 'subgroup_analysis_results.csv')
            if os.path.exists(s_path):
                df = pd.read_csv(s_path)
                # Keep relevant columns
                if 'Subgroup' in df.columns and 'Value' in df.columns and 'CATE' in df.columns:
                    df['Cohort'] = cohort
                    sub_data.append(df[['Cohort', 'Subgroup', 'Value', 'CATE']])
        
        if sub_data:
            df_t3 = pd.concat(sub_data)
            df_t3.to_csv(os.path.join(OUTPUT_DIR, 'Table_3_Subgroup_CATE.csv'), index=False, encoding='utf-8-sig')
            print("Table 3 生成成功。")
        else:
            print("Table 3 生成失败 (未找到亚组分析文件)。")
    except Exception as e:
        print(f"Table 3 收集出错: {e}")

    # ---------------------------------------------------------
    # Table S2: Missing Data (Little's MCAR)
    # ---------------------------------------------------------
    print("正在收集 Table S2 (Missing Data)...")
    ts2_path = r'imputation_npj_results/table2b_littles_mcar.csv'
    if os.path.exists(ts2_path):
        shutil.copy(ts2_path, os.path.join(OUTPUT_DIR, 'Table_S2_Missing_Data_MCAR.csv'))
        print("Table S2 收集成功。")
    else:
        print("Table S2 未找到。")

    # ---------------------------------------------------------
    # Table S3: Sensitivity Analysis (Cutoffs)
    # ---------------------------------------------------------
    print("正在收集 Table S3 (Sensitivity Analysis)...")
    # This might be in a text file or CSV. Let's check Cohort B's sensitivity folder.
    # Or look for 'bias_sensitivity.csv'
    ts3_path = r'Cohort_B_Depression_to_Comorbidity/07_sensitivity/sensitivity_exercise/bias_sensitivity.csv'
    if os.path.exists(ts3_path):
        shutil.copy(ts3_path, os.path.join(OUTPUT_DIR, 'Table_S3_Sensitivity_Bias.csv'))
        print("Table S3 (Bias) 收集成功。")
    else:
        print("Table S3 (Bias) 未找到。")

    # ---------------------------------------------------------
    # Table S5: External Validation
    # ---------------------------------------------------------
    print("正在收集 Table S5 (External Validation)...")
    try:
        ev_data = []
        for cohort, folder in [('A', 'Cohort_A_Healthy_Prospective'), 
                               ('B', 'Cohort_B_Depression_to_Comorbidity'), 
                               ('C', 'Cohort_C_Cognition_to_Comorbidity')]:
            ev_path = os.path.join(folder, '04b_external_validation', 'external_validation_summary.csv')
            if os.path.exists(ev_path):
                df = pd.read_csv(ev_path)
                df['Cohort'] = cohort
                ev_data.append(df)
        
        if ev_data:
            df_ts5 = pd.concat(ev_data)
            df_ts5.to_csv(os.path.join(OUTPUT_DIR, 'Table_S5_External_Validation.csv'), index=False, encoding='utf-8-sig')
            print("Table S5 生成成功。")
        else:
            print("Table S5 生成失败。")
    except Exception as e:
        print(f"Table S5 收集出错: {e}")

    # ---------------------------------------------------------
    # Table S6: Cross Validation (Causal Methods)
    # ---------------------------------------------------------
    print("正在收集 Table S6 (Causal Methods Comparison)...")
    # This is usually generated by run_causal_methods_comparison.py
    # Check for 'causal_methods_comparison_summary.csv' in root or results
    ts6_paths = [
        r'causal_methods_comparison_summary.csv',
        r'results/causal_methods_comparison_summary.csv',
        r'evaluation_results/causal_methods_comparison_summary.csv'
    ]
    found_ts6 = False
    for p in ts6_paths:
        if os.path.exists(p):
            shutil.copy(p, os.path.join(OUTPUT_DIR, 'Table_S6_Causal_Methods_Comparison.csv'))
            print(f"Table S6 收集成功: {p}")
            found_ts6 = True
            break
    if not found_ts6:
        print("Table S6 未找到。")

    print(f"\n>>> 所有表格收集完成！请查看目录: {OUTPUT_DIR}")

if __name__ == "__main__":
    collect_tables()
