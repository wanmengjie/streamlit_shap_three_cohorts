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
        r'results/tables/table1_baseline_characteristics.csv',
        r'LIU_JUE_STRATEGIC_SUMMARY/table1_baseline_characteristics.csv',
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
        print("Table 1 未找到。正在尝试重新生成...")
        try:
            # Run the generation script
            import run_baseline_table_only
            # Check if it generated
            if os.path.exists(r'results/tables/table1_baseline_characteristics.csv'):
                shutil.copy(r'results/tables/table1_baseline_characteristics.csv', os.path.join(OUTPUT_DIR, 'Table_1_Baseline_Characteristics.csv'))
                print("Table 1 重新生成并收集成功。")
            elif os.path.exists(r'LIU_JUE_STRATEGIC_SUMMARY/table1_baseline_characteristics.csv'):
                shutil.copy(r'LIU_JUE_STRATEGIC_SUMMARY/table1_baseline_characteristics.csv', os.path.join(OUTPUT_DIR, 'Table_1_Baseline_Characteristics.csv'))
                print("Table 1 重新生成并收集成功 (from LIU_JUE_STRATEGIC_SUMMARY)。")
            else:
                print("Table 1 重新生成失败。")
        except Exception as e:
            print(f"Table 1 重新生成出错: {e}")

    # ---------------------------------------------------------
    # Table 2: Prediction Performance (Aggregated - 三队列)
    # ---------------------------------------------------------
    print("正在收集 Table 2 (Prediction Performance - 三队列)...")
    try:
        perf_data = []
        for cohort, folder in [('A', 'Cohort_A_Healthy_Prospective'), 
                               ('B', 'Cohort_B_Depression_to_Comorbidity'), 
                               ('C', 'Cohort_C_Cognition_to_Comorbidity')]:
            p_path = os.path.join(folder, '01_prediction', 'model_performance_full_is_comorbidity_next.csv')
            if os.path.exists(p_path):
                df = pd.read_csv(p_path)
                best_row = df.iloc[0]
                perf_data.append({
                    'Cohort': cohort,
                    'Best Model': best_row['Model'],
                    'AUC (95% CI)': best_row.get('AUC', best_row.get('AUC_95CI', 'N/A')),
                    'Brier': best_row.get('Brier', best_row.get('Brier_Score', 'N/A'))
                })
        
        # 若 Cohort 文件夹缺数据，从 cpm_evaluation_results 补充
        if len(perf_data) < 3:
            cpm_path = r'cpm_evaluation_results/table2_main_performance_combined.csv'
            if os.path.exists(cpm_path):
                df_cpm = pd.read_csv(cpm_path)
                if 'Cohort' in df_cpm.columns:
                    for c in ['A', 'B', 'C']:
                        if not any(p['Cohort'] == c for p in perf_data):
                            sub = df_cpm[df_cpm['Cohort'] == c]
                            if len(sub) > 0:
                                best = sub.loc[sub['AUC'].idxmax()]
                                auc_val = best.get('AUC_95CI', f"{best['AUC']:.4f}")
                                brier_val = best.get('Brier_95CI', f"{best['Brier_Score']:.4f}")
                                if isinstance(auc_val, float):
                                    auc_val = f"{auc_val:.4f}"
                                if isinstance(brier_val, float):
                                    brier_val = f"{brier_val:.4f}"
                                perf_data.append({
                                    'Cohort': c,
                                    'Best Model': best['Model'],
                                    'AUC (95% CI)': auc_val,
                                    'Brier': brier_val
                                })
        
        if perf_data:
            df_t2 = pd.DataFrame(perf_data).sort_values('Cohort')
            df_t2.to_csv(os.path.join(OUTPUT_DIR, 'Table_2_Prediction_Performance.csv'), index=False, encoding='utf-8-sig')
            print(f"Table 2 生成成功 (A/B/C 共 {len(perf_data)} 队列)。")
        else:
            print("Table 2 生成失败 (未找到模型性能文件)。")
    except Exception as e:
        print(f"Table 2 收集出错: {e}")

    # ---------------------------------------------------------
    # Table 3: Subgroup CATE (Aggregated - All 3 Cohorts)
    # ---------------------------------------------------------
    print("正在收集 Table 3 (Subgroup CATE)...")
    try:
        sub_data = []
        # We need A, B and C
        for cohort, folder in [('A', 'Cohort_A_Healthy_Prospective'),
                               ('B', 'Cohort_B_Depression_to_Comorbidity'), 
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
            # Fix "1-2" issue
            if 'Value' in df_t3.columns:
                df_t3['Value'] = df_t3['Value'].replace('1-2', '1 to 2')
            
            # Use a slightly different filename to avoid permission issues if open
            df_t3.to_csv(os.path.join(OUTPUT_DIR, 'Table_3_Subgroup_CATE_All_Cohorts.csv'), index=False, encoding='utf-8-sig')
            print("Table 3 生成成功 (包含 A, B, C 三个队列)。")
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
    # Table S5: External Validation (三队列)
    # ---------------------------------------------------------
    print("正在收集 Table S5 (External Validation - 三队列)...")
    try:
        ev_data = []
        rt = 'results/tables'
        for cohort, fname in [('A', 'table6_external_validation_axisA.csv'),
                             ('B', 'table6_external_validation_axisB.csv'),
                             ('C', 'table6_external_validation_axisC.csv')]:
            rpath = os.path.join(rt, fname)
            if os.path.exists(rpath):
                df = pd.read_csv(rpath)
                df['Cohort'] = cohort
                ev_data.append(df)
        if not ev_data:
            for cohort, folder in [('A', 'Cohort_A_Healthy_Prospective'), 
                                   ('B', 'Cohort_B_Depression_to_Comorbidity'), 
                                   ('C', 'Cohort_C_Cognition_to_Comorbidity')]:
                ev_path = os.path.join(folder, '04b_external_validation', 'external_validation_summary.csv')
                if os.path.exists(ev_path):
                    df = pd.read_csv(ev_path)
                    df['Cohort'] = cohort
                    ev_data.append(df)
        
        if ev_data:
            df_ts5 = pd.concat(ev_data, ignore_index=True)
            df_ts5.to_csv(os.path.join(OUTPUT_DIR, 'Table_S5_External_Validation.csv'), index=False, encoding='utf-8-sig')
            print(f"Table S5 生成成功 (A/B/C 共 {len(df_ts5)} 行)。")
        else:
            print("Table S5 生成失败。")
    except Exception as e:
        print(f"Table S5 收集出错: {e}")

    # ---------------------------------------------------------
    # Table S6: Cross Validation (Causal Methods) - 含 95%CI
    # ---------------------------------------------------------
    print("正在收集 Table S6 (Causal Methods Comparison - 95%CI)...")
    ts6_paths = [
        r'LIU_JUE_STRATEGIC_SUMMARY/causal_methods_comparison_summary.csv',
        r'causal_methods_comparison_summary.csv',
        r'results/causal_methods_comparison_summary.csv',
        r'evaluation_results/causal_methods_comparison_summary.csv'
    ]
    found_ts6 = False
    for p in ts6_paths:
        if os.path.exists(p):
            df_ts6 = pd.read_csv(p)
            # 添加 95%CI 列，四位小数
            if 'ate_lb' in df_ts6.columns and 'ate_ub' in df_ts6.columns:
                for col in ['ate', 'ate_lb', 'ate_ub']:
                    if col in df_ts6.columns:
                        df_ts6[col] = df_ts6[col].round(4)
                df_ts6['95%CI'] = df_ts6.apply(
                    lambda r: f"({r['ate_lb']:.4f}, {r['ate_ub']:.4f})"
                    if pd.notna(r.get('ate_lb')) and pd.notna(r.get('ate_ub'))
                    else '', axis=1
                )
            df_ts6.to_csv(os.path.join(OUTPUT_DIR, 'Table_S6_Causal_Methods_Comparison.csv'), index=False, encoding='utf-8-sig')
            print(f"Table S6 收集成功 (含95%CI): {p}")
            found_ts6 = True
            break
    
    if not found_ts6:
        print("Table S6 未找到。正在尝试重新生成...")
        try:
            # Run the generation script logic
            from charls_complete_preprocessing import preprocess_charls_data
            from charls_causal_methods_comparison import run_all_axes_comparison
            
            # Load data
            print("Loading data for Table S6 generation...")
            df = preprocess_charls_data('CHARLS.csv', age_min=60, write_output=False)
            if df is not None:
                # Run comparison
                print("Running causal methods comparison...")
                run_all_axes_comparison(df, output_root='LIU_JUE_STRATEGIC_SUMMARY')
                
                # Check result
                if os.path.exists(r'LIU_JUE_STRATEGIC_SUMMARY/causal_methods_comparison_summary.csv'):
                    shutil.copy(r'LIU_JUE_STRATEGIC_SUMMARY/causal_methods_comparison_summary.csv', os.path.join(OUTPUT_DIR, 'Table_S6_Causal_Methods_Comparison.csv'))
                    print("Table S6 重新生成并收集成功。")
                else:
                    print("Table S6 重新生成失败 (未生成文件)。")
            else:
                print("Table S6 重新生成失败 (数据加载失败)。")
        except Exception as e:
            print(f"Table S6 重新生成出错: {e}")

    print(f"\n>>> 所有表格收集完成！请查看目录: {OUTPUT_DIR}")

if __name__ == "__main__":
    collect_tables()
