import os
import shutil
import sys
import time

# Add current directory to path
sys.path.append(os.getcwd())

# Define output directories
SUPP_DIR = os.path.join("FINAL_PAPER_FIGURES", "Supplementary")
os.makedirs(SUPP_DIR, exist_ok=True)

def update_supplementary():
    print(">>> 开始更新附件图表 (Supplementary Figures)...")

    # ---------------------------------------------------------
    # Figure S1: Missing Data Heatmap
    # ---------------------------------------------------------
    print("正在更新 Figure S1 (Missing Data Heatmap)...")
    src_s1 = r'imputation_npj_results/fig1_missing_heatmap.png'
    if os.path.exists(src_s1):
        dst_s1 = os.path.join(SUPP_DIR, 'Figure_S1_Missing_Heatmap.png')
        shutil.copy(src_s1, dst_s1)
        print(f"Figure S1 更新成功: {dst_s1}")
    else:
        print(f"Figure S1 更新失败: 未找到源文件 {src_s1}")

    # ---------------------------------------------------------
    # Figure S2: Propensity Score Overlap
    # ---------------------------------------------------------
    print("正在更新 Figure S2 (Propensity Score Overlap)...")
    # Try to run the drawing script
    try:
        from draw_propensity_overlap import draw_propensity_overlap
        # The script saves to OUTPUT_ROOT/fig_propensity_overlap.png
        # We need to know what OUTPUT_ROOT is. Based on config.py it's usually 'results' or similar.
        # Let's check config.py or just let it run and find the file.
        from config import OUTPUT_ROOT
        
        draw_propensity_overlap()
        
        src_s2 = os.path.join(OUTPUT_ROOT, 'fig_propensity_overlap.png')
        if os.path.exists(src_s2):
            dst_s2 = os.path.join(SUPP_DIR, 'Figure_S2_Propensity_Overlap.png')
            shutil.copy(src_s2, dst_s2)
            print(f"Figure S2 更新成功: {dst_s2}")
        else:
            print(f"Figure S2 生成后未找到文件: {src_s2}")
            
    except Exception as e:
        print(f"Figure S2 生成出错: {e}")
        print("尝试查找旧的 Figure S2...")
        # Fallback: check if one exists in results
        possible_paths = [
            r'results/fig_propensity_overlap.png',
            r'evaluation_results/fig_propensity_overlap.png'
        ]
        for p in possible_paths:
            if os.path.exists(p):
                shutil.copy(p, os.path.join(SUPP_DIR, 'Figure_S2_Propensity_Overlap.png'))
                print(f"已使用现有文件作为 Figure S2: {p}")
                break

    # ---------------------------------------------------------
    # Figure S3: Imputation Diagnostics
    # ---------------------------------------------------------
    print("正在更新 Figure S3 (Imputation Diagnostics)...")
    src_s3 = r'imputation_npj_results/fig4_distribution_comparison.png'
    if os.path.exists(src_s3):
        dst_s3 = os.path.join(SUPP_DIR, 'Figure_S3_Imputation_Diagnostics.png')
        shutil.copy(src_s3, dst_s3)
        print(f"Figure S3 更新成功: {dst_s3}")
    else:
        print(f"Figure S3 更新失败: 未找到源文件 {src_s3}")

    # ---------------------------------------------------------
    # Additional: Sensitivity Analysis Figures (e.g., E-Value)
    # ---------------------------------------------------------
    print("正在更新其他敏感性分析图表...")
    # E-Value plot for Cohort B (Exercise)
    src_evalue = r'Cohort_B_Depression_to_Comorbidity/07_sensitivity/sensitivity_exercise/fig_s14_placebo_e_value.png'
    if os.path.exists(src_evalue):
        dst_evalue = os.path.join(SUPP_DIR, 'Figure_S_E_Value.png')
        shutil.copy(src_evalue, dst_evalue)
        print(f"E-Value Plot 更新成功: {dst_evalue}")
    
    # Bias Sensitivity
    src_bias = r'Cohort_B_Depression_to_Comorbidity/07_sensitivity/sensitivity_exercise/fig_bias_sensitivity.png'
    if os.path.exists(src_bias):
        dst_bias = os.path.join(SUPP_DIR, 'Figure_S_Bias_Sensitivity.png')
        shutil.copy(src_bias, dst_bias)
        print(f"Bias Sensitivity Plot 更新成功: {dst_bias}")

    print(f"\n>>> 附件图表更新完成！请查看目录: {SUPP_DIR}")

if __name__ == "__main__":
    update_supplementary()
