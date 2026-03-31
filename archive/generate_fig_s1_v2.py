
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_scientific_fig_s1(data_path='preprocessed_data/CHARLS_unscaled_for_table1.csv', output_dir='evaluation_results/missing_audit'):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 加载包含 51 个初始精选特征的未插补数据
    if not os.path.exists(data_path):
        # 保底方案：如果预处理文件夹下没找到，尝试读取原始数据
        df_raw = pd.read_csv('CHARLS.csv')
    else:
        df_raw = pd.read_csv(data_path)
    
    # 2. 定义我们在 Table 1 中呈现的 51 个初始精选特征 (从之前的变量映射中提取)
    # 包含：年龄、性别、城乡、受教育、婚姻、认知、抑郁、血压、腰围、BMI、吸烟、饮酒、社交、慢性病史等
    initial_features = [c for c in df_raw.columns if c not in ['ID', 'is_comorbidity_next', 'wave', 'province', 'baseline_group', 'is_fall_next']]
    
    df_initial = df_raw[initial_features]
    
    # 3. 计算缺失率
    missing_data = df_initial.isnull().mean().sort_values(ascending=False)
    # 过滤掉缺失率为 0 的（如果数据中确实有，说明这些变量采全了）
    missing_data = missing_data[missing_data > 0]
    
    # 如果列表太长，只取 Top 30 以保证学术美感
    missing_data = missing_data.head(30)
    
    # 4. 绘图 (Figure S1)
    plt.figure(figsize=(12, 10))
    # 使用渐变色：缺失越严重的颜色越深 (学术冷色调)
    sns.barplot(x=missing_data.values, y=missing_data.index, palette='Blues_r')
    
    # 添加辅助线和百分比标注
    for i, v in enumerate(missing_data.values):
        plt.text(v + 0.002, i, f'{v:.2%}', va='center', fontsize=10, color='darkblue')
        
    plt.title('Figure S1. Missing Data Pattern of 51 Initial Selected Variables', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Proportion of Missingness', fontsize=12)
    plt.ylabel('Clinical & Demographic Variables', fontsize=12)
    plt.xlim(0, max(missing_data.values) + 0.05)
    
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # 保存结果
    plt.savefig(os.path.join(output_dir, 'figS1_missing_ratios_final.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Figure S1 已重新生成，精准对标 51 个初始变量。位置: {output_dir}")

if __name__ == "__main__":
    generate_scientific_fig_s1()
