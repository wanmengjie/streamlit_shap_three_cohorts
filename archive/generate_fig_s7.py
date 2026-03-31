
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_figure_s7(df_path, consensus_path, output_dir='evaluation_results/distributions'):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 加载数据与共识特征
    df = pd.read_csv(df_path)
    audit_df = pd.read_csv(consensus_path)
    
    # 选取投票数最高的 Top 9 特征进行展示 (Figure S7 通常展示最关键的几个)
    top_features = audit_df.sort_values('Selection_Count', ascending=False)['Unnamed: 0'].head(9).tolist()
    
    # 2. 绘图设置
    plt.figure(figsize=(15, 15))
    Y = 'is_comorbidity_next'
    
    for i, col in enumerate(top_features):
        plt.subplot(3, 3, i+1)
        
        # 判断是连续变量还是分类变量
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) > 5:
            # 连续变量用 Violin Plot
            sns.violinplot(x=Y, y=col, data=df, palette='Set2', split=True, inner="quart")
            plt.title(f'Distribution of {col}')
        else:
            # 分类变量用 Count Plot (带百分比)
            counts = df.groupby([Y, col]).size().unstack(fill_value=0)
            counts_pct = counts.div(counts.sum(axis=1), axis=0) * 100
            counts_pct.plot(kind='bar', stacked=True, ax=plt.gca(), color=sns.color_palette('Pastel1'))
            plt.title(f'Proportion of {col}')
            plt.ylabel('Percentage (%)')
            plt.legend(title=col, bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.xlabel('Incident Comorbidity (0: No, 1: Yes)')
        plt.tight_layout()

    plt.savefig(os.path.join(output_dir, 'figS7_top_predictors_dist.png'), dpi=300)
    plt.close()
    print(f"✅ Figure S7 已生成至: {output_dir}")

if __name__ == "__main__":
    generate_figure_s7(
        'preprocessed_data/CHARLS_final_preprocessed.csv',
        'evaluation_results/methodology_audit/feature_selection_audit_final.csv'
    )
