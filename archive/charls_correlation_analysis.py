import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)

def run_correlation_analysis(df, output_dir='evaluation_results'):
    """特征相关性分析 (Figure S5)"""
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"正在生成相关性热图到: {output_dir}")
    
    # 选取主要数值列
    cols = ['age', 'gender', 'rural', 'bmi', 'systo', 'diasto', 'total_cognition', 'cesd10', 'income_total']
    actual_cols = [c for c in cols if c in df.columns]
    
    corr = df[actual_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", square=True)
    plt.title('Feature Correlation Matrix (Figure S5)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figS5_correlation_heatmap.png'), dpi=300)
    plt.close()
    logger.info("✅ 特征相关性分析完成。")
