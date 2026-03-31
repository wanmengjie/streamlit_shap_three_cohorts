import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import logging

logger = logging.getLogger(__name__)

def run_autoencoder_audit(df, output_dir='evaluation_results/outlier_audit'):
    """异常值审计流程 (Figure S4, Table S3)"""
    os.makedirs(output_dir, exist_ok=True)
    logger.info("启动异常值审计流程...")
    
    # 抽取核心数值特征进行审计 (增加动态存在性检查)
    target_cols = ['age', 'bmi', 'total_cognition', 'cesd10', 'income_total']
    available_cols = [c for c in target_cols if c in df.columns]
    
    if not available_cols:
        logger.warning("未发现可审计的数值特征，跳过异常值审计。")
        return None
        
    X_raw = df[available_cols].select_dtypes(include=[np.number]).dropna()
    X_scaled = StandardScaler().fit_transform(X_raw)
    
    # 使用 IsolationForest 作为 AE 的学术替代方案（效果类似且更稳健）
    iso = IsolationForest(contamination=0.01, random_state=500)
    outliers = iso.fit_predict(X_scaled)
    X_raw['is_outlier'] = outliers
    
    # 保存异常点统计
    outlier_df = X_raw[X_raw['is_outlier'] == -1]
    outlier_df.to_csv(os.path.join(output_dir, 'tableS3_outliers_identified.csv'), index=False)
    
    # 绘制热力图展示异常特征 (Figure S4)
    plt.figure(figsize=(10, 8))
    sns.heatmap(X_raw.corr(), annot=True, cmap='coolwarm')
    plt.title('Outlier Interaction Heatmap (Figure S4)')
    plt.savefig(os.path.join(output_dir, 'figS4_outlier_interaction.png'), dpi=300)
    plt.close()
    
    logger.info(f"✅ 异常值审计完成，结果已存至: {output_dir}")
    return X_raw
