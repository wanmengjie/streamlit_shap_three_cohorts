import pandas as pd
import numpy as np
try:
    from config import RANDOM_SEED
except ImportError:
    RANDOM_SEED = 500
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer

def generate_real_fig_s3(raw_path='CHARLS.csv', output_dir='evaluation_results/missing_audit'):
    os.makedirs(output_dir, exist_ok=True)
    print(">>> 启动真实插补敏感性分析 (Median vs MICE)...")
    
    # 1. 加载数据
    df_raw = pd.read_csv(raw_path)
    
    # 选取核心建模特征 (与 charls_feature_lists 一致)
    from charls_feature_lists import get_exclude_cols
    exclude = get_exclude_cols(df_raw, target_col='is_comorbidity_next')
    X_cols = [c for c in df_raw.columns if c not in exclude and df_raw[c].dtype in [np.float64, np.int64]]
    
    # 确保目标变量没有缺失
    df_clean = df_raw.dropna(subset=['is_comorbidity_next'])
    X = df_clean[X_cols]
    y = df_clean['is_comorbidity_next'].astype(int)
    
    # 划分测试集 (保持一致)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED)
    
    results = []
    
    # 方案 A: Simple Median Imputation
    print("正在评估: Median Imputation...")
    imp_median = SimpleImputer(strategy='median')
    X_train_med = imp_median.fit_transform(X_train)
    X_test_med = imp_median.transform(X_test)
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_SEED)
    rf.fit(X_train_med, y_train)
    auc_med = roc_auc_score(y_test, rf.predict_proba(X_test_med)[:, 1])
    results.append({'Method': 'Median', 'AUC': auc_med, 'Color': '#1f77b4'})
    
    # 方案 B: Iterative (MICE-like) Imputation
    print("正在评估: Iterative (MICE) Imputation...")
    # 为了速度，我们限制迭代次数
    imp_mice = IterativeImputer(max_iter=5, random_state=RANDOM_SEED)
    X_train_mice = imp_mice.fit_transform(X_train)
    X_test_mice = imp_mice.transform(X_test)
    
    rf.fit(X_train_mice, y_train)
    auc_mice = roc_auc_score(y_test, rf.predict_proba(X_test_mice)[:, 1])
    results.append({'Method': 'MICE', 'AUC': auc_mice, 'Color': '#ff7f0e'})
    
    # 2. 绘图 (Figure S3)
    res_df = pd.DataFrame(results)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Method', y='AUC', data=res_df, palette='Set2')
    
    # 设定 y 轴范围以突出差异
    y_min = min(auc_med, auc_mice) - 0.05
    plt.ylim(max(0.5, y_min), 1.0)
    
    for i, v in enumerate(res_df['AUC']):
        plt.text(i, v + 0.01, f'AUC: {v:.4f}', ha='center', fontweight='bold', fontsize=12)
        
    plt.title('Figure S3. Sensitivity Analysis of Imputation Methods (AUC Comparison)', fontsize=14, fontweight='bold')
    plt.ylabel('Model Performance (AUC)', fontsize=12)
    
    plt.savefig(os.path.join(output_dir, 'figS3_imputation_sensitivity.png'), dpi=300)
    plt.close()
    
    print(f"✅ Figure S3 (基于真实数据) 已生成: {output_dir}")
    print(f"Median AUC: {auc_med:.4f} vs MICE AUC: {auc_mice:.4f}")

if __name__ == "__main__":
    generate_real_fig_s3()
