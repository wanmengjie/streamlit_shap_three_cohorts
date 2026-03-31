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

def generate_robust_fig_s3(data_path='preprocessed_data/CHARLS_final_preprocessed.csv', output_dir='evaluation_results/missing_audit'):
    os.makedirs(output_dir, exist_ok=True)
    print(">>> 启动鲁棒性插补敏感性分析 (Injected Missingness: Median vs MICE)...")
    
    # 1. 加载已经处理好结局标签的数据
    df = pd.read_csv(data_path)
    Y = 'is_comorbidity_next'
    from charls_feature_lists import get_exclude_cols
    exclude = get_exclude_cols(df, target_col=Y)
    X_cols = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64]]
    
    X = df[X_cols]
    y = df[Y].astype(int)
    
    # 2. 人为注入 10% 的随机缺失 (模拟真实世界的缺失，但保持结局已知)
    np.random.seed(RANDOM_SEED)
    X_missing = X.copy()
    mask = np.random.rand(*X_missing.shape) < 0.1
    X_missing[mask] = np.nan
    
    X_train, X_test, y_train, y_test = train_test_split(X_missing, y, test_size=0.3, random_state=RANDOM_SEED)
    
    results = []
    
    # 方案 A: Median
    print("正在评估: Median Imputation...")
    imp_median = SimpleImputer(strategy='median')
    X_train_med = imp_median.fit_transform(X_train)
    X_test_med = imp_median.transform(X_test)
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_SEED)
    rf.fit(X_train_med, y_train)
    auc_med = roc_auc_score(y_test, rf.predict_proba(X_test_med)[:, 1])
    results.append({'Method': 'Median', 'AUC': auc_med})
    
    # 方案 B: MICE (Iterative)
    print("正在评估: Iterative (MICE) Imputation...")
    imp_mice = IterativeImputer(max_iter=5, random_state=RANDOM_SEED)
    X_train_mice = imp_mice.fit_transform(X_train)
    X_test_mice = imp_mice.transform(X_test)
    
    rf.fit(X_train_mice, y_train)
    auc_mice = roc_auc_score(y_test, rf.predict_proba(X_test_mice)[:, 1])
    results.append({'Method': 'MICE', 'AUC': auc_mice})
    
    # 3. 绘图 (Figure S3)
    res_df = pd.DataFrame(results)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Method', y='AUC', data=res_df, palette='Set2')
    
    # 设定 y 轴范围
    y_min = min(auc_med, auc_mice) - 0.02
    plt.ylim(max(0.5, y_min), 1.0)
    
    for i, v in enumerate(res_df['AUC']):
        plt.text(i, v + 0.005, f'AUC: {v:.4f}', ha='center', fontweight='bold', fontsize=12)
        
    plt.title('Figure S3. Imputation Sensitivity Check (Injecting 10% Missingness)', fontsize=14, fontweight='bold')
    plt.ylabel('Model Performance (AUC)', fontsize=12)
    plt.savefig(os.path.join(output_dir, 'figS3_imputation_sensitivity.png'), dpi=300)
    plt.close()
    
    print(f"✅ Figure S3 已通过『人为注入缺失法』修复！AUC对比: Median({auc_med:.4f}) vs MICE({auc_mice:.4f})")

if __name__ == "__main__":
    generate_robust_fig_s3()
