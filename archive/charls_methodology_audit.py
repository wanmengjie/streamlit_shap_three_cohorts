import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import RFE
from boruta import BorutaPy
from sklearn_genetic import GAFeatureSelectionCV
from sklearn.impute import SimpleImputer
import logging

logger = logging.getLogger(__name__)

def run_methodology_audit(df, output_dir='methodology_audit'):
    """
    执行高严谨性方法学审计：四重特征筛选共识
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(">>> 启动四重方法学审计模块 (Consensus Feature Selection)...")

    Y = 'is_comorbidity_next'
    # 排除非特征列 (与 charls_feature_lists.get_exclude_cols 一致)
    from charls_feature_lists import get_exclude_cols
    exclude = get_exclude_cols(df, target_col=Y)
    
    X_cols = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64]]
    
    # [严谨性加固] 显式处理 NaN
    imputer = SimpleImputer(strategy='median')
    X_clean = pd.DataFrame(imputer.fit_transform(df[X_cols]), columns=X_cols)
    y_clean = df[Y].astype(int)

    consensus_results = {}

    # A. Boruta
    try:
        logger.info("正在执行 Boruta 筛选...")
        rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
        boruta = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=500)
        boruta.fit(X_clean.values, y_clean.values)
        consensus_results['Boruta'] = [X_cols[i] for i, x in enumerate(boruta.support_) if x]
    except Exception as e:
        logger.warning(f"Boruta 失败: {e}")

    # B. LASSO
    try:
        logger.info("正在执行 LASSO 筛选...")
        from sklearn.preprocessing import StandardScaler
        X_std = StandardScaler().fit_transform(X_clean)
        lasso = LassoCV(cv=5).fit(X_std, y_clean)
        consensus_results['LASSO'] = [X_cols[i] for i, coef in enumerate(lasso.coef_) if abs(coef) > 1e-5]
    except Exception as e:
        logger.warning(f"LASSO 失败: {e}")

    # C. RF-RFE
    try:
        logger.info("正在执行 RFE 筛选...")
        selector = RFE(RandomForestClassifier(n_estimators=100, max_depth=5, random_state=500), n_features_to_select=10, step=5)
        selector = selector.fit(X_clean, y_clean)
        consensus_results['RFE'] = [X_cols[i] for i, x in enumerate(selector.support_) if x]
    except Exception as e:
        logger.warning(f"RFE 失败: {e}")

    # D. GA
    try:
        logger.info("正在执行 Genetic Algorithm (GA) 筛选...")
        ga_selector = GAFeatureSelectionCV(
            estimator=RandomForestClassifier(n_estimators=50, max_depth=5),
            cv=3, scoring="roc_auc", population_size=10, generations=5,
            n_jobs=-1, verbose=0
        )
        ga_selector.fit(X_clean, y_clean)
        consensus_results['GA'] = [X_cols[i] for i, x in enumerate(ga_selector.best_features_) if x]
    except Exception as e:
        logger.warning(f"GA 筛选失败: {e}")

    # 生成共识矩阵
    try:
        all_features = sorted(list(set([f for sub in consensus_results.values() for f in sub])))
        if not all_features:
            logger.warning("未筛选出任何共识特征，返回空结果。")
            return {}
            
        matrix = pd.DataFrame(0, index=all_features, columns=consensus_results.keys())
        for method, selected in consensus_results.items():
            matrix.loc[selected, method] = 1
        
        plt.figure(figsize=(10, 12))
        sns.heatmap(matrix, annot=True, cmap='YlGnBu', cbar=False)
        plt.title('Feature Selection Consensus Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'figS6_quadruple_consensus.png'), dpi=300)
        plt.close()
        matrix.to_csv(os.path.join(output_dir, 'feature_selection_audit_final.csv'), index=False, encoding='utf-8-sig')
    except Exception as e:
        logger.warning(f"共识矩阵产出失败: {e}")

    return consensus_results
