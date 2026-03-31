import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import logging
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

logger = logging.getLogger(__name__)

def run_cognitive_subdomain_analysis(df, output_dir='causal_results/cognitive_subdomains'):
    """
    深度挖掘：抑郁对认知亚类领域 (记忆、定向、执行) 的因果损害
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(">>> 启动认知亚类结局深度因果分析...")

    subdomains = {}
    
    if not subdomains:
        logger.info("已根据指令跳过有问题的认知亚类结局深度分析。")
        return
    
    valid_subdomains = {}
    df = df.sort_values(['ID', 'wave'])
    for name, col in subdomains.items():
        if col in df.columns:
            df[f'next_{col}'] = df.groupby('ID')[col].shift(-1)
            valid_subdomains[name] = f'next_{col}'

    if not valid_subdomains:
        logger.warning("未找到认知子领域列，跳过深度子类分析。")
        return

    results = []
    T = 'is_depression'
    from utils.charls_feature_lists import get_exclude_cols
    exclude = get_exclude_cols(df, target_col='is_comorbidity_next', treatment_col=T)
    X_cols = [c for c in df.columns if c not in exclude and 'next' not in c.lower() and df[c].dtype in [np.float64, np.int64]]

    for name, Y_col in valid_subdomains.items():
        logger.info(f"正在计算亚类: {name}...")
        df_sub = df.dropna(subset=[Y_col, T] + X_cols)
        if len(df_sub) < 500: 
            logger.warning(f"亚类 {name} 样本量不足 ({len(df_sub)})，跳过。")
            continue
        
        Y_series = df_sub[Y_col].astype(float)
        T_series = df_sub[T].astype(int)
        X_sub = df_sub[X_cols]
        
        try:
            dml = CausalForestDML(
                model_y=RandomForestRegressor(n_estimators=100, max_depth=5, n_jobs=-1),
                model_t=RandomForestClassifier(n_estimators=100, max_depth=5, n_jobs=-1),
                discrete_treatment=True, cv=3
            )
            dml.fit(Y=Y_series, T=T_series, X=X_sub, W=None)
            ate = dml.ate(X_sub)
            lb, ub = dml.ate_interval(X_sub)
            
            results.append({
                'Subdomain': name,
                'ATE': ate,
                'Lower': lb,
                'Upper': ub
            })
        except Exception as e:
            logger.error(f"亚类 {name} 计算失败: {e}")

    if not results:
        logger.warning("所有亚类分析均未产出有效结果。")
        return

    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(output_dir, 'cognitive_subdomain_ate.csv'), index=False, encoding='utf-8-sig')

    plt.figure(figsize=(10, 6))
    plt.errorbar(res_df['ATE'], range(len(res_df)), 
                 xerr=[res_df['ATE']-res_df['Lower'], res_df['Upper']-res_df['ATE']], 
                 fmt='o', color='darkblue', capsize=5, markersize=8)
    plt.yticks(range(len(res_df)), res_df['Subdomain'])
    plt.axvline(0, color='red', linestyle='--')
    plt.xlabel('Estimated Causal Effect (ATE)')
    plt.title('Impact of Depression on Different Cognitive Subdomains')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_cognitive_subdomain_forest.png'), dpi=300)
    plt.close()

    logger.info(f"子领域分析完成！结果已保存至: {output_dir}")
