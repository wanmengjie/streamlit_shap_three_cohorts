import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import logging

logger = logging.getLogger(__name__)

def run_causal_refinement(df, consensus_features, original_ate, treatment_col='is_depression', output_dir='causal_results/refinement'):
    """
    基于共识特征的因果精炼分析：验证因果效应在精简变量集下的稳健性
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f">>> 启动因果精炼分析 ({treatment_col})：基于核心特征重新估计 ATE...")

    T = treatment_col
    Y = 'is_comorbidity_next'
    
    # 准备精简后的协变量集 (排除处理变量本身)
    W_cols = [f for f in consensus_features if f != T]
    X = df[W_cols].select_dtypes(include=[np.number])
    # 填充 NaN
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X_filled = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    T_series = df[T].astype(int)
    Y_series = df[Y].astype(float)

    # 拟合精简版因果森林
    dml_refined = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=100, max_depth=5, random_state=500, n_jobs=-1),
        model_t=RandomForestClassifier(n_estimators=100, max_depth=5, random_state=500, n_jobs=-1),
        discrete_treatment=True,
        cv=3,
        random_state=500
    )
    
    try:
        # 使用填充后的 X_filled
        dml_refined.fit(Y=Y_series, T=T_series, X=X_filled, W=None)
        refined_ate = dml_refined.effect(X_filled).mean()
        logger.info(f"[{T}] 精炼模型 ATE: {refined_ate:.4f} (原始模型 ATE: {original_ate:.4f})")

        # 绘图对比
        plt.figure(figsize=(8, 6))
        categories = ['Full Model', 'Refined Model']
        ates = [original_ate, refined_ate]
        
        bars = plt.bar(categories, ates, color=['lightsteelblue', 'darkblue'], alpha=0.8)
        plt.axhline(0, color='black', linewidth=1)
        plt.ylabel('Estimated Treatment Effect (ATE)')
        plt.title(f'Causal Robustness ({T}): ATE Stability')
        
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.002, f'{yval:.4f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'figS9_ate_stability_comparison.png'), dpi=300)
        plt.close()
        
        # 保存数值结果
        comparison_df = pd.DataFrame({
            'Model': categories,
            'ATE': ates,
            'Feature_Count': [36, len(consensus_features)]
        })
        comparison_df.to_csv(os.path.join(output_dir, 'ate_refinement_results.csv'), index=False, encoding='utf-8-sig')
        
        return refined_ate

    except Exception as e:
        logger.error(f"因果精炼分析失败: {e}")
        return None
