import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
import shap
import logging

logger = logging.getLogger(__name__)

def generate_case_studies(df, output_dir='evaluation_results/case_studies'):
    """
    生成典型病例画像：高风险 vs 低风险，并附带 SHAP 可视化
    """
    # 动态识别因果列
    causal_col = next((c for c in df.columns if c.startswith('causal_impact_')), 'causal_impact')
    treatment_col = 'is_depression' if 'depression' in causal_col else 'is_cognitive_impairment'
    
    analysis_tag = treatment_col.replace('is_', '')
    output_dir = os.path.join(output_dir, f'case_studies_{analysis_tag}')
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f">>> 正在生成 {treatment_col} 的典型病例画像...")

    T = treatment_col
    Y = 'is_comorbidity_next'
    
    # 1. 识别典型病例
    # 高风险组：发病概率最高的人
    high_risk_cases = df[(df[T] == 1) & (df[Y] == 1)].sort_values(by=causal_col, ascending=False).head(3)
    # 低风险组：发病概率最低的人
    low_risk_cases = df[(df[T] == 1) & (df[Y] == 0)].sort_values(by=causal_col, ascending=True).head(3)

    if len(high_risk_cases) == 0 or len(low_risk_cases) == 0:
        logger.warning(f"由于样本量不足，无法生成 {treatment_col} 的典型病例画像。")
        return False

    # 2. 准备 SHAP 解释模型（局部）
    from utils.charls_feature_lists import get_exclude_cols
    exclude = get_exclude_cols(df, target_col=Y, treatment_col=T) + [causal_col]
    X_cols = [c for c in df.columns if c not in exclude]
    X = df[X_cols].select_dtypes(include=[np.number])
    y = df[Y]
    
    # 填补缺失值用于训练解释器
    X_imputed = X.fillna(X.median())
    model = XGBClassifier(n_estimators=50, max_depth=3, random_state=500).fit(X_imputed, y)
    explainer = shap.TreeExplainer(model)

    with open(os.path.join(output_dir, f'case_study_reports_{analysis_tag}.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Clinical Case Study Report for {treatment_col}\n")
        f.write("==================================\n\n")

        all_cases = pd.concat([high_risk_cases, low_risk_cases])
        for i, (idx, case) in enumerate(all_cases.iterrows()):
            type_str = "High Risk" if i < 3 else "Low Risk"
            f.write(f"Case {i+1} [{type_str}]: ID {int(case['ID'])}\n")
            f.write(f"----------------------------------\n")
            f.write(f"Demographics: Age {case['age']:.1f}, Gender {'Male' if case['gender']==1 else 'Female'}, {'Rural' if case['rural']==1 else 'Urban'}\n")
            f.write(f"Estimated Causal Impact: {case[causal_col]:.4f}\n")
            
            # 生成个体的特征贡献图
            shap_val = explainer.shap_values(X_imputed.loc[[idx]])[0]
            
            # 取贡献最大的 Top 5 特征
            top_idx = np.argsort(np.abs(shap_val))[-5:]
            top_feats = X.columns[top_idx]
            top_vals = shap_val[top_idx]

            plt.figure(figsize=(10, 5))
            colors = ['red' if x > 0 else 'blue' for x in top_vals]
            plt.barh(top_feats, top_vals, color=colors)
            plt.title(f"Case {i+1} ({type_str}) - {treatment_col}: Top Risk Drivers")
            plt.xlabel("SHAP Value (Contribution to Risk)")
            plt.tight_layout()
            
            save_name = f'case_{i+1}_{analysis_tag}_high_risk.png' if i < 3 else f'case_{i+1}_{analysis_tag}_low_risk.png'
            plt.savefig(os.path.join(output_dir, save_name), dpi=300)
            plt.close()
            
            f.write(f"Top Risk Drivers: {', '.join(top_feats[::-1])}\n")
            f.write("\n")

    logger.info(f"病例画像完成，报告见：{output_dir}")
    return True
