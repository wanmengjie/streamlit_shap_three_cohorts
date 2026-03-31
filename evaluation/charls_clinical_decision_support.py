import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

def run_clinical_decision_support(df, model=None, output_dir='clinical_decision_support', target_col='is_comorbidity_next', treatment_col='exercise'):
    """
    开发反事实决策支持模块：展示个体化干预路径。
    与管线因果分析一致，默认干预变量为 exercise；反事实为「若增加运动」「若改善睡眠」「综合」。
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f">>> 启动临床决策支持模块 (Counterfactual Support, Target: {target_col})...")

    Y = target_col
    
    # 彻底对齐建模排除名单 (与 charls_feature_lists.get_exclude_cols 一致)
    from utils.charls_feature_lists import get_exclude_cols
    current_exclude = list(set(get_exclude_cols(df, target_col=Y) + [Y]))

    if model is None:
        model_path = os.path.join('evaluation_results', 'best_predictive_model.joblib')
        if not os.path.exists(model_path):
            logger.warning("未找到最强预测模型，无法执行决策支持。")
            return
        model = joblib.load(model_path)
    
    # 从 Pipeline 或 CalibratedClassifierCV( pipeline ) 获取训练时使用的特征列
    _model = model.estimator if hasattr(model, 'estimator') else model
    orig_X_cols = None
    missing_for_model = []
    if hasattr(_model, 'named_steps') and 'preprocessor' in _model.named_steps:
        try:
            preprocessor = _model.named_steps['preprocessor']
            if hasattr(preprocessor, 'transformers_'):
                input_cols = []
                for _, _, cols in preprocessor.transformers_:
                    input_cols.extend(list(cols) if isinstance(cols, (list, np.ndarray)) else [cols])
                model_expected_cols = list(dict.fromkeys(input_cols))
                orig_X_cols = [c for c in model_expected_cols if c in df.columns]
                missing_for_model = [c for c in model_expected_cols if c not in df.columns]
                if missing_for_model:
                    logger.debug(f"决策支持：df 缺少模型期望列 {missing_for_model}，将用 0 填充")
        except Exception as ex:
            logger.debug(f"从 Pipeline 提取特征列失败: {ex}")
    if orig_X_cols is None or len(orig_X_cols) == 0:
        orig_X_cols = [c for c in df.columns if c not in current_exclude and pd.api.types.is_numeric_dtype(df[c])]
    if len(orig_X_cols) == 0:
        logger.warning("决策支持：无可用特征列，跳过。")
        return True

    # 选取高风险典型病例 (当前目标下发病的人)
    high_risk_candidates = df[df[Y] == 1]
    if len(high_risk_candidates) == 0:
        high_risk_candidates = df.sample(min(100, len(df)))
    
    # 随机选 3 个样本进行反事实模拟
    high_risk = high_risk_candidates.sample(min(3, len(high_risk_candidates)))
    
    all_cols = orig_X_cols + missing_for_model
    for i, (idx, row) in enumerate(high_risk.iterrows()):
        # 1. 现状风险：构建与模型期望列一致的 DataFrame（缺失列用 0 填充）
        row_vals = {}
        for c in orig_X_cols:
            row_vals[c] = row[c] if c in row.index and pd.notna(row.get(c)) else 0
        for c in missing_for_model:
            row_vals[c] = 0
        current_df = pd.DataFrame([row_vals], columns=all_cols)
        
        try:
            base_prob = model.predict_proba(current_df)[0, 1]
            
            # 2. 反事实场景模拟（与管线因果干预一致：treatment_col 默认为 exercise）
            # 场景 A: 若增加运动 (干预变量置为 1)
            cf_exercise = current_df.copy()
            if treatment_col in cf_exercise.columns:
                cf_exercise[treatment_col] = 1
            prob_exercise = model.predict_proba(cf_exercise)[0, 1]
            
            # 场景 B: 若改善睡眠（sleep 连续变量，置为 7 小时表示充足睡眠）
            cf_sleep = current_df.copy()
            if 'sleep' in cf_sleep.columns:
                cf_sleep['sleep'] = 7
            prob_sleep = model.predict_proba(cf_sleep)[0, 1]
            
            # 场景 C: 综合干预（运动 + 睡眠）
            cf_both = current_df.copy()
            if treatment_col in cf_both.columns:
                cf_both[treatment_col] = 1
            if 'sleep' in cf_both.columns:
                cf_both['sleep'] = 7
            prob_both = model.predict_proba(cf_both)[0, 1]
            
            # 绘制风险下降路径图
            plt.figure(figsize=(10, 6))
            labels = ['Baseline', 'Increase Exercise', 'Improve Sleep', 'Combined']
            probs = [base_prob, prob_exercise, prob_sleep, prob_both]
            
            colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71']
            sns.barplot(x=labels, y=probs, palette=colors, hue=labels, legend=False)
            plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
            plt.ylabel('Risk Probability')
            plt.title(f'Counterfactual Decision Support (Case {i+1}, ID: {int(row["ID"])})')
            plt.ylim(0, 1)
            for j, p in enumerate(probs):
                plt.text(j, p + 0.02, f'{p:.1%}', ha='center', fontweight='bold')
            
            plt.savefig(os.path.join(output_dir, f'fig5d_counterfactual_case_{i+1}.png'), dpi=300)
            plt.close()
            
        except Exception as e:
            logger.warning(f"为病例 {idx} 生成决策支持图失败: {e}")

    logger.info(f"决策支持模块完成！已为 {len(high_risk)} 个病例生成干预预测图。")
    return True
