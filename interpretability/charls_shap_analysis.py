import pandas as pd
import numpy as np
import os
import logging
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from utils.charls_feature_lists import get_exclude_cols
from config import BBOX_INCHES, TREATMENT_COL, INTERVENABLE_KEYWORDS, TOP_N_FOR_STRONG_NARRATIVE

logger = logging.getLogger(__name__)


def _is_intervenable(col_name):
    """列名包含可干预关键词则视为可干预因素"""
    c = str(col_name).lower()
    return any(kw in c for kw in INTERVENABLE_KEYWORDS)

def run_shap_analysis_v2(df, model, output_dir='shap_results', target_col='is_comorbidity_next'):
    """
    针对冠军模型进行全方位 SHAP 归因分析 (策略对比版)
    """
    logger.info(f">>> 启动冠军模型 SHAP 可解释性分析 (Target: {target_col})...")
    os.makedirs(output_dir, exist_ok=True)

    # 彻底对齐建模排除名单 (预防泄露)
    exclude_cols = get_exclude_cols(df, target_col)
    
    X_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[X_cols].select_dtypes(include=[np.number])
    if X.shape[1] == 0 or len(X) < 5:
        logger.warning("SHAP：无可用特征或样本过少，跳过。")
        return False

    # 处理 Pipeline 或 CalibratedClassifierCV(Pipeline) 包装的模型
    actual_model = model
    X_shap = X
    if hasattr(model, 'estimator') and hasattr(model.estimator, 'named_steps'):
        pipe = model.estimator
        actual_model = pipe.named_steps['clf']
        X_shap_arr = pipe.named_steps['preprocessor'].transform(X)
        try:
            f_names = pipe.named_steps['preprocessor'].get_feature_names_out()
            X_shap = pd.DataFrame(X_shap_arr, columns=f_names)
        except Exception:
            X_shap = pd.DataFrame(X_shap_arr)
    elif hasattr(model, 'named_steps') and 'clf' in model.named_steps:
        actual_model = model.named_steps['clf']
        X_shap_arr = model.named_steps['preprocessor'].transform(X)
        try:
            f_names = model.named_steps['preprocessor'].get_feature_names_out()
            X_shap = pd.DataFrame(X_shap_arr, columns=f_names)
        except Exception:
            X_shap = pd.DataFrame(X_shap_arr)
    
    try:
        model_str = str(type(actual_model))
        shap_values = None
        
        # 解释器选择逻辑（TreeExplainer 支持 sklearn 树模型、XGB/LGBM/CatBoost）
        # 补全：ExtraTrees/GBDT/HistGBM/DT 若未列入会降级至 KernelExplainer，易失败或超时
        if any(m in model_str for m in ['RandomForest', 'XGB', 'LGBM', 'CatBoost', 'ExtraTrees', 'GradientBoosting', 'HistGradientBoosting', 'DecisionTree']):
            explainer = shap.TreeExplainer(actual_model)
            shap_values = explainer.shap_values(X_shap)
        elif 'LogisticRegression' in model_str:
            explainer = shap.LinearExplainer(actual_model, X_shap)
            shap_values = explainer.shap_values(X_shap)
        else:
            logger.warning("降级至 KernelExplainer...")
            X_sample = shap.sample(X_shap, 50)
            explainer = shap.KernelExplainer(actual_model.predict_proba, X_sample)
            X_shap = X_shap.head(100)
            shap_values = explainer.shap_values(X_shap, nsamples=100)
        
        # 统一 SHAP 值格式
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        elif len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]

        # [核心修改] 清理特征名称中的前缀 (pass__, num__) 以便于阅读
        # X_shap.columns 包含了带有前缀的列名（由 ColumnTransformer 生成）
        # 我们创建一个映射字典，用于图表展示
        clean_feat_names = [c.replace('pass__', '').replace('num__', '') for c in X_shap.columns]
        # 更新 X_shap 的列名，以便 shap.summary_plot 使用清洁的名称
        X_shap.columns = clean_feat_names

        # --- 逻辑闭环：SHAP 排名 + 可干预性验证 ---
        # 因果分析的前提：可干预因素（如运动）应在 SHAP 中排名靠前
        feat_names = X_shap.columns.tolist()
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        rank_df = pd.DataFrame({
            'feature': feat_names,
            'mean_abs_shap': mean_abs_shap,
            'is_intervenable': [_is_intervenable(c) for c in feat_names],
        })
        rank_df = rank_df.sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
        rank_df['rank'] = rank_df.index + 1

        # 保存完整排名（含可干预标记）
        rank_path = os.path.join(output_dir, f'shap_feature_ranking_{target_col}.csv')
        rank_df.to_csv(rank_path, index=False, encoding='utf-8-sig')
        logger.info(f"SHAP 特征排名已保存: {rank_path}")

        # 验证主干预因素（TREATMENT_COL）的排名
        treatment_rank = None
        for i, row in rank_df.iterrows():
            if TREATMENT_COL in str(row['feature']).lower():
                treatment_rank = int(row['rank'])
                break
        intervenable_in_top = rank_df[rank_df['is_intervenable']].head(TOP_N_FOR_STRONG_NARRATIVE)

        # Generate intervenability validation report
        report_lines = [
            "=" * 60,
            "SHAP Ranking and Intervenability Validation Report",
            "=" * 60,
            "",
            "Logic: Intervenable factors (e.g. exercise) should rank high in SHAP for causal narrative.",
            "",
            f"Primary intervention: {TREATMENT_COL}",
            f"Primary intervention SHAP rank: {treatment_rank if treatment_rank else 'Not in features'}",
            f"Expected Top-{TOP_N_FOR_STRONG_NARRATIVE}: {'Yes' if treatment_rank and treatment_rank <= TOP_N_FOR_STRONG_NARRATIVE else 'No - consider subgroup/specific population effects'}",
            "",
            "Top 15 intervenable factors:",
        ]
        for _, r in intervenable_in_top.iterrows():
            report_lines.append(f"  Rank {r['rank']:2d}: {r['feature']} (mean_abs_shap={r['mean_abs_shap']:.4f})")
        if intervenable_in_top.empty:
            report_lines.append("  (None)")
        report_lines.extend(["", "=" * 60])
        report_txt = "\n".join(report_lines)
        report_path = os.path.join(output_dir, f'shap_intervention_validation_{target_col}.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_txt)
        logger.info(report_txt)

        # B. 可干预性高亮图：Top-N 特征，可干预的用不同颜色
        n_show = min(20, len(rank_df))
        plot_df = rank_df.head(n_show)
        fig, ax = plt.subplots(figsize=(10, max(6, n_show * 0.35)))
        colors = ['#2ecc71' if b else '#95a5a6' for b in plot_df['is_intervenable']]
        y_pos = np.arange(len(plot_df))[::-1]
        ax.barh(y_pos, plot_df['mean_abs_shap'], color=colors, alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{r['feature']} {'[Intervenable]' if r['is_intervenable'] else ''}" for _, r in plot_df.iterrows()], fontsize=9)
        ax.set_xlabel('Mean |SHAP|')
        ax.set_title(f'SHAP Feature Importance (green=intervenable, gray=non-intervenable) | Target: {target_col}')
        ax.legend(handles=[
            mpatches.Patch(color='#2ecc71', label='Intervenable'),
            mpatches.Patch(color='#95a5a6', label='Non-intervenable'),
        ], loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'fig5b_shap_intervenability_{target_col}.png'), dpi=300, bbox_inches=BBOX_INCHES)
        plt.close()

        # A. Summary Plot（原有）
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_shap, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'fig5a_shap_summary_{target_col}.png'), dpi=300, bbox_inches=BBOX_INCHES)
        plt.close()

        logger.info(f"冠军模型 SHAP Summary 与可干预性验证已生成至: {output_dir}")
        
    except Exception as e:
        logger.error(f"SHAP 分析失败: {e}", exc_info=True)

    return True
