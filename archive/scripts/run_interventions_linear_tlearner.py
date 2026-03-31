# -*- coding: utf-8 -*-
"""
干预变量因果分析：主方法 TLearner（因果机器学习），PSM/PSW 作为验证。
对 exercise、drinkev 等 5 个干预变量，在三队列上运行，输出：
- interventions_linear_tlearner_summary.csv（TLearner + PSM + PSW 汇总）
- interventions_tlearner_primary.csv（TLearner 主结果，论文主表）
- 各干预/队列下的 causal_methods_comparison_*.csv（T vs PSM vs PSW 对比）
- 各干预/队列下的 assumption_* 假设检验文件
"""
import os
import sys
import logging
import pandas as pd

# 归档于 archive/scripts/，需多一层 dirname 到项目根
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import USE_IMPUTED_DATA, IMPUTED_DATA_PATH, RAW_DATA_PATH, AGE_MIN, COLS_TO_DROP
from data.charls_complete_preprocessing import preprocess_charls_data, reapply_cohort_definition
from scripts.run_all_interventions_analysis import prepare_interventions, INTERVENTIONS
from causal.charls_causal_multi_method import estimate_causal_multi_method
from causal.charls_causal_assumption_checks import run_all_assumption_checks
from causal.charls_causal_methods_comparison import run_causal_methods_comparison

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_interventions_linear_tlearner(output_dir='interventions_linear_tlearner'):
    """对 5 个干预变量在三队列上运行 TLearner（主）+ PSM/PSW（验证），并执行假设检验"""
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载数据
    df_clean = None
    if USE_IMPUTED_DATA and os.path.exists(IMPUTED_DATA_PATH):
        try:
            df_clean = pd.read_csv(IMPUTED_DATA_PATH, encoding='utf-8-sig')
            df_clean = df_clean.drop(columns=[c for c in COLS_TO_DROP if c in df_clean.columns], errors='ignore')
            df_clean = df_clean[df_clean['age'] >= AGE_MIN]
            df_clean = reapply_cohort_definition(df_clean, 10, 10)
            logger.info(f"已加载插补数据，n={len(df_clean)}")
        except Exception as e:
            logger.warning(f"插补数据加载失败: {e}")
    if df_clean is None:
        df_clean = preprocess_charls_data(RAW_DATA_PATH, age_min=AGE_MIN, write_output=False)
    if df_clean is None:
        logger.error("数据加载失败")
        return pd.DataFrame()

    df_clean = prepare_interventions(df_clean.copy())

    # 2. 5 个干预变量 × 3 队列：TLearner 主方法，PSM/PSW 验证
    all_results = []
    for col, _, label in INTERVENTIONS:
        if col not in df_clean.columns:
            logger.warning(f"跳过 {col}：列不存在")
            continue
        for axis_name, df_sub in [
            ('Cohort_A', df_clean[df_clean['baseline_group'] == 0]),
            ('Cohort_B', df_clean[df_clean['baseline_group'] == 1]),
            ('Cohort_C', df_clean[df_clean['baseline_group'] == 2]),
        ]:
            df_sub = df_sub.dropna(subset=[col]).copy()
            if len(df_sub) < 50 or df_sub[col].nunique() < 2:
                continue
            out_sub = os.path.join(output_dir, col, axis_name)
            os.makedirs(out_sub, exist_ok=True)
            logger.info(f">>> {axis_name} {col} ({label}) n={len(df_sub)}")
            try:
                # TLearner 主方法
                res_df = estimate_causal_multi_method(
                    df_sub, treatment_col=col, output_dir=out_sub,
                    methods=['TLearner']
                )
                if len(res_df) > 0:
                    tlearner_row = res_df[res_df['method'] == 'TLearner']
                    ate_tl = tlearner_row['ate'].iloc[0] if len(tlearner_row) > 0 else None
                    ate_lb_tl = tlearner_row['ate_lb'].iloc[0] if len(tlearner_row) > 0 else None
                    ate_ub_tl = tlearner_row['ate_ub'].iloc[0] if len(tlearner_row) > 0 else None

                    # 假设检验：重叠、平衡、E-value
                    run_all_assumption_checks(
                        df_sub, col, out_sub,
                        ate=ate_tl, ate_lb=ate_lb_tl, ate_ub=ate_ub_tl
                    )

                    # PSM/PSW 验证：与 TLearner 对比
                    comp_df = run_causal_methods_comparison(
                        df_sub, col, out_sub,
                        dml_ate=ate_tl, dml_lb=ate_lb_tl, dml_ub=ate_ub_tl,
                        ml_method_name='TLearner'
                    )

                    # 汇总：TLearner + PSM + PSW
                    for _, r in res_df.iterrows():
                        sig = 1 if (r['ate_lb'] > 0 or r['ate_ub'] < 0) else 0
                        all_results.append({
                            'exposure': col, 'label': label, 'axis': axis_name,
                            'method': r['method'], 'ate': r['ate'], 'ate_lb': r['ate_lb'], 'ate_ub': r['ate_ub'],
                            'significant': sig, 'n': len(df_sub),
                        })
                    if comp_df is not None and len(comp_df) > 0:
                        for _, r in comp_df.iterrows():
                            if r['Method'] in ('PSM', 'PSW'):
                                ate_val = r['ATE']
                                lb_val = r.get('ATE_lb', ate_val - 0.05)
                                ub_val = r.get('ATE_ub', ate_val + 0.05)
                                sig = 1 if (not pd.isna(lb_val) and not pd.isna(ub_val) and (lb_val > 0 or ub_val < 0)) else 0
                                all_results.append({
                                    'exposure': col, 'label': label, 'axis': axis_name,
                                    'method': r['Method'], 'ate': ate_val, 'ate_lb': lb_val, 'ate_ub': ub_val,
                                    'significant': sig, 'n': len(df_sub),
                                })
            except Exception as e:
                logger.warning(f"  {axis_name} {col} 失败: {e}")

    if all_results:
        summary = pd.DataFrame(all_results)
        out_csv = os.path.join(output_dir, 'interventions_linear_tlearner_summary.csv')
        summary.to_csv(out_csv, index=False, encoding='utf-8-sig')
        logger.info(f"干预变量 TLearner(主)+PSM/PSW(验证) 汇总已保存: {out_csv}")
        # 单独输出 TLearner 主结果，便于论文主表引用
        tlearner_only = summary[summary['method'] == 'TLearner']
        if len(tlearner_only) > 0:
            tlearner_only.to_csv(os.path.join(output_dir, 'interventions_tlearner_primary.csv'), index=False, encoding='utf-8-sig')
            logger.info(f"TLearner 主结果已保存: interventions_tlearner_primary.csv")
        return summary
    return pd.DataFrame()


if __name__ == '__main__':
    run_interventions_linear_tlearner()
