# -*- coding: utf-8 -*-
"""
导出预测模型所用变量的 baseline 到 CSV。
预测模型特征 = 排除 target、leakage(cesd/cognition)、psyche、rgrip、grip_strength_avg、sleep_adequate 后的数值列。
（sleep 连续变量用于预测，sleep_adequate 仅用于因果干预）
"""
import pandas as pd
import os

def main():
    base = os.path.dirname(os.path.abspath(__file__))
    for sub in ['FINAL_PAPER_TABLES', 'evaluation_results']:
        table1_path = os.path.join(base, sub, 'table1_baseline_characteristics.csv')
        if os.path.exists(table1_path):
            break
    else:
        print("未找到 table1_baseline_characteristics.csv")
        return

    t1 = pd.read_csv(table1_path, encoding='utf-8-sig')
    # 预测模型排除的变量（来自 charls_feature_lists）：
    # rgrip, grip_strength_avg, psyche, sleep_adequate(仅因果用), cesd/cognition(定义变量), is_comorbidity_next(结局)
    exclude_rows = [
        'Right grip, kg', 'Psychiatric condition', 'Grip strength, kg',
        'Adequate sleep (≥6h)',  # sleep_adequate 仅用于因果干预，预测用 sleep 连续变量
        'Cognition score (defining)', 'CES-D-10 (defining)', 'Incident comorbidity, follow-up',
        '—— Defining variables ——', '—— Outcome ——'  # 无内容的分组标题
    ]
    mask = ~t1['Variable'].str.strip().isin(exclude_rows)
    out = t1[mask].copy()
    # P 列：p<0.001 显示为 <0.001
    if 'P' in out.columns:
        def _fmt_p(v):
            if pd.isna(v) or str(v).strip() == '':
                return ''
            try:
                p = float(v)
                return '<0.001' if p < 0.001 else f'{p:.4f}'
            except (ValueError, TypeError):
                return str(v)
        out['P'] = out['P'].apply(_fmt_p)
    out_dir = os.path.join(base, 'FINAL_PAPER_TABLES')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'table1_prediction_model_variables_baseline.csv')
    out.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"已导出预测模型变量 baseline: {out_path}")
    print(f"共 {len(out)} 行（已排除: Right grip, Psychiatric condition, Grip strength, Adequate sleep, Cognition score, CES-D-10, Incident comorbidity）")

if __name__ == '__main__':
    main()
