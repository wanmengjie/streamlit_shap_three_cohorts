import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
import logging

logger = logging.getLogger(__name__)

def run_subgroup_analysis(df, output_dir='evaluation_results/subgroup_analysis'):
    """
    【学术精修版】深度挖掘：城乡差异 (Rural) 与 高龄效应 (Age 75+)
    """
    logger.info(">>> 启动学术级亚组深度挖掘...")
    os.makedirs(output_dir, exist_ok=True)

    Y = 'is_comorbidity_next'
    
    # 动态识别因果效应列
    causal_col = next((c for c in df.columns if c.startswith('causal_impact_')), 'causal_impact')
    if causal_col not in df.columns:
        logger.error(f"未找到因果效应列，分析中止。")
        return False
        
    logger.info(f"正在基于 {causal_col} 深度解析亚组异质性...")
    
    # 归一化得分用于指标评估
    y_prob = df[causal_col]
    y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-9)
    y_true = df[Y]

    subgroup_metrics = []

    MIN_TOTAL = 30  # 总样本至少 30

    # 1. 深度城乡差异 (Rural vs Urban)
    if 'rural' in df.columns:
        for val in [0, 1]:
            mask = (df['rural'] == val)
            n_events = (y_true[mask] == 1).sum()
            if mask.sum() >= MIN_TOTAL:
                label = "Urban" if val == 0 else "Rural"
                auc = roc_auc_score(y_true[mask], y_prob[mask]) if len(np.unique(y_true[mask])) > 1 else 0.5
                avg_cate = df[mask][causal_col].mean()
                warn = "Caution: Underpowered" if n_events < 30 else "OK"
                subgroup_metrics.append({'Subgroup': 'Residence', 'Value': label, 'AUC': auc, 'CATE': avg_cate, 'Count': mask.sum(), 'N_events': int(n_events), 'Sample_Size_Warning': warn})

    # 2. 深度高龄效应 (精准锁定 75+)
    if 'age' in df.columns:
        df['age_group'] = pd.cut(df['age'], bins=[0, 65, 75, 120], labels=['<65', '65-75', '75+'])
        for val in df['age_group'].unique():
            if pd.isna(val): continue
            mask = (df['age_group'] == val)
            n_events = (y_true[mask] == 1).sum()
            if mask.sum() >= MIN_TOTAL:
                auc = roc_auc_score(y_true[mask], y_prob[mask]) if len(np.unique(y_true[mask])) > 1 else 0.5
                avg_cate = df[mask][causal_col].mean()
                warn = "Caution: Underpowered" if n_events < 30 else "OK"
                subgroup_metrics.append({'Subgroup': 'Age_Group', 'Value': str(val), 'AUC': auc, 'CATE': avg_cate, 'Count': mask.sum(), 'N_events': int(n_events), 'Sample_Size_Warning': warn})

    # 3. 性别差异
    if 'gender' in df.columns:
        for val in df['gender'].unique():
            mask = (df['gender'] == val)
            n_events = (y_true[mask] == 1).sum()
            if mask.sum() >= MIN_TOTAL:
                label = "Female" if val == 0 else "Male"
                auc = roc_auc_score(y_true[mask], y_prob[mask]) if len(np.unique(y_true[mask])) > 1 else 0.5
                avg_cate = df[mask][causal_col].mean()
                warn = "Caution: Underpowered" if n_events < 30 else "OK"
                subgroup_metrics.append({'Subgroup': 'Gender', 'Value': label, 'AUC': auc, 'CATE': avg_cate, 'Count': mask.sum(), 'N_events': int(n_events), 'Sample_Size_Warning': warn})

    # 4. 教育水平（审稿拓展）
    if 'edu' in df.columns:
        for val in df['edu'].dropna().unique():
            mask = (df['edu'] == val)
            n_events = (y_true[mask] == 1).sum()
            if mask.sum() >= MIN_TOTAL:
                auc = roc_auc_score(y_true[mask], y_prob[mask]) if len(np.unique(y_true[mask])) > 1 else 0.5
                avg_cate = df[mask][causal_col].mean()
                warn = "Caution: Underpowered" if n_events < 30 else "OK"
                subgroup_metrics.append({'Subgroup': 'Education', 'Value': f'Edu_{val}', 'AUC': auc, 'CATE': avg_cate, 'Count': mask.sum(), 'N_events': int(n_events), 'Sample_Size_Warning': warn})

    # 5. 慢性病数量（0/1-2/3+）
    if 'chronic_burden' in df.columns:
        df_c = df.copy()
        df_c['chronic_group'] = pd.cut(df_c['chronic_burden'].fillna(0), bins=[-0.1, 0, 2, 20], labels=['0', '1-2', '3+'])
        for val in df_c['chronic_group'].dropna().unique():
            mask = (df_c['chronic_group'] == val)
            n_events = (y_true[mask] == 1).sum()
            if mask.sum() >= MIN_TOTAL:
                auc = roc_auc_score(y_true[mask], y_prob[mask]) if len(np.unique(y_true[mask])) > 1 else 0.5
                avg_cate = df.loc[mask, causal_col].mean()
                warn = "Caution: Underpowered" if n_events < 30 else "OK"
                subgroup_metrics.append({'Subgroup': 'Chronic', 'Value': str(val), 'AUC': auc, 'CATE': avg_cate, 'Count': mask.sum(), 'N_events': int(n_events), 'Sample_Size_Warning': warn})

    # 6. 自评健康（审稿拓展）
    if 'srh' in df.columns:
        for val in df['srh'].dropna().unique()[:5]:
            mask = (df['srh'] == val)
            n_events = (y_true[mask] == 1).sum()
            if mask.sum() >= MIN_TOTAL:
                auc = roc_auc_score(y_true[mask], y_prob[mask]) if len(np.unique(y_true[mask])) > 1 else 0.5
                avg_cate = df[mask][causal_col].mean()
                warn = "Caution: Underpowered" if n_events < 30 else "OK"
                subgroup_metrics.append({'Subgroup': 'SRH', 'Value': f'SRH_{val}', 'AUC': auc, 'CATE': avg_cate, 'Count': mask.sum(), 'N_events': int(n_events), 'Sample_Size_Warning': warn})

    sub_df = pd.DataFrame(subgroup_metrics)
    sub_df.to_csv(os.path.join(output_dir, 'subgroup_analysis_results.csv'), index=False, encoding='utf-8-sig')

    # 亚组效应异质性检验（交互项 p 值）
    try:
        from sklearn.linear_model import LinearRegression
        treatment_col = causal_col.replace('causal_impact_', '')
        if treatment_col in df.columns:
            T = df[treatment_col].fillna(0).astype(int)
            Y = df[Y].astype(float)
            sub_var = None
            if 'rural' in df.columns and df['rural'].nunique() == 2:
                sub_var = df['rural']
            elif 'gender' in df.columns and df['gender'].nunique() == 2:
                sub_var = df['gender']
            if sub_var is not None and T.nunique() >= 2:
                X_int = np.column_stack([T, sub_var, T * sub_var])
                X_int = np.nan_to_num(X_int, nan=0)
                reg = LinearRegression().fit(X_int, Y)
                from scipy import stats
                y_pred = reg.predict(X_int)
                resid = Y - y_pred
                n, k = X_int.shape
                var_b = np.var(resid) * np.linalg.inv(X_int.T @ X_int + 1e-8 * np.eye(k)).diagonal()
                se = np.sqrt(var_b)
                t_stat = reg.coef_[2] / (se[2] + 1e-9)
                p_interaction = 2 * (1 - stats.t.cdf(abs(t_stat), n - k))
                with open(os.path.join(output_dir, 'heterogeneity_test.txt'), 'w', encoding='utf-8') as f:
                    f.write(f"Interaction test (T*Subgroup): coef={reg.coef_[2]:.4f}, p={p_interaction:.4f}\n")
            else:
                with open(os.path.join(output_dir, 'heterogeneity_test.txt'), 'w', encoding='utf-8') as f:
                    f.write("Heterogeneity test: insufficient variation in subgroup\n")
    except Exception as e:
        logger.debug(f"异质性检验跳过: {e}")

    if len(sub_df) == 0:
        logger.warning("亚组无满足最小样本量的分组，跳过绘图。")
        return sub_df

    # --- 绘图：Fig 4d 学术级森林图布局 ---
    plt.figure(figsize=(14, 7))
    # 左图：模型性能稳定性 (AUC)
    plt.subplot(1, 2, 1)
    sns.barplot(data=sub_df, x='Value', y='AUC', hue='Subgroup', palette='viridis')
    plt.axhline(0.7, color='red', linestyle='--', alpha=0.5)
    plt.title('Model Robustness (AUC) across Subgroups', fontsize=12)
    plt.xticks(rotation=45)
    plt.ylim(0.4, 1.0)

    # 右图：因果异质性 (CATE) - 寻找“高获益人群”
    plt.subplot(1, 2, 2)
    sns.barplot(data=sub_df, x='Value', y='CATE', hue='Subgroup', palette='rocket')
    plt.title('Causal Heterogeneity (ATE) across Subgroups', fontsize=12)
    plt.ylabel('Mean Causal Effect (Benefit)')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_subgroup_academic_forest.png'), dpi=300)
    plt.close()

    # 💡 导师点评：自动生成一份“学术发现摘要”
    logger.info(f">>> 深度亚组分析完成！结果已存入：{output_dir}")
    return sub_df

def draw_performance_radar(perf_df, output_dir='evaluation_results'):
    """
    绘制模型性能雷达图 (Fig 4a)
    适配最新的 metrics 指标 (AUC, Accuracy, F1, Precision, Recall)
    """
    from math import pi
    # 提取核心展示指标
    categories = ['AUC', 'Accuracy', 'F1', 'Precision', 'Recall']
    available_cats = [c for c in categories if c in perf_df.columns]
    if not available_cats or len(perf_df) == 0:
        logger.warning("perf_df 为空或缺少雷达图所需指标列，跳过绘制。")
        return

    N = len(available_cats)
    plt.figure(figsize=(9, 9))
    ax = plt.subplot(111, polar=True)
    
    # 选取前 4 个模型展示 (包括 Ensemble 集成模型)
    top_models = perf_df.head(4)
    colors = ['#d9534f', '#5bc0de', '#5cb85c', '#f0ad4e']
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    for i, (idx, row) in enumerate(top_models.iterrows()):
        values = []
        for c in available_cats:
            # 处理 "0.7391 (0.71-0.77)" 这种带置信区间的字符串，只取数值
            val_str = str(row[c])
            try:
                val = float(val_str.split(' ')[0])
                values.append(val)
            except (ValueError, IndexError):
                values.append(0)
        
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=row['Model'], color=colors[i % len(colors)])
        ax.fill(angles, values, color=colors[i % len(colors)], alpha=0.1)
    
    plt.xticks(angles[:-1], available_cats, color='grey', size=10)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
    plt.ylim(0, 1)
    
    plt.title('Predictive Performance Radar (Fig 4a)', fontsize=15, fontweight='bold', pad=30)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig4a_performance_radar.png'), dpi=300)
    plt.close()
