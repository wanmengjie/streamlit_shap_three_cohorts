import pandas as pd
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from econml.dml import CausalForestDML
import shap

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_cate_visualization(df, treatment_col='is_depression', output_dir='evaluation_results/cate_analysis'):
    """
    生成 DML 因果森林的异质性处理效应 (CATE) 可视化
    """
    logger.info(f">>> 启动 CATE 异质性效应深度分析 (Treatment: {treatment_col})...")
    os.makedirs(output_dir, exist_ok=True)

    T = treatment_col
    Y = 'is_comorbidity_next'
    
    # 提取特征 (与 charls_feature_lists.get_exclude_cols 保持一致)
    from utils.charls_feature_lists import get_exclude_cols, CONTINUOUS_FOR_SCALING
    exclude_cols = get_exclude_cols(df, target_col=Y, treatment_col=T)
    X_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in [np.float64, np.int64]]
    
    # 将另一个核心变量（非 Treatment 的那个）放回协变量集 X
    other_core = 'is_cognitive_impairment' if T == 'is_depression' else 'is_depression'
    if other_core in df.columns and other_core not in X_cols:
        X_cols.append(other_core)
        
    # [审稿人深度修正] 显式处理 NaN，确保因果森林拟合稳健性
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    
    # 定义原始特征矩阵 X
    X = df[X_cols]
    
    # 填充缺失值 (中位数)
    imputer = SimpleImputer(strategy='median')
    X_filled = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # 标准化连续变量，与 charls_feature_lists.CONTINUOUS_FOR_SCALING 一致（有序计数列仅插补、不缩放）
    actual_cont = [c for c in CONTINUOUS_FOR_SCALING if c in X_filled.columns]
    X_scaled = X_filled.copy()
    if actual_cont:
        X_scaled[actual_cont] = StandardScaler().fit_transform(X_filled[actual_cont])
        
    X_to_fit = X_scaled
    T_series = df[T].astype(int)
    Y_series = df[Y].astype(float)

    # 1. 重新拟合 Causal Forest (为了获取模型对象进行深度解释)
    logger.info("正在拟合因果森林以提取异质性效应...")
    dml_causal_forest = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=100, max_depth=5, random_state=500, n_jobs=-1),
        model_t=RandomForestClassifier(n_estimators=100, max_depth=5, random_state=500, n_jobs=-1),
        discrete_treatment=True,
        cv=3,
        random_state=500
    )
    # 使用填充后的数据
    dml_causal_forest.fit(Y=Y_series, T=T_series, X=X_to_fit, W=None)
    
    # 获取个体水平的效应 (CATE)
    cate_estimates = dml_causal_forest.effect(X_to_fit)
    
    # 2. 可视化 CATE 分布 (Histogram)
    plt.figure(figsize=(10, 6))
    sns.histplot(cate_estimates, kde=True, color='teal', bins=30)
    plt.axvline(x=cate_estimates.mean(), color='red', linestyle='--', label=f'ATE: {cate_estimates.mean():.4f}')
    plt.title(f'Distribution of CATE (Impact of {T} on Comorbidity)')
    plt.xlabel(f'Estimated Treatment Effect (Impact of {T})')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'fig_cate_distribution.png'), dpi=300)
    plt.close()

    # 3. 计算 CATE 的 SHAP 值 (全样本版，以获得最精确的解释)
    logger.info("正在计算 CATE 的 SHAP 值 (全样本)...")
    X_shap = X_to_fit
    
    shap_results = dml_causal_forest.shap_values(X_shap)
    
    # 提取实际的 SHAP 值矩阵
    # EconML 的返回结构通常是 { 'Y_name': { 'T_name': shap_explanation_object } } 
    # 或者 { 'T_name': shap_explanation_object }
    if isinstance(shap_results, dict):
        # 深度展开字典直到找到非字典对象 (即 SHAP Explanation 或 Array)
        first_val = list(shap_results.values())[0]
        if isinstance(first_val, dict):
            s_vals_obj = list(first_val.values())[0]
        else:
            s_vals_obj = first_val
    else:
        s_vals_obj = shap_results

    # 进一步处理：如果是 SHAP Explanation 对象，提取其 values 矩阵或直接传递对象
    # shap.summary_plot 在较新版本可以直接接受 Explanation 对象，但为了兼容性我们检查一下
    logger.info(f"SHAP 返回类型: {type(s_vals_obj)}")

    plt.figure(figsize=(12, 8))
    try:
        # 使用采样后的 X_shap
        shap.summary_plot(s_vals_obj, X_shap, show=False)
    except Exception as e:
        logger.warning(f"直接绘制 SHAP 失败 ({e})，尝试提取 values 矩阵...")
        if hasattr(s_vals_obj, 'values'):
            shap.summary_plot(s_vals_obj.values, X_shap, show=False)
        else:
            # 最后的保底尝试
            shap.summary_plot(np.array(s_vals_obj), X_shap, show=False)

    plt.title(f'Drivers of Heterogeneity: Which features determine the impact of {T}?')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_cate_shap_summary.png'), dpi=300)
    plt.close()

    # 导出 CATE 的 SHAP 数值矩阵
    if 's_vals_obj' in locals():
        if hasattr(s_vals_obj, 'values'):
            s_vals_matrix = s_vals_obj.values
        else:
            s_vals_matrix = np.array(s_vals_obj)
        
        shap_cate_df = pd.DataFrame(s_vals_matrix, columns=X_cols)
        shap_cate_df.to_csv(os.path.join(output_dir, 'cate_shap_values.csv'), index=False, encoding='utf-8-sig')

    # 4. 关键特征与 CATE 的关系 (Partial Dependence style)
    # 提取用于排序的均值绝对值
    if hasattr(s_vals_obj, 'values'):
        vals = np.abs(s_vals_obj.values).mean(0)
    elif hasattr(s_vals_obj, 'shape'):
        vals = np.abs(s_vals_obj).mean(0)
    else:
        vals = np.abs(np.array(s_vals_obj)).mean(0)
    top_indices = np.argsort(vals)[-2:][::-1]
    top_features = [X_cols[i] for i in top_indices]

    for feat in top_features:
        plt.figure(figsize=(10, 6))
        # 散点图显示特征与效应的关系
        plt.scatter(X[feat], cate_estimates, alpha=0.4, color='purple', s=10)
        # 添加趋势线 (lowess)
        sns.regplot(x=X[feat], y=cate_estimates, scatter=False, color='orange', lowess=True)
        plt.title(f'Heterogeneity Analysis: Impact vs {feat}')
        plt.xlabel(feat)
        plt.ylabel('Estimated Treatment Effect (CATE)')
        plt.grid(alpha=0.2)
        plt.savefig(os.path.join(output_dir, f'fig_cate_vs_{feat}.png'), dpi=300)
        plt.close()

    # 5. 效应分组 (High vs Low Sensitivity)
    df_cate = X.copy()
    df_cate['cate'] = cate_estimates
    df_cate['sensitivity_group'] = pd.qcut(df_cate['cate'], q=3, labels=['Low Impact', 'Medium Impact', 'High Impact'])
    
    group_stats = df_cate.groupby('sensitivity_group')[X_cols].mean().T
    group_stats.to_csv(os.path.join(output_dir, 'cate_group_characteristics.csv'), index=False, encoding='utf-8-sig')
    
    # 6. 新增：CATE 交互热图 (Interaction Heatmap)
    logger.info("正在生成 CATE 交互热图...")
    top_4_feats = [X_cols[i] for i in np.argsort(vals)[-4:][::-1]]
    inter_data = []
    for f1 in top_4_feats:
        row = []
        for f2 in top_4_feats:
            if f1 == f2:
                row.append(0)
            else:
                # 简单交互模拟：计算 CATE 在两个特征交叉分组下的均值
                f1_med = X[f1].median()
                f2_med = X[f2].median()
                mask = (X[f1] > f1_med) & (X[f2] > f2_med)
                if mask.sum() > 10:
                    row.append(cate_estimates[mask].mean() - cate_estimates.mean())
                else:
                    row.append(0)
        inter_data.append(row)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(inter_data, annot=True, xticklabels=top_4_feats, yticklabels=top_4_feats, cmap='RdBu_r', center=0)
    plt.title('CATE Interaction Heatmap: Synergistic Effects of Risk Factors')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig3_cate_interaction_heatmap.png'), dpi=300)
    plt.close()

    logger.info(f"CATE 分析完成。图片已保存至: {output_dir}")
    return cate_estimates

if __name__ == "__main__":
    # 简单测试逻辑
    if os.path.exists('preprocessed_data/CHARLS_final_preprocessed.csv'):
        df = pd.read_csv('preprocessed_data/CHARLS_final_preprocessed.csv')
        run_cate_visualization(df)
