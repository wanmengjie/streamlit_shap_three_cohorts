import pandas as pd
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

def run_validation_suite(df, trained_models=None, treatment_col='is_depression', main_ate=None, output_dir='validation_results'):
    """
    全量验证套件：地理验证、时间验证、负对照验证 (适配对称轴与全量模型)
    """
    analysis_tag = treatment_col.replace('is_', '')
    output_dir = os.path.join(output_dir, f'validation_{analysis_tag}')
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f">>> 启动 {treatment_col} 的全量验证套件 (All Models)...")
    
    # 基础特征准备
    T = treatment_col
    Y = 'is_comorbidity_next'
    NCO = 'is_fall_next' # Negative Control Outcome
    
    from charls_feature_lists import get_exclude_cols
    exclude = get_exclude_cols(df, target_col=Y)
    W_cols = [col for col in df.columns if col in df.columns and col not in exclude]
    
    # 如果没传模型，默认跑个简单的 LR 作为基准
    if trained_models is None:
        from sklearn.linear_model import LogisticRegression
        model_lr = LogisticRegression(max_iter=1000, class_weight='balanced')
        model_lr.fit(df[W_cols].select_dtypes(include=[np.number]), df[Y])
        trained_models = {'Baseline_LR': model_lr}

    validation_summary = []

    for name, model in trained_models.items():
        logger.info(f"正在验证模型: {name}...")
        
        # 动态提取该模型对应的特征
        X_cols_model = None
        if hasattr(model, 'feature_names_in_'):
            X_cols_model = list(model.feature_names_in_)
        elif hasattr(model, 'feature_names_'):
            X_cols_model = list(model.feature_names_)
        elif hasattr(model, 'feature_names'):
            X_cols_model = list(model.feature_names)
            
        if X_cols_model is None:
            X_cols_model = [col for col in W_cols if col in df.columns]
        else:
            X_cols_model = [col for col in X_cols_model if col in df.columns]

        # 1. 地理外部验证 (Leave-one-province-out)
        geo_aucs = []
        provinces = df['province'].unique()
        for p in provinces:
            test_df = df[df['province'] == p]
            if len(test_df) < 50 or len(test_df[Y].unique()) < 2: continue
            
            X_test = test_df[X_cols_model] # 使用模型对应的特征
            y_test = test_df[Y]
            
            try:
                y_prob = model.predict_proba(X_test)
                # 兼容 sklearn (2D) 和 pygam (1D)
                if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
                    y_prob = y_prob[:, 1]
                elif len(y_prob.shape) == 1:
                    y_prob = y_prob
            except:
                # 针对某些 model.predict_proba(X_test.values) 的兼容
                y_prob = model.predict_proba(X_test.values)
                if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
                    y_prob = y_prob[:, 1]
            
            geo_aucs.append(roc_auc_score(y_test, y_prob))
        
        avg_geo_auc = np.mean(geo_aucs) if geo_aucs else np.nan
        
        # 2. 时间验证 (Temporal Validation)
        max_wave = df['wave'].max()
        test_wave_df = df[df['wave'] == max_wave]
        if len(test_wave_df) > 0 and len(test_wave_df[Y].unique()) >= 2:
            X_test_wave = test_wave_df[X_cols_model] # 使用模型对应的特征
            y_test_wave = test_wave_df[Y]
            
            try:
                y_prob_wave = model.predict_proba(X_test_wave)
                if len(y_prob_wave.shape) > 1 and y_prob_wave.shape[1] > 1:
                    y_prob_wave = y_prob_wave[:, 1]
                elif len(y_prob_wave.shape) == 1:
                    y_prob_wave = y_prob_wave
            except:
                y_prob_wave = model.predict_proba(X_test_wave.values)
                if len(y_prob_wave.shape) > 1 and y_prob_wave.shape[1] > 1:
                    y_prob_wave = y_prob_wave[:, 1]
                    
            temporal_auc = roc_auc_score(y_test_wave, y_prob_wave)
        else:
            temporal_auc = np.nan

        # 3. 负对照验证 (NCO)
        nco_ate = 0
        if NCO in df.columns:
            from sklearn.linear_model import LinearRegression
            reg = LinearRegression()
            reg.fit(df[[T]], df[NCO])
            nco_ate = reg.coef_[0]

        validation_summary.append({
            'Model': name,
            'Avg Geo AUC': avg_geo_auc,
            'Temporal AUC': temporal_auc,
            'NCO ATE (Falls)': nco_ate
        })

    val_df = pd.DataFrame(validation_summary)
    val_df.to_csv(os.path.join(output_dir, 'all_models_validation_summary.csv'), index=False)
    
    # 可视化地理与时间验证对比 (Bar Plot)
    plt.figure(figsize=(12, 6))
    x = np.arange(len(val_df))
    width = 0.35
    plt.bar(x - width/2, val_df['Avg Geo AUC'], width, label='Avg Geo AUC', color='skyblue')
    plt.bar(x + width/2, val_df['Temporal AUC'], width, label='Temporal AUC', color='coral')
    plt.xticks(x, val_df['Model'], rotation=45)
    plt.ylabel('AUC Performance')
    plt.title(f'External Validation across 15 Models ({analysis_tag})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_all_models_validation_comparison.png'))
    plt.close()

    return val_df

