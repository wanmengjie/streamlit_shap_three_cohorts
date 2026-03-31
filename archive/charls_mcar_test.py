import pandas as pd
import numpy as np
import os
import logging
from scipy.stats import ttest_ind

logger = logging.getLogger(__name__)

def run_mcar_test(output_dir='evaluation_results/missing_audit'):
    """MCAR 测试：检验结局缺失是否为随机缺失 (Table S2)"""
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv('CHARLS.csv')
    
    # 模拟 Little's MCAR 测试：比较结局变量缺失/不缺失的两组在其他协变量上的分布
    target_var = 'cesd10' # 以抑郁评分为例
    if target_var not in df.columns:
        logger.warning(f"缺失目标变量 {target_var}，跳过 MCAR")
        return None
        
    df['is_missing'] = df[target_var].isnull().astype(int)
    results = []
    
    for col in ['age', 'gender', 'rural', 'edu']:
        if col in df.columns:
            g0 = df[df['is_missing'] == 0][col].dropna()
            g1 = df[df['is_missing'] == 1][col].dropna()
            if len(g0) > 0 and len(g1) > 0:
                t_stat, p_val = ttest_ind(g0, g1)
                results.append({'Variable': col, 'T-stat': t_stat, 'P-value': p_val})
                
    mcar_df = pd.DataFrame(results)
    mcar_df.to_csv(os.path.join(output_dir, 'tableS2_mcar_test_results.csv'), index=False)
    logger.info(f"✅ MCAR 假设测试表已生成：{output_dir}")
    return mcar_df
