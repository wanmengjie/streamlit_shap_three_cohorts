import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

def generate_variable_detail_table(df, output_dir='evaluation_results'):
    """生成变量细节对照表 (Table S1)"""
    os.makedirs(output_dir, exist_ok=True)
    
    details = []
    for col in df.columns:
        details.append({
            'Variable_ID': col,
            'Description': 'Semantic description based on CHARLS wave4 codebook',
            'Mean_or_Mode': df[col].mode()[0] if df[col].dtype == object else f"{df[col].mean():.2f}",
            'Type': 'Categorical' if df[col].dtype == object or df[col].nunique() < 5 else 'Continuous'
        })
        
    dict_df = pd.DataFrame(details)
    dict_df.to_csv(os.path.join(output_dir, 'variable_detail_dictionary.csv'), index=False)
    logger.info(f"✅ 变量细节对照表已生成: {output_dir}")
    return dict_df
