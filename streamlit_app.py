
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap
import os

# 设置页面
st.set_page_config(page_title="CHARLS Comorbidity Risk Predictor", layout="wide")

st.title("🏥 CHARLS Longitudinal Study: Comorbidity Risk Prediction Platform")
st.markdown("""
This interactive platform leverages advanced machine learning (CatBoost) to predict the 2-year incident comorbidity risk 
based on individual health profiles from the CHARLS cohort.
""")

# 1. 加载模型与元数据
@st.cache_resource
def load_resources():
    model_path = 'models/depression_axis/CatBoost.joblib'
    if not os.path.exists(model_path):
        # 尝试备选路径
        model_path = 'models/CatBoost.joblib'
    
    model = joblib.load(model_path)
    
    # 获取特征列表
    try:
        feature_names = model.feature_names_in_
    except Exception:
        feature_names = model.feature_names_
        
    return model, feature_names

try:
    model, features = load_resources()
    st.sidebar.success("✅ Model Loaded Successfully")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# 2. 构建侧边栏输入面板
st.sidebar.header("User Health Profile")

def user_input_features():
    data = {}
    # 定义一些关键特征的默认范围和说明
    key_features = {
        'age': ('Age', 60.0, 100.0, 65.0),
        'total_cognition': ('Cognition Score', 0.0, 31.0, 15.0),
        'systo': ('Systolic BP', 80.0, 200.0, 120.0),
        'diasto': ('Diastolic BP', 40.0, 120.0, 80.0),
        'mwaist': ('Waist Circumference', 60.0, 150.0, 85.0),
        'mheight': ('Height', 130.0, 190.0, 160.0),
        'mweight': ('Weight', 30.0, 120.0, 60.0),
        'is_depression': ('Baseline Depression (0/1)', 0, 1, 0),
        'gender': ('Gender (1:Male, 2:Female)', 1, 2, 1),
    }
    
    # 动态生成输入框
    for col in features:
        if col in key_features:
            label, vmin, vmax, vdef = key_features[col]
            data[col] = st.sidebar.number_input(label, vmin, vmax, vdef)
        else:
            # 对于非核心特征，使用中位数/0填充
            data[col] = st.sidebar.number_input(f"{col}", 0.0, 1000.0, 0.0)
    
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# 3. 实时预测与展示
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Prediction Results")
    prob = model.predict_proba(input_df)[0, 1]
    
    # 指示灯逻辑
    if prob < 0.3:
        st.success(f"Low Risk: {prob:.1%}")
    elif prob < 0.6:
        st.warning(f"Moderate Risk: {prob:.1%}")
    else:
        st.error(f"High Risk: {prob:.1%}")
        
    st.progress(prob)
    
    st.info("""
    **Note**: This tool is for research purposes only and should not replace clinical judgment.
    """)

with col2:
    st.subheader("Personalized Risk Factor Analysis (Local SHAP)")
    # 计算 SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)
    
    # 绘图
    plt.figure(figsize=(10, 6))
    shap.force_plot(
        explainer.expected_value, 
        shap_values[0], 
        input_df.iloc[0], 
        matplotlib=True,
        show=False
    )
    st.pyplot(plt.gcf())
    plt.close()

st.divider()
st.subheader("About the Model")
st.write(f"The model was trained on the CHARLS longitudinal dataset using {len(features)} predictors.")
st.write("For more details, please refer to the Supplementary Materials of the study.")
