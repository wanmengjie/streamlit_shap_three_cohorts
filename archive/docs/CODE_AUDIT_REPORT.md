# 代码全量检查报告

## 一、检查结果汇总

### 1. 数据预处理模块 ✅

| 检查点 | 状态 | 说明 |
|--------|------|------|
| 表1样本流失逻辑 | ✅ | 96,628→14,386，与 charls_complete_preprocessing.py 一致 |
| 三轴线分组 | ✅ | baseline_group 0/1/2，CES-D≥10、认知≤10 |
| CES-D 列名兼容 | ✅ 已修正 | 增加 cesd/cesd10  fallback |
| 认知列兼容 | ✅ 已修正 | semantic_features 含 total_cognition、total_cog |
| 缺失值处理 | ✅ | dropna 用于关键列，建模时中位数插补 |
| 路径 | ✅ | 相对路径 CHARLS.csv、preprocessed_data/ |
| 编码 | ✅ 已修正 | to_csv 增加 encoding='utf-8' |

### 2. 预测建模模块 ✅

| 检查点 | 状态 | 说明 |
|--------|------|------|
| 5折分组按ID | ✅ | GroupKFold + groups=ID |
| 15种模型 | ✅ | LR, RF, XGB, GBDT, ExtraTrees, AdaBoost, DT, Bagging, KNN, MLP, NB, SVM, HistGBM, LightGBM, CatBoost |
| 冠军模型保存 | ✅ 已新增 | joblib.dump 至 01_prediction/champion_model.joblib |
| 随机种子 | ✅ | config.RANDOM_SEED=500 |

### 3. SHAP 可解释性 ✅

| 检查点 | 状态 | 说明 |
|--------|------|------|
| 基于冠军模型 | ✅ | run_shap_analysis_v2(model=best_model) |
| 分层SHAP | ✅ | charls_shap_stratified.py |
| 交互分析 | ✅ | run_shap_interaction |

### 4. 因果推断模块 ✅

| 检查点 | 状态 | 说明 |
|--------|------|------|
| Causal Forest DML | ✅ | cv=5, groups=cluster_ids |
| 7类干预 | ✅ | exercise, sleep_adequate, smokev, drinkev, is_socially_isolated, bmi_normal, chronic_low |
| PSM/PSW | ✅ | 1:1 最近邻，caliper=0.2×SD(PS) |
| ATE可靠性 | ✅ | ATE∈[-1,1] |
| DID | ✅ 已移除 | 设计不匹配，已从主流程移除 |

### 5. 敏感性+亚组 ✅

| 检查点 | 状态 | 说明 |
|--------|------|------|
| 截断值情景 | ✅ | CES-D≥8/10/12, Cog≤8/10/12, 完整病例 |
| 亚组分层 | ✅ | 居住地、年龄、性别、教育、慢性病、自评健康 |
| 异质性检验 | ✅ | 干预×亚组交互项 |

### 6. GitHub 开源适配 ✅

| 检查点 | 状态 | 说明 |
|--------|------|------|
| main.py | ✅ 已新增 | 一键运行入口 |
| requirements.txt | ✅ 已新增 | 精确版本依赖 |
| README.md | ✅ 已新增 | 研究简介、结构、运行步骤 |
| data_description.md | ✅ 已新增 | 变量定义、筛选逻辑 |
| results/ 结构 | ✅ 已新增 | tables, figures, models |
| 结果 consolidate | ✅ 已新增 | 主流程结束后复制至 results/ |
| sample_data.csv | ✅ 已新增 | 前100行示例数据 |

---

## 二、复现验证清单

运行 `python main.py` 后，可核对以下结果与论文一致：

| 论文结果 | 预期值 | 输出位置 |
|----------|--------|----------|
| 表1 入射队列 | 14,386 | preprocessed_data/attrition_flow.csv |
| 表3 轴线A 冠军 | CatBoost AUC≈0.73 | Axis_A_Healthy_Prospective/01_prediction/ |
| 表3 轴线B 冠军 | CatBoost AUC≈0.70 | Axis_B_Depression_to_Comorbidity/01_prediction/ |
| 表3 轴线C 冠军 | ExtraTrees AUC≈0.63 | Axis_C_Cognition_to_Comorbidity/01_prediction/ |
| 表4 运动ATE(轴线B) | ≈-0.037 | LIU_JUE_STRATEGIC_SUMMARY/all_interventions_summary.csv 或 08_multi_exposure/ |
| 表6 外部验证 | AUC 0.60–0.67 | Axis_*/04b_external_validation/external_validation_summary.csv |
| 表7 PSM/PSW | ATE≈-0.036/-0.034 | LIU_JUE_STRATEGIC_SUMMARY/causal_methods_comparison_summary.csv |

---

## 三、可复现性保障

1. **随机种子**：config.RANDOM_SEED=500，numpy/random 已固定
2. **路径**：全部相对路径，适配任意环境
3. **依赖**：requirements.txt 含版本号
4. **数据**：CHARLS 需自行申请，data_description.md 说明变量与筛选
