# 方法学详细描述（供论文方法部分引用）

## 一、预测建模

### 1.1 模型比较与超参数调优
- **交叉验证**：5 折分组交叉验证（GroupKFold），按个体 ID 分组，避免同一人多次出现在训练/测试集
- **超参数搜索**：RandomizedSearchCV，n_iter=25（树模型 12），评分指标为 ROC-AUC
- **预处理**：连续变量 StandardScaler 标准化，缺失值 SimpleImputer(median) 插补
- **类别不平衡**：scale_pos_weight = n_neg/n_pos（XGB/LightGBM/CatBoost）

### 1.2 Causal Forest DML
- **实现**：econml.dml.CausalForestDML
- **树数量**：n_estimators=1000
- **交叉拟合**：cv=5
- **Y 模型**：CatBoostRegressor，n_estimators=200, depth=4
- **T 模型**：CatBoostClassifier，n_estimators=200, depth=4
- **聚类**：按 communityID 聚类（若无则按 ID）

### 1.3 SHAP 分析
- **解释器**：TreeExplainer（树模型）/ LinearExplainer（LR）/ KernelExplainer（其他）
- **SHAP 交互**：shap_interaction_values()，仅树模型支持

### 1.4 Bootstrap 置信区间
- **重采样次数**：1000
- **有效迭代阈值**：至少 10 次有效迭代才计算百分位数 CI

---

## 二、因果推断

### 2.1 PSM
- **倾向得分**：LogisticRegression(C=1e-2)
- **匹配**：1:1 最近邻，caliper=0.2×SD(logit(PS))

### 2.2 PSW
- **权重**：IPW，w = 1/PS (T=1) 或 1/(1-PS) (T=0)
- **截断**：w ∈ [0.1, 50]

### 2.3 E-Value
- **公式**：E = RR + sqrt(RR×(RR-1))，RR 由线性回归近似

---

## 三、外部验证

### 3.1 时间验证
- 训练集：wave < max_wave；验证集：wave = max_wave

### 3.2 区域验证
- 东中西部划分（CHARLS 省份编码映射）
- 训练：东部+中部；验证：西部

---

## 四、校准与 DCA
- **校准斜率**：y_true ~ logit(p) 的回归斜率，理想值 1.0
- **Brier 分解**：Uncertainty、Reliability、Resolution
- **DCA 最优阈值**：净获益最大的概率 cutoff
