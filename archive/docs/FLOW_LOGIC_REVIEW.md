# 流程逻辑审阅

## 一、主流程顺序（正确）

```
1. 预处理 (preprocess_charls_data)
   → 流失表写 preprocessed_data/attrition_flow.csv
   → df_clean (baseline_group 0/1/2)

2. Fig 2 概念框架图（静态，不依赖数据）

3. Table 1 (generate_baseline_table)

4. 复制流失表 → 画流程图

5. 轴线 A/B/C 并行逻辑（各自独立）
   - Axis A: compare_models → SHAP → causal
   - Axis B/C: causal → compare_models → SHAP → eval → decision → subgroup

6. 汇总图（incidence, ATE, 额外汇总, Fig 3 ROC）
```

**结论**：顺序合理，无依赖倒置。

---

## 二、数据流检查（正确）

| 步骤 | 数据流 | 状态 |
|------|--------|------|
| estimate_causal_impact | 修改 df_sub 增加 causal_impact_exercise，失败时不修改 | ✓ |
| run_subgroup_analysis | 依赖 df_sub 中的 causal_impact_*，因果失败时跳过 | ✓ |
| compare_models | 使用 get_exclude_cols 排除 causal_impact_*，预测不泄露 | ✓ |
| clinical_evaluation | 从模型取 X_cols，与 predict_proba 输入对齐 | ✓ |
| roc_data.json | compare_models 保存最优模型在 test 集上的 y_true/y_prob | ✓ |

---

## 三、发现的问题与建议

### 1. 【建议】draw_combined_subgroup_cate 路径依赖 cwd

**位置**：`charls_extra_figures.py` 第 153–156 行

**问题**：路径 `'Axis_B_Depression_to_Comorbidity/06_subgroup/...'` 为相对路径，依赖当前工作目录为项目根。若从其他目录运行脚本会找不到文件。

**建议**：使用 `os.path.dirname(os.path.abspath(__file__))` 或项目根路径拼接，或接受“主流程从项目根运行”的约定并在 README 中说明。

---

### 2. 【方法学说明】亚组分析中的“AUC”

**位置**：`charls_subgroup_analysis.py` 第 40–41 行

**说明**：亚组分析用 CATE（因果效应）归一化后作为“概率”计算 roc_auc_score。这是把 CATE 当作风险评分，用于衡量“CATE 对结局的区分能力”，与预测模型的 AUC 含义不同。方法上可接受，但应在方法部分写明：“亚组内 AUC 为 CATE 对结局的区分度，非预测模型 AUC。”

---

### 3. 【已正确】因果失败时的处理

当 `estimate_causal_impact` 失败时返回 `(None, (0,0,0))`，`df_sub` 不被修改，`causal_col` 为 None，亚组分析正确跳过。汇总图使用 (0,0,0) 作为 ATE，不会报错。

---

### 4. 【已正确】预测模型不泄露

`get_exclude_cols` 排除 `causal_impact_*`、`cesd`、`total_cog` 等，预测模型不会使用因果效应或定义变量，无信息泄露。

---

## 四、小结

- **未发现逻辑错误**：数据流、顺序、失败处理均合理。
- **建议**：在 README 中说明“主流程需从项目根目录运行”；方法部分补充亚组 AUC 的界定。
