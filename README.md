# 基于因果机器学习的老年抑郁-认知共病预测与干预效应评估

基于 CHARLS 纵向数据的实证研究：预测建模 + SHAP 可解释性 + Causal Forest DML + PSM/PSW 因果推断。

## 研究简介

识别老年抑郁-认知共病高危人群，评估运动、睡眠、吸烟、饮酒等可干预因素的因果效应，为精准预防提供实证基础。

## 仓库结构

```
├── main.py                 # 主运行脚本（一键运行全流程）
├── run_all_charls_analyses.py  # 核心管线
├── config.py               # 全局配置
├── requirements.txt        # 依赖包
├── data/
│   └── data_description.md # 数据说明（变量定义、筛选逻辑）
├── preprocessed_data/      # 预处理输出（运行后生成）
├── results/                # 结果输出（运行后生成）
│   ├── tables/             # 论文表格 CSV
│   ├── figures/            # 论文图表 PNG
│   └── models/             # 冠军模型
├── Axis_A_Healthy_Prospective/   # 轴线 A 详细结果
├── Axis_B_Depression_to_Comorbidity/
├── Axis_C_Cognition_to_Comorbidity/
└── LIU_JUE_STRATEGIC_SUMMARY/   # 汇总结果
```

## 环境配置

```bash
pip install -r requirements.txt
```

## 数据获取

CHARLS 原始数据需从 [CHARLS 官网](http://charls.pku.edu.cn/) 申请获取。将 `CHARLS.csv` 置于项目根目录。

## 运行步骤

```bash
python main.py
```

或直接运行：

```bash
python run_all_charls_analyses.py
```

**运行时间**：约 30–90 分钟（视硬件而定）  
**硬件要求**：建议 8GB+ 内存

## 交互式预测演示（Streamlit 网页）

三队列 **CPM 冠军模型** 的风险概率 + **个体 SHAP** 说明见根目录脚本（需先跑完预测管线以生成各队列 `champion_model.joblib` 与插补表）：

```bash
pip install streamlit
streamlit run streamlit_shap_three_cohorts.py
```

**Windows（Cursor / 内置终端易断连时）**：在项目根目录执行 `powershell -ExecutionPolicy Bypass -File scripts/run_streamlit_three_cohorts.ps1`，会在**新窗口**中启动 Streamlit，关闭编辑器终端一般不会结束该进程。SHAP 图保存在 `streamlit_shap_output/`。

详细依赖、数据路径与界面说明：**`docs/SHAP_Streamlit_三队列使用说明.md`**（与 **`README_运行与输出说明.md`** 中步骤 6 一致）。**仅供研究与答辩演示，不可替代临床判断。**

旧版单模型演示（如 `streamlit_app.py`、`charls_streamlit_app.py`）路径与冠军模型可能已过期，**投稿复现请以 `streamlit_shap_three_cohorts.py` 为准**。

## 结果复现

| 论文表格 | 输出文件 |
|----------|----------|
| 表 1 样本流失 | `preprocessed_data/attrition_flow.csv` |
| 表 3 预测性能 | `Axis_*/01_prediction/model_performance_full_*.csv` |
| 表 4 ATE | `LIU_JUE_STRATEGIC_SUMMARY/all_interventions_summary.csv` |
| 表 6 外部验证 | `Axis_*/04b_external_validation/external_validation_summary.csv` |
| 表 7 PSM/PSW | `LIU_JUE_STRATEGIC_SUMMARY/causal_methods_comparison_summary.csv` |

## 分析流程

1. **数据预处理**：年龄≥60、CES-D/认知非缺失、入射队列划分（三轴线）
2. **预测建模**：15 种 ML 模型 + 5 折分组交叉验证
3. **SHAP 可解释性**：特征重要性、交互效应、分层分析
4. **因果推断**：Causal Forest DML + PSM/PSW 交叉验证
5. **敏感性分析**：截断值变化、完整病例
6. **亚组分析**：居住地、年龄、性别、教育等分层

## 引用

如使用本代码，请引用对应论文。

## 联系方式

如有问题，请通过 GitHub Issues 反馈。
