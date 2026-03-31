# CHARLS 因果机器学习管线：运行与输出说明

## Cohort 与目录命名（论文 ↔ 代码）

- 论文与主流程中的分析子样本统一称 **Cohort A / B / C**（健康前瞻、抑郁→共病、认知→共病）。
- 仓库里若仍看到 **`Axis_A_*` / `Axis_B_*` / `Axis_C_*`** 文件夹名，与 **Cohort A/B/C 一一对应**，属于历史命名；**不必改文件夹名**即可复现旧路径。
- 当前 `config.py` 中主输出目录为 **`COHORT_A_DIR` / `COHORT_B_DIR` / `COHORT_C_DIR`**（如 `Cohort_A_Healthy_Prospective/`），与 `AXIS_*` 为**别名**；统一分析入口为 **`run_cohort_protocol`**（定义于 `run_all_charls_analyses.py`）。
- 汇总与步骤逻辑以 `README` 下文及 `config.COHORT_STEP_DIRS` 为准。

## 运行顺序

1. **主分析**（必选）  
   ```bash
   python run_all_charls_analyses.py
   ```  
   **中途崩溃续跑（例如 CSV 被 Excel 占用写失败）：** 关闭占用 `Cohort_*/*/01_prediction/*.csv` 的程序后，在项目根目录执行  
   `python scripts/resume_charls_cohorts.py B,C`  
   将跳过前置插补与 Table1 等前缀步骤，只重跑 B、C；A 的汇总从磁盘已有结果读取。也可在 `config.py` 设置 `RUN_COHORTS_ONLY = ['B','C']` 与 `MAIN_SKIP_STEPS_BEFORE_COHORTS = True` 后直接 `python run_all_charls_analyses.py`。跑完后请将 `RUN_COHORTS_ONLY` 改回 `None`。  

   会完成：预处理 → Table 1 → **三队列 Cohort A/B/C**（预测、SHAP、因果、临床评价、决策支持、亚组）→ 截断值敏感性 → **多暴露因果分析**（运动、睡眠、吸烟、饮酒、社会隔离）→ 汇总图。

   **插补是否自动跑：** 由 `config.py` 的 **`RUN_IMPUTATION_BEFORE_MAIN`** 决定。为 **`True`** 时，每次主流程会先跑 npj 插补再分析（**总耗时 = 插补 + 主分析**，常达小时级）。为 **`False`** 时仅读取已有 `imputation_npj_results/pipeline_trace/step1_imputed_full.csv`。当前以仓库内 `config` 为准。亦可单独运行：  
   `python archive/charls_imputation_npj_style.py`  
   **单次插补（当前默认）：** `N_MULTIPLE_IMPUTATIONS=0`、`USE_RUBIN_POOLING=False` 时只生成/使用 `step1_imputed_full.csv`，不生成 `step1_imputed_m*.csv`、不跑 Rubin 合并；与 `archive/charls_imputation_npj_style.py` 内同名常量需一致。  
   **务必用最新插补：** 保持 **`RUN_IMPUTATION_BEFORE_MAIN=True`** 时，每次主流程会先重跑插补并**覆盖** `step1_imputed_full`；若改为 `False`，日志会对比预处理表与插补 CSV 修改时间，预处理更新而未重跑插补时会 **WARNING**。加载成功时日志会打印插补文件的**修改时间**，便于核对版本。

2. **敏感性分析**（已集成主流程，亦可单独运行）  
   ```bash
   python run_sensitivity_scenarios.py
   ```  
   会跑不同截断值（抑郁/认知）和「干预无缺失」完整病例，结果写入汇总目录。  
   **说明**：脚本结尾会恢复主定义 (10,10) 并写回 `preprocessed_data/`，之后可随时再跑主分析。

3. **仅重绘流程图**（已跑过主分析、只需更新图时）  
   ```bash
   python draw_attrition_flowchart.py
   ```  

4. **仅重绘 Fig 2 概念框架图**  
   ```bash
   python draw_conceptual_framework.py
   ```  

5. **仅重绘 Fig 3 ROC 叠加图**（需已有 `roc_data.json`，即主流程跑完）  
   ```bash
   python draw_roc_combined.py
   ```

6. **三队列 SHAP + 预测概率 在线演示（Streamlit，供导师/答辩）**  
   需已有 `step1_imputed_full.csv` 与各队列 `01_prediction/champion_model.joblib`。在项目根目录执行：  
   ```bash
   streamlit run streamlit_shap_three_cohorts.py
   ```  
   说明见 **`docs/SHAP_Streamlit_三队列使用说明.md`**。

## 主要输出目录

| 目录/文件 | 内容 |
|-----------|------|
| `LIU_JUE_STRATEGIC_SUMMARY/` | 汇总：Table 1、**Fig 2 概念框架图**、发病率图、B/C 的 ATE 图、**Fig 3 ROC 叠加图**、流失表、纳入/排除流程图、额外汇总图、敏感性汇总表与图、**08_multi_exposure**（多暴露因果汇总） |
| `Cohort_A_Healthy_Prospective/`（旧名 `Axis_A_*`） | Cohort A：01 预测 02 SHAP 03 因果 |
| `Cohort_B_Depression_to_Comorbidity/`（旧名 `Axis_B_*`） | Cohort B：01 预测 … 06 亚组（步骤见 `config.COHORT_STEP_DIRS`） |
| `Cohort_C_Cognition_to_Comorbidity/`（旧名 `Axis_C_*`） | Cohort C：同上 |
| `preprocessed_data/` | 预处理后数据、流失表（主分析用） |

## 文档索引（方法/定义/审稿）

- **causal/README_因果分析说明.md**：因果分析流程、假设检验含义、输出文件索引（重叠、平衡、E-value）。  
- **ANALYSIS_DEFINITIONS.md**：结局/暴露/截断值定义，与代码一致，便于写 Methods。  
- **METHODS_DRAFT.md**：方法学中英草稿（人群、定义、因果假定、缺失、敏感性、伦理）。  
- **PUBLIC_HEALTH_ANALYSIS_IMPROVEMENTS.md**：公共卫生视角的可提升点与优先级。  
- **REVIEW_数据分析审稿意见_更新版.md**：审稿式检查与已修复/待注意项。

## 环境与复现

- Python 3.x；依赖见项目 requirements（若有）。  
- 主流程入口已设 `random.seed(500)`、`np.random.seed(500)`，复现需相同 Python 与依赖版本。  
- 数据文件：根目录下 `CHARLS.csv`（主分析与敏感性脚本均从此读取）。

### 投稿 / 修回前：可复现快照（推荐）

每次锁定结果前，在项目**根目录**执行：

```bash
python scripts/save_reproducibility_snapshot.py
```

- 默认写入 **`runs/repro_snapshots/YYYYMMDD_HHMMSS/`**（可用 `--out results/repro_snapshots` 改基路径）。
- 内容包括：**git 分支与 commit**、`git status --porcelain`、**关键 `config` 开关**（如 `USE_IMPUTED_DATA`、`USE_RUBIN_POOLING`、`N_MULTIPLE_IMPUTATIONS` 等）、**`python -m pip freeze`**、**`conda env export`**（若无 conda 则为说明性输出）。
- 审稿人或日后自己核对「当时怎么跑的」时，以该文件夹 + 同 commit 代码为准。

感谢使用；若有审稿意见或新需求，可对照上述文档逐条落实。
