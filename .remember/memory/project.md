# 项目偏好与约定（因果机器学习 / CHARLS 管线）

## 语言与写作

- **对话与说明**：中文为主；论文正文与图表标签以**英文**为准（与 `config.INTERVENTION_LABELS_EN`、投稿稿一致）。
- **术语**：论文与代码中的分析子样本称 **Cohort A/B/C**。**Streamlit 演示**（`streamlit_shap_three_cohorts.py`）界面**仅英文**（侧栏 **Baseline healthy / depression / cognitive impairment cohort** 等与 A/B/C 对应）。仓库内若仍出现 **Axis_*** 目录名，视为与 Cohort 一一对应的**历史文件夹命名**（见 `README_运行与输出说明.md` 文首说明）。

## 图表与结果文件

- **首选路径**：`results/tables/`、`results/figures/` 下以 **`*_cohort*`** 为主文件名；**`*_axis*`** 为 consolidate **双写兼容副本**（同内容），论文引用以 cohort 为准。
- **核对清单**：投稿前对照 `PAPER_图表文件核对清单.md`、`PAPER_写作前最终检查清单.md`。

## 期刊与稿件

- **主投稿稿**：以 `PAPER_Manuscript_Submission_Ready.md` 及同轮锁定 PDF/Word 为准；风格对齐 npj / 综合医学期刊表述时可参考 `PAPER_npj_style_polished*.md`。
- **方法学事实**：插补层次（bulk MICE vs 训练折内 Pipeline）以 `config.py` 注释与 `docs/MEDICAL_CODE_CAUTIONS.md` 为准，文稿不得与之矛盾。

## 数据与路径

- **原始数据**：默认根目录 `CHARLS.csv`（见 `config.RAW_DATA_PATH`）；**勿改写原始文件**；预处理与插补输出写入约定子目录（如 `preprocessed_data/`、`imputation_npj_results/`）。
- **可复现快照**：投稿/修回锁定前运行 `python scripts/save_reproducibility_snapshot.py`，保存 git、config 与环境记录。
- **插补与主流程（终稿预设见 `config.py`）**：`N_MULTIPLE_IMPUTATIONS=5`、`USE_RUBIN_POOLING=True`、`USE_TEMPORAL_SPLIT=True`、`CALIBRATE_CHAMPION_PROBA=True`。**预测（CPM）**：预处理缺失宽表 + `compare_models` 内 **`IterativeImputer` Pipeline**（训练折 fit）；**因果等** 在 `USE_IMPUTED_DATA=True` 时仍用 `step1_imputed_full` 队列切片。脚本 `load_supervised_prediction_df()` 与主流程预测步对齐；`load_df_for_analysis()` 仍以插补表为主（因果脚本）。需**最新 bulk 插补**时保持 `RUN_IMPUTATION_BEFORE_MAIN=True`。

## AI / 协作

- 任务涉及方法与论文一致性时，需对照 `.remember/memory/self.md` 中的错误与修正记录，避免重复方法学表述错误。
