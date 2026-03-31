# 全量代码潜在问题检查报告 2026-03-16

基于主流程 `run_all_charls_analyses.py` 及全部关联模块的系统检查。

---

## 一、导入路径错误（独立脚本）

以下脚本使用 `from charls_feature_lists import ...`，但 `charls_feature_lists` 位于 `utils/` 子目录，**独立运行时会 ModuleNotFoundError**：

| 文件 | 正确导入 |
|------|----------|
| `charls_policy_forest_plot.py` L45 | `from utils.charls_feature_lists import get_exclude_cols` |
| `charls_did_analysis.py` L11 | `from utils.charls_feature_lists import get_exclude_cols` |
| `charls_cate_visualization.py` L26 | `from utils.charls_feature_lists import get_exclude_cols` |
| `charls_ablation_study.py` L12 | `from utils.charls_feature_lists import get_exclude_cols` |
| `charls_visual_enhancement.py` L47 | `from utils.charls_feature_lists import get_exclude_cols` |
| `charls_case_study.py` L40 | `from utils.charls_feature_lists import get_exclude_cols` |

**说明**：主流程不直接调用上述脚本，仅影响独立运行或手动调用。

---

## 二、异常处理不规范

### 1. 裸 `except:` 或 `except:` 吞掉所有异常

| 文件 | 行 | 建议 |
|------|-----|------|
| `charls_policy_forest_plot.py` | 63, 68 | 改为 `except Exception as e:` 并记录日志 |
| `evaluation/charls_subgroup_analysis.py` | 205 | 同上 |
| `causal/charls_causal_methods_comparison.py` | 220-221, 297-298 | 至少记录 warning |
| `evaluation/charls_clinical_evaluation.py` | 41-42, 154, 170 | 同上 |
| `evaluation/charls_clinical_decision_support.py` | 51-52 | 同上 |
| `causal/charls_low_sample_optimization.py` | 34-35 | 同上 |

### 2. `except Exception: pass` 静默失败

多处 `except Exception: pass` 会隐藏真实错误，建议至少 `logger.debug(...)` 或 `logger.warning(...)`。

---

## 三、数据与路径

### 1. 硬编码 CHARLS.csv

- `run_all_charls_analyses.py`、`run_sensitivity_scenarios.py`、`charls_dose_response.py` 等使用 `'CHARLS.csv'`
- 建议：在 `config.py` 中增加 `RAW_DATA_PATH = 'CHARLS.csv'`，统一引用

### 2. charls_imputation_audit.run_imputation_sensitivity

- L17 直接 `pd.read_csv('CHARLS.csv')`，且使用 `cesd10` 目标（与主流程 `is_comorbidity_next` 不同）
- 该函数未被主流程调用，仅用于旧版插补审计；主流程使用 `run_imputation_sensitivity_preprocessed`

### 3. charls_cate_visualization 独立运行

- L198-199 硬编码 `preprocessed_data/CHARLS_final_preprocessed.csv`
- 若文件不存在或路径错误，独立运行会失败

---

## 四、charls_policy_forest_plot 特有问题

1. **导入错误**：`from charls_feature_lists` → 应改为 `from utils.charls_feature_lists`
2. **裸 except**：L63、L68 使用 `except:`，应改为 `except Exception as e:` 并记录
3. **干预变量**：使用 `intervention_sleep`、`intervention_no_exercise` 等列，这些列需由外部脚本构造，若 df 中不存在会直接 `continue` 跳过，可能无输出

---

## 五、主流程已确认无问题（来自 CODE_ANALYSIS_AUDIT）

| 模块 | 检查项 | 状态 |
|------|--------|------|
| 数据加载 | USE_IMPUTED_DATA + 路径不存在时回退预处理 | ✅ |
| 插补敏感性 | 主分析用插补时单独 preprocess 得带缺失 df | ✅ |
| 临床决策支持 | sleep_adequate 反事实与管线一致 | ✅ |
| 因果 DML | T 排除于 X，仅连续变量 StandardScaler | ✅ |
| 分类变量 | CATEGORICAL_NO_SCALE，edu/gender 不缩放 | ✅ |
| 剂量反应 | sleep 缺失 dropna 排除，exercise 二分类定序 | ✅ |
| 预测模型 | Imputer/Scaler 封装在 Pipeline 内 | ✅ |
| LEAKAGE_KEYWORDS | 已修正为 memory（非 memeory） | ✅ |

---

## 六、修复优先级建议

| 优先级 | 项目 | 影响 |
|--------|------|------|
| P1 | charls_policy_forest_plot / charls_did / charls_cate / charls_ablation 导入路径 | 独立运行必现失败 |
| P2 | charls_policy_forest_plot 裸 except | 异常时难以排查 |
| P3 | 硬编码 CHARLS.csv 抽到 config | 可维护性 |
| P4 | 静默 except 增加日志 | 可观测性 |

---

## 七、主流程运行建议

- 以项目根目录为工作目录运行：`python run_all_charls_analyses.py`
- 确保 `CHARLS.csv` 或 `imputation_npj_results/pipeline_trace/step1_imputed_full.csv` 存在
- 主流程涉及的模块均使用正确的 `from utils.charls_feature_lists import ...`，无导入问题
