# 三队列 SHAP 在线演示（Streamlit）

## 文件

- **`streamlit_shap_three_cohorts.py`**（项目根目录）

## 运行

在**项目根目录**执行：

```powershell
cd "C:\Users\lenovo\Desktop\因果机器学习"
streamlit run streamlit_shap_three_cohorts.py
```

浏览器会自动打开；也可手动访问终端里提示的地址（端口可能是 **8501、8502** 等，以终端为准）。

**若内置终端一关页面就断连**：在项目根执行  
`powershell -ExecutionPolicy Bypass -File scripts/run_streamlit_three_cohorts.ps1`  
会在**新 PowerShell 窗口**里启动服务，与 Cursor 终端解耦。

## 前置条件

1. **分析数据（与训练同源）**：应用内部调用 **`utils.charls_script_data_loader.load_df_for_analysis()`**，对插补表会执行 `reapply_cohort_definition`、`prepare_exposures` 等，**不要**绕过该入口直接裸读 `step1_imputed_full.csv`，否则列集、暴露编码与主流程不一致，`Pipeline.transform` 可能失败。**预测特征矩阵**由 `get_exclude_cols` 构建，**不含** `had_comorbidity_before`（队列定义变量，与 `utils/charls_feature_lists.EXCLUDE_COLS_BASE` 一致）；但 **宽表仍须经同一 loader** 得到正确的 `baseline_group` 与模型输入列顺序。
2. **插补/预处理文件**：`config.IMPUTED_DATA_PATH`（默认 `imputation_npj_results/pipeline_trace/step1_imputed_full.csv`）或回退 `preprocessed_data/CHARLS_final_preprocessed.csv`；加载后需能得到 **`baseline_group`**（0=A, 1=B, 2=C）。

3. **各队列冠军模型**：  
   - `Cohort_A_Healthy_Prospective/01_prediction/champion_model.joblib`  
   - `Cohort_B_Depression_to_Comorbidity/01_prediction/champion_model.joblib`  
   - `Cohort_C_Cognition_to_Comorbidity/01_prediction/champion_model.joblib`  

   需先跑通 `run_all_charls_analyses.py`（或至少各队列预测步）。

4. **Python 包**：`streamlit`, `shap`, `pandas`, `numpy`, `joblib`, `matplotlib`, `scikit-learn` 等（与主项目一致）。

## 界面语言

**仅英文**：Streamlit 界面为**单语英文**（无语言切换、无 `?lang=` URL 参数）。侧栏、按钮、说明与图表轴标题均为英文。

## 冠军模型与磁盘一致性（本应用不展示）

界面已精简，**不再展示**「Table 2 记录冠军 vs joblib」校验。若需核对：仍可按仓库内 `model_complexity_efficiency.txt` 或 table2 按 AUC 排序与 `champion_model.joblib` 对照；不一致时请重跑该队列预测步。

## 页面结构

| 页面 | 内容 |
|------|------|
| **侧栏三选一** | 无单独说明页；默认打开 **Baseline healthy cohort**（Cohort A）。侧栏标签：**Baseline healthy / depression / cognitive impairment cohort**（英文）。 |
| **各队列页** | 加载该队列 `champion_model.joblib` + 子样本；特征表两列：**Variable**、**Value**，**All features** 一张表纵向滚动（列顺序与 Table 1 BPS 一致）。修改特征后点击 **Run prediction** 更新下方 **Risk** 与 **SHAP**（首次进入该队列仍会先自动预测一次）。越界输入黄色警告并裁剪。 |

## 部署给导师（公网 URL）

- **Streamlit Community Cloud**：将仓库推 GitHub，在 [share.streamlit.io](https://share.streamlit.io) 绑定仓库与入口文件 `streamlit_shap_three_cohorts.py`。  
- **注意**：公网演示**不得**挂载可识别个体数据；仅用脱敏/合成或已公开的聚合示例（当前脚本读取本地 CSV，适合**本机或课题组内网**）。

## 与论文一致

- 各队列 **独立 CPM 冠军**，与主文 **Table 3** 对应；磁盘性能表文件名仍为 `table2_{A|B|C}_main_performance.csv`（流水线历史命名）。核对冠军时请按 **AUC 排序** 或 `model_complexity_efficiency.txt`，**首行不一定是冠军**。  
- 结局表述与稿件一致：**next-wave incident DCC**（抑郁–认知共病），非笼统「两年」口径。  
- SHAP 在 **预处理后的特征空间** 上计算，逻辑对齐 `interpretability/charls_shap_analysis.py`（含 Pipeline 解包）。  
- **SVM** 等模型使用 `KernelExplainer`，首次计算可能较慢。

## 故障排除

- **找不到模型**：确认对应 `Cohort_* / 01_prediction / champion_model.joblib` 存在。  
- **插补数据缺失**：运行前置插补或关闭 `RUN_IMPUTATION_BEFORE_MAIN` 前确保已有 `step1_imputed_full.csv`。  
- **`shap` 导入失败**：`pip install shap`。
- **Connection error / 页面一直 CONNECTING**：确认 Streamlit 进程仍在运行；URL 端口与终端一致；可试 `http://127.0.0.1:端口`。项目 `.streamlit/config.toml` 已关 `runOnSave`、关 WebSocket 压缩、**`fileWatcherType = "none"`**（减轻 Windows 下监视大目录导致的不稳定）。
- **SHAP 图不在页面上**：每次运行后在项目内 **`streamlit_shap_output/shap_cohort_A.png`**（B/C 同理）生成 PNG，按页面绿色框里的路径用资源管理器打开（避免大图塞进页面导致断连）。
