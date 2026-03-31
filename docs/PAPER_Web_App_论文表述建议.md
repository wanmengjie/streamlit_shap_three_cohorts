# 论文中如何简洁表述 Web 演示工具（Streamlit）

界面已按**附录 / 补充材料**取向压平装饰（无强阴影、无渐变按钮、主内容宽约 900px，便于截图嵌入 PDF）。

## 中文稿可套用句式（按需删减）

> 我们提供基于 **Streamlit** 的交互式补充工具（本地运行脚本 `streamlit_shap_three_cohorts.py`），用于在三个基线子队列上加载 **CPM 冠军模型**（`champion_model.joblib`），通过**特征表**手动输入并查看预测概率及 **SHAP** 局部解释；后台按子样本 min–max 校验，越界时**界面警告**并裁剪后参与预测，**不用于临床决策**。

## 英文稿可套用句式

> We provide a **Streamlit**-based supplementary web tool (`streamlit_shap_three_cohorts.py`) that loads cohort-specific **CPM champion** models (`champion_model.joblib`) and supports **manual feature entry** (defaults to cohort medians); inputs are checked against subsample min–max in the background, with a **warning** and clipping if out of range; **not intended for clinical use**.

## 方法或附录中建议一笔带过的技术点

| 要点 | 一句话 |
|------|--------|
| 结局与表号 | 输出为 **next-wave incident DCC** 概率；**CPM 冠军**与主文 **Table 3** 对齐（磁盘上 `table2_*_main_performance.csv` 为流水线文件名）。 |
| 队列 | 与正文一致，A/B/C 各加载独立 `champion_model.joblib`。 |
| 特征 | 与训练同源 `load_df_for_analysis`；界面仅展示数值型列。 |
| SHAP | Tree/Linear/Kernel Explainer 依模型类型自动选择。 |
| 可复现 | 需本地数据与模型路径；公网部署勿上传可识别个体数据。 |

## 截图建议

- **主文**：1 张即可——任选一队列页的 **「风险输出 + SHAP 图」** 同框（或分栏），避免首页向导占版面。  
- **附录**：可补 1 张「说明页」交代三队列入口与免责声明。  
- 浏览器缩放约 **100%**，语言与正文一致（中/英），窗口宽度约 **1100–1280 px** 与当前版式匹配。

## 若期刊要求「无交互链接」

可写：**代码与运行说明见仓库 / 补充材料**；审稿阶段再提供内网或脱敏演示链接。
