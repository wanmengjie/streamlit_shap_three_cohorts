# 基线表（Table 1）规范

**用途**：以后做数据分析时，基线特征表统一按本规范生成，保证格式一致、可直接用于论文或报告。

---

## 1. 表结构

- **行**：变量名（按下面 2 的模块与顺序排列）
- **列**：分组 1 | 分组 2 | 分组 3 | P
- **分组列名**：由分组变量取值决定（如 Healthy / Depression only / Cognition impaired only），或自定义
- **P 列**：组间比较 p 值（可选）

---

## 2. 变量类型与展示方式

| 类型 | 展示格式 | 统计检验 | 说明 |
|------|----------|----------|------|
| **样本量** | N（整数） | — | 首行，每组一列 |
| **连续变量** | mean ± SD | Kruskal-Wallis | 偏态可改为 median [IQR]，需在配置中标明 |
| **二值变量** | n (%) | Pearson χ² | 只展示某一类（如 Female）或两类都展示（Female / Male） |
| **多分类变量** | 每水平一行，每行 n (%) | Pearson χ²（整变量一次） | 水平名可用 level 0/1/2 或编码本文字 |
| **随访结局** | n (%) 或 mean ± SD | 视变量类型 | 放在表末 |

---

## 3. 模块顺序（固定）

按以下顺序排布变量，缺失的变量自动跳过：

1. **N**  
   各组样本量。

2. **人口学（连续）**  
   Age, BMI, Waist, Family size 等，格式：mean ± SD。

3. **人口学（二值）**  
   - 性别：**Female**、**Male** 各一行，n (%)。  
   - 其余二值：如 Rural residence, Socially isolated 等，n (%)。

4. **分类变量（多水平）**  
   Education, Marital status, Self-rated health, Life satisfaction 等：每个水平一行，行名为 `变量名: level k`，格式 n (%)。P 值只在该变量第一行给出。

5. **生活方式**  
   连续（如 Sleep）用 mean ± SD；二值（如 Regular exercise, Smoking, Drinking）用 n (%)。

6. **体格与生理**  
   Puff, SBP, DBP, Pulse, Left grip, Right grip 等，mean ± SD。

7. **健康状况**  
   - 综合：Chronic disease count，mean ± SD。  
   - 各病种：Hypertension, Diabetes, Cancer, …，各一行 n (%)。  
   - 握力：Grip strength，mean ± SD。

8. **经济与自评/合成**  
   Log(income+1)、合成指标等，mean ± SD。

9. **定义变量**  
   用于分组的变量（如 Cognition score, CES-D-10），标注 “(defining)”，mean ± SD。

10. **随访结局**  
    如 Incident comorbidity，n (%)。

---

## 4. 技术约定

- **分组列**：默认列名为 `group_col` 的取值（如 0/1/2）对应的标签；可在代码中配置 `GROUP_LABELS`。
- **缺失**：按列缺失自动忽略；若某变量在数据中不存在，则不在表中出现。
- **P 值**：连续用 Kruskal-Wallis，分类用 Pearson χ²；无 scipy 时 P 列为空。
- **输出**：  
  - 主文件：`table1_baseline_characteristics.csv`  
  - 兼容旧名：`table1_academic_final.csv`  
  编码：UTF-8 with BOM（便于 Excel 打开）。

---

## 5. 新项目如何沿用

1. **复制本规范**（本文件）和实现脚本（如 `charls_table1_stats.py`）。  
2. **在脚本顶部修改配置**：  
   - 分组列名 `group_col` 与 `GROUP_LABELS`。  
   - 各模块变量列表：连续变量、二值变量、分类变量、慢性病列表等（见脚本内 `TABLE1_CONFIG`）。  
3. **数据要求**：  
   - 必须包含分组列（如 `baseline_group`）。  
   - 变量名与配置中列名一致；若编码为 0/1/2…，分类变量会以 level 0/1/2 显示，可后续替换为编码本文字。

---

## 6. 配置与代码对应关系

实现脚本内通过 **TABLE1_CONFIG** 驱动表的生成，各键与本节对应：

- **continuous**：连续变量，表内为 mean ± SD。  
- **binary**：二值变量，(列名, 展示名, 取值表示“是”)。  
- **sex_col**：性别列名；会生成 Female（值=1）与 Male（值=0）两行。  
- **categorical**：多分类变量，(列名, 展示名)。可通过 `CATEGORICAL_LEVEL_LABELS` 配置各水平的可读标签（如 marry: 0→未婚/离异/丧偶, 1→在婚），否则显示 "level k"。  
- **lifestyle_continuous** / **lifestyle_binary**：生活方式中的连续/二值。  
- **physical**：体格与生理，连续。  
- **chronic_count_col**：慢性病计数列名。  
- **chronic_disease_cols**：各病种 (列名, 展示名)。  
- **optional_continuous**：经济、合成等连续。  
- **defining_continuous**：定义变量（如认知、抑郁量表）。  
- **outcome_col** / **outcome_label**：随访结局列名与展示名。

新增或删除变量时，只改配置、不改逻辑即可保持规范一致。

---

## 7. 实现与配置位置

- **规范文档**：本文件 `TABLE1_BASELINE_SPEC.md`。
- **实现脚本**：`charls_table1_stats.py`，入口函数 `generate_baseline_table(df, output_dir, add_pvalues)`。
- **配置**：同一脚本顶部 `TABLE1_CONFIG` 与 `GROUP_COL` / `GROUP_LABELS`。新数据或新变量只需改此处，表结构与输出格式不变。
