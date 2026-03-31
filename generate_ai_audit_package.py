#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成给外部AI审查的精简代码包
移除独立注释行和空行，保留核心逻辑及行内注释（方法学/文献引用等上下文）
"""

import os
import re
import ast
from pathlib import Path
from datetime import datetime


class CodeAuditor:
    def __init__(self, source_dir=".", output_dir="ai_audit_package"):
        self.source = Path(source_dir)
        self.output = Path(output_dir)
        self.output.mkdir(exist_ok=True)
        self.file_stats = []

    def clean_code(self, filepath):
        """清理代码：保留逻辑与行内注释，移除独立注释行和空行"""
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        original_lines = len(content.split('\n'))

        content = re.sub(r'"""[\s\S]*?"""', '', content)
        content = re.sub(r"'''[\s\S]*?'''", '', content)

        lines = content.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('#!') or stripped.startswith('# -*-'):
                cleaned_lines.append(line)
                continue
            if stripped.startswith('#'):
                continue
            if line.strip():
                cleaned_lines.append(line)

        cleaned_content = '\n'.join(cleaned_lines)
        final_lines = len(cleaned_lines)

        return cleaned_content, original_lines, final_lines

    def extract_critical_configs(self):
        """提取关键配置参数"""
        configs = []
        config_patterns = [
            r'(RANDOM_SEED|random_state)\s*=\s*(\d+)',
            r'(n_estimators|max_depth|n_neighbors)\s*=\s*(\d+)',
            r'(honest|train_only|ANALYSIS_LOCK)\s*=\s*(True|False)',
            r'(USE_IMPUTED_DATA|IMPUTED_DATA_PATH)\s*=\s*([^\n]+)',
            r'(AGE_MIN|CESD_CUTOFF|COGNITION_CUTOFF)\s*=\s*(\d+)'
        ]

        for py_file in self.source.rglob("*.py"):
            if '__pycache__' in str(py_file):
                continue
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                for pattern in config_patterns:
                    matches = re.finditer(pattern, content)
                    for match in matches:
                        configs.append({
                            'file': py_file.name,
                            'param': match.group(1),
                            'value': match.group(2).strip()
                        })
            except Exception:
                continue

        return configs

    def generate_logic_flow(self):
        """生成逻辑流程摘要"""
        flow = []
        main_files = [
            'run_all_charls_analyses.py',
            'charls_recalculate_causal_impact.py',
            'charls_sensitivity_analysis.py'
        ]

        for fname in main_files:
            fpath = self.source / fname
            if fpath.exists():
                try:
                    with open(fpath, 'r', encoding='utf-8') as f:
                        tree = ast.parse(f.read())

                    functions = [node.name for node in ast.walk(tree)
                                if isinstance(node, ast.FunctionDef)
                                and not node.name.startswith('_')]

                    imports = []
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            imports.extend(n.name for n in node.names)
                        elif isinstance(node, ast.ImportFrom) and node.module:
                            imports.append(node.module)

                    flow.append({
                        'file': fname,
                        'functions': functions[:10],
                        'imports': list(dict.fromkeys(imports))[:8]
                    })
                except Exception:
                    flow.append({'file': fname, 'functions': [], 'imports': []})

        return flow

    def _validate_clean_files(self):
        """打包前验证 .clean 文件可解析为合法 Python AST"""
        critical = [
            'run_all_charls_analyses.clean',
            'config.clean',
            'charls_recalculate_causal_impact.clean',
            'charls_sensitivity_analysis.clean',
        ]
        failed = []
        logic_dir = self.output / "logic"
        clean_files = list(logic_dir.glob("*.clean")) if logic_dir.exists() else list(self.output.glob("*.clean"))
        for fname in clean_files:
            try:
                with open(fname, 'r', encoding='utf-8') as f:
                    ast.parse(f.read())
            except SyntaxError as e:
                failed.append((fname.name, str(e)))
        if failed:
            print("⚠️ AST 解析失败（清理可能破坏了语法）:")
            for name, err in failed:
                print(f"   - {name}: {err}")
            if any(f[0] in critical for f in failed):
                raise RuntimeError("关键 .clean 文件无法解析，请检查 clean_code 逻辑")
        else:
            print("✓ 所有 .clean 文件通过 AST 解析验证")

    def create_package(self):
        """创建审计包"""
        print("🔍 扫描Python文件...")

        logic_dir = self.output / "logic"
        context_dir = self.output / "context"
        logic_dir.mkdir(exist_ok=True)
        context_dir.mkdir(exist_ok=True)

        core_files = []
        for py_file in self.source.rglob("*.py"):
            if '__pycache__' in str(py_file) or 'test' in str(py_file).lower():
                continue

            cleaned, orig, final = self.clean_code(py_file)

            if final > 10:
                output_file = logic_dir / f"{py_file.stem}.clean"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(f"# {py_file.name} (Original: {orig} lines, Cleaned: {final} lines)\n")
                    f.write(cleaned)

                self.file_stats.append({
                    'original': py_file.name,
                    'cleaned': f"logic/{output_file.name}",
                    'lines_original': orig,
                    'lines_cleaned': final,
                    'reduction': f"{(1-final/orig)*100:.1f}%"
                })
                core_files.append(py_file.name)

        self._validate_clean_files()
        configs = self.extract_critical_configs()
        with open(context_dir / "CRITICAL_CONFIGS.txt", 'w', encoding='utf-8') as f:
            f.write("# 关键配置参数提取（含业务含义）\n\n")
            for cfg in configs:
                f.write(f"{cfg['file']:30s} | {cfg['param']:20s} | {cfg['value']}\n")

        flow = self.generate_logic_flow()
        with open(context_dir / "LOGIC_FLOW.txt", 'w', encoding='utf-8') as f:
            f.write("# 数据流与函数调用关系\n\n")
            for item in flow:
                f.write(f"## {item['file']}\n")
                f.write(f"Functions: {', '.join(item['functions'])}\n")
                f.write(f"Key Imports: {', '.join(item['imports'])}\n\n")

        self._write_domain_knowledge(context_dir)

        with open(self.output / "INDEX.txt", 'w', encoding='utf-8') as f:
            f.write(f"# CHARLS Causal ML Code Audit Package\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"Total Files: {len(self.file_stats)}\n\n")
            f.write("## 目录结构\n")
            f.write("logic/     - 代码逻辑 (.clean)\n")
            f.write("context/   - 业务上下文 (LOGIC_FLOW, CRITICAL_CONFIGS, DOMAIN_KNOWLEDGE)\n")
            f.write("AI_REVIEW_PROMPT.txt - 审查指令\n\n")
            f.write("## File Manifest\n")
            for stat in self.file_stats:
                f.write(f"{stat['cleaned']:45s} | Orig: {stat['lines_original']:4d} | "
                       f"Clean: {stat['lines_cleaned']:4d} | {stat['reduction']}\n")

        self._generate_ai_prompt()

        print(f"✅ 审计包已生成: {self.output.absolute()}")
        print(f"📊 共处理 {len(self.file_stats)} 个文件")
        print(f"📄 请查看 {self.output}/AI_REVIEW_PROMPT.txt 获取审查指令")

    def _write_domain_knowledge(self, context_dir):
        """生成 CHARLS 数据库特有说明（缺失值编码、变量定义等）"""
        content = """# CHARLS 数据库与业务上下文

## 数据来源
- China Health and Retirement Longitudinal Study (CHARLS)，中国健康与养老追踪调查
- 波次：wave 1–4，间隔约 2 年
- 年龄：≥60 岁纳入分析（AGE_MIN=60）

## 缺失值编码与处理
- 干预变量（exercise/sleep/smokev/drinkev 等）：部分代码对缺失使用 fillna(0)，即按未暴露处理；论文附录 S2 要求核心干预变量完整病例分析
- 协变量：SimpleImputer(strategy='median') 或 MissForest 插补（USE_IMPUTED_DATA 时）
- 结局 is_comorbidity_next：缺失行 dropna，不插补

## 关键变量定义
- is_depression: CES-D-10 ≥ cesd_cutoff（默认 10）
- is_cognitive_impairment: total_cognition/total_cog ≤ cognition_cutoff（默认 10）
- is_comorbidity: 抑郁与认知受损同时存在
- is_comorbidity_next: 下一波是否发生共病（主结局）
- baseline_group: 0=健康(A), 1=仅抑郁(B), 2=仅认知受损(C)
- exercise: 规律运动，二值 0/1
- sleep: 睡眠时长（小时，连续），用于预测；sleep_adequate(≥6h) 仅用于因果干预

## 三队列划分
- Cohort A: baseline_group==0，基线无抑郁无认知受损
- Cohort B: baseline_group==1，基线仅抑郁
- Cohort C: baseline_group==2，基线仅认知受损

## 泄露风险排除（LEAKAGE_KEYWORDS）
列名含 cesd/total_cog/cognition/memory/executive/score/test 等一律排除，避免用结局相关测量预测结局。
"""
        with open(context_dir / "DOMAIN_KNOWLEDGE.txt", 'w', encoding='utf-8') as f:
            f.write(content)

    def _generate_ai_prompt(self):
        """生成给外部AI的审查提示词"""
        prompt = """# AI代码审查指令（System Prompt）

你是一位严格的因果推断方法学审稿人，正在审查一项基于CHARLS队列的抑郁-认知共病研究代码。

## 审查范围
- 研究设计：三队列表型分层（Healthy/Depression-only/Cognition-impaired）
- 方法学：Causal Forest DML + PSM + PSW 三角验证
- 数据：中国健康与退休纵向研究（CHARLS）

## 审查重点（按优先级）

### P0 - 致命错误（必须修复）
1. **数据泄露（Data Leakage）**
   - 检查插补（imputation）是否在训练集上拟合并应用到测试集
   - 检查敏感性分析是否隔离了测试集（train_only参数）
   - 检查特征工程是否使用了未来信息（future leakage）

2. **因果识别假设违反**
   - 是否验证了重叠假设（Overlap/Positivity）？
   - 是否处理了未测量混杂（E-value计算）？
   - 是否确保SUTVA（无干扰假设）合理性？

### P1 - 方法学缺陷
3. **随机性与可重复性**
   - 所有随机种子是否固定（RANDOM_SEED）？
   - 交叉验证的分层是否合理（GroupKFold with ID）？
   - Bootstrap重采样次数是否充足（n≥1000）？

4. **统计效能与稳定性**
   - 小样本亚组分析是否有保护机制（n<30警告）？
   - Causal Forest是否使用Honest估计（honest=True）？
   - 极端权重是否修剪（IPW trimming）？

### P2 - 代码质量
5. **死代码与冗余**
   - 未使用的函数/变量
   - 重复计算（如DML在阶段3和阶段7是否重复）
   - 硬编码参数（应提取到config）

6. **逻辑一致性**
   - 队列划分（A/B/C）是否使用基线变量（非结局）
   - 暴露定义（exercise）是否一致
   - NNT计算是否基于正确的ATE（-0.029 vs -0.018）

## 输出格式要求

对每个发现的问题，请按以下格式报告：

```
RISK_LEVEL: CRITICAL / HIGH / MEDIUM / LOW

[优先级] 问题标题
文件: 文件名.clean
位置: 函数名 / 行号（若可定位）
描述: 简要说明问题及潜在影响
建议: 具体修复建议或替代方案
---
```

示例：
```
RISK_LEVEL: HIGH

[P0] 数据泄露风险：敏感性分析未隔离测试集
文件: run_sensitivity_scenarios.clean
位置: run_one_scenario / 约第70行
描述: df_clean 直接用于敏感性分析，未按 train_only 划分，可能导致测试集信息泄露。
建议: 传入 _get_train_subset(df) 而非 df_clean，与 compare_models 的 80/20 划分保持一致。
---
```

请逐文件审查以下文件，输出完整问题清单。

## 待审查文件清单
logic/
- config.clean: 配置参数
- run_all_charls_analyses.clean: 主流程
- charls_recalculate_causal_impact.clean: 因果推断核心
- charls_sensitivity_analysis.clean: 敏感性分析

context/（业务上下文，审查时务必参考）
- LOGIC_FLOW.txt: 数据流与函数调用关系
- CRITICAL_CONFIGS.txt: 关键参数及业务含义
- DOMAIN_KNOWLEDGE.txt: CHARLS 数据库特有说明（缺失值编码、变量定义）

请开始审查。
"""

        with open(self.output / "AI_REVIEW_PROMPT.txt", 'w', encoding='utf-8') as f:
            f.write(prompt)


if __name__ == "__main__":
    auditor = CodeAuditor()
    auditor.create_package()
