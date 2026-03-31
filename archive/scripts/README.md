# 归档脚本说明

以下脚本已从 `scripts/` 移至此处，主流程不再调用。用于方法对比、实验性分析或历史参考。

**运行方式**：在项目根目录执行
```powershell
python archive/scripts/run_xxx.py
```

| 脚本 | 用途 | 主流程替代 |
|------|------|------------|
| run_interventions_linear_tlearner.py | 线性 T-Learner + PSM/PSW 对比 | run_all_interventions + run_all_axes_comparison |
| run_multi_method_causal.py | 6 种因果方法对比 | run_all_axes_comparison (PSM/PSW/XLearner) |
| run_xlearner_causal_only.py | 单干预 X-Learner 快速版 | run_xlearner_all_interventions |
| run_subgroup_and_joint_causal.py | 亚组 + 联合干预因果 | run_subgroup_analysis (主流程 06_subgroup) |
| test_grip_walking_causal.py | 握力/步行速度因果探索 | run_all_physio_causal |
