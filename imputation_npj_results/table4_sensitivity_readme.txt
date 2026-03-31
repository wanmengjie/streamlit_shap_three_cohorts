table4_sensitivity_validation.csv / table4_diagnostics.csv 解读
============================================================
Original_mean: 插补前该列「有观测」的均值
Imputed_mean: 插补后该列「非缺失」的均值（若 N_still_missing>0 则与 Original 非同一批样本）
Mean_of_imputed_cells_only: 仅「原缺失、插补后填入」的格子均值，用于判断插补值是否偏倚
Flag: OK=正常; remaining_NaN=插补后仍缺失>100; moderate_shift=均值差异 20-50%%; large_shift=>50%%
若 large_shift 且 Mean_of_imputed_cells_only 与 Original_mean 差异大，建议检查该变量单位/波次一致性或宽表跳过列。
