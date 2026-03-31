# -*- coding: utf-8 -*-
"""单独运行：清理 CatBoost 遗留的 temp_cat_* 临时目录"""
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

from causal.charls_recalculate_causal_impact import cleanup_temp_cat_dirs

if __name__ == '__main__':
    n = cleanup_temp_cat_dirs()
    if n == 0:
        print("未发现 temp_cat_* 临时目录，无需清理。")
    else:
        print(f"已清理 {n} 个临时目录。")
