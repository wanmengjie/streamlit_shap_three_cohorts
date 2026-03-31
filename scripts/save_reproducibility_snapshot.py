# -*- coding: utf-8 -*-
"""
投稿 / 修回锁定前：一键保存可复现快照（git、config 开关、Python 环境）。

用法（在项目根目录）:
    python scripts/save_reproducibility_snapshot.py
    python scripts/save_reproducibility_snapshot.py --out results/repro_snapshots

默认输出目录: runs/repro_snapshots/YYYYMMDD_HHMMSS/
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# 项目根目录（本文件位于 scripts/）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 从 config 导出的关键开关（与论文方法学表述强相关）
CONFIG_ATTRS = [
    "RANDOM_SEED",
    "ANALYSIS_LOCK",
    "USE_IMPUTED_DATA",
    "IMPUTED_DATA_PATH",
    "RUN_IMPUTATION_BEFORE_MAIN",
    "RUN_COHORTS_ONLY",
    "MAIN_SKIP_STEPS_BEFORE_COHORTS",
    "WARN_IMPUTED_OLDER_THAN_PREPROCESSED",
    "N_MULTIPLE_IMPUTATIONS",
    "IMPUTED_MI_DIR",
    "USE_RUBIN_POOLING",
    "USE_TEMPORAL_SPLIT",
    "EXERCISE_STABLE_ONLY",
    "CAUSAL_METHOD",
    "CALIPER_PSM",
    "PARALLEL_COHORTS",
    "USE_GPU",
    "SAVE_INTERMEDIATE",
    "TARGET_COL",
    "TREATMENT_COL",
    "CESD_CUTOFF",
    "COGNITION_CUTOFF",
    "RAW_DATA_PATH",
    "OUTPUT_ROOT",
    "RESULTS_ROOT",
    "RESULTS_TABLES",
    "COHORT_A_DIR",
    "COHORT_B_DIR",
    "COHORT_C_DIR",
    # 兼容别名（若存在则一并记录）
    "AXIS_A_DIR",
    "AXIS_B_DIR",
    "AXIS_C_DIR",
]


def _run_text(cmd: list[str], cwd: Path, timeout: int = 120) -> str:
    try:
        p = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        out = (p.stdout or "") + (p.stderr or "" if p.returncode != 0 else "")
        if p.returncode != 0:
            return f"[exit {p.returncode}]\n{out}"
        return out.strip() or "(empty)"
    except FileNotFoundError:
        return "(command not found)"
    except subprocess.TimeoutExpired:
        return "(timeout)"
    except Exception as ex:
        return f"(error: {ex})"


def _collect_git(cwd: Path) -> dict:
    return {
        "git_version": _run_text(["git", "--version"], cwd),
        "branch": _run_text(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd),
        "commit_full": _run_text(["git", "rev-parse", "HEAD"], cwd),
        "commit_short": _run_text(["git", "rev-parse", "--short", "HEAD"], cwd),
        "status_porcelain": _run_text(["git", "status", "--porcelain"], cwd),
        "last_log": _run_text(["git", "log", "-1", "--oneline"], cwd),
    }


def _collect_config() -> dict:
    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        import config as cfg  # noqa: WPS433

        rows = {}
        for name in CONFIG_ATTRS:
            if hasattr(cfg, name):
                rows[name] = repr(getattr(cfg, name))
            else:
                rows[name] = "<not set>"
        return rows
    except Exception as ex:
        return {"_error": str(ex)}
    finally:
        if str(PROJECT_ROOT) in sys.path:
            sys.path.remove(str(PROJECT_ROOT))


def _pip_freeze() -> str:
    return _run_text([sys.executable, "-m", "pip", "freeze"], PROJECT_ROOT, timeout=180)


def _conda_export() -> str:
    # 若未安装 conda 或不在环境中，会失败 —— 仍写入说明性文本
    return _run_text(["conda", "env", "export"], PROJECT_ROOT, timeout=180)


def main() -> None:
    parser = argparse.ArgumentParser(description="Save reproducibility snapshot (git, config, env).")
    parser.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "runs" / "repro_snapshots"),
        help="Base directory; a timestamped subfolder will be created.",
    )
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    # Python
    py_info = {
        "executable": sys.executable,
        "version": sys.version,
    }
    (out_dir / "python_version.json").write_text(
        json.dumps(py_info, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Git
    git_data = _collect_git(PROJECT_ROOT)
    (out_dir / "git_info.json").write_text(
        json.dumps(git_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        f"branch: {git_data.get('branch', '')}",
        f"commit: {git_data.get('commit_full', '')}",
        f"short:  {git_data.get('commit_short', '')}",
        f"last:   {git_data.get('last_log', '')}",
        "",
        "status --porcelain:",
        git_data.get("status_porcelain", ""),
    ]
    (out_dir / "git_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    # Config
    cfg_dict = _collect_config()
    (out_dir / "config_snapshot.json").write_text(
        json.dumps(cfg_dict, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    cfg_lines = [f"{k} = {v}" for k, v in sorted(cfg_dict.items())]
    (out_dir / "config_snapshot.txt").write_text("\n".join(cfg_lines), encoding="utf-8")

    # Environment
    (out_dir / "environment_pip_freeze.txt").write_text(_pip_freeze(), encoding="utf-8")
    (out_dir / "environment_conda_export.txt").write_text(_conda_export(), encoding="utf-8")

    readme = f"""# Reproducibility snapshot

Generated: {stamp}
Project root: {PROJECT_ROOT}

## Files

| File | Content |
|------|---------|
| git_summary.txt / git_info.json | Branch, commit, `git status --porcelain` |
| config_snapshot.txt / .json | Selected `config.py` attributes (imputation, Rubin, causal, cohort dirs) |
| python_version.json | `sys.executable` and `sys.version` |
| environment_pip_freeze.txt | `python -m pip freeze` |
| environment_conda_export.txt | `conda env export` (may be error text if conda unused) |

## How reviewers / you can use this

1. Checkout the recorded **commit** (if clean tree) or diff against it.
2. Restore Python env from **pip freeze** and/or **conda export**.
3. Match **config_snapshot** to the paper Methods (imputation, Rubin, temporal split, etc.).

Re-run analysis from project root: `python run_all_charls_analyses.py`
"""
    (out_dir / "README_snapshot.md").write_text(readme, encoding="utf-8")

    print(f"Reproducibility snapshot written to:\n  {out_dir}")


if __name__ == "__main__":
    main()
