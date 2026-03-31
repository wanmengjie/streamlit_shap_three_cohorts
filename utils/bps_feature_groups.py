# -*- coding: utf-8 -*-
"""
论文 Table 1 与 `data/charls_table1_stats.BPS_SECTIONS` 一致：
按生物–心理–社会（BPS）维度对预测特征列排序并给出分组键（供 Streamlit i18n）。
"""
from __future__ import annotations

# BPS 小节标题 → streamlit_shap_three_cohorts.py 中 STRINGS 的 i18n 键
_SECTION_TITLE_TO_I18N = {
    "Biological factors": "bps_sec_bio",
    "Psychological factors": "bps_sec_psych",
    "Social factors": "bps_sec_social",
    "Lifestyle (intervenable)": "bps_sec_lifestyle",
    "Defining variables": "bps_sec_defining",
}


def _walk_bps_section(sec_cfg: dict, sk: str, out: list[tuple[str, str]], seen: set[str]) -> None:
    """将小节内出现的列按配置顺序追加到 out（去重）。"""

    def push(col: str | None) -> None:
        if col and col not in seen:
            seen.add(col)
            out.append((col, sk))

    for col, _ in sec_cfg.get("continuous", []):
        push(col)
    sc = sec_cfg.get("sex_col")
    if sc:
        push(sc)
    for t in sec_cfg.get("binary", []):
        push(t[0])
    for col, _ in sec_cfg.get("categorical", []):
        push(col)
    for t in sec_cfg.get("lifestyle_binary", []):
        push(t[0])
    for col, _ in sec_cfg.get("lifestyle_continuous", []):
        push(col)
    for col, _ in sec_cfg.get("physical", []):
        push(col)
    for t in sec_cfg.get("chronic_disease_cols", []):
        push(t[0])
    for col, _ in sec_cfg.get("optional_continuous", []):
        push(col)
    for col_or_key, _ in sec_cfg.get("defining_continuous", []):
        if col_or_key == "total_cognition":
            push("total_cognition")
            push("total_cog")
        else:
            push(col_or_key)


def bps_ordered_pairs() -> list[tuple[str, str]]:
    """(列名, i18n 分组键) 列表，顺序与 Table 1 BPS 一致；不含结局列。"""
    from data.charls_table1_stats import BPS_SECTIONS

    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for sec_title, sec_cfg in BPS_SECTIONS:
        if sec_title == "Outcome":
            continue
        sk = _SECTION_TITLE_TO_I18N.get(sec_title, "bps_sec_other")
        _walk_bps_section(sec_cfg, sk, out, seen)
    return out


def order_columns_for_editor(cols_list: list[str]) -> tuple[list[str], dict[str, str]]:
    """
    将当前模型实际使用的列名重排为 BPS 顺序，并返回每列对应的 i18n 键。
    未出现在 BPS 配置中的列（如 had_comorbidity_before）置于末尾，键为 bps_sec_other。
    """
    col_set = {str(c) for c in cols_list}
    key_by_col: dict[str, str] = {}
    ordered: list[str] = []
    for c, sk in bps_ordered_pairs():
        if c in col_set:
            ordered.append(c)
            key_by_col[c] = sk
    for c in cols_list:
        cs = str(c)
        if cs not in key_by_col:
            key_by_col[cs] = "bps_sec_other"
        if cs not in ordered:
            ordered.append(cs)
    return ordered, key_by_col
