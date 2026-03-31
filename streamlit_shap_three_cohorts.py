# -*- coding: utf-8 -*-
"""
Three-cohort CHARLS CPM champions: next-wave incident DCC risk + local SHAP (Streamlit, **English UI only**).

Aligned with `PAPER_Manuscript_Submission_Ready.md`: outcome **next-wave incident DCC**; main-text CPM **Table 3**
(pipeline filenames may still be `table2_*_main_performance.csv`).

Run from project root:
    streamlit run streamlit_shap_three_cohorts.py
"""
from __future__ import annotations

import base64
import html as html_module
import io
import logging
import os
import sys

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

logging.basicConfig(level=logging.WARNING)

try:
    import shap
except ImportError:
    shap = None

from config import (
    COHORT_A_DIR,
    COHORT_B_DIR,
    COHORT_C_DIR,
    COHORT_STEP_DIRS,
    IMPUTED_DATA_PATH,
    RANDOM_SEED,
    TARGET_COL,
    USE_IMPUTED_DATA,
)
from utils.bps_feature_groups import order_columns_for_editor
from utils.charls_feature_lists import get_exclude_cols

from data.charls_table1_stats import BPS_SECTIONS, CATEGORICAL_LEVEL_LABELS

# ---------------------------------------------------------------------------
#  UI strings (English only)
# ---------------------------------------------------------------------------
STRINGS: dict[str, str] = {
    "page_title": "Comorbidity Risk Prediction in Older Adults",
    "app_title": "Comorbidity Risk Prediction in Older Adults",
    "app_subtitle": "Depression–Cognition Comorbidity (DCC) · CHARLS Longitudinal Study · CPM + SHAP",
    "hero_main_title": "Comorbidity Risk Prediction in Older Adults",
    "brand_short": "CHARLS · supplement",
    "home_eyebrow": "Supplementary interactive demo",
    "home_tagline": "Three-cohort next-wave incident DCC prediction and local SHAP",
    "home_about_title": "Purpose",
    "home_quick_title": "Select cohort",
    "btn_enter_cohort": "Open cohort {}",
    "btn_go_sidebar": "Pick “Cohort {}” in the sidebar",
    "home_card_sub": "Table 1 N≈ {}",
    "lbl_champion": "CPM champion (main-text Table 3)",
    "lbl_estimator": "Classifier in joblib",
    "section_model": "Model",
    "expander_model": "Model metadata (Table 3 / joblib)",
    "section_controls": "Demo individual & controls",
    "pd_step1_title": "Features",
    "pd_step1_desc": "Variables ordered like Table 1 **biopsychosocial** blocks; columns not listed in BPS appear under “Other”.",
    "bps_sec_bio": "Biological",
    "bps_sec_psych": "Psychological",
    "bps_sec_social": "Social",
    "bps_sec_lifestyle": "Lifestyle",
    "bps_sec_defining": "Defining",
    "bps_sec_other": "Other",
    "btn_run_predict": "Run prediction",
    "btn_start_assessment": "Start assessment",
    "sidebar_clinical_params": "Clinical assessment parameters",
    "sidebar_clinical_hint": "**Numeric**: continuous variables (sliders). **Categorical & binary**: Table 1–aligned levels in this subsample. Values stay within subsample min–max / observed levels.",
    "sidebar_section_numeric": "Numeric (continuous)",
    "sidebar_section_categorical": "Categorical & binary",
    "section_assessment_result": "Assessment result",
    "section_i_assessment": "I. Assessment result",
    "section_ii_shap_text": "II. SHAP interpretation",
    "section_ii_shap_plot": "II. SHAP visualization",
    "section_iii_clinical": "III. Clinical recommendations",
    "section_iv_features": "IV. Current feature values",
    "section_key_factors": "Key contributing factors",
    "section_shap_expl": "SHAP explanation",
    "section_recommendations": "Clinical recommendations",
    "stale_inputs_warn": "Inputs changed since the last run. Click **Start assessment** to refresh the model output and SHAP plot.",
    "risk_criteria_expander": "Risk classification criteria",
    "risk_criteria_body": """
**Low** — predicted probability under 20%. **Moderate** — 20–40%. **High** — 40% or higher.  
These cutoffs are for **demo visualization only**; they are not validated clinical thresholds. Outcome: **next-wave incident DCC** (CPM from main-text Table 3).
    """,
    "risk_tier_low": "Low risk",
    "risk_tier_med": "Moderate risk",
    "risk_tier_high": "High risk",
    "factor_col_feature": "Clinical factor",
    "factor_col_impact": "Impact direction",
    "impact_increase": "Increases DCC risk",
    "impact_decrease": "Decreases DCC risk",
    "dcc_probability_line": "Next-wave DCC probability: {:.1%}",
    "predict_click_prompt": "Click **Start assessment** below to run the model and show risk, key factors, and the SHAP explanation.",
    "manual_range_warn_intro": "These inputs are **outside** the subsample min–max and were **clipped** for prediction & SHAP:",
    "manual_range_warn_more": "… **{}** more variable(s) clipped (not listed).",
    "pd_step2_title": "Risk output",
    "pd_step2_desc": "CPM predicted probability of **next-wave incident DCC** (illustration only; not for clinical use).",
    "pd_step3_title": "SHAP",
    "pd_step3_desc": "Local SHAP for the positive class (next-wave DCC).",
    "tip_live_update": "Edits on the left refresh the outputs below.",
    "input_mode": "Feature source",
    "mode_sample": "Pick one row from subsample",
    "mode_manual": "Manual entry (vertical list)",
    "manual_hint": "Each **row** is one feature: names are read-only on the left; edit values on the right. Scroll vertically to review all inputs.",
    "manual_subtitle": "Feature values",
    "manual_n_feat": "**{}** numeric features (same columns as training)",
    "manual_filter": "Filter by variable name (optional, case-insensitive)",
    "manual_filter_empty": "No match — showing all variables.",
    "manual_col_var": "Variable",
    "manual_col_val": "Value",
    "btn_load_row": "Load slider row",
    "btn_load_median": "Load cohort medians",
    "manual_no_true_y": "No observed outcome label for this synthetic profile.",
    "manual_no_true_y_detail": "The displayed probability is the model’s predicted risk of next-wave incident DCC only; it is not a clinical diagnosis and is not validated against an individual’s true future outcome in this demo.",
    "section_output": "Prediction",
    "sidebar_menu": "Menu",
    "sidebar_nav": "Page",
    "nav_a": "Baseline healthy cohort",
    "nav_b": "Baseline depression cohort",
    "nav_c": "Baseline cognitive impairment cohort",
    "sidebar_pick_cohort": "Choose cohort (sidebar)",
    "sidebar_root": "Project root",
    "sidebar_run_hint": "Run",
    "home_header": "Three-cohort prediction and local SHAP",
    "home_body": """
Matches the manuscript’s **three incident-eligible baseline phenotype cohorts**. Each loads its **CPM champion** (`champion_model.joblib`, main-text **Table 3**) for **next-wave incident depression–cognition comorbidity (DCC)**.

- Not for clinical use.  
- Demo: random subsample of the **MICE-completed** analysis table; do not upload identifiable records if hosted publicly.  
- Missing files: run `run_all_charls_analyses.py` (prediction step per cohort).
    """,
    "home_wizard_title": "Cohort mapping (main-text definitions)",
    "wiz_dep": "Baseline CES-D ≥ 10 (depression)?",
    "wiz_cog": "Baseline global cognition ≤ 10 (impairment)?",
    "wiz_opt_unknown": "Unsure / N/A",
    "wiz_opt_no": "No",
    "wiz_opt_yes": "Yes",
    "wiz_suggest_a": "→ Use **Cohort A (healthy)** model.",
    "wiz_suggest_b": "→ Use **Cohort B (depression only)** model.",
    "wiz_suggest_c": "→ Use **Cohort C (cognitive impairment only)** model.",
    "wiz_both_yes": "Baseline comorbidity is outside the **incident three-cohort** design; no model here.",
    "wiz_else": "Choose **Cohort A / B / C** in the sidebar.",
    "sidebar_suggest": "Wizard suggests",
    "cohort_caption_n": "｜ Table 1 ref. N ≈ {}",
    "cohort_ref_n_only": "Table 1 reference N ≈ {}",
    "err_no_shap": "Install shap: `pip install shap`",
    "err_no_model": "Champion model not found:\n`{}`\nRun the cohort prediction step first.",
    "err_no_data": "Cannot load analysis data (same as training via `load_df_for_analysis`) or missing `baseline_group`:\n`{}`",
    "err_load_analysis": "Failed to load analysis data: {}",
    "err_empty_x": "Numeric feature matrix is empty.",
    "err_transform": "Preprocessor transform failed: {}",
    "err_explainer": "SHAP explainer failed: {}",
    "err_predict": "predict_proba failed: {}",
    "err_shap": "SHAP computation failed: {}",
    "info_expected_champion": "**Table 3 / CPM recorded champion**: {}",
    "info_loaded_estimator": "**Classifier inside joblib**: `{}`",
    "warn_mismatch": "**Mismatch**: recorded champion “{}” vs joblib classifier “{}”. Re-run the **full prediction step** for this cohort (`run_all_charls_analyses.py` or `scripts/run_cpm_table2_only.py --full --cohort X`) to refresh `champion_model.joblib`. The **first row** of `table2_*_main_performance.csv` (pipeline file backing **main-text Table 3**) is **not** necessarily the champion; this app uses **AUC-sorted** performance tables or `model_complexity_efficiency.txt`.",
    "section_pick": "Select a demo individual",
    "seed_label": "Random seed (cohort {})",
    "btn_random": "Random individual",
    "slider_idx": "Or choose row index (within subsample)",
    "section_pred": "Predicted risk",
    "metric_risk": "Next-wave incident DCC risk (model output)",
    "cap_true_y": "Row label `{}` = {} (for sanity check only)",
    "section_shap": "Local SHAP (positive class · top features)",
    "shap_spin": "Computing SHAP… (SVM may be slow)",
    "shap_dim_warn": "SHAP length ≠ features; showing overlap only.",
    "topk": "Show top-K features",
    "expander_feats": "Feature values (raw input space, partial)",
    "btn_download_shap_png": "Download SHAP figure (PNG)",
    "col_value": "value",
    "shap_xlabel": "SHAP value (positive → higher next-wave DCC risk)",
    "shap_title": "Local SHAP | Cohort {} | row {}",
    "shap_title_manual": "Local SHAP | Cohort {}",
    "cohort_kicker": "Cohort {}",
    "shap_trouble_title": "Why SHAP or the score might seem broken",
    "shap_trouble_body": """
**What this app does**  
- **Predicted risk** = `predict_proba` for **one synthetic row** built from your **sidebar sliders** (not a real respondent ID).  
- **SHAP** = local attribution for **that same row** in the **preprocessed feature space** (after the model’s `Pipeline` transform). It explains the classifier output, **not** a causal treatment effect.

**If you see `Failed to fetch dynamically imported module` (red box)**  
- That is a **browser / Streamlit frontend** issue loading JavaScript chunks — **not** a Python SHAP failure.  
- Try: **clear site data** for `localhost:8501`, **Ctrl+F5**, or an **InPrivate/Incognito** window; use `http://127.0.0.1:8501` instead of `localhost`.  
- Reinstall Streamlit: `pip install -U streamlit` and restart the server.  
- The SHAP bar chart is embedded as **one `st.markdown` HTML block** (`<img src="data:image/png;base64,…">`) — **no** `st.image`, `st.pyplot`, or `st.download_button`, because those often trigger lazy-loaded JS (`static/js/index.*.js`) and **Failed to fetch dynamically imported module**. Save via **right-click → Save image as…**.

**If the plot area is blank**  
- **Hard-refresh** (Ctrl+F5) or a **private window** after code changes.  
- If you use **Kernel SHAP** (e.g. SVM), the first run can take **many minutes**; do not stop the terminal.

**If the score does not update**  
- After moving sliders, click **Start assessment** again (or **Load cohort medians**, which also triggers a refresh).

**If the Streamlit tab shows “Connection error”**  
- The server stopped (Ctrl+C, crash, or sleep). Restart: `streamlit run streamlit_shap_three_cohorts.py`.

**GPU XGBoost note**  
- If the champion is **XGBoost on CUDA** while SHAP uses **CPU** arrays, you may see warnings or rare failures; check the terminal for tracebacks.
    """,
    # SHAP-aligned narrative (same top-|SHAP| order as the bar chart)
    "section_shap_suggestions": "Takeaways from this SHAP ranking",
    "shap_sugg_blurb": "Local attributions for this profile (not causal); 🔴 increases modeled risk, 🔵 decreases it — ordered risk-up first, then risk-down.",
    "shap_sugg_up": "🔴 Increases modeled DCC probability for this instance.",
    "shap_sugg_dn": "🔵 Decreases modeled DCC probability for this instance.",
    "shap_sugg_val": "input: {}",
}


def t(key: str, *args) -> str:
    s = STRINGS.get(key, key)
    if args:
        try:
            return s.format(*args)
        except Exception:
            return s
    return s


def _strip_feat_name(name: str) -> str:
    return str(name).replace("pass__", "").replace("num__", "").strip().lower()


# UI: short academic glosses for SHAP interpretation (extend as needed)
_FEATURE_SHAP_NOTES: dict[str, str] = {
    "retire": "Retirement status may co-occur with changes in daily structure, social roles, and mental health, which population studies link to mood and cognition outcomes.",
    "edu": "Education often proxies cognitive reserve and health literacy; higher attainment is frequently associated with more favorable cognitive and depressive symptom trajectories in older cohorts.",
    "age": "Chronological age is a core demographic correlate of both late-life depression and cognitive decline in longitudinal surveys.",
    "sleep": "Self-reported sleep duration and quality overlap with mood regulation and cognitive performance; extremes may signal risk relevant to DCC models.",
    "iadl": "Instrumental ADL limitation reflects functional dependence; greater limitation often tracks higher healthcare needs and neuropsychiatric burden.",
    "adlab_c": "Basic ADL items capture severe functional impairment, which is strongly associated with multimorbidity and cognitive vulnerability.",
    "income_total": "Household resources relate to care access, stress, and health behaviours, which can confound and mediate mental and cognitive outcomes.",
    "rural": "Urban–rural residence may capture healthcare access, social density, and environmental factors that differ across mental health profiles.",
    "arthre": "Arthritis and chronic pain can impair activity and sleep, indirectly affecting mood and perceived cognitive function.",
    "disability": "Self-reported disability aggregates physical and functional constraints often correlated with depression and cognitive complaints.",
    "cesd": "CES-D–style depressive symptom scores directly inform the depression component of comorbidity risk in phenotype-based cohorts.",
    "cesd10": "Short-form depressive symptom scales remain sensitive to subthreshold mood burden relevant to incident comorbidity.",
    "cognition": "Global cognition summary scores reflect the cognitive impairment axis used to define baseline risk groups and incident comorbidity.",
    "tot_cognition": "Composite cognition measures integrate multiple domains and are standard predictors in ageing mental health models.",
    "bmi": "Body mass index links to cardiometabolic health, frailty, and inflammation pathways implicated in mood and cognition.",
    "smokev": "Smoking status relates to vascular and inflammatory risk with cross-sectional and longitudinal mental health associations.",
    "drinkev": "Alcohol use patterns interact with sleep, mood, and cognition; non-linear relationships are common in older adults.",
    "exercise": "Physical activity is a modifiable behaviour repeatedly associated with lower depressive symptoms and better cognitive trajectories.",
    "socwk": "Social activity frequency may buffer isolation and stress, with protective associations in geriatric mental health research.",
    "child": "Number of children can proxy informal support networks affecting care and psychological wellbeing in ageing.",
    "married": "Marital status reflects cohabitation and support; partnership loss is a known stressor for mood in late life.",
    "hibpe": "Hypertension marks vascular burden relevant to both vascular cognitive impairment and affective symptoms.",
    "diabe": "Diabetes associates with metabolic and microvascular pathways linked to depression and cognitive decline.",
    "hearte": "Heart disease signals systemic cardiovascular risk overlapping with mood and cognition in older populations.",
    "stroke": "History of stroke is a strong marker of brain injury and subsequent risk for depression and dementia spectrum outcomes.",
    "psychological": "Summary psychological distress scales aggregate anxiety and depressive facets used in biopsychosocial risk models.",
}


def _feature_academic_note(feat_name: str) -> str:
    k = _strip_feat_name(feat_name)
    return _FEATURE_SHAP_NOTES.get(
        k,
        "This variable enters the CPM as coded in training; its SHAP contribution reflects model dependence for this profile—integrate with domain knowledge and clinical context.",
    )


def _html_simple_table(df: pd.DataFrame, *, max_rows: int = 200) -> str:
    """Plain HTML table for use inside st.markdown(..., unsafe_allow_html=True).
    Avoids st.dataframe, which can break when Streamlit fails to load JS chunks
    (TypeError: Failed to fetch dynamically imported module …).
    """
    if df is None or len(df) == 0:
        return "<p><em>No rows.</em></p>"
    sub = df.head(max_rows)
    th = "".join(f"<th>{html_module.escape(str(c))}</th>" for c in sub.columns)
    trs: list[str] = []
    for _, row in sub.iterrows():
        tds = "".join(f"<td>{html_module.escape(str(v))}</td>" for v in row)
        trs.append(f"<tr>{tds}</tr>")
    return (
        '<table style="border-collapse:collapse;width:100%;font-size:0.85rem;'
        'border:1px solid #e0ddd8;">'
        f"<thead><tr>{th}</tr></thead><tbody>{''.join(trs)}</tbody></table>"
    )


def _html_details(summary: str, body_text: str) -> str:
    """Native <details> instead of st.expander (avoids extra Streamlit JS chunks)."""
    body = html_module.escape(body_text.strip()).replace("\n", "<br/>")
    return (
        '<details style="margin:0.5rem 0;font-size:0.82rem;line-height:1.45;">'
        f'<summary style="cursor:pointer;font-weight:600;">{html_module.escape(summary)}</summary>'
        f'<div style="margin-top:0.45rem;">{body}</div></details>'
    )


def _html_details_rich(summary: str, inner_html: str) -> str:
    """<details> with trusted inner HTML (e.g. tables we built ourselves)."""
    return (
        '<details style="margin:0.75rem 0;font-size:0.85rem;">'
        f'<summary style="cursor:pointer;font-weight:600;">{html_module.escape(summary)}</summary>'
        f'<div style="margin-top:0.5rem;">{inner_html}</div></details>'
    )


def _row_value_for_shap_feature(shap_col: str, row_values: dict[str, float]) -> str | None:
    """Map SHAP / preprocessor column name back to sidebar row value when possible."""
    raw = str(shap_col)
    stripped = raw.replace("pass__", "").replace("num__", "")
    for k in (raw, stripped):
        if k in row_values:
            try:
                return f"{float(row_values[k]):.4g}"
            except (TypeError, ValueError):
                return str(row_values[k])
    return None


def _html_shap_derived_suggestions(
    vec: np.ndarray,
    names: list,
    row_values: dict[str, float] | None,
    top_k: int,
) -> str:
    """UI: risk-up (red) first, then risk-down (blue); plain-language note per feature."""
    v = np.nan_to_num(np.asarray(vec, dtype=np.float64).ravel(), nan=0.0, posinf=0.0, neginf=0.0)
    n_all = min(int(v.size), len(names))
    if n_all <= 0 or top_k <= 0:
        return ""
    k = int(min(int(top_k), n_all))
    order_abs = np.argsort(np.abs(v[:n_all]))[-k:][::-1]
    pos_j = [int(j) for j in order_abs if v[j] >= 0.0]
    neg_j = [int(j) for j in order_abs if v[j] < 0.0]
    pos_j.sort(key=lambda j: abs(float(v[j])), reverse=True)
    neg_j.sort(key=lambda j: abs(float(v[j])), reverse=True)
    ordered = pos_j + neg_j
    rv = row_values or {}
    items: list[str] = []
    for j in ordered:
        nm = str(names[j])
        s = float(v[j])
        esc = html_module.escape(nm)
        disp = _row_value_for_shap_feature(nm, rv)
        val_line = ""
        if disp is not None:
            val_line = (
                f'<div style="font-size:0.78rem;color:var(--ink-soft);margin:0.15rem 0 0.2rem;">'
                f'Current value: <strong>{html_module.escape(disp)}</strong></div>'
            )
        dir_html = t("shap_sugg_up") if s >= 0.0 else t("shap_sugg_dn")
        note = html_module.escape(_feature_academic_note(nm))
        border = "#C62828" if s >= 0.0 else "#1565C0"
        items.append(
            f'<li style="margin-bottom:0.75rem;padding-left:0.5rem;border-left:3px solid {border};list-style:none;">'
            f'<div style="font-weight:700;color:var(--ink);">{esc}</div>'
            f'{val_line}'
            f'<div style="font-size:0.8rem;margin:0.2rem 0 0.15rem;">{dir_html}</div>'
            f'<div style="font-size:0.76rem;color:var(--ink-muted);line-height:1.45;">{note}</div>'
            f"</li>"
        )
    blurb = (
        f'<p style="margin:0 0 0.65rem;font-size:0.78rem;color:var(--ink-soft);line-height:1.5;">'
        f'{t("shap_sugg_blurb")}</p>'
    )
    ul = '<ul style="margin:0;padding-left:0;">' + "".join(items) + "</ul>"
    return blurb + ul


def _top_shap_triple(vec: np.ndarray, names: list) -> list[tuple[str, float]]:
    """Top 3 features by |SHAP| for dynamic clinical text."""
    v = np.nan_to_num(np.asarray(vec, dtype=np.float64).ravel(), nan=0.0, posinf=0.0, neginf=0.0)
    n = min(int(v.size), len(names))
    if n == 0:
        return []
    k = min(3, n)
    idx = np.argsort(np.abs(v[:n]))[-k:][::-1]
    return [(str(names[int(i)]), float(v[int(i)])) for i in idx]


def _recommendations_dynamic_html(proba: float, vec: np.ndarray, names: list) -> str:
    """Tier-specific guidance + top-3 SHAP names; keeps academic disclaimer."""
    top3 = _top_shap_triple(vec, names)
    top_txt = ", ".join(html_module.escape(n) for n, _ in top3) if top3 else "—"

    base = [
        "<strong>Not for clinical use.</strong> Research demonstration only; does not diagnose or treat.",
        '<strong>Model output only.</strong> Predicted next-wave incident DCC risk for this profile; not a confirmed individual prognosis.',
    ]

    if proba >= 0.40:
        extra = [
            "<strong>High modeled risk (demo tier):</strong> Favor closer follow-up and comprehensive geriatric assessment when clinically appropriate; discuss mood, cognition, function, and medications.",
            f"<strong>Top |SHAP| features this run:</strong> {top_txt}. Treat positive-SHAP items as <em>model-highlighted</em> discussion points for higher predicted probability—not causal treatment targets.",
            "<strong>Multidimensional framing:</strong> Combine sleep, physical activity, social connection, safety (falls/medications), and chronic-disease control within usual-care pathways.",
        ]
    elif proba >= 0.20:
        extra = [
            "<strong>Moderate modeled risk (demo tier):</strong> Consider structured mood and cognition screening in routine visits; reinforce protective behaviours.",
            f"<strong>Feature focus from this SHAP profile:</strong> {top_txt}. Prioritize clinical review of factors with positive SHAP if they match your assessment.",
            "<strong>Intervention emphasis:</strong> Address modifiable drivers (e.g. sleep, activity, pain, social engagement) where feasible alongside standard care.",
        ]
    else:
        neg = [n for n, sv in top3 if sv < 0]
        neg_s = ", ".join(html_module.escape(x) for x in neg[:3]) if neg else "patterns in the SHAP list"
        extra = [
            "<strong>Lower modeled probability (demo tier):</strong> Continue routine preventive care; uncertainty remains—probability is not zero.",
            f"<strong>Protective patterns in this run:</strong> Features with negative local SHAP (e.g. {neg_s}) are associated with <em>lower</em> modeled risk here; sustaining healthy behaviours still matters.",
            f"<strong>Monitoring:</strong> Keep {top_txt} in context for future visits if symptoms or function change.",
        ]

    items = "".join(f'<li style="margin-bottom:0.35rem;">{x}</li>' for x in (base + extra))
    return f'<ol style="margin:0;padding-left:1.25rem;">{items}</ol>'


# ---------------------------------------------------------------------------
#  UI: CSS + 小组件（美观版式）
# ---------------------------------------------------------------------------
def _inject_app_css() -> None:
    # ── 完整 CSS 系统 ──────────────────────────────────────────────────────────
    # 配色：主色 #2D5031（品牌绿）、风险红 #C62828、保护蓝 #1565C0
    # 字体：Source Sans 3 → Microsoft YaHei → Segoe UI（中英文通用）
    st.markdown(
        """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:ital,opsz,wght@0,8..32,300;0,8..32,400;0,8..32,600;0,8..32,700&display=swap');

  /* ── 设计令牌 ─────────────────────────────────────────────────────── */
  :root {
    --brand:        #2D5031;
    --brand-dk:     #1e3720;
    --brand-lt:     #3d6b42;
    --risk-red:     #C62828;
    --risk-red-lt:  #fff5f5;
    --protect-blue: #1565C0;
    --ink:          #1a1917;
    --ink-muted:    #52504c;
    --ink-soft:     #7a756f;
    --paper:        #F5F4F2;
    --surface:      #ffffff;
    --line:         #E0E0E0;
    --line-strong:  #c8c4bc;
    --r:            8px;
    --r-sm:         5px;
    --sh-sm:        0 1px 4px rgba(0,0,0,.07),0 1px 2px rgba(0,0,0,.04);
    --sh:           0 3px 12px rgba(0,0,0,.10),0 1px 4px rgba(0,0,0,.06);
  }

  /* ── 全局字体与背景 ──────────────────────────────────────────────── */
  html, body, [class*="stApp"] {
    font-family: 'Source Sans 3','Microsoft YaHei','PingFang SC','Segoe UI',sans-serif;
  }
  .stApp { background: var(--paper); }
  .main .block-container {
    padding-top: 0;
    padding-bottom: 1rem;
    max-width: 1380px;
    padding-left: 1.25rem;
    padding-right: 1.25rem;
  }
  .main .stMarkdown p, .main p { color: var(--ink-muted); font-size: 0.92rem; line-height: 1.6; }
  .main h1 { font-size: 1.4rem; font-weight: 700; color: var(--ink); }
  .main h2 { font-size: 1.05rem; font-weight: 700; color: var(--ink); }
  .main h3 { font-size: 0.88rem; font-weight: 600; color: var(--ink-muted); margin: 0.75rem 0 0.3rem; }

  /* ── 品牌顶栏 ─────────────────────────────────────────────────────── */
  .app-header {
    background: var(--brand);
    border-radius: 0;
    padding: 0.55rem 1.25rem;
    margin: 0 -1.25rem 0.75rem -1.25rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }
  .app-header-icon { font-size: 1.55rem; line-height: 1; flex-shrink: 0; }
  .app-header-title {
    font-size: 1.12rem; font-weight: 700; color: #fff;
    letter-spacing: -0.01em; line-height: 1.2;
  }
  .app-header-sub {
    font-size: 0.65rem; color: rgba(255,255,255,.62);
    margin-top: 0.1rem; line-height: 1.35;
    letter-spacing: 0.02em;
  }

  /* ── 侧边栏 ─────────────────────────────────────────────────────── */
  [data-testid="stSidebar"] { background: #F0F0F0 !important; border-right: 1px solid var(--line); }
  [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p,
  [data-testid="stSidebar"] span, [data-testid="stSidebar"] label { color: #363330 !important; }
  [data-testid="stSidebar"] .stSlider label { font-size: 0.74rem !important; font-weight: 600 !important; color: #4a4744 !important; }
  [data-testid="stSidebar"] .stSelectbox label { font-size: 0.74rem !important; font-weight: 600 !important; color: #4a4744 !important; }
  [data-testid="stSidebar"] [role="radiogroup"] label { font-weight: 500 !important; font-size: 0.9rem !important; }
  [data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: var(--surface) !important; border-color: var(--line-strong) !important; border-radius: var(--r-sm) !important;
  }
  /* 侧栏品牌名 */
  .pd-sidebar-top {
    font-size: 0.7rem; font-weight: 700; color: var(--brand);
    padding: 0.15rem 0 0.6rem; letter-spacing: 0.08em; text-transform: uppercase;
    border-bottom: 2px solid var(--brand); margin-bottom: 0.6rem;
  }
  /* 侧栏分组标题 */
  .sb-group-hdr {
    font-size: 0.65rem; font-weight: 700; color: var(--ink-soft);
    letter-spacing: 0.08em; text-transform: uppercase;
    margin: 1.2rem 0 0.45rem; padding-bottom: 0.3rem;
    border-bottom: 1px solid var(--line);
  }

  /* ── 队列条 ─────────────────────────────────────────────────────── */
  .cohort-bar {
    display: flex; align-items: center; gap: 0.9rem;
    background: var(--surface);
    border: 1px solid var(--line);
    border-left: 4px solid var(--coh-accent, #8f9188);
    border-radius: var(--r); padding: 0.65rem 1.1rem;
    margin-bottom: 1rem; box-shadow: var(--sh-sm);
  }
  .cohort-bar-name { font-size: 1rem; font-weight: 700; color: var(--ink); letter-spacing: -0.01em; }
  .cohort-bar-meta { font-size: 0.78rem; color: var(--ink-soft); }

  /* ── 分区标题 ────────────────────────────────────────────────────── */
  .sec-title {
    font-size: 0.65rem; font-weight: 700; color: var(--ink-soft);
    letter-spacing: 0.1em; text-transform: uppercase;
    margin: 1.35rem 0 0.7rem; display: flex; align-items: center; gap: 0.45rem;
  }
  .sec-title::before {
    content: ''; flex-shrink: 0; width: 3px; height: 0.7em;
    background: var(--brand); border-radius: 2px;
  }
  .sec-title.first { margin-top: 0.25rem; }
  .sec-title.sec-roman { font-size: 0.62rem; letter-spacing: 0.12em; }

  /* ── 卡片容器（统一 8px 圆角 + 间距） ───────────────────────────── */
  .card {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: var(--r);
    padding: 1.2rem 1.3rem;
    box-shadow: var(--sh-sm);
    margin-bottom: 16px;
  }
  .card:last-child { margin-bottom: 0; }

  /* ── 风险展示 ────────────────────────────────────────────────────── */
  .risk-number {
    font-size: 4.35rem; font-weight: 800; letter-spacing: -0.05em;
    line-height: 1; display: block; margin-bottom: 0.15rem;
  }
  .risk-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.07em;
    text-transform: uppercase; padding: 0.25rem 0.65rem;
    border-radius: 99px; margin-bottom: 0.75rem;
  }
  .risk-badge::before { content: ''; width: 6px; height: 6px; border-radius: 50%; background: currentColor; flex-shrink: 0; }
  .risk-low  { color: #1e5c30; }
  .risk-med  { color: #8a4500; }
  .risk-high { color: var(--risk-red); }
  .risk-badge.risk-low  { background: #edf7f0; color: #1e5c30; }
  .risk-badge.risk-med  { background: #fff3e0; color: #8a4500; }
  .risk-badge.risk-high { background: var(--risk-red-lt); color: var(--risk-red); }
  .risk-footnote { font-size: 0.76rem; color: var(--ink-soft); line-height: 1.5; margin-top: 0.6rem; }

  /* ── 按钮 ────────────────────────────────────────────────────────── */
  div.stButton > button {
    border-radius: var(--r-sm) !important; box-shadow: none !important;
    font-weight: 700 !important; font-size: 0.88rem !important;
    transition: background .15s, transform .1s !important;
  }
  /* Start assessment: 主色填充 */
  .main .run-btn div.stButton > button {
    background: var(--brand) !important; color: #fff !important;
    border: none !important; padding: 0.5rem 1.6rem !important;
    font-size: 0.9rem !important;
  }
  .main .run-btn div.stButton > button:hover { background: var(--brand-dk) !important; }
  /* Load medians: 轮廓 */
  [data-testid="stSidebar"] div.stButton > button {
    background: transparent !important; color: var(--brand) !important;
    border: 1.5px solid var(--brand) !important;
    font-size: 0.82rem !important; padding: 0.35rem 0.9rem !important;
  }
  [data-testid="stSidebar"] div.stButton > button:hover {
    background: var(--brand) !important; color: #fff !important;
  }

  /* ── 提示文字 ─────────────────────────────────────────────────────── */
  .run-hint { font-size: 0.85rem; color: var(--ink-soft); line-height: 1.5; margin-top: 0.35rem; }

  /* ── SHAP wrapper ────────────────────────────────────────────────── */
  .shap-inline-chart img {
    max-width: 100% !important; height: auto !important;
    border-radius: var(--r) !important; display: block !important;
    border: 1px solid var(--line) !important;
  }
  .slider-median-hint {
    font-size: 0.68rem !important; color: #6d6a66 !important;
    margin: -0.35rem 0 0.55rem 0 !important; line-height: 1.35 !important;
  }

  /* ── Expander / details ─────────────────────────────────────────── */
  [data-testid="stExpander"] details { border: 1px solid var(--line) !important; border-radius: var(--r-sm) !important; background: var(--surface) !important; }
  [data-testid="stExpander"] summary { font-weight: 600 !important; font-size: 0.8rem !important; color: var(--ink-muted) !important; }

  /* ── 其余 Streamlit 组件修正 ──────────────────────────────────────── */
  [data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--surface) !important; border: 1px solid var(--line) !important;
    border-radius: var(--r) !important; box-shadow: var(--sh-sm) !important;
    padding: 0.75rem 1rem !important;
  }
  hr[data-testid="stHorizontalRule"] { margin: 0.65rem 0 !important; border: none !important; border-top: 1px solid var(--line) !important; }
  div[data-testid="column"] { padding-top: 0.05rem; padding-bottom: 0.05rem; }
  [data-testid="stMetricValue"] { font-size: 1.8rem !important; font-weight: 700 !important; color: var(--ink) !important; }
  [data-testid="stMetricLabel"] { font-size: 0.76rem !important; font-weight: 600 !important; color: var(--ink-soft) !important; }
  [data-testid="stImage"] img { max-width:100%!important; opacity:1!important; visibility:visible!important; display:block!important; }
  div[data-testid="stDataEditor"] { border: 1px solid var(--line) !important; border-radius: var(--r-sm) !important; box-shadow: none !important; }
  .main .stRadio label, .main .stSlider label, .main .stNumberInput label { color: var(--ink-muted) !important; }

  /* ── legacy class aliases (used by older code paths) ─────────────── */
  .assessment-result-card  { background:var(--surface); border:1px solid var(--line); border-radius:var(--r); padding:1.2rem 1.3rem; box-shadow:var(--sh-sm); }
  .clinical-section-title  { font-size:0.65rem; font-weight:700; color:var(--ink-soft); margin:1.35rem 0 0.7rem; letter-spacing:.1em; text-transform:uppercase; }
  .clinical-subtitle       { font-size:0.88rem; color:var(--ink-soft); line-height:1.5; }
  .cohort-label, .cohort-header-bar { display:none; }   /* replaced by .cohort-bar */
  .risk-pill               { display:block; font-size:3.6rem; font-weight:800; letter-spacing:-.04em; line-height:1; }
  .risk-tier-line          { display:inline-flex; align-items:center; gap:.35rem; font-size:.7rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase; margin:.4rem 0 .55rem; }
  .risk-tier-line::before  { content:''; width:7px; height:7px; border-radius:50%; background:currentColor; flex-shrink:0; }
  .risk-tier-line.risk-high{ color:var(--risk-red); }
  .risk-tier-line.risk-med { color:#8a4500; }
  .risk-tier-line.risk-low { color:#1e5c30; }
  .pd-sidebar-top          { font-size:.7rem; font-weight:700; color:var(--brand); padding:.15rem 0 .6rem; letter-spacing:.08em; text-transform:uppercase; border-bottom:2px solid var(--brand); margin-bottom:.6rem; }
  .feat-count, .stat-grid, .stat-card, .input-hint-box,
  .paper-step-h5, .paper-step-desc, .pd-section-label { font-size:0.82rem; color:var(--ink-muted); }
</style>
        """,
        unsafe_allow_html=True,
    )


def _cohort_hero(title: str, subtitle: str, accent: str, cohort_key: str) -> None:
    a = accent if isinstance(accent, str) and accent.startswith("#") else "#8f9188"
    kicker = t("cohort_kicker", cohort_key)
    st.markdown(
        f"""
<div class="pd-cohort-hero" style="--coh-accent: {a};">
  <p class="pd-cohort-kicker">{html_module.escape(kicker)}</p>
  <h2>{html_module.escape(title)}</h2>
  <p class="pd-cohort-sub">{html_module.escape(subtitle)}</p>
</div>
        """,
        unsafe_allow_html=True,
    )


def _section_title(text: str) -> None:
    st.markdown(f"### {text}")


def _pd_flow_step(step_no: int, title: str, desc: str, accent_hex: str) -> None:
    """Paper-oriented: numbered title + optional short description."""
    _ = accent_hex
    desc_html = f'<p class="paper-step-desc">{html_module.escape(desc)}</p>' if desc else ""
    st.markdown(
        f'<h5 class="paper-step-h5">{int(step_no)}. {html_module.escape(title)}</h5>'
        + desc_html,
        unsafe_allow_html=True,
    )


def _prediction_panel_container():
    """Streamlit ≥1.33 可用 border=True 形成「结果卡」。"""
    try:
        return st.container(border=True)
    except TypeError:
        return st.container()


def _configure_matplotlib_fonts() -> None:
    """避免中文标题/标签在图中显示为方块；英文模式仍保留中文字体回退以防混排。"""
    import matplotlib as mpl

    mpl.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "PingFang SC",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "Segoe UI",
        "DejaVu Sans",
    ]
    mpl.rcParams["axes.unicode_minus"] = False


def _apply_plot_style(accent: str) -> None:
    _configure_matplotlib_fonts()
    # 全局 rcParams：白底学术风格
    plt.rcParams.update(
        {
            "figure.facecolor": "#ffffff",
            "axes.facecolor":   "#ffffff",
            "axes.edgecolor":   "#e0e0e0",
            "axes.labelcolor":  "#52504c",
            "axes.titlecolor":  "#1a1917",
            "xtick.color":      "#7a756f",
            "ytick.color":      "#7a756f",
            "grid.color":       "#eeeeee",
            "grid.alpha":       1.0,
            "axes.grid":        True,
            "axes.spines.top":  False,
            "axes.spines.right":False,
        }
    )
    _configure_matplotlib_fonts()


def _project_root() -> str:
    return _ROOT


def _unpack_for_shap(model):
    actual_model = model
    if hasattr(model, "estimator") and hasattr(model.estimator, "named_steps"):
        pipe = model.estimator
        actual_model = pipe.named_steps["clf"]
        preprocessor = pipe.named_steps["preprocessor"]

        def transform(X_num: pd.DataFrame) -> pd.DataFrame:
            arr = preprocessor.transform(X_num)
            try:
                names = preprocessor.get_feature_names_out()
                return pd.DataFrame(arr, columns=names, index=X_num.index)
            except Exception:
                return pd.DataFrame(arr, index=X_num.index)

        return actual_model, transform
    if hasattr(model, "named_steps") and "clf" in model.named_steps:
        actual_model = model.named_steps["clf"]
        preprocessor = model.named_steps["preprocessor"]

        def transform(X_num: pd.DataFrame) -> pd.DataFrame:
            arr = preprocessor.transform(X_num)
            try:
                names = preprocessor.get_feature_names_out()
                return pd.DataFrame(arr, columns=names, index=X_num.index)
            except Exception:
                return pd.DataFrame(arr, index=X_num.index)

        return actual_model, transform
    return actual_model, lambda x: x


def _clean_shap_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).replace("pass__", "").replace("num__", "") for c in out.columns]
    return out


def _coerce_float_matrix_for_shap(X: pd.DataFrame) -> pd.DataFrame:
    """
    SHAP / XGBoost expect a strict float matrix. On Streamlit Cloud, pipeline output can
    leave stringified scalars like ``'[1.406E-1]'`` (bracketed scientific notation).

    We always stringify then strip brackets (repeat in case of nested ``[[...]]``) so we
    never skip cleanup when pandas reports a column as numeric yet cells are still strings.
    """
    if X.empty:
        return X
    n, m = int(X.shape[0]), int(X.shape[1])
    arr = np.empty((n, m), dtype=np.float64)
    for j, c in enumerate(X.columns):
        s = X.iloc[:, j].astype(str).str.strip()
        for _ in range(4):
            s = s.str.replace(r"^\[(.*)\]$", r"\1", regex=True)
        col = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64, copy=False)
        arr[:, j] = np.nan_to_num(col, nan=0.0, posinf=0.0, neginf=0.0)
    return pd.DataFrame(arr, columns=list(X.columns), index=X.index)


def _float_vec_from_cache(raw) -> np.ndarray:
    """Parse SHAP vector from session cache (handles object / bracketed string elements)."""
    flat = np.asarray(raw).ravel()
    if flat.size == 0:
        return flat.astype(np.float64)
    if flat.dtype != object:
        return np.nan_to_num(
            np.asarray(flat, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
        )
    out: list[float] = []
    for x in flat.tolist():
        if isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool):
            out.append(float(x))
            continue
        s = str(x).strip()
        if s.startswith("[") and s.endswith("]"):
            inner = s[1:-1].strip().split()
            s = inner[0] if inner else ""
        try:
            out.append(float(s))
        except ValueError:
            out.append(0.0)
    return np.nan_to_num(np.asarray(out, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def _build_explainer(actual_model, X_bg: pd.DataFrame):
    X_bg = _coerce_float_matrix_for_shap(X_bg)
    model_str = str(type(actual_model))
    if any(
        m in model_str
        for m in [
            "RandomForest",
            "XGB",
            "LGBM",
            "CatBoost",
            "ExtraTrees",
            "GradientBoosting",
            "HistGradientBoosting",
            "DecisionTree",
            "AdaBoost",
        ]
    ):
        # Default TreeExplainer output is often **margin (logit)**, not predict_proba.
        # That breaks the force plot (x-axis is probability) and can confuse the UI.
        _tries = (
            lambda: shap.TreeExplainer(actual_model, X_bg, model_output="probability"),
            lambda: shap.TreeExplainer(actual_model, data=X_bg, model_output="probability"),
            lambda: shap.TreeExplainer(actual_model, X_bg),
            lambda: shap.TreeExplainer(actual_model, data=X_bg),
        )
        last_err: Exception | None = None
        for mk in _tries:
            try:
                return mk(), "tree"
            except Exception as e:
                last_err = e
                continue
        if last_err is not None:
            raise last_err
    if "LogisticRegression" in model_str:
        explainer = shap.LinearExplainer(actual_model, X_bg)
        return explainer, "linear"
    bg = shap.sample(X_bg, min(80, len(X_bg)))
    explainer = shap.KernelExplainer(actual_model.predict_proba, bg)
    return explainer, "kernel"


def _shap_values_for_class1(explainer, kind: str, X_shap: pd.DataFrame, nsamples: int = 80):
    if kind == "kernel":
        raw = explainer.shap_values(X_shap, nsamples=nsamples)
    else:
        raw = explainer.shap_values(X_shap)
    if isinstance(raw, list):
        arr = raw[1] if len(raw) > 1 else raw[0]
    else:
        arr = raw
    if hasattr(arr, "shape") and len(arr.shape) == 3:
        arr = arr[:, :, 1]
    return np.asarray(arr)


def _matplotlib_shap_barh_plot(
    vec: np.ndarray,
    names: list,
    cohort_key: str,
    *,
    top_k: int = 10,
    explainer_kind: str = "",
) -> plt.Figure:
    """学术风格本地 SHAP 条形图：白底、高对比红/蓝、弱化网格。"""
    _configure_matplotlib_fonts()
    v = np.asarray(vec, dtype=np.float64).ravel()
    n_all = min(int(v.size), len(names))
    if n_all == 0:
        fig, ax = plt.subplots(figsize=(6, 2), facecolor="#ffffff")
        ax.text(0.5, 0.5, "No SHAP values", ha="center", va="center", fontsize=11)
        ax.set_axis_off()
        return fig

    labs = [str(names[i]) for i in range(n_all)]
    v = v[:n_all]
    k = int(min(int(top_k), n_all))
    order = np.argsort(np.abs(v))[-k:][::-1]
    vals = v[order].astype(float)
    ylab = [labs[i] if len(labs[i]) <= 38 else labs[i][:35] + "…" for i in order]
    y_pos = np.arange(k)

    bar_colors = ["#C62828" if val >= 0 else "#1565C0" for val in vals]

    fig_h = max(3.4, 0.44 * k + 1.65)
    fig, ax = plt.subplots(figsize=(7.2, fig_h), facecolor="#ffffff")
    ax.set_facecolor("#ffffff")

    for y, val, col in zip(y_pos, vals, bar_colors):
        ax.barh(y, val, height=0.62, color=col, alpha=1.0, edgecolor="none", zorder=2)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(ylab, fontsize=9.5, color="#1a1917")
    ax.invert_yaxis()

    # 中线
    ax.axvline(0.0, color="#9e9e9e", linewidth=0.9, zorder=1, linestyle="-")

    # 轴样式
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#e0e0e0")
    ax.spines["bottom"].set_color("#e0e0e0")
    ax.tick_params(axis="x", colors="#7a756f", labelsize=8.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("SHAP value (positive class · this instance)", fontsize=8.5, color="#9e9e9e", labelpad=5)
    ek = str(explainer_kind).strip() or "—"
    ax.set_title(
        f"Local SHAP · Cohort {cohort_key}\nTop {k} features by |SHAP| · explainer: {ek}",
        fontsize=9.5,
        fontweight="600",
        color="#1a1917",
        pad=8,
        loc="left",
    )
    ax.grid(axis="x", color="#f0f0f0", linewidth=0.6, alpha=0.55, zorder=0)

    # 数值标注
    lo, hi = float(np.min(vals)), float(np.max(vals))
    span = max(abs(lo), abs(hi), 1e-9)
    pad_x = span * 0.16
    ax.set_xlim(lo - pad_x, hi + pad_x)
    for y, val in zip(y_pos, vals):
        off = span * 0.025
        if val >= 0:
            ax.text(val + off, y, f"+{val:.4g}", va="center", ha="left", fontsize=7.5, color="#C62828")
        else:
            ax.text(val - off, y, f"{val:.4g}", va="center", ha="right", fontsize=7.5, color="#1565C0")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def _streamlit_show_shap_figure(fig: plt.Figure, raw_png: bytes, cohort_key: str) -> None:
    """
    Show SHAP PNG **only** via `st.markdown` + inline data-URI `<img>`.

    `st.image`, `st.pyplot`, and `st.download_button` trigger lazy-loaded Streamlit JS chunks
    (`static/js/index.*.js`) that often fail with **Failed to fetch dynamically imported module**
    on some setups — same error you see after SHAP check.
    """
    del fig  # caller closes; avoid pyplot path entirely
    if len(raw_png) < 400:
        st.error(
            f"SHAP figure export looks invalid ({len(raw_png)} bytes). "
            "Check the terminal for matplotlib errors."
        )
        return

    b64 = base64.standard_b64encode(raw_png).decode("ascii")
    st.markdown(
        f'<div class="shap-inline-chart">'
        f'<img alt="Local SHAP bar chart" src="data:image/png;base64,{b64}" '
        f'style="max-width:100%;height:auto;display:block;" />'
        f'</div>',
        unsafe_allow_html=True,
    )


def _bps_ui_kind(col: str) -> str:
    """
    Map a model feature column to sidebar control type using Table 1 BPS metadata.
    Returns 'continuous' (slider) or 'discrete' (selectbox: categorical / binary / sex / chronic flags).
    """
    c = str(col)
    for title, sec_cfg in BPS_SECTIONS:
        if title == "Outcome":
            continue
        if sec_cfg.get("sex_col") == c:
            return "discrete"
        for t in sec_cfg.get("binary", []) or []:
            if t and t[0] == c:
                return "discrete"
        for t in sec_cfg.get("lifestyle_binary", []) or []:
            if t and t[0] == c:
                return "discrete"
        for t in sec_cfg.get("chronic_disease_cols", []) or []:
            if t and t[0] == c:
                return "discrete"
        for coln, _ in sec_cfg.get("categorical", []) or []:
            if coln == c:
                return "discrete"
        for key in ("continuous", "optional_continuous", "lifestyle_continuous", "physical"):
            for item in sec_cfg.get(key, []) or []:
                if item and item[0] == c:
                    return "continuous"
        for item in sec_cfg.get("defining_continuous", []) or []:
            if not item:
                continue
            col_or_key = item[0]
            if col_or_key == "total_cognition" and c in ("total_cognition", "total_cog"):
                return "continuous"
            if col_or_key == c:
                return "continuous"
    return "continuous"


def _discrete_levels(X_all: pd.DataFrame, col: str) -> list[float]:
    s = pd.to_numeric(X_all[col], errors="coerce").dropna()
    if len(s) == 0:
        return [0.0, 1.0]
    return sorted({float(x) for x in s.unique()})


def _nearest_discrete_level(levels: list[float], v: float) -> float:
    if not levels:
        return float(v)
    return float(min(levels, key=lambda x: abs(x - float(v))))


def _bps_sex_column(col: str) -> bool:
    c = str(col)
    for _title, sec_cfg in BPS_SECTIONS:
        if sec_cfg.get("sex_col") == c:
            return True
    return False


def _level_in_discrete(levels: list[float], v: float, *, tol: float = 1e-6) -> bool:
    fv = float(v)
    return any(abs(fv - x) <= tol for x in levels)


def _categorical_option_label(col: str, v: float, levels: list[float]) -> str:
    c = str(col)
    if c in CATEGORICAL_LEVEL_LABELS:
        ik = int(round(float(v)))
        lab = CATEGORICAL_LEVEL_LABELS[c].get(ik)
        if lab:
            return f"{lab} ({ik})"
    if (
        len(levels) == 2
        and {round(x, 6) for x in levels} <= {0.0, 1.0}
        and not _bps_sex_column(c)
    ):
        return "No (0)" if abs(float(v)) < 0.5 else "Yes (1)"
    if _bps_sex_column(c):
        return f"Sex code {int(round(float(v)))}"
    return f"{float(v):g}"


def _numeric_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    exclude = get_exclude_cols(df, target_col=TARGET_COL)
    cols = [c for c in df.columns if c not in exclude]
    X = df[cols].select_dtypes(include=[np.number])
    return X


def _value_outside_bounds(val: float, lo: float, hi: float) -> bool:
    """判断是否明显超出 [lo, hi]（含浮点容差）。"""
    if lo > hi:
        lo, hi = hi, lo
    span = max(abs(hi - lo), abs(lo), abs(hi), 1.0)
    tol = 1e-9 * span
    return val < lo - tol or val > hi + tol


def _variable_minmax(X_all: pd.DataFrame, cols_list: list) -> dict[str, tuple[float, float]]:
    """各特征在子样本 X_all 中的 min/max（用于校验与裁剪，界面不展示）。"""
    out: dict[str, tuple[float, float]] = {}
    for c in cols_list:
        s = pd.to_numeric(X_all[c], errors="coerce").dropna()
        if len(s) == 0:
            lo = hi = 0.0
        else:
            lo, hi = float(s.min()), float(s.max())
        if lo > hi:
            lo, hi = hi, lo
        out[str(c)] = (lo, hi)
    return out


def _build_feature_editor_df(
    X_all: pd.DataFrame,
    display_order: list[str],
    wide_seed: pd.DataFrame | None,
) -> pd.DataFrame:
    """纵向表：variable | value；BPS 分组仅用于 Tab，不在表中重复「类别」列。"""
    order = list(display_order)
    bounds = _variable_minmax(X_all, order)
    if wide_seed is not None and len(wide_seed) >= 1:
        ser = wide_seed.iloc[0].reindex(order)
        vals = pd.to_numeric(ser, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    else:
        vals = (
            X_all[order].median(numeric_only=True).reindex(order).fillna(0.0).to_numpy(dtype=np.float64)
        )
    for i, c in enumerate(order):
        lo, hi = bounds[str(c)]
        vals[i] = float(np.clip(vals[i], lo, hi))
    return pd.DataFrame(
        {
            "variable": [str(c) for c in order],
            "value": vals,
        }
    )


def _vertical_to_wide_row(vert: pd.DataFrame, column_order: list) -> pd.DataFrame:
    """纵向表拼回单行宽表，供 predict_proba / transform。"""
    order = list(column_order)
    m: dict[str, float] = {}
    for _, r in vert.iterrows():
        v = str(r["variable"])
        val = pd.to_numeric(r.get("value"), errors="coerce")
        m[v] = 0.0 if pd.isna(val) else float(val)
    arr = [m.get(str(c), 0.0) for c in order]
    return pd.DataFrame([arr], columns=order, dtype=np.float64)


def _feature_table_signature(vert: pd.DataFrame) -> tuple[tuple[str, float], ...]:
    """Stable fingerprint: sorted by variable name + rounded values (avoids data_editor float noise)."""
    v = vert["variable"].astype(str)
    num = pd.to_numeric(vert["value"], errors="coerce").fillna(0.0)
    pairs = sorted(
        zip(v.tolist(), [round(float(x), 6) for x in num.tolist()]),
        key=lambda p: p[0],
    )
    return tuple(pairs)


def _slider_sess_key(cohort_key: str, i: int) -> str:
    return f"sl_feat_{cohort_key}_{i}"


def _params_signature_from_sliders(
    display_order: list,
    cohort_key: str,
    bounds_map: dict,
    X_all: pd.DataFrame,
) -> tuple[tuple[str, float], ...]:
    pairs: list[tuple[str, float]] = []
    for i, c in enumerate(display_order):
        sk = _slider_sess_key(cohort_key, i)
        raw = st.session_state.get(sk)
        if raw is None:
            if _bps_ui_kind(c) == "discrete":
                lv = _discrete_levels(X_all, c)
                med = float(pd.to_numeric(X_all[c], errors="coerce").median())
                raw = _nearest_discrete_level(lv, med if np.isfinite(med) else lv[0])
            else:
                lo, hi = bounds_map[str(c)]
                if hi <= lo:
                    hi = lo + 1.0
                raw = float(np.clip(float(X_all[c].median()), lo, hi))
        pairs.append((str(c), round(float(raw), 6)))
    return tuple(sorted(pairs, key=lambda x: x[0]))


def _row_X_from_sliders(cols_list: list, display_order: list, cohort_key: str) -> pd.DataFrame:
    val_map: dict[str, float] = {}
    for i, c in enumerate(display_order):
        val_map[str(c)] = float(st.session_state[_slider_sess_key(cohort_key, i)])
    arr = [val_map[str(c)] for c in cols_list]
    return pd.DataFrame([arr], columns=cols_list, dtype=np.float64)


def _expected_value_proba_class1(explainer) -> float:
    """从 explainer 中取正类的 expected_value（概率尺度）。"""
    try:
        ev = explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            ev = float(np.asarray(ev).ravel()[1] if len(np.asarray(ev).ravel()) > 1 else np.asarray(ev).ravel()[0])
        else:
            ev = float(ev)
        if 0.0 <= ev <= 1.0:
            return ev
    except Exception:
        pass
    return float("nan")


def _base_value_for_force_plot(proba: float, vec: np.ndarray, ev_raw: float) -> float:
    """计算与 Σφ 一致的 baseline：优先用 explainer.expected_value；否则 f(x) − Σφ。"""
    if np.isfinite(ev_raw) and 0.0 <= ev_raw <= 1.0:
        return ev_raw
    return float(proba - float(np.sum(vec)))


def _risk_tier_labels(proba: float) -> tuple[str, str, str]:
    if proba >= 0.40:
        return ("risk-high", "⚠️", t("risk_tier_high"))
    if proba >= 0.20:
        return ("risk-med", "◆", t("risk_tier_med"))
    return ("risk-low", "✓", t("risk_tier_low"))


def _recommendations_html(proba: float) -> str:
    """结构化 HTML 建议（用于卡片内渲染）。"""
    base = [
        ("<strong>Not for clinical use.</strong> This tool illustrates a research model (CHARLS CPM champion); it does not diagnose or treat."),
        ('<strong>Interpret cautiously.</strong> \u201cNext-wave incident DCC\u201d is a statistical target from the manuscript, not an individual prognosis.'),
    ]
    if proba >= 0.40:
        extra = [
            "<strong>High modeled risk (demo tier):</strong> Discuss comprehensive geriatric assessment and mood/cognition follow-up with a qualified clinician if clinically indicated.",
            "<strong>Lifestyle &amp; safety:</strong> Encourage sleep, physical activity, and social engagement; address falls and medication review in usual care.",
        ]
    elif proba >= 0.20:
        extra = [
            "<strong>Moderate modeled risk (demo tier):</strong> Consider periodic mood and cognition screening in routine care; reinforce protective behaviours.",
            "<strong>Monitor comorbidities</strong> that appear as top SHAP factors alongside clinical judgment.",
        ]
    else:
        extra = [
            "<strong>Lower modeled probability (demo tier):</strong> Maintain routine preventive care; risk is probabilistic and uncertainty remains.",
            "<strong>Use SHAP factors</strong> as hypotheses for discussion, not standalone decision rules.",
        ]
    items = "".join(
        f'<li style="margin-bottom:0.3rem;">{txt}</li>' for txt in (base + extra)
    )
    return f'<ol style="margin:0;padding-left:1.25rem;">{items}</ol>'


def _recommendations_markdown(proba: float) -> str:
    lines = [
        "1. **Not for clinical use.** This tool illustrates a research model (CHARLS CPM champion); it does not diagnose or treat.",
        "2. **Interpret cautiously.** “Next-wave incident DCC” is a statistical target from the manuscript, not an individual prognosis.",
    ]
    if proba >= 0.40:
        lines += [
            "3. **High modeled risk (demo tier):** Discuss comprehensive geriatric assessment and mood/cognition follow-up with a qualified clinician if clinically indicated.",
            "4. **Lifestyle & safety:** Encourage sleep, physical activity, and social engagement as appropriate; address falls and medication review in usual care.",
        ]
    elif proba >= 0.20:
        lines += [
            "3. **Moderate modeled risk (demo tier):** Consider periodic mood and cognition screening in routine care; reinforce protective behaviors (activity, sleep, social connection).",
            "4. **Monitor comorbidities** that appear as top factors in the SHAP panel alongside clinical judgment.",
        ]
    else:
        lines += [
            "3. **Lower modeled probability (demo tier):** Still maintain routine preventive care; risk is probabilistic and uncertainty remains.",
            "4. **Use SHAP factors** as hypotheses for discussion, not as standalone decision rules.",
        ]
    return "\n".join(lines)


@st.cache_resource
def _load_champion_model(model_path: str, _mtime: float):
    if not os.path.isfile(model_path):
        return None
    return joblib.load(model_path)


def _analysis_data_fingerprint() -> tuple:
    """缓存失效：插补或预处理任一路径更新则重载。"""
    root = _project_root()
    imp = os.path.join(root, IMPUTED_DATA_PATH)
    pre = os.path.join(root, "preprocessed_data", "CHARLS_final_preprocessed.csv")
    mi = os.path.getmtime(imp) if os.path.isfile(imp) else 0.0
    mp = os.path.getmtime(pre) if os.path.isfile(pre) else 0.0
    return (bool(USE_IMPUTED_DATA), mi, mp)


@st.cache_data(show_spinner=True)
def _load_cohort_subsample(_data_fp: tuple, baseline_group: int, max_rows: int = 8000):
    """
    与 compare_models / 主流程一致：经 `load_df_for_analysis`
    （MICE-completed 表 + `reapply_cohort_definition` + `prepare_exposures`）。
    勿直接裸读 `step1_imputed_full.csv`，否则列集与训练路径不一致，`transform` 可能失败。
    预测特征经 `get_exclude_cols` 构建（含排除 `had_comorbidity_before` 等），与 `champion_model.joblib` 一致。
    """
    try:
        from utils.charls_script_data_loader import load_df_for_analysis

        df = load_df_for_analysis()
    except Exception as ex:
        return ("error", str(ex))
    if df is None or len(df) == 0:
        return ("error", "load_df_for_analysis returned empty")
    if "baseline_group" not in df.columns:
        return ("error", "missing baseline_group after load_df_for_analysis")
    sub = df[df["baseline_group"] == baseline_group].copy()
    if len(sub) == 0:
        return ("error", f"no rows for baseline_group=={baseline_group}")
    if len(sub) > max_rows:
        sub = sub.sample(n=max_rows, random_state=RANDOM_SEED)
    return ("ok", sub)


COHORT_META = [
    {
        "key": "A",
        "baseline": 0,
        "dir": COHORT_A_DIR,
        "title_en": "Cohort A — Baseline healthy",
        "n_paper": 8828,
        "accent": "#8f9e8b",
    },
    {
        "key": "B",
        "baseline": 1,
        "dir": COHORT_B_DIR,
        "title_en": "Cohort B — Depression only",
        "n_paper": 3123,
        "accent": "#8a9bab",
    },
    {
        "key": "C",
        "baseline": 2,
        "dir": COHORT_C_DIR,
        "title_en": "Cohort C — Cognitive impairment only",
        "n_paper": 2435,
        "accent": "#a89098",
    },
]


def render_cohort_tab(meta: dict):
    key = meta["key"]
    accent = meta.get("accent", "#8f9188")
    root = _project_root()
    pred_dir = os.path.join(root, meta["dir"], COHORT_STEP_DIRS["prediction"])
    model_path = os.path.join(pred_dir, "champion_model.joblib")

    if shap is None:
        st.error(t("err_no_shap"))
        return

    mtime = os.path.getmtime(model_path) if os.path.isfile(model_path) else 0.0
    model = _load_champion_model(model_path, mtime)
    if model is None:
        st.error(t("err_no_model", model_path))
        return

    load_res = _load_cohort_subsample(_analysis_data_fingerprint(), meta["baseline"])
    if not load_res or load_res[0] != "ok":
        msg = load_res[1] if load_res and load_res[0] == "error" else "unknown"
        st.error(t("err_load_analysis", msg))
        st.caption(t("err_no_data", os.path.join(root, IMPUTED_DATA_PATH)))
        return
    df_sub = load_res[1]

    X_all = _numeric_feature_matrix(df_sub)
    if X_all.shape[1] == 0:
        st.error(t("err_empty_x"))
        return

    actual_model, transform = _unpack_for_shap(model)
    try:
        X_bg_full = transform(X_all)
    except Exception as e:
        st.error(t("err_transform", e))
        return
    X_bg = _clean_shap_columns(X_bg_full)
    X_bg = _coerce_float_matrix_for_shap(X_bg)
    if len(X_bg) > 600:
        X_bg = X_bg.sample(n=600, random_state=RANDOM_SEED)

    try:
        explainer, kind = _build_explainer(actual_model, X_bg)
    except Exception as e:
        st.error(t("err_explainer", e))
        return

    # ----- Sidebar: clinical parameters（分组折叠，减轻滚动） -----
    st.sidebar.markdown("<div style='margin-top:0.5rem;'></div>", unsafe_allow_html=True)
    st.sidebar.markdown(
        f'<p class="sb-group-hdr">{html_module.escape(t("sidebar_clinical_params"))}</p>',
        unsafe_allow_html=True,
    )

    cols_list = list(X_all.columns)
    display_order, _ = order_columns_for_editor(cols_list)
    bounds_map = _variable_minmax(X_all, cols_list)

    with st.sidebar.expander(t("sidebar_section_numeric"), expanded=True):
        for i, c in enumerate(display_order):
            if _bps_ui_kind(c) != "continuous":
                continue
            sk = _slider_sess_key(key, i)
            lab = str(c) if len(str(c)) <= 68 else str(c)[:65] + "…"
            lo, hi = bounds_map[str(c)]
            if hi <= lo:
                hi = lo + 1.0
            if sk not in st.session_state:
                st.session_state[sk] = float(np.clip(float(X_all[c].median()), lo, hi))
            st.slider(lab, float(lo), float(hi), key=sk, format="%.4g")
            med_c = float(np.clip(float(pd.to_numeric(X_all[c], errors="coerce").median()), lo, hi))
            cur_c = float(st.session_state[sk])
            st.markdown(
                f'<p class="slider-median-hint">Current: <strong>{cur_c:.4g}</strong> · '
                f'Cohort median: <strong>{med_c:.4g}</strong></p>',
                unsafe_allow_html=True,
            )

    with st.sidebar.expander(t("sidebar_section_categorical"), expanded=False):
        for i, c in enumerate(display_order):
            if _bps_ui_kind(c) != "discrete":
                continue
            sk = _slider_sess_key(key, i)
            lab = str(c) if len(str(c)) <= 68 else str(c)[:65] + "…"
            levels = _discrete_levels(X_all, c)
            if sk not in st.session_state:
                med = float(pd.to_numeric(X_all[c], errors="coerce").median())
                st.session_state[sk] = _nearest_discrete_level(
                    levels, med if np.isfinite(med) else levels[0]
                )
            else:
                cur = float(st.session_state[sk])
                if not _level_in_discrete(levels, cur):
                    st.session_state[sk] = _nearest_discrete_level(levels, cur)
            st.selectbox(
                lab,
                options=levels,
                key=sk,
                format_func=lambda v, col=c, lv=levels: _categorical_option_label(col, float(v), lv),
            )
            med_d = float(pd.to_numeric(X_all[c], errors="coerce").median())
            cur_d = float(st.session_state[sk])
            st.markdown(
                f'<p class="slider-median-hint">Current: <strong>{cur_d:.4g}</strong> · '
                f'Cohort median (numeric): <strong>{med_d:.4g}</strong></p>',
                unsafe_allow_html=True,
            )

    if st.sidebar.button(t("btn_load_median"), key=f"load_median_{key}", use_container_width=True):
        for j, col in enumerate(display_order):
            skj = _slider_sess_key(key, j)
            if _bps_ui_kind(col) == "discrete":
                lv = _discrete_levels(X_all, col)
                med = float(pd.to_numeric(X_all[col], errors="coerce").median())
                st.session_state[skj] = _nearest_discrete_level(lv, med if np.isfinite(med) else lv[0])
            else:
                lo, hi = bounds_map[str(col)]
                if hi <= lo:
                    hi = lo + 1.0
                st.session_state[skj] = float(np.clip(float(X_all[col].median()), lo, hi))
        st.session_state[f"{key}_run_ver"] = int(st.session_state.get(f"{key}_run_ver", 1)) + 1
        st.rerun()

    # ----- Main: 队列条 + 按钮行 -----
    st.markdown(
        f'<div class="cohort-bar" style="--coh-accent:{accent};">'
        f'<span class="cohort-bar-name">{html_module.escape(meta["title_en"])}</span>'
        f'<span class="cohort-bar-meta">Cohort {html_module.escape(key)} &nbsp;·&nbsp; N ≈ {meta["n_paper"]:,}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if f"{key}_run_ver" not in st.session_state:
        st.session_state[f"{key}_run_ver"] = 1
    if f"{key}_computed_ver" not in st.session_state:
        st.session_state[f"{key}_computed_ver"] = 0

    # 按钮 + 提示文字
    _btn_c, _hint_c = st.columns([1, 5], gap="small")
    with _btn_c:
        st.markdown('<div class="run-btn">', unsafe_allow_html=True)
        if st.button(t("btn_start_assessment"), key=f"start_asm_{key}", use_container_width=True):
            st.session_state[f"{key}_run_ver"] = int(st.session_state[f"{key}_run_ver"]) + 1
        st.markdown("</div>", unsafe_allow_html=True)
    with _hint_c:
        st.markdown(
            '<p class="run-hint">Adjust parameters in the sidebar, then click <strong>Start assessment</strong> to compute risk and SHAP</p>',
            unsafe_allow_html=True,
        )

    sig = _params_signature_from_sliders(display_order, key, bounds_map, X_all)
    need_compute = int(st.session_state[f"{key}_run_ver"]) != int(st.session_state[f"{key}_computed_ver"])
    cache_key = f"{key}_result_cache"

    if need_compute:
        row_X = _row_X_from_sliders(cols_list, display_order, key)
        with st.spinner("Running model & SHAP…"):
            try:
                proba = float(model.predict_proba(row_X)[0, 1])
            except Exception as e:
                st.error(t("err_predict", e))
                return
            X_one_t = transform(row_X)
            X_one = _clean_shap_columns(X_one_t)
            X_one = _coerce_float_matrix_for_shap(X_one)
            try:
                if kind == "kernel":
                    sv = _shap_values_for_class1(explainer, kind, X_one, nsamples=60)
                else:
                    sv = _shap_values_for_class1(explainer, kind, X_one)
                vec = np.asarray(sv).ravel()
                names = X_one.columns.tolist()
                if len(vec) != len(names):
                    st.warning(t("shap_dim_warn"))
                    m = min(len(vec), len(names))
                    vec, names = vec[:m], names[:m]
            except Exception as e:
                st.error(t("err_shap", e))
                return
            # Base value for caption only — must not block SHAP bar chart if this step fails
            ev_base = float(proba - float(np.sum(vec)))
            try:
                ev_raw = _expected_value_proba_class1(explainer)
                ev_base = _base_value_for_force_plot(float(proba), vec, ev_raw)
            except Exception:
                pass

        order_abs = np.argsort(np.abs(vec))[::-1][:10]
        fac_feat = [names[j] for j in order_abs]
        fac_imp = [t("impact_increase") if vec[j] >= 0 else t("impact_decrease") for j in order_abs]
        factors_df = pd.DataFrame({t("factor_col_feature"): fac_feat, t("factor_col_impact"): fac_imp})
        row_dict = row_X.iloc[0].to_dict()
        row_simple: dict[str, float] = {}
        for k, v in row_dict.items():
            x = pd.to_numeric(v, errors="coerce")
            if pd.notna(x):
                row_simple[str(k)] = float(x)

        st.session_state[cache_key] = {
            "proba": float(proba),
            "base_proba": float(ev_base),
            "vec": np.asarray(vec, dtype=np.float64).ravel().tolist(),
            "names": [str(x) for x in names],
            "factors_records": factors_df.to_dict(orient="records"),
            "factors_columns": list(factors_df.columns),
            "row_values": row_simple,
        }
        st.session_state[f"{key}_computed_ver"] = int(st.session_state[f"{key}_run_ver"])
        st.session_state[f"{key}_cache_sig"] = sig

    cache = st.session_state.get(cache_key)
    # Stale session: versions "match" but cache missing (code upgrade, cleared storage, etc.) → force recompute.
    if cache is None and not need_compute:
        st.session_state[f"{key}_computed_ver"] = 0
        st.rerun()

    if cache is None:
        st.info(t("predict_click_prompt"))
        return

    if sig != st.session_state.get(f"{key}_cache_sig"):
        st.warning(t("stale_inputs_warn"))

    proba = float(cache["proba"])
    vec = _float_vec_from_cache(cache["vec"])
    names = list(cache["names"])
    if len(vec) != len(names):
        m = min(len(vec), len(names))
        if m == 0:
            st.error("SHAP vector and feature names are empty or mismatched.")
            return
        st.warning(t("shap_dim_warn"))
        vec = vec[:m]
        names = names[:m]
    if cache.get("factors_records") is not None:
        factors_df = pd.DataFrame(
            cache["factors_records"],
            columns=cache.get("factors_columns") or [t("factor_col_feature"), t("factor_col_impact")],
        )
    else:
        factors_df = cache.get("factors_df")
        if factors_df is None:
            factors_df = pd.DataFrame(columns=[t("factor_col_feature"), t("factor_col_impact")])

    risk_cls, sym, tier_name = _risk_tier_labels(proba)
    n_feat = len(names)
    top_k_plot = min(10, n_feat) if n_feat > 0 else 0
    tier_emoji = {"risk-low": "🟢", "risk-med": "🟡", "risk-high": "🔴"}.get(risk_cls, "")
    row_vals_map = cache.get("row_values") or {}

    # ── 主区 60/40：左 = I 评估 + II SHAP 解读 + III 建议；右 = SHAP 图 ───
    c_left, c_right = st.columns([3, 2], gap="medium")

    with c_left:
        st.markdown(
            f'<p class="sec-title sec-roman first">{html_module.escape(t("section_i_assessment"))}</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="card">'
            f'<span class="risk-number {risk_cls}">{proba:.1%}</span>'
            f'<div class="risk-badge {risk_cls}">{tier_emoji} {html_module.escape(tier_name)}</div>'
            f'<p class="risk-footnote">{html_module.escape(t("manual_no_true_y"))}</p>'
            f'<p class="risk-footnote" style="margin-top:0.35rem;">'
            f'{html_module.escape(t("manual_no_true_y_detail"))}</p>'
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            _html_details(t("risk_criteria_expander"), t("risk_criteria_body")),
            unsafe_allow_html=True,
        )

        if n_feat > 0 and vec.size > 0:
            st.markdown(
                f'<p class="sec-title sec-roman">{html_module.escape(t("section_ii_shap_text"))}</p>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="card" style="padding:0.95rem 1.1rem;font-size:0.84rem;'
                f'line-height:1.55;color:var(--ink-muted);">'
                f'{_html_shap_derived_suggestions(vec, names, row_vals_map, top_k_plot)}'
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown(
            f'<p class="sec-title sec-roman">{html_module.escape(t("section_iii_clinical"))}</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="card" style="font-size:0.88rem;line-height:1.7;color:var(--ink-muted);">'
            f'{_recommendations_dynamic_html(proba, vec, names)}'
            f"</div>",
            unsafe_allow_html=True,
        )

    with c_right:
        if n_feat == 0 or vec.size == 0:
            st.error("SHAP vector is empty; check model preprocessor output.")
        else:
            st.markdown(
                f'<p class="sec-title sec-roman first">{html_module.escape(t("section_ii_shap_plot"))}</p>',
                unsafe_allow_html=True,
            )
            kn = (
                '<p style="font-size:0.74rem;color:var(--ink-soft);margin:0 0 0.5rem;line-height:1.45;">'
                "Kernel SHAP: first run may take several minutes.</p>"
                if kind == "kernel"
                else ""
            )
            st.markdown(
                f'<div class="card" style="padding:0.75rem 0.85rem;">'
                f'<p style="font-size:0.78rem;color:var(--ink-muted);margin:0 0 0.35rem;line-height:1.45;">'
                f"Top <strong>{top_k_plot}</strong> by |SHAP| · explainer: "
                f'<code style="font-size:0.72rem;">{html_module.escape(str(kind))}</code> · '
                f'<span style="color:#C62828;">🔴</span> risk up · '
                f'<span style="color:#1565C0;">🔵</span> risk down</p>'
                f"{kn}"
                f"{_html_details(t('shap_trouble_title'), t('shap_trouble_body'))}"
                f"</div>",
                unsafe_allow_html=True,
            )
            fig = None
            png_bytes = b""
            try:
                _apply_plot_style(accent)
                fig = _matplotlib_shap_barh_plot(
                    vec, names, key, top_k=top_k_plot, explainer_kind=str(kind)
                )
                buf = io.BytesIO()
                fig.savefig(
                    buf,
                    format="png",
                    dpi=120,
                    bbox_inches="tight",
                    pad_inches=0.22,
                    facecolor="#ffffff",
                    edgecolor="none",
                )
                png_bytes = buf.getvalue()
                st.session_state[f"{key}_shap_png"] = png_bytes
                st.markdown(
                    '<div class="card" style="padding:0.55rem 0.65rem;margin-bottom:12px;">',
                    unsafe_allow_html=True,
                )
                _streamlit_show_shap_figure(fig, png_bytes, key)
                st.markdown("</div>", unsafe_allow_html=True)
            except Exception as e:
                st.markdown(
                    f'<p style="color:var(--risk-red);font-weight:600;padding:0.4rem 0;">'
                    f"SHAP plot failed: {html_module.escape(str(e))}</p>",
                    unsafe_allow_html=True,
                )
            finally:
                if fig is not None:
                    plt.close(fig)
            if len(png_bytes) >= 400:
                st.download_button(
                    label=t("btn_download_shap_png"),
                    data=png_bytes,
                    file_name=f"shap_local_cohort_{key}.png",
                    mime="image/png",
                    key=f"dl_shap_{key}",
                    use_container_width=True,
                )

    # ── IV. 当前输入特征表（底部全宽，便于核对） ─────────────────────────
    rv = row_vals_map
    if rv:
        df_feat = pd.DataFrame(
            [(str(k), float(v)) for k, v in rv.items()],
            columns=[t("manual_col_var"), t("manual_col_val")],
        )
        st.markdown(
            f'<p class="sec-title sec-roman" style="margin-top:0.25rem;">'
            f'{html_module.escape(t("section_iv_features"))}</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="card" style="padding:0.75rem 0.9rem;">'
            f'<div style="max-height:320px;overflow:auto;">{_html_simple_table(df_feat)}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )


def main():
    if "nav_idx" not in st.session_state:
        st.session_state["nav_idx"] = 0
    # Legacy nav: 0=intro 1=A 2=B 3=C → current: 0=A 1=B 2=C (one-time migration)
    if st.session_state.get("_nav_schema_v2") != "abc":
        try:
            ni = int(st.session_state.get("nav_idx", 0))
            if 1 <= ni <= 3:
                st.session_state.nav_idx = ni - 1
            elif ni > 3:
                st.session_state.nav_idx = 2
            else:
                st.session_state.nav_idx = 0
        except (TypeError, ValueError):
            st.session_state.nav_idx = 0
        st.session_state["_nav_schema_v2"] = "abc"

    st.set_page_config(
        page_title=STRINGS["page_title"],
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="🧠",
    )

    _inject_app_css()

    # ── 侧边栏：品牌 + 队列选择（无 project root） ──────────────────────
    st.sidebar.markdown(
        '<p class="pd-sidebar-top">CHARLS · Supplement</p>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        '<p class="sb-group-hdr" style="margin-top:0.5rem;">Choose cohort</p>',
        unsafe_allow_html=True,
    )
    nav_labels = [t("nav_a"), t("nav_b"), t("nav_c")]
    nav_i = st.sidebar.radio(
        "page_nav",
        [0, 1, 2],
        format_func=lambda i: nav_labels[i],
        key="nav_idx",
        label_visibility="collapsed",
    )

    # ── 顶部品牌栏（全宽深绿） ───────────────────────────────────────────
    st.markdown(
        f'<div class="app-header">'
        f'<div class="app-header-icon">🧠</div>'
        f'<div>'
        f'<div class="app-header-title">{html_module.escape(t("app_title"))}</div>'
        f'<div class="app-header-sub">{html_module.escape(t("app_subtitle"))}</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if nav_i == 0:
        render_cohort_tab(COHORT_META[0])
    elif nav_i == 1:
        render_cohort_tab(COHORT_META[1])
    else:
        render_cohort_tab(COHORT_META[2])


if __name__ == "__main__":
    main()
