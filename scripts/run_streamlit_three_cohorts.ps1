# English-only strings in this file (Windows PowerShell 5.x mishandles UTF-8 without BOM).
# Opens Streamlit in a separate PowerShell window (stays alive if Cursor's integrated terminal resets).
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-NoProfile",
    "-Command",
    "Set-Location -LiteralPath '$ProjectRoot'; streamlit run streamlit_shap_three_cohorts.py"
)
