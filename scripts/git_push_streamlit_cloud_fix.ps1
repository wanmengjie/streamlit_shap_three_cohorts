# English-only strings (Windows PowerShell 5.x + UTF-8 without BOM).
# Push Streamlit Cloud-related files to GitHub.
#
# Usage (from repo root):
#   .\scripts\git_push_streamlit_cloud_fix.ps1
#   .\scripts\git_push_streamlit_cloud_fix.ps1 -Message "fix: SHAP float coercion"
#   .\scripts\git_push_streamlit_cloud_fix.ps1 -ExtraFiles @("utils/charls_prepare_exposures.py")
#
# Manual one-liner (same folder as this repo):
#   Set-Location "C:\Users\lenovo\Desktop\因果机器学习"; git add streamlit_shap_three_cohorts.py; git commit -m "your message"; git push origin main

param(
    [string] $Message = "chore: Streamlit app + data loader for Cloud",
    [string[]] $ExtraFiles = @()
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$Files = @(
    "streamlit_shap_three_cohorts.py",
    "utils/charls_script_data_loader.py"
) + $ExtraFiles

foreach ($f in $Files) {
    if (-not (Test-Path (Join-Path $Root $f))) {
        Write-Error "Missing file: $f"
    }
}

git add -- $Files
$staged = git diff --cached --name-only
if (-not $staged) {
    Write-Host "Nothing staged (no changes in listed files). Current branch:" -ForegroundColor Yellow
    git status -sb
    exit 0
}

git commit -m $Message
git push origin main

Write-Host "Done. If push asks for login, authorize in browser or set SSH/HTTPS credentials." -ForegroundColor Green
