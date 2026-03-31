# English-only strings in this file (Windows PowerShell 5.x mishandles UTF-8 without BOM).
# Push charls_script_data_loader.py to GitHub (Streamlit Cloud demo / fallback fixes).
# Usage: cd to repo root, then: .\scripts\git_push_streamlit_cloud_fix.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$Files = @(
    "utils/charls_script_data_loader.py"
)

foreach ($f in $Files) {
    if (-not (Test-Path (Join-Path $Root $f))) {
        Write-Error "Missing file: $f"
    }
}

git add -- $Files
$status = git status --short
if ($status -notmatch "charls_script_data_loader") {
    Write-Host "Nothing to commit for charls_script_data_loader (already committed or no changes)." -ForegroundColor Yellow
    git status -sb
    exit 0
}

git commit -m 'chore: update charls_script_data_loader for Streamlit Cloud'
git push origin main

Write-Host "Done. If push asks for login, authorize in browser or set SSH/HTTPS credentials." -ForegroundColor Green
