# English-only strings in this file (Windows PowerShell 5.x mishandles UTF-8 without BOM).
# Run Streamlit and mirror ALL console output to a log file (helps when the process exits with no traceback).
# Usage (from anywhere):
#   powershell -ExecutionPolicy Bypass -File scripts/run_streamlit_with_log.ps1

$ErrorActionPreference = "Continue"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $ProjectRoot

$logDir = Join-Path $ProjectRoot "streamlit_logs"
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}
$logFile = Join-Path $logDir ("streamlit_{0:yyyyMMdd_HHmmss}.log" -f (Get-Date))

Write-Host "Logging to: $logFile"
Write-Host "Tip: if this window returns to PS> immediately, open the log file; the last lines usually show why."
Write-Host ""

# Extra diagnostics (harmless if unused)
$env:PYTHONFAULTHANDLER = "1"
$env:STREAMLIT_LOGGER_LEVEL = "debug"

# 2>&1 merges stderr into the stream; Tee-Object captures everything shown in the console
streamlit run streamlit_shap_three_cohorts.py --logger.level=debug 2>&1 | Tee-Object -FilePath $logFile

$code = $LASTEXITCODE
Write-Host ""
Write-Host "Streamlit exited with code: $code"
Write-Host "Full log: $logFile"
