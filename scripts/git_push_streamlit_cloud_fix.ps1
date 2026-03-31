# 将「云端缺 CHARLS/插补表时的演示数据回退」等改动推送到 GitHub。
# 用法：在 PowerShell 中执行
#   cd "你的项目根\因果机器学习"
#   .\scripts\git_push_streamlit_cloud_fix.ps1
# 若需一并提交其它已修改文件，先手动 git add，再改下面 $Files 或注释掉固定列表仅用 git add -u。

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$Files = @(
    "utils/charls_script_data_loader.py"
)

foreach ($f in $Files) {
    if (-not (Test-Path (Join-Path $Root $f))) {
        Write-Error "缺少文件: $f"
    }
}

git add -- $Files
$status = git status --short
if ($status -notmatch "charls_script_data_loader") {
    Write-Host "没有可提交的暂存变更（可能已提交或无改动）。当前状态:" -ForegroundColor Yellow
    git status -sb
    exit 0
}

git commit -m "fix(data): fallback to data/sample_data when CHARLS/imputed missing (Streamlit Cloud)"
git push origin main

Write-Host "Done. 若 push 要求登录，请在浏览器完成 GitHub 授权，或配置 SSH/HTTPS 凭据。" -ForegroundColor Green
