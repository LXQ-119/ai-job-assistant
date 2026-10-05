# 启动 Streamlit 前端（默认 http://localhost:8501）
# 用法：.\scripts\run_ui.ps1
# 注意：必须先让后端在另一个终端里跑着，否则前端会提示连不上后端。
#
# 本机**没有使用虚拟环境**，Python 直接来自 PATH（实际是 E:\dev\Python312）。

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Resolve-Python {
    foreach ($candidate in @("python", "py")) {
        $cmd = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($null -eq $cmd) { continue }
        if ($cmd.Source -like "*WindowsApps*") { continue }
        return $cmd.Source
    }
    $fallback = "E:\dev\Python312\python.exe"
    if (Test-Path $fallback) { return $fallback }
    throw "找不到可用的 Python。请确认 E:\dev\Python312 存在，或 python 在 PATH 中。"
}

$python = Resolve-Python

# 顺手检查后端是否还活着，省掉一次"为什么前端报错"的困惑
try {
    $health = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -TimeoutSec 3
    Write-Host "后端在线 ｜ 模型：$($health.model) ｜ 索引块数：$($health.knowledge_chunks)" -ForegroundColor Green
    if (-not $health.api_key_configured) {
        Write-Host "注意：后端还没配 LLM_API_KEY，问答和简历解析会返回 502。" -ForegroundColor Yellow
    }
} catch {
    Write-Host "警告：后端没有响应。请在另一个终端先启动后端（启动项目.bat 或 .\scripts\run_api.ps1）" -ForegroundColor Yellow
}

Write-Host "前端启动中…… http://localhost:8501`n" -ForegroundColor Green
& $python -m streamlit run ui/app.py
