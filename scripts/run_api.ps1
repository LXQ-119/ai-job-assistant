# 启动后端 API（默认 http://127.0.0.1:8000）
# 用法：.\scripts\run_api.ps1
#
# 本机**没有使用虚拟环境**，Python 直接来自 PATH（实际是 E:\dev\Python312）。
# 如果 PowerShell 提示"禁止运行脚本"，改用项目根目录的 启动项目.bat（.bat 不受执行策略限制）。

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Resolve-Python {
    <#
      找一个真正可用的 Python。
      必须跳过微软商店的占位别名（WindowsApps\python.exe）——它只会弹商店，不会执行。
    #>
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
Write-Host "使用 Python：$python" -ForegroundColor DarkGray

if (-not (Test-Path ".env")) {
    Write-Host "警告：没有 .env 文件，模型接口会返回 502。" -ForegroundColor Yellow
    Write-Host "      执行：copy .env.example .env   然后填入 LLM_API_KEY" -ForegroundColor Yellow
}

Write-Host "后端启动中…… 接口文档 http://127.0.0.1:8000/docs" -ForegroundColor Green
Write-Host "按 Ctrl+C 停止`n" -ForegroundColor DarkGray

& $python -m app.main
