# 环境检查 / 补齐依赖
# 用法：在项目根目录执行  .\scripts\setup.ps1
#
# 说明：本机**没有使用虚拟环境**。Python 装在 E:\dev\Python312 并已加入 PATH
#       （前插在微软商店占位别名之前，所以 python 命令直接可用）。
#
#       这个脚本做三件事：
#         1) 确认 Python 可用（自动跳过 WindowsApps 的占位别名）
#         2) 安装/补齐 requirements.txt 里的依赖
#         3) 如果还没有 .env，从 .env.example 生成一份
#
# 如果 PowerShell 提示"禁止运行脚本"（你的执行策略是 Restricted），改用：
#     项目根目录的  启动项目.bat   或   scripts\环境自检.bat
# （.bat 不受 PowerShell 执行策略限制）

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
Write-Host "项目目录：$root" -ForegroundColor Cyan

# --- 1. 找 Python ---
function Resolve-Python {
    foreach ($candidate in @("python", "py")) {
        $cmd = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($null -eq $cmd) { continue }
        # 微软商店的占位别名只会弹商店，必须跳过
        if ($cmd.Source -like "*WindowsApps*") { continue }
        return $cmd.Source
    }
    $fallback = "E:\dev\Python312\python.exe"
    if (Test-Path $fallback) { return $fallback }
    throw "找不到可用的 Python。请确认 E:\dev\Python312 存在，或 python 在 PATH 中。"
}

$python = Resolve-Python
Write-Host "使用 Python：$python" -ForegroundColor Green
& $python --version

# --- 2. 安装依赖 ---
Write-Host "`n检查并补齐依赖……" -ForegroundColor Cyan
# 临时目录必须指到 E 盘：DSH 沙箱给的临时目录不能建子目录也不能删除，
# 而 pip 解包 wheel 需要这两项能力。
$env:TEMP = "E:\dev\tmp"
$env:TMP = "E:\dev\tmp"
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null

& $python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "依赖安装有问题，请把上面的报错发出来。" -ForegroundColor Red
} else {
    Write-Host "依赖已就绪。" -ForegroundColor Green
}

# --- 3. 生成 .env ---
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "`n已生成 .env —— 请打开它填入 LLM_API_KEY（否则模型接口返回 502）" -ForegroundColor Yellow
} else {
    Write-Host "`n.env 已存在，保留不动。" -ForegroundColor DarkGray
}

# --- 4. 离线测试 ---
Write-Host "`n运行单元测试……" -ForegroundColor Cyan
& $python -m pytest -q -p no:cacheprovider

Write-Host "`n完成。接下来：" -ForegroundColor Green
Write-Host "  启动项目.bat            （一键开后端+前端）"
Write-Host "  或 .\scripts\run_api.ps1 / .\scripts\run_ui.ps1（分别启动）"
