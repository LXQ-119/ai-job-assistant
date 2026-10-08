"""模型可插拔功能的冒烟测试。

测整条链路：
    1. 读当前配置
    2. 测试连接（故意给个错的 → 应该失败；给对的 → 应该成功）
    3. 切换到运行时配置 → /health 应该立刻反映出来
    4. 回到 .env 默认值

用法：
    python -m scripts.provider_smoke
    python -m scripts.provider_smoke http://127.0.0.1:8010
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402
from dotenv import dotenv_values  # noqa: E402

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
LINE = "=" * 74


def active_of() -> dict:
    return httpx.get(f"{BASE}/providers", timeout=15).json()["active"]


def main() -> None:
    env = dotenv_values(ROOT / ".env")
    real_key = (env.get("LLM_API_KEY") or "").strip()

    print(f"\n{LINE}\n目标后端：{BASE}\n{LINE}")

    print("\n【1】当前配置")
    current = active_of()
    print(f"  {current['provider_name']} ／ {current['model']}")
    print(f"  来源：{current['source_label']}　密钥：{current['api_key_masked']}")
    original = {
        "provider": current["provider"],
        "base_url": current["base_url"],
        "api_key": real_key,
        "model": current["model"],
    }

    print("\n【2a】测试连接 —— 故意给一个错的地址")
    bad = httpx.post(
        f"{BASE}/providers/test",
        json={
            "provider": "custom",
            "base_url": "https://definitely-not-a-real-host.example",
            "api_key": "sk-fake",
            "model": "no-such-model",
        },
        timeout=90,
    ).json()
    print(f"  ok={bad['ok']}  {bad['message'][:110]}")
    print("  " + ("✅ 正确报错（不是 500 崩溃）" if not bad["ok"] else "❌ 应该失败的"))

    print("\n【2b】测试连接 —— 用当前真实的配置")
    good = httpx.post(f"{BASE}/providers/test", json=original, timeout=90).json()
    print(f"  ok={good['ok']}  {good['latency_ms']} ms  模型回：{good['reply'][:40]!r}")
    print("  " + ("✅ 连通" if good["ok"] else f"❌ {good['message'][:100]}"))

    print("\n【2c】测试连接 —— Key 留空，但地址和当前一致")
    fallback = httpx.post(
        f"{BASE}/providers/test", json={**original, "api_key": ""}, timeout=90
    ).json()
    print(f"  ok={fallback['ok']}  {fallback['latency_ms']} ms  模型回：{fallback['reply'][:40]!r}")
    print(
        "  "
        + (
            "✅ 正确沿用了当前生效的 Key（用户不用重新粘一遍）"
            if fallback["ok"]
            else f"❌ 没能沿用：{fallback['message'][:90]}"
        )
    )

    print("\n【3】切换到「运行时配置」（模拟网页上点切换）")
    switched = httpx.post(
        f"{BASE}/providers/switch",
        json={**original, "provider": "deepseek"},
        timeout=30,
    ).json()
    after = switched["active"]
    print(f"  切换后来源：{after['source_label']}")

    health = httpx.get(f"{BASE}/health", timeout=15).json()
    print(f"  /health 报告：{health['model']} ／ 服务商 {health['provider']}")
    print("  " + ("✅ 立即生效，没重启" if after["source"] == "runtime" else "❌ 没生效"))

    print("\n【4】切回 .env 默认值")
    reset = httpx.post(f"{BASE}/providers/reset", json={}, timeout=30).json()
    print(f"  现在来源：{reset['active']['source_label']}")
    print("  " + ("✅ 已还原" if reset["active"]["source"] == "env" else "❌ 没还原"))

    print(f"\n{LINE}")
    print("安全检查：运行时配置文件必须被 git 忽略（里面有 API Key）")
    print(LINE)
    import subprocess

    runtime = ROOT / "data" / "runtime" / "model.json"
    check = subprocess.run(
        ["git", "check-ignore", "-v", str(runtime)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if check.returncode == 0:
        print(f"  ✅ 已被忽略：{check.stdout.strip()}")
    else:
        print(f"  ❌ 没有被忽略！{runtime} 里的 API Key 会被提交到 git —— 必须马上修 .gitignore")

    tracked = subprocess.run(
        ["git", "ls-files", "data/runtime/"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if tracked.stdout.strip():
        print(f"  ❌ 已经有文件被跟踪了：{tracked.stdout.strip()}")
    else:
        print("  ✅ 仓库里没有跟踪任何 data/runtime 下的文件")


if __name__ == "__main__":
    main()
