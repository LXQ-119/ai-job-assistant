"""接口冒烟测试 —— 用 Python 发请求，避免 PowerShell 5.1 的编码坑。

踩过的坑：PowerShell 的 Invoke-RestMethod 发中文 JSON 时，
中文会被破坏成 "?"，模型收到的是乱码问题，然后回你一句
"你的消息好像没发完整"。**这不是接口的问题，是测试工具的问题。**

用法：
    python -m scripts.api_smoke
    python -m scripts.api_smoke http://127.0.0.1:8010
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
LINE = "=" * 74


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'✅' if ok else '❌'} {label}" + (f"　{detail}" if detail else ""))


def main() -> None:
    print(f"\n{LINE}\n目标后端：{BASE}\n{LINE}")

    print("\n【基础接口】")
    try:
        health = httpx.get(f"{BASE}/health", timeout=10).json()
        check("GET  /health", True, f"模型 {health['model']}，知识库 {health['knowledge_chunks']} 块")
    except Exception as exc:  # noqa: BLE001
        check("GET  /health", False, str(exc))
        return

    try:
        status = httpx.get(f"{BASE}/kb/status", timeout=10).json()
        check("GET  /kb/status", True, f"{len(status['sources'])} 个文件")
    except Exception as exc:  # noqa: BLE001
        check("GET  /kb/status", False, str(exc))

    print("\n【写死的 RAG：/kb/query】")
    for question in ["我给自己定的检索目标是多少？", "红烧肉怎么做才好吃？"]:
        try:
            data = httpx.post(
                f"{BASE}/kb/query", json={"question": question}, timeout=120
            ).json()
            first = data["answer"].strip().splitlines()[0]
            print(f"  问：{question}")
            print(f"    翻到 {data['retrieved']} 张卡 ｜ ${data['cost_usd']}")
            print(f"    {first[:70]}")
        except Exception as exc:  # noqa: BLE001
            check(question, False, str(exc))

    print("\n【Agentic RAG：/agent/ask】")
    cases = [
        ("查笔记", "我从小米那次面试里学到了什么？"),
        ("算数", "1234 乘以 5678 等于多少？"),
        ("直接答", "李白是哪个朝代的诗人？"),
    ]
    ok_count = 0
    for label, question in cases:
        try:
            data = httpx.post(
                f"{BASE}/agent/ask", json={"question": question}, timeout=240
            ).json()
            final = data.get("final")
            if not final:
                check(f"{label}：{question}", False, str(data.get("error")))
                continue
            ok_count += 1
            tools = " + ".join(final["tools_used"]) or "（没调工具）"
            print(f"  ✅ {label}｜{question}")
            print(f"     走的工具：{tools}　轮数 {final['steps']}　${final['cost_usd']}")
            print(f"     答：{final['content'].strip().splitlines()[0][:66]}")
        except Exception as exc:  # noqa: BLE001
            check(f"{label}：{question}", False, str(exc))

    print(f"\n{LINE}")
    print(f"Agent 接口通过 {ok_count}/{len(cases)}")
    print(LINE)


if __name__ == "__main__":
    main()
