"""验证后端接口返回的中文编码是否正确。

为什么要单独验这个：我在 PowerShell 里调用时中文显示成了乱码
（`IDF æ˜¯...` 这种 UTF-8 字节被当成 Latin-1 解读的典型症状）。
必须确认是**终端显示问题**还是**接口真的返回了坏编码**——
如果是后者，用户在网页上看到的也会是乱码。
"""

from __future__ import annotations

import httpx

API = "http://127.0.0.1:8000"


def main() -> int:
    # --- 1. 原始字节：确认响应体到底是不是合法 UTF-8 ---
    response = httpx.post(
        f"{API}/kb/query",
        json={"question": "切块时的 overlap 有什么作用？", "top_k": 3},
        timeout=90,
    )
    print(f"HTTP {response.status_code}")
    print(f"content-type: {response.headers.get('content-type')}")

    raw = response.content
    print(f"响应体 {len(raw)} 字节，前 3 字节: {raw[:3]!r}")

    try:
        text = raw.decode("utf-8")
        print("UTF-8 解码: 成功 ✅（说明接口编码没问题）")
    except UnicodeDecodeError as exc:
        print(f"UTF-8 解码: 失败 ❌ {exc}")
        return 1

    # --- 2. 检查有没有出现乱码特征字符 ---
    mojibake_markers = ["æ", "å", "ç", "è", "é", "ï¼", "ã"]
    hits = [m for m in mojibake_markers if m in text]
    if hits:
        print(f"发现乱码特征字符: {hits} ❌")
    else:
        print("乱码特征字符: 无 ✅")

    # --- 3. 正常打印内容 ---
    data = response.json()
    print("\n回答：")
    print("  " + data["answer"].replace("\n", "\n  "))
    print("\n引用来源：")
    for citation in data["citations"]:
        print(f"  - {citation['source']} · {citation['heading']}  (相关度 {citation['score']})")
    print(f"\n耗时 {data['latency_ms']} ms    成本 ${data['cost_usd']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
