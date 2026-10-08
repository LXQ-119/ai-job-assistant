"""模型服务商：可插拔的接入层。

## 为什么"支持多家模型"不需要写很多代码

因为**几乎所有主流服务商都兼容 OpenAI 协议** —— 同一套 SDK，只换三个东西：

    base_url   接口地址
    api_key    密钥
    model      模型名

所以不需要为每家写适配器，只需要把这三样做成**运行时配置**。

    DeepSeek        https://api.deepseek.com
    OpenAI          https://api.openai.com/v1
    硅基流动         https://api.siliconflow.cn/v1
    阿里百炼         https://dashscope.aliyuncs.com/compatible-mode/v1
    智谱 GLM        https://open.bigmodel.cn/api/paas/v4
    月之暗面 Kimi    https://api.moonshot.cn/v1
    Ollama（本地）   http://127.0.0.1:11434/v1      ← 不需要 Key

## 配置分两层

    .env                      默认值。首次启动、或者没在网页上切过时用它
    data/runtime/model.json   运行时选择。网页上切的那份，**不用重启就生效**

第二层在 .gitignore 里 —— 因为它**含有 API Key**。
把密钥写进一个会被提交的文件，是这类项目最常见的事故。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .config import get_settings, mask_secret


@dataclass(frozen=True)
class Provider:
    """一个服务商的预设。填好 base_url 和常见模型名，用户少填两栏。"""

    key: str
    name: str
    base_url: str
    models: list[str] = field(default_factory=list)
    needs_key: bool = True
    note: str = ""


PRESETS: dict[str, Provider] = {
    "deepseek": Provider(
        key="deepseek",
        name="DeepSeek（深度求索）",
        base_url="https://api.deepseek.com",
        models=["deepseek-flash", "deepseek-v4-pro"],
        note="本项目默认。注意 base_url 不要带 /v1",
    ),
    "siliconflow": Provider(
        key="siliconflow",
        name="硅基流动 SiliconFlow",
        base_url="https://api.siliconflow.cn/v1",
        models=["deepseek-ai/DeepSeek-V3", "Qwen/Qwen2.5-72B-Instruct"],
        note="国内聚合平台，模型多，常用来做 embeddings",
    ),
    "dashscope": Provider(
        key="dashscope",
        name="阿里百炼（通义千问）",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        models=["qwen-plus", "qwen-turbo", "qwen-max"],
        note="OpenAI 兼容模式，路径必须带 /compatible-mode/v1",
    ),
    "zhipu": Provider(
        key="zhipu",
        name="智谱 GLM",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        models=["glm-4-flash", "glm-4-plus"],
        note="glm-4-flash 有免费额度，适合试跑",
    ),
    "moonshot": Provider(
        key="moonshot",
        name="月之暗面 Kimi",
        base_url="https://api.moonshot.cn/v1",
        models=["moonshot-v1-8k", "moonshot-v1-32k"],
    ),
    "openai": Provider(
        key="openai",
        name="OpenAI",
        base_url="https://api.openai.com/v1",
        models=["gpt-4o-mini", "gpt-4o"],
        note="国内需要能连通网络",
    ),
    "ollama": Provider(
        key="ollama",
        name="Ollama（本地模型）",
        base_url="http://127.0.0.1:11434/v1",
        models=["qwen2.5:7b", "llama3.1:8b"],
        needs_key=False,
        note="完全本地跑，不花钱、不联网。Key 随便填个 ollama 即可",
    ),
    "custom": Provider(
        key="custom",
        name="自定义（任意 OpenAI 兼容接口）",
        base_url="",
        models=[],
        note="只要兼容 OpenAI 协议就能接：vLLM、LM Studio、公司内网网关……",
    ),
}


@dataclass
class ActiveModel:
    """当前生效的模型配置。"""

    provider: str
    base_url: str
    api_key: str
    model: str
    source: str  # "runtime"（网页上切的）| "env"（.env 里的默认值）

    @property
    def name(self) -> str:
        preset = PRESETS.get(self.provider)
        return preset.name if preset else self.provider

    @property
    def needs_key(self) -> bool:
        """本地模型（Ollama）不需要 Key，不能因为没填就报错。"""
        preset = PRESETS.get(self.provider)
        return preset.needs_key if preset else True

    @property
    def effective_key(self) -> str:
        """真正传给 SDK 的 key。

        OpenAI SDK 不接受空字符串，所以不需要 Key 的服务商要塞个占位符 ——
        不填的话本地 Ollama 会被卡在"客户端初始化"这一步。
        """
        if self.api_key:
            return self.api_key
        return "not-needed" if not self.needs_key else ""

    def to_public(self) -> dict:
        """给界面看的版本 —— **密钥永远打码**。"""
        return {
            "provider": self.provider,
            "provider_name": self.name,
            "base_url": self.base_url,
            "model": self.model,
            "api_key_masked": mask_secret(self.api_key),
            "api_key_configured": bool(self.api_key),
            "source": self.source,
            "source_label": "网页上切换的" if self.source == "runtime" else ".env 默认值",
        }


# ---------------------------------------------------------------------------
# 运行时配置的读写
# ---------------------------------------------------------------------------


def runtime_path() -> Path:
    return get_settings().data_dir / "runtime" / "model.json"


def load_runtime() -> dict | None:
    path = runtime_path()
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict) or not payload.get("base_url"):
        return None
    return payload


def save_runtime(*, provider: str, base_url: str, api_key: str, model: str) -> None:
    path = runtime_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "provider": provider,
        "base_url": base_url.strip().rstrip("/"),
        "api_key": api_key.strip(),
        "model": model.strip(),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def clear_runtime() -> None:
    """删掉运行时配置，回到 .env 的默认值。"""
    path = runtime_path()
    if path.exists():
        path.unlink()


def _normalize(url: str) -> str:
    return url.strip().rstrip("/").lower()


def resolve_api_key(base_url: str, provided: str) -> str:
    """决定这次到底用哪个 Key。

    规则：**留空 + 地址和当前生效的一致 → 沿用当前生效的 Key。**

    为什么需要这个：网页上打开「模型设置」时，Key 输入框是空的
    （原始 Key 从不回传前端）。如果不做这个回退，用户想测一下
    "当前正在用的这个模型通不通"，还得把 Key 重新粘一遍 —— 很蠢。
    """
    key = (provided or "").strip()
    if key:
        return key

    active = get_active()
    if _normalize(base_url) == _normalize(active.base_url):
        return active.api_key
    return ""


# ---------------------------------------------------------------------------
# 取当前生效的配置
# ---------------------------------------------------------------------------


def get_active() -> ActiveModel:
    """运行时配置优先，没有就用 .env 的默认值。

    这个函数在**每次模型调用**时都会被调用，所以不能做重活 ——
    只是读一个几百字节的 JSON，可以接受。
    （真要优化可以加缓存 + 切换时失效，但那是过早优化。）
    """
    settings = get_settings()

    runtime = load_runtime()
    if runtime:
        return ActiveModel(
            provider=runtime.get("provider", "custom"),
            base_url=runtime["base_url"],
            api_key=runtime.get("api_key", ""),
            model=runtime.get("model") or settings.llm_model,
            source="runtime",
        )

    return ActiveModel(
        provider="deepseek" if "deepseek" in settings.llm_base_url else "custom",
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        source="env",
    )


def describe(active: ActiveModel | None = None) -> str:
    """给日志用的连接信息，密钥打码。"""
    item = active or get_active()
    return (
        f"provider={item.provider} base_url={item.base_url} "
        f"model={item.model} api_key={mask_secret(item.api_key)} "
        f"source={item.source}"
    )
