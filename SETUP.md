# 环境说明与启动指南

> **这份文档记录的是你机器上"实际已装好的状态"**，不是通用教程。
> 照着做就能跑，不需要重新安装任何东西。

---

## 一、当前已装好的环境

| 项目 | 实际位置 | 状态 |
|---|---|---|
| **项目代码** | `C:\Users\17839\Desktop\ai-job-assistant` | ✅ 15 个单元测试全通过 |
| **Python 3.12.10** | `E:\dev\Python312` | ✅ 已装，pip 25.0.1 |
| **项目依赖** | `E:\dev\Python312\Lib\site-packages` | ✅ 全部装好 |
| **Python 安装器** | `E:\dev\_installers\python-3.12.10-amd64.exe` | 留着，以后修复/重装用 |
| **pip 缓存 / 临时目录** | `E:\dev\pip-cache`、`E:\dev\tmp` | ✅ |
| **Git 2.47.1** | `D:\Git`（你原有的） | ✅ 已加入 PATH |
| **用户 PATH** | `E:\dev\Python312`、`E:\dev\Python312\Scripts`、`D:\Git\cmd` | ✅ 已写入，**新开终端生效** |

**没有使用虚拟环境（venv）** —— 因为这个 Python 是专为本项目装的，独占使用。
（沙箱环境下 `python -m venv` 的 ensurepip 会因临时目录权限失败，这是当时的实际原因。）

### 验证环境

新开一个 PowerShell 窗口：

```powershell
python --version      # 应输出 Python 3.12.10
git --version         # 应输出 git version 2.47.1
```

---

## 二、启动项目（三种方式，任选一种）

> 先记住一件事：**Python 项目不需要编译**，没有 `mvn package`、没有 `.jar`、也不用配 Tomcat。
> 就是直接 `python -m xxx` 跑源码。而且这个项目是**两个进程**：后端 8000 + 前端 8501。

### 方式一：双击（最简单）

双击项目根目录的 **`启动项目.bat`** —— 会自动弹两个窗口，分别跑后端和前端。
（`.bat` 不受 PowerShell 执行策略限制，所以最省事。）

### 方式二：手动开两个终端

```powershell
cd C:\Users\17839\Desktop\ai-job-assistant
```

**窗口 A —— 后端：**
```powershell
python -m app.main
```
接口文档：http://127.0.0.1:8000/docs

**窗口 B —— 前端**（新开一个终端，同样先 `cd`）：
```powershell
python -m streamlit run ui/app.py
```
界面：http://localhost:8501

首次使用：前端左侧点「**重建知识库索引**」，然后到问答页提问。

### 方式三：VS Code（推荐日常用这个）

```powershell
code C:\Users\17839\Desktop\ai-job-assistant
```

或者在 VS Code 里 `文件 → 打开文件夹`，选 `C:\Users\17839\Desktop\ai-job-assistant`。

打开后按 **F5**，会弹出调试配置让你选（这些配置写在 `.vscode/launch.json` 里）：

| 配置 | 作用 |
|---|---|
| ① 后端 FastAPI（端口 8000） | 启动后端，**可以打断点** |
| ② 前端 Streamlit（端口 8501） | 启动前端界面 |
| ③ Agent 例子：01 加法（离线模式） | 跑 `examples/01_add_agent.py`，不需要 API Key |
| ④ 当前打开的文件 | 调试任意一个 py 文件 |

**两个都要跑**：先按 F5 选①，再按一次 F5 选②（VS Code 会开两个终端）。

> **为什么用 VS Code 而不是 IDEA**：你机器上的 IDEA 是 Ultimate 版，装 Python 插件理论上也能跑，
> 但 VS Code 是目前 Python / AI 应用开发最普遍的选择，对「Python + Markdown + Dockerfile + 配置」
> 这种混合项目支持最全，而且更轻。
>
> **扩展说明**：已装好 `ms-python.python`、`vscode-pylance`、`debugpy`、`vscode-python-envs` 四个。
> 因为安装时遇到环境的证书拦截（`ERR_TLS_CERT_ALTNAME_INVALID`），是**离线解压 + 手工写扩展索引**
> 装上的。副作用：**这些扩展不会自动更新**。需要更新时可以在 VS Code 的扩展面板里重新安装，
> 或者告诉我。安装包归档在 `E:\dev\_installers\vscode-extensions\`。

> 如果 PowerShell 提示「禁止运行脚本」，那是因为你的执行策略是 Restricted。
> 本项目所有命令都是直接调 `python`，**不需要改执行策略**，也不用管 `scripts\*.ps1`。

---

## 三、配置 API Key（目前唯一还没做的事）

现在模型相关功能（问答、简历解析）会返回 **502**，因为没有 Key。

```powershell
cd C:\Users\17839\Desktop\ai-job-assistant
copy .env.example .env
notepad .env
```

在 `LLM_API_KEY=` 后面填入你的 DeepSeek Key（https://platform.deepseek.com 获取），保存。

**验证读取（不花钱）：**
```powershell
python -c "from app.config import get_settings, mask_secret; s=get_settings(); print(s.llm_base_url, s.llm_model, mask_secret(s.llm_api_key))"
```
显示 `<empty>` 说明 `.env` 没生效 —— 最常见原因是记事本存成了 `.env.txt`，
请在资源管理器里打开「显示文件扩展名」确认。

---

## 四、跑测试和评测

```powershell
cd C:\Users\17839\Desktop\ai-job-assistant

# 单元测试（不需要 Key、不联网、不花钱）
python -m pytest -v

# 检索评测（不需要 Key，产出 Hit@k / MRR）
python -m scripts.eval_rag --top-k 4
```

报告写到 `reports\` 目录，文件名带时间戳。

> ⚠️ **当前评测集太简单**：示例文档 12 个小节对应 12 道题，几乎一一对应，
> 所以 Hit@k 跑出来是 100% —— 这个数字**不能写进简历**，面试官一问就露馅。
> 要换成有区分度的题（改写问法、跨小节、故意问不到的负例），
> 才能跑出真实的"62% → 89%"那种提升曲线。详见 [INTERVIEW.md](INTERVIEW.md)。

---

## 五、你这台机器上的特殊之处（都是实测发现的）

### 1. D 盘根目录只读，`D:\dev` 权限已修好

```
D:\            Everyone 只有只读 → 无法在根目录建文件夹（需要管理员）
D:\dev         已授予 Everyone:(OI)(CI)F + LXQ\LXQ:(OI)(CI)F  → 可正常读写（当前为空）
E:\dev         已授予 Everyone:(OI)(CI)F → 可正常读写（Python 就装在这里）
```

> 历史问题已修复：早期用 `icacls D:\dev /grant "LXQ:..."` 时，Windows 把 `LXQ`
> 解析成了**计算机账户**（机器 SID，少了 `-1001`），导致授权无效。
> 已删除错误条目并改为 `LXQ\LXQ`。修复日志：`E:\dev\acl-fix.log`。

> **项目位置变更记录**
> 项目一度放在 `D:\dev\ai-job-assistant`，后已移回桌面工作区
> `C:\Users\17839\Desktop\ai-job-assistant`。
> 原因：DSH 的文件工具**只能访问工作区**，项目放在 D 盘时我改文件必须走命令行拼接
> 内容，又慢又容易在引号转义上出错。搬回桌面后文件工具恢复，开发效率明显更高。
> `D:\dev` 现在是空目录，留着备用。

### 2. 装 Python 包时必须把临时目录指到 E 盘

这是 **DSH 沙箱**的限制（不是 Windows 的问题）：沙箱给的临时目录**只能平铺写文件，
不能建子目录、也不能删除**，而 pip 解包 wheel 必须在子目录里创建并清理文件，所以会报：

```
ERROR: Could not install packages due to an OSError: [Errno 13] Permission denied: '...pip-unpack-xxxx\xxx.whl'
```

**以后自己装包时这样写：**

```powershell
$env:TEMP = "E:\dev\tmp"
$env:TMP  = "E:\dev\tmp"
python -m pip install 包名 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

（`PIP_CACHE_DIR` 已永久设为 `E:\dev\pip-cache`，这部分不用管。）

### 3. 控制台中文显示乱码

PowerShell 控制台默认 GBK，Python 输出的 UTF-8 中文会变成 `���`。
**只是显示问题，不影响运行。** 想看正常中文：

```powershell
$env:PYTHONIOENCODING = "utf-8"
chcp 65001
```

### 4. pytest 会报 `.pytest_cache` 权限警告

沙箱不允许创建该缓存目录，**无害，可忽略**。想彻底消掉：

```powershell
python -m pytest -v -p no:cacheprovider
```

---

## 六、排错速查

| 现象 | 原因 | 解决 |
|---|---|---|
| `python` 打开商店 | 微软商店占位别名 | 用 `py`，或新开终端（PATH 已配好） |
| `No module named app` | 不在项目根目录 | `cd C:\Users\17839\Desktop\ai-job-assistant` |
| 接口返回 **502** | 没配 Key / Key 错 / 余额不足 | 按第三节配 `.env`，看后端窗口报错原文 |
| 前端连不上后端 | 后端没启动 | 确认窗口 A 还开着，访问 http://127.0.0.1:8000/health |
| Key 显示 `<empty>` | 文件名成了 `.env.txt` | 打开「显示文件扩展名」后改名 |
| `pip install` 报 Permission denied | 沙箱临时目录限制 | 见第五节第 2 条 |
| 装包很慢 | 默认源在境外 | 加 `-i https://pypi.tuna.tsinghua.edu.cn/simple` |
| 数 `app.routes` 数不到接口 | FastAPI 0.142 的新行为 | 用 `TestClient` 实际调用验证，别数路由 |

---

## 七、完成标志

- [x] Python 3.12.10 装在 E 盘，依赖全部装好
- [x] 项目位于 `C:\Users\17839\Desktop\ai-job-assistant`
- [x] `python -m pytest -v` → **15 passed**
- [x] API 冒烟测试全通过（`/health` `/kb/status` `/metrics` `/kb/index` 均 200）
- [x] 索引建成（12 个块），评测脚本能跑出报告
- [x] `D:\dev` 权限已修复
- [ ] 配好 `.env` 的 API Key ← **下一步**
- [ ] 把 `data/knowledge/` 换成你自己的面经笔记
- [ ] 重做一份有区分度的评测集，跑出真实数字
