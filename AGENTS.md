# Agents.md — 开发准则

## 工具链与运行环境
- **包管理**：统一使用 `uv` 管理依赖。
- **运行脚本**（如 `run.bat` / `run.sh`）：**禁止**通过 `uv run` 启动，必须直接使用项目虚拟环境中的解释器（如 `.venv\Scripts\python` 或 `./.venv/bin/python`），以确保进程行为和 IDE 调试一致。

## Python 版本与类型注解
- **Python 3.14 强制要求**。
- **延迟注解求值默认启用**：不再需要 `from __future__ import annotations`。
- **类型提示风格**：全面使用 Python 3.14 原生类型语法（如 `list[int]`、`dict[str, int]`、`| None` 等），避免使用 `typing.List` 等旧式写法。

## 静态检查与格式化
- **双工具联合检查**：每次提交前执行 `ty check` 和 `pyright`（可并行运行）。
  - **pyright 模式**：默认启用**严格模式（strict）**；若项目已在 `pyproject.toml` 或其它配置中指定了类型检查等级（如 `typeCheckingMode`），则遵循项目配置，不覆盖。
- 对二者报错保持**批判性审视**——虽误报率极低，但出现不合理警告时以实际运行逻辑为准。
- **格式化**：使用 `ruff format` 统一代码风格，不做手动调整。

## 版本控制
- **提交前检查 `.gitignore`**：确保已忽略所有无需版本控制的文件（如虚拟环境 `.venv/`、`__pycache__/`、`*.pyc`、`dist/`、`build/`、`*.egg-info/`、IDE 配置目录等）。若发现遗漏，及时更新 `.gitignore` 后再提交。
- **完成修改后立即提交并推送**（`git add -u && git commit -m "<message>" && git push`），保持远端与本地同步。