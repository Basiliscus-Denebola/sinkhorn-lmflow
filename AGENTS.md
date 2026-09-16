# 开发约定

## 开发环境

- 本项目统一使用名为 `OT` 的 Conda 环境进行开发、运行和测试。
- 环境路径：`D:\miniconda_envs\OT`。
- 开始开发前执行 `conda activate OT`。
- 对于未激活环境的独立命令，使用 `conda run -n OT <命令>`，例如 `conda run -n OT python -m pytest`。
- 安装 Python 依赖时使用 `conda run -n OT python -m pip install <包名>`，确保依赖安装到 `OT` 环境。
- 不使用 `base` 或其他环境替代 `OT`；若环境不可用，先报告问题。
- 用户已确认保留 `OT` 的 Python 3.9，本项目兼容 Python >= 3.9；此约定优先于原始 prompt 的 Python >= 3.11 要求。
- 可直接调用 `D:\miniconda_envs\OT\python.exe`，确保独立命令使用同一环境。
