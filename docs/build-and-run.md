# 构建与运行

## 1. 基础依赖

- Python 3.10 或更高版本；
- CMake 3.20 或更高版本；
- Ninja；
- LLVM 与 MLIR 20（当前构建在 20.1.8 上验证）；
- C++17 编译器。

真实 Triton/TileLang/CUDA 闭包还需要兼容的 PyTorch、Triton、CUDA 工具链和 NVIDIA GPU。TileLang Source 需要可用的 TileLang 安装。RVV 目标执行需要对应 sysroot、LLVM 工具链和目标机或运行环境。运行多前端 Dataset demo 时，必须通过 `INTENTIR_DATASET_ROOT` 或 CLI 参数显式指定外部 Dataset，demo 不再假设默认路径。

## 2. Python 环境

核心 Python 包没有强制运行时依赖。验证与参考执行安装：

```bash
python -m pip install -e '.[verify]'
```

需要运行 Lift/Replay agent 时再安装可选依赖：

```bash
python -m pip install -e '.[verify,lift-agent]'
```

Provider 密钥放在被 Git 忽略的 `llm_providers.local.json`，或通过调用端安全注入；不得写入证据或提交到仓库。

## 3. 构建 `intent-opt`

以系统 LLVM/MLIR 20 为例：

```bash
cmake -S . -B build -G Ninja \
  -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir \
  -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm
cmake --build build --target intent-opt
```

生成的优化器默认位于：

```text
build/tools/intent-opt/intent-opt
```

Python 后端还会使用 `mlir-opt`、`mlir-translate`、LLVM 运行库和 Clang。非标准安装可通过相应环境变量或工具入口的显式参数覆盖，不要把本机绝对路径编码进 `J`、`C` 或生产快照。

## 4. 清单诊断

以下命令检查仓库与 Dataset 的可解析性；它们不是闭包进度：

```bash
PYTHONPATH=python python tools/scrc_batch.py --validate-manifest --json
PYTHONPATH=python python tools/scrc_batch.py --list --json
PYTHONPATH=python python tools/scrc_frontend.py --list --json
PYTHONPATH=python python tools/frontend_dataset.py --validate
```

对真实运行始终指定仓库之外的 evidence 根目录：

```bash
PYTHONPATH=python python tools/scrc_frontend.py \
  --case '<registered-case-id>' \
  --dataset-root '/path/to/Dataset' \
  --provider-config '/secure/path/llm_providers.local.json' \
  --run-root '/path/to/evidence/scrc-frontends' \
  --json
```

该命令只调用 Triton/TTIR 的 Lift/Replay provider。真实闭包只能由该 fresh Lift → Replay → Gate 流程产生；历史 artifact 不进入活动验证路径。CUDA、CuTe、TileLang 不通过此入口运行，必须分别使用各自的 compiler/Gate1 runner；CUDA/TileLang 的 Gate2 继续使用 native adapter，CuTe 不回退到 Triton Gate2。native Dataset 缺失时保持 unavailable，不能回退为 Triton/TTIR 输入。

## 5. CUDA 后端

直接从生产快照降低、执行与基准：

```bash
PYTHONPATH=python python tools/benchmark_production_cuda.py \
  --snapshot-dir '/path/to/accepted-snapshots' \
  --out '/path/to/evidence/cuda-benchmark'
```

也可以在编译机导出 bundle，再在目标 GPU 上执行：

```bash
PYTHONPATH=python python tools/export_cuda_benchmark_bundle.py \
  --snapshot-dir '/path/to/accepted-snapshots' \
  --out '/path/to/bundles/cuda'

PYTHONPATH=python python tools/run_cuda_benchmark_bundle.py \
  --bundle '/path/to/bundles/cuda' \
  --out '/path/to/evidence/cuda-target'
```

## 6. RVV 后端

RVV 使用预编译对象 bundle，使目标机 runner 只依赖 Python 标准库与系统链接/执行环境：

```bash
PYTHONPATH=python python tools/export_rvv_benchmark_bundle.py \
  --snapshot-dir '/path/to/accepted-snapshots' \
  --out '/path/to/bundles/rvv'

PYTHONPATH=python python tools/run_rvv_benchmark_bundle.py \
  --bundle '/path/to/bundles/rvv' \
  --out '/path/to/evidence/rvv-target'
```

以各工具的 `--help` 为参数权威来源；目标特性、sysroot、基线 sidecar 和测量 regime 必须由运行环境显式提供。

## 7. 验收边界

构建成功、Python 可导入、清单可解析和 TypeCheck 通过都是必要诊断，但都不增加闭包计数。活动树不运行或恢复历史测试。真正的验收来自真实 Kernel 的 Gate1、Gate2 三边比较和保存完备的 closure artifacts。
