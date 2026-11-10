# IntentIR

IntentIR 是一个把高性能 Kernel 的源程序提升为可验证语义、再从该语义生成 CPU/CUDA/RISC-V 实现的研究系统。仓库只维护一条权威主线：前端 Lift、SCRC 闭包和生产后端共享同一套 `J/C/π` 合同，保持原有的 Source/compiler/observer 运行边界。

系统的核心边界是：

```text
Lift(Source) -> (J, C, π)
Replay(J, source-bound artifact.c; U_f) -> R
Gate1(Source, R, π)
Gate2(Source, R, RefJ) = Source–Replay ∧ Source–J ∧ Replay–J
```

- `J` 是与源语言无关的语义归档，描述张量、访问、谓词、归约、效果和输出语义。
- `C` 是当前 Source/后端流程实际使用的实现选择，包括 tile、launch 和编译器调度参数；它不改变 `J`。真实生产后端使用各自生成的 `target_C`，Replay 验证只使用已经和 Source/Harness 绑定的 `artifact.c`。
- `π` 是 Gate1 使用的冻结、带类型对应关系。它可以表达程序轴、访问、guard、布尔和归约关系，但不能修改 Source 或 Replay 的观测结果。
- Triton/TTIR、CUDA、CuTe、TileLang 和 RISC-V 保留各自的 source/compiler/observer/lowering 路径；它们可以共享 `J/C/π` 的基本合同，但不被转送到一个新的共享运行管线。
- Gate1 必须先闭合；Gate2 随后在同一 Harness 估值上执行 Source、Replay 和 `J` 的参考语义，三条比较边必须同时通过。

## 仓库结构

- `include/`、`lib/`：IntentIR 方言、转换与 CPU/CUDA/RISC-V lowering。
- `python/intentir/ir/`：`J` 的规范化、物化和生产快照边界。
- `python/intentir/extraction/`：Source 到 `(J,C,π)` 的 Lift 入口。
- `python/intentir/verify/`：TypeCheck、Gate1、Gate2、真实 workload 与证据生命周期。
- `python/intentir/backend/`：CUDA 与 RISC-V 生产后端，以及 Gate2 使用的 CPU 标量参考路径。
- `tools/`：SCRC、闭包复核、后端导出、搜索和基准入口。
- `corpus/`、`spec/`：生产 Source、目标规格与机器可读清单。
- `docs/`：当前公开架构、构建、流程与唯一最终任务。
- `local/`：被 Git 忽略的内部理解、真实实验结果和历史归档；它不是公共合同的一部分。

## 快速开始

```bash
python -m pip install -e '.[verify]'

cmake -S . -B build -G Ninja \
  -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir \
  -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm
cmake --build build --target intent-opt

PYTHONPATH=python python tools/scrc_batch.py --validate-manifest --json
PYTHONPATH=python python tools/scrc_frontend.py --list --json
PYTHONPATH=python python tools/frontend_dataset.py --validate
```

`scrc_frontend.py` 的共享清单和完整生命周期仅适用于 Triton/TTIR。CUDA、CuTe
和 TileLang 保留各自的 compiler/observer/Gate runner；native Dataset 由对应的
前端入口独立检查和执行，不转送到 Triton/TTIR 的 SCRC runtime。

## 可运行的 rundemo

仓库根目录的 [`rundemo/`](rundemo/) 存放可以直接调用的真实流程封装。它们只调用
现有工具入口，不生成 mock 输入，也不创建 `python/tests`：

```bash
bash rundemo/01_validate_scrc_catalog.sh
INTENTIR_DATASET_ROOT=/path/to/Dataset bash rundemo/02_validate_multifrontend_dataset.sh
INTENTIR_DATASET_ROOT=/path/to/Dataset bash rundemo/03_run_liger_softmax_scrc.sh
INTENTIR_DATASET_ROOT=/path/to/Dataset bash rundemo/04_probe_cuda_dataset.sh
INTENTIR_DATASET_ROOT=/path/to/Dataset bash rundemo/05_probe_tilelang_dataset.sh
```

各 demo 的职责不同：

- `01` 校验 28 个 Triton/TTIR SCRC case 的真实清单与 Source resolution；
- `02` 校验外部 CUDA/TileLang Dataset manifest；
- `03` 运行真实 `high_performance:liger_softmax` 的 SCRC、Replay、Gate1 和 Gate2；
- `04` 使用 CUDA Dataset、真实 ABI 和 `CudaCompiler` 做 native CUDA 编译探针；
- `05` 使用 Dataset 中固定的 TileLang Python 环境做 native TileLang 编译探针。

`02`–`05` 不再假设 Dataset 的默认位置，运行前必须通过
`INTENTIR_DATASET_ROOT` 指定已有 Dataset；`02` 也支持将 Dataset 根目录作为第一个
位置参数传入。

`03` 需要当前被 Git 忽略的 `llm_providers.local.json`，默认读取仓库根目录；也可
通过 `INTENTIR_PROVIDER_CONFIG` 指定已有配置，或仅对当前进程注入
`OPENAI_BASE_URL`、`OPENAI_API_KEY`。它的 evidence 默认写入
`../intentir-evidence/rundemo/scrc/`，不会写入仓库；也可通过
`INTENTIR_EVIDENCE_ROOT` 或 `INTENTIR_DEMO_RUN_ROOT` 指定路径。所有 demo 都支持 `--help`；
CUDA/TileLang demo 在相应工具链不可用时会返回非零并报告真实阻塞原因。

完整依赖、真实闭包命令和后端工作流见：

- [系统架构](docs/architecture.md)
- [构建与运行](docs/build-and-run.md)
- [工作流](docs/workflows.md)
- [唯一最终任务](docs/final-task.md)

本 README 是项目说明、核心合同、真实 Kernel 示例和当前结论的主入口。更细的
示例拆解放在[项目指南目录](docs/project-guide/)；当前闭包事实和唯一权威运行
记录以最终任务为准：

- [真实 Kernel 示例](docs/project-guide/examples.md)
- [唯一最终任务](docs/final-task.md)

## 一个真实 Kernel 例子：Liger softmax

这个例子来自真实的 `high_performance:liger_softmax` workload，不是 mock 或
单元测试。Source 的核心计算是：

```python
mask = offs < n_cols
x = tl.load(X_ptr + row_id * X_row_stride + offs,
            mask=mask, other=-float("inf"))
m = tl.max(x, axis=0)
e = tl.exp(x - m)
d = tl.sum(e, axis=0)
y = e / d
tl.store(Y_ptr + row_id * Y_row_stride + offs, y, mask=mask)
```

对应的 J 语义边界保留了同一份逻辑输入和两个下游消费者：

```text
a0: load t0, predicate = (k1 < x1), fill = -inf
v1: i1 guard derived from iota/lt
n2: reduce(max, t0), outside = -inf
n5: reduce(sum, v4), outside = 0
a1: predicated store t2
```

这里 `a0` 是一个共享的 node-less load，但它有两个明确的
`consumer_extensions`。`n2` 和 `n5` 的 outside identity 不相同，不能合并成一个
fill，也不能复制成两个物理 load。这个关系是 Gate2 J reference 和生产 typed
contract projection 共同消费的边界。

本例的 C 绑定是：

```text
C slot: c0 / constexpr / i32
C value: c0 = 128
```

在真实生产后端，`target_C` 由对应 CUDA/CPU/RISC-V 后端生成和消费；在本例的
Replay/Gate1/Gate2 验证中，Replay 使用 Source/Harness 已经绑定的同一份
`artifact.c`，不会重新生成独立的 `target_C`。

## 本轮结构改进

- guarded load 从“只能绑定一个消费者”改为支持一个共享 load 对多个显式
  `consumer_extensions`，并保留每个消费者的 `outside`、`identity_law`、
  `operand_index` 和 `tile_axis`。
- 区分 Source 物理 load 的 `fill_value` 与下游 reduce/scan 的 outside identity，
  避免用 `-inf` 错误替代 sum 的 `0`。
- 修正 J reference 对完整 rank singleton broadcast 的处理，避免重复插入轴造成
  `(2, 2, 127)` 这样的错误形状。
- 将同一 guarded-load 合同同步到生产 typed contract projection；CUDA 结构分析不再
  因 node-less shared load 被误报为 unsupported。
- Replay continuation 的默认后端固定为 `responses`，避免未显式传参时误走
  MiniSWE/Codex transport 路径。

## 当前保留的真实闭包证据

`high_performance:liger_swiglu` 是 `docs/project-guide/real-evidence` 中唯一保留证据的真实 Kernel：

```text
main_status: SCRC_CERTIFIED
Gate1: replay_closed
Gate2: gate2_pass
Source–Replay: pass
Source–J: pass
Replay–J: pass
```

完整 evidence 位于：

`docs/project-guide/real-evidence/liger_swiglu/closure/`

该证据记录 CandidateCheck 通过、TypeCheck 实际调用一次、masked-J necessity 已证明，
且 Gate2 三条比较边全部通过。它不代表其他活动 Kernel 已完成闭包；当前全局
闭包数和 campaign 停止位置仅以 `docs/final-task.md` 为准。

## 验证原则

活动树不包含测试目录，也不以 mock、smoke、协议测试或 TypeCheck 数量替代研究结论。TypeCheck 和构建只负责机械前提；进展只由真实 Kernel 的 Gate1 闭包、Gate2 三条比较边、闭包 Kernel 数量和最终交付物衡量。历史测试、旧文档和 Trellis 状态只保存在本地归档中。
