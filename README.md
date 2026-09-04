# IntentIR

IntentIR 是一个把高性能 Kernel 的源程序提升为可验证语义、再从该语义生成 CPU/CUDA/RISC-V 实现的研究系统。仓库只维护一条权威主线：前端 Lift 冻结后端无关的 `J=A+B`；Source 声明可机械投影为值无关的 Layer-C 接口，生产后端再从同一 J 推导目标接口并使用用途不同、可替换的 C 插件。

系统的核心边界是：

```text
LiftSemantic(Source) -> (J=A+B, E_C, π_source)

BindSource(J, E_C, trusted compiler-view) -> C_source or ∅
Replay(J; U_f) -> R                    when active C_source is empty
Replay(J, C_source; U_f) -> R          otherwise, with every active slot bound
Gate1(Source, R, π_source)
Gate2(Source, R, RefJ) = Source–Replay ∧ Source–J ∧ Replay–J

BindTarget(J, E_C, target, capabilities) -> C_target
ValidatePlugin(J, E_C, C_target, target) -> accept or fail closed
Lower(J, C_target, target) -> target program
```

- `J=A+B` 是与源语言和目标后端无关的语义归档。Layer A 拥有逻辑张量、operator、def-use、算法 control/recurrence 和结果；Layer B 拥有逻辑 domain、access、predicate、effect、alias 和输出关系。
- `E_C` 是附着在稳定 J handle 上的强类型实现扩展点。当前代码已支持值无关的标量点、required/optional、subject/axis attachment 和接口 digest；Source C 声明会机械投影为该接口，CUDA/RVV 也会按 J 推导各自接口。memory/layout/parallel/pipeline/fusion 等丰富结构目前仍主要由后端内部候选 registry 的闭合 key+digest 表达，尚未全部迁为独立持久化 grammar。
- `C_source` 只是 Source 当前前端实现的绑定，仅服务 Replay/Gate。trusted compiler-view 排除无实际消费者的非活动槽；留下的活动槽必须全部使用真实、类型匹配的值绑定。无活动 C 时不合成实现值，直接从 `J.signature` 确定性生成可编译占位符。
- `C_target` 是按 target/capabilities 选择或生成的可替换后端插件。它绑定当前 target 实际要求的标量参数，通过不透明 candidate key+digest 引用后端内部已重新枚举、重新准入的实现结构，并绑定 target、J、接口和 capability-snapshot digest；明确不要求具体值的参数继续使用 J 确定性生成的编译占位符，不伪造 C value。插件不得新增 operator、predicate、effect 或 result，也不得改变 J 的可观测语义。
- `π_source` 是 Gate1 使用的冻结、带类型对应关系。它可以表达程序轴、访问、guard、布尔和归约关系，但不进入生产 lowering，也不能修改 Source 或 Replay 的观测结果。
- CUDA、RVV 或其他后端必须消费同一份 J canonical bytes/digest；切换后端只替换 `C_target`。缺失、额外、错类型、错 attachment、错 target 或不受 capability 支持的插件都在 Lower 前失败关闭。
- 当前 Lift 已把 semantic compiler-view 与 Source-realization compiler-view 分开验证，但为兼容 Source compile/observer/retained workspace 仍是一次模型调用，模型仍可见 flat compatibility compiler-view；真正物理隔离的两阶段 Lift 尚未完成。
- Gate1 必须先闭合；Gate2 随后在同一 Harness 估值上执行 Source、Replay 和 `J` 的参考语义，三条比较边必须同时通过。

## 仓库结构

- `include/`、`lib/`：IntentIR 方言、转换与 CPU/CUDA/RISC-V lowering。
- `python/intentir/ir/`：`J` 的规范化、物化和生产快照边界。
- `python/intentir/extraction/`：Source 到 backend-neutral `J=A+B`、Layer-C 扩展点和 Source Gate witness 的 Lift 入口。
- `python/intentir/verify/`：TypeCheck、Gate1、Gate2、真实 workload 与证据生命周期。
- `python/intentir/backend/`：Layer-C target plugin 生成/验证、CUDA 与 RISC-V 生产后端，以及 Gate2 使用的 CPU 标量参考路径。
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
```

完整依赖、真实闭包命令和后端工作流见：

- [系统架构](docs/architecture.md)
- [构建与运行](docs/build-and-run.md)
- [工作流](docs/workflows.md)
- [唯一最终任务](docs/final-task.md)

## 验证原则

活动树不包含测试目录，也不以 mock、smoke、协议测试或 TypeCheck 数量替代研究结论。TypeCheck 和构建只负责机械前提；进展只由真实 Kernel 的 Gate1 闭包、Gate2 三条比较边、闭包 Kernel 数量和最终交付物衡量。历史测试、旧文档和 Trellis 状态只保存在本地归档中。
