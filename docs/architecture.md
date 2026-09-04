# 系统架构

## 1. 系统边界

IntentIR 把“理解一个源 Kernel”和“为目标硬件实现它”拆成两个可独立验证的过程。Lift 先冻结完整、后端无关的语义归档 `J=A+B`；Source C 的声明可机械投影为值无关的强类型 Layer-C 接口，target backend 也会从同一 J 推导其接口。Source 闭包使用当前前端的 `C_source` 与 `π_source`；生产 lowering 则根据 target/capabilities 选择或生成可替换的 `C_target` 插件。

```text
                                  +--> BindSource --> C_source or ∅ --> Replay/Gate
                                  |
Source + Harness --> LiftSemantic --> J=A+B + E_C
                                  |
                                  +--> BindTarget(target, capabilities)
                                                   |
                                                   v
                                               C_target
                                                   |
                                                   v
                                            ValidatePlugin
                                                   |
                                                   v
                                        Materialize J + Lower
```

Lift 不因当前 GPU、CPU、RVV 或前端 Harness 改写 A/B。对同一个 J 切换 CUDA、RVV 或其他后端时，下列不变式必须成立：

```text
canonical_bytes(J_cuda) == canonical_bytes(J_rvv)
digest(J_cuda) == digest(J_rvv)

C_cuda != C_rvv                         # 允许且通常应当不同
Lower(J, C_cuda) != Lower(J, C_rvv)     # 目标程序可以不同
RefSemantics(J)                         # 可观测语义始终相同
```

## 2. 对象所有权

| 对象 | 拥有的事实 | 不得承担的职责 |
| --- | --- | --- |
| Layer A | 逻辑张量、dtype/shape、operator、def-use、算法 control/recurrence、返回值 | tile、warp/thread/block、vector width、layout、memory level、launch、pipeline 或目标 API |
| Layer B | 逻辑 domain、access/index、predicate/fill、effect、alias/storage identity、update/result relation、后端无关的 realization permission | program/lane 坐标、地址形成、具体 atomic/memory/layout 实现；兼容期仍存在的 `tile_axis` 物理载体须按迁移顺序移除 |
| `J=A+B` | 完整、可独立参考执行的观测语义和稳定 attachment handle | 目标选择、Harness 本次估值、源文件路径、算法名称驱动的后端决策 |
| `E_C` | 当前已实现的值无关标量扩展点、required/optional、stable subject/axis attachment 与 interface digest | 作为数学语义洞，或把尚未实现的完整结构 grammar 说成已经持久化 |
| `C_source` | 当前 Source 前端实现的活动参数和结构绑定 | 作为 CUDA/RVV 生产插件，或影响 J 成员关系 |
| `C_target` | target/J/interface/capability-snapshot identity、required 参数绑定、后端候选 registry 的闭合 key+digest | 为不需要实际值的点伪造参数，新增算法 operator/predicate/effect/result，携带任意源码或自由文本策略，修改 J |
| `π_source` | Gate1 的冻结、带类型 Source–Replay 观测对应 | 进入 Replay 或生产 lowering，改写观测，携带后端 recipe |
| Harness | 实际 ABI、输入、shape 估值、Source grid 和 Source compiler-view | 成为 J 语义候选集，将一次估值固化为 A/B |
| Gate2 `Q` | 外部数值比较政策和容差 | 写入 J/C 或为错误结果放宽语义 |

`E_C` 所说的“占位符”是 realization 接口，不是不可执行的数学洞。J 仍必须完整表达算法语义，使 `RefJ` 和 Gate2 能够在不知道某个目标布局或 tile 的情况下执行。

## 3. A/B 与 C 的语义防火墙

A/B 只保留 Source 算法可观测的逻辑事实。对于 dtype/shape、loop、pad/tile、domain/index、storage 等既可能表达语义、也可能表达物理实现的字段，Lift 必须按下列规则分类：

- 改变 logical tensor、数学 operator、逻辑 control/recurrence、access/effect 或 result 的事实属于 A/B。
- 只改变如何在目标上实现同一语义的参数、计算方式和计算结构属于 C。
- 算法本身的 loop/reduce/pad/tile 可以留在 A；strip-mining、vectorization、parallel reduction、physical padding、fusion/split 和 software pipeline 必须通过 C 在临时 lowering IR 上实现。
- 公开 logical dtype/shape 留在 A/B；register/vector/fragment type、physical padded shape、address space 和 accumulator realization 进入 C。
- semantic storage identity、alias 和 loop-carried lifetime 可以留在 B；register/shared/global 放置、scratch allocation、reuse 和 double buffering 进入 C。
- logical domain/index/predicate/fill 留在 B；program/block/thread/lane 坐标、coalescing、swizzle、vector mask 和 physical tail 进入 C。

C 不是任意 JSON 或源码侧通道。最终目标是让每种结构化选择都来自闭合 grammar，并具有明确类型、target capability 要求、稳定 J attachment 和语义保持条件。当前实现完成了标量接口与 attachment 校验；更丰富的 memory/layout/parallel/pipeline/fusion 结构暂由后端内部候选 registry 表达，插件只运输不透明 key 和覆盖完整内部候选的 digest，不运输 pass/source/compiler text。

### 3.1 当前兼容字段与迁移顺序

`Boundary.reductions[*].tile_axis`、`Boundary.consumer_extensions[*].tile_axis`、`ScopeKind.PROGRAM_INSTANCE` 以及部分 pointer/opaque/compiler ABI carrier 仍被 functional projection、TypeCheck、Replay 或 RefEval 消费。它们不能直接删除。安全迁移顺序固定为：

1. 为 Source/target Layer C 建立稳定 subject/axis/effect/authorization attachment；
2. 兼容期双读旧 J 字段与新 C attachment，并要求二者一致；
3. 迁移 backend、functional projection、RefEval、TypeCheck 和 Replay 消费者；
4. 最后从 J wire/Lift schema 删除仅表示物理 realization 的字段。

## 4. Source Replay/Gate 路径

`C_source` 描述当前 Source 前端的实际 realization，仅用于 source-blind Replay 与 Gate1。trusted compiler-view 在构造和规范化阶段排除没有实际消费者的非活动扩展点；下游不再逐槽删除、补值、压缩或重编号。

- active `C_source` 为空时，Replay 只接收参数化 J 和固定工具环境 `U_f`；编译所需占位值由 `J.signature` 确定性生成，不伪装成 C 的实际参数。
- active `C_source` 非空时，Replay 接收完整 C，每个活动槽都必须有真实、类型匹配的绑定。`constexpr c*` 仍是 callable ABI 占位符，C value 只是本次 Source 编译绑定。
- packet、compiler-index、episode workspace 和 Replay final ingress 必须对同一份规范化 J/C/U_f 失败关闭。
- Replay 不可见 Source、`π_source`、Gate 差异或失败历史；C 不得补写 J 语义。

Source observer 和 Replay observer 独立产生原始观测；observer 不读取 `π_source`。normalizer 依据同一个冻结 witness 对两侧分别规范化，不得通过联合查看两侧制造相等。

## 5. Target 插件与 Lowering

target backend 只从 canonical J、`E_C`、target identity 和明确 capabilities 生成或选择 `C_target`，不读取 Source 名称、Source compiler-view、`π_source` 或历史 per-kernel 标签。当前 CUDA/RVV 插件冻结 target、J/interface digest、capability-snapshot digest、标量 bindings 以及候选 key+digest；resolver 会从可信 registry 重新枚举并重新准入候选。候选内部的丰富结构尚未逐字段复制到插件 wire，因此当前保证是“不能由 caller 注入或篡改、可精确重建”，不是“所有结构都已有独立公开 grammar”。

当前 scalar 接口的“完整”以 required points 为边界：需要实际参数的点必须由 C 真实绑定；optional 且 toolchain 允许保持参数化的点直接使用 J 确定性生成的编译占位符，不为凑齐插件合成假值。target-conditioned point 是后续结构 grammar 工作，尚未在接口中实现。若后续阶段开始要求具体值，必须重新 BindTarget 和 ValidatePlugin，Lower 不得暗中补值。

`ValidatePlugin` 在 Lower 之前至少验证：

1. C 的 target identity 与当前 backend 一致，capability-snapshot digest 能由当前后端派生事实精确重算；
2. C 与正确的 J digest/extension-point contract 绑定；
3. required binding 完整，且无额外、错类型或未知选择；
4. 每个 subject/axis/authorization attachment 均引用存在的 J 声明；
5. 内部候选所要求的 capability 均由当前 target contract 派生，候选 key+digest 在重建后的 universe 中唯一命中；
6. C 不携带新的 logical operator、predicate、effect、result、任意源码或自由文本策略。

验证通过后，后端先由 J 确定性物化 target-neutral Intent MLIR，再由插件解析到可信内部候选并应用到临时 lowering IR，最后进入目标后端、LLVM、PTX 等路径。无插件的 CUDA public Lower 只允许 deterministic generic 路径；需要 specialized/auto 结构时必须有插件。Lowering 不得回写 J。

## 5.1 Lift 当前解耦程度

SCRC Lift 已将 `semantic_lift_view` 与 `source_realization_view` 分开，并分别由 `_validate_source_semantic_view` 与 `_validate_source_realization_config` 校验；前者不含 schedule value、grid、actual shape/value 或 compiler role。Lift 成功后，Source C 声明会机械投影为一等的内存对象 `LayerCInterface` 并计算 digest。

仍需如实区分两项兼容限制：当前 Gate1 artifact wire 仍是 `{schema,j,c,pi}`，尚未把 `E_C` 作为独立 root 持久化；Lift worker 仍是单次模型调用，并因 Source compile、TTIR observer 与旧 CandidateCheck 兼容而可见 flat compiler-view。nested projection 与 flat copy 会做 drift rejection，但这还不是物理上完全不可见的两阶段 Lift。

## 6. Gate1 与 Gate2

Gate1 比较 Source 与 Replay 的规范化编译器观测。正常臂必须闭合；任何 Replay ingress、编译、观测、规范化或比较失败都不得进入 Gate2。

Gate2 在冻结 Harness 上运行三个执行腿：

1. 原始 Source；
2. Gate1 已接受的精确 Replay 字节；
3. 从同一 J 构造的参考执行 `RefJ`。

比较策略由外部 Q 决定，不写入 J/C。Source–Replay、Source–J、Replay–J 必须全部通过；TypeCheck、构建、插件合同检查或任何单边通过都不构成 Kernel 闭包。

## 7. 前端与后端

活动 SCRC 清单支持 Triton、TileLang 和 CUDA 真实 Source。前端只负责提供 Source/Harness/compiler-view 投影和 `C_source`，不得将其程序坐标、API 或物理 tile 固化到 J。公开可运行能力以 `tools/scrc_frontend.py --list --json` 的实际输出为准。

- CUDA 后端拥有 NVIDIA target plugin、Intent MLIR 到 GPU/PTX/cubin 的 lowering、Driver 执行与真实 target search。
- RISC-V 后端拥有 RVV target plugin、Intent MLIR 到 LLVM/RVV 的 lowering、对象导出与目标机 search/timing。
- CPU 标量路径主要服务 Gate2 `RefJ`；它不是 CUDA/RVV 生产性能的替代证据。

研究消融可以显式擦除 C binding 或限制插件候选空间，但这属于外部实验臂；后端不得从 Source 名称、历史手工标签或其他会话恢复被擦除的信息。

## 8. 可移植性边界

Git 跟踪的仓库包含构建系统、Python 包、生产 Source、机器可读规格和公共文档。Dataset、大模型 provider 配置、GPU/RVV 运行证据、内部研究笔记和历史归档均位于仓库之外或 `local/`，通过显式路径注入。任何新机器都应能在没有这些私有文件时构建核心系统并列出静态能力；真实闭包和目标基准则需要相应 Dataset、工具链和硬件。
