# 工作流

## 1. 统一流程合同

所有前端、Source 闭包和生产后端都以同一份后端无关语义归档为边界：

```text
LiftSemantic(Source)
  -> J=A+B
  -> typed Layer-C extension points E_C
  -> π_source

BindSource(J, E_C, trusted compiler-view)
  -> C_source or ∅
  -> only for Source Replay/Gate

BindTarget(J, E_C, target, capabilities)
  -> C_target
  -> replaceable backend plugin

ValidatePlugin(J, E_C, C_target, target)
  -> accept or fail closed

Lower(J, C_target, target)
  -> target program
```

`J=A+B` 必须完整表达算法的可观测语义。`E_C` 只声明 realization 接口；`C_source`、`C_cuda` 和 `C_rvv` 是用途与身份不同的绑定实例，不能互相冒充：

```text
C_source != C_cuda != C_rvv
```

它们可以遵守同一个强类型接口合同，个别数值也可能碰巧相等，但 Source 绑定不能直接作为新后端插件使用。

## 2. LiftSemantic 解耦工作流

LiftSemantic 的目标职责只理解 Source 的算法语义，并按以下顺序生成归档。当前实现已经拆分 semantic/source-realization projection 与 validator，但仍由同一次模型调用产出兼容 `{j,c,pi}` wire；因此以下流程同时包含已实现边界和剩余物理隔离目标：

1. 提取 logical tensor、dtype/shape、operator、def-use、算法 control/recurrence 和 result，形成 Layer A。
2. 提取 logical domain/index、access、predicate/fill、effect、alias/storage identity 和 update/result relation，形成 Layer B。
3. 对每个可能因硬件目标或 lowering 策略而改变的参数、计算方式或计算结构，保留稳定的 J attachment，并声明强类型 `E_C`；不得把当前 tile、warp、thread、vector、layout、memory level、launch、pipeline、fusion/split 或目标 API 固化到 A/B。
4. 保证删除具体 realization 后，J 仍可由 `RefJ` 独立执行。算法本身的 loop、reduce、pad 或逻辑 tile 留在 A/B；只服务物理实现的 strip-mining、并行归约、physical padding、swizzle、double buffering 等进入 `E_C`。
5. 分别规范化 J 与 `E_C` 合同，保存 J canonical bytes/digest 和 extension-point contract digest。当前 Source 声明在 Lift 成功后机械投影为内存中的 `LayerCInterface`；Gate1 wire 尚未新增独立 `E_C` root。`π_source` 单独冻结，只供 Source–Replay 证明。

当前 extension point 已有稳定 slot/role、闭合标量类型、required/optional 和 J subject/axis attachment；接口有 canonical digest。完整目标还包括独立的结构 grammar、点间约束和 target-conditioned binding。占位符是 realization interface，不是允许 C 修补数学语义的洞。

若某个位置不需要实际参数，Lift 不创建伪造的 C value。它保留 J 中的参数化语义和 `E_C` 的可选接口，后续直接从 J 确定性物化编译占位符。若 target compiler 最终要求具体值，则该位置在进入 Lower 前必须由相应 C 插件真实绑定。

## 3. Source Bind、Replay 与 conditional-C

`BindSource` 只读取 J、`E_C` 和本次实际 Source 编译的 trusted compiler-view。它不得读取目标后端搜索结果，也不得把 Source 的物理实现写回 J。

1. compiler-view 判定哪些 extension point 在当前 Source 实现中有实际消费者。
2. 无实际消费者的点在 Source binding 中保持非活动；下游不得逐槽删除、压缩或重编号。
3. 每个活动点都必须取得真实、类型匹配的编译绑定；禁止猜值、默认补值和使用占位值伪装实际参数。
4. 产出规范化的 `C_source`；若没有活动点，则产出 `∅`，而不是空 C 对象。

conditional-C 的 packet 与 episode 合同固定如下：

```text
active C_source = ∅
  -> packet {j,u_f}
  -> 9-file episode
  -> no c.json
  -> compile placeholders generated deterministically from J.signature

active C_source != ∅
  -> packet {j,c,u_f}
  -> 10-file episode
  -> c.json == packet.c == artifact.c
  -> every active slot has a real, complete, type-correct value
```

在 J-only 分支，pointer/tensor/scalar/constexpr 所需的构造编译值只从 `J.signature` 确定性生成；这些值是让参数化程序可编译的占位符，不是 Layer-C 的硬件事实。在 complete-C 分支，`set(C_source.values)` 必须精确等于活动槽集合；缺失值、`null`、额外值、错类型、错 attachment 或非法 compiler role 均失败关闭。

Replay 为 Source-blind：只能读取本 arm 的 J、可选完整 `C_source` 和固定工具环境 `U_f`，不可见 Source、`π_source`、Gate 差异、失败历史或其他候选。packet、compiler-index、episode workspace 与 final ingress 必须验证它们来自同一份 J/C/U_f，不能将派生文件当作独立事实源。

## 4. Gate1、Gate2 与真实 Kernel 闭包

每个正式 SCRC 轮次按以下顺序执行：

1. 在静态清单或多前端 Dataset 中固定真实 Source、Harness、compiler-view、设备和 lineage。
2. 执行 LiftSemantic 和 BindSource，构造新的 9/10 文件 source-blind Replay episode。
3. final Replay ingress 检查安全语法、exact ABI、constexpr annotation、非法 carrier、C attachment/消费和程序轴范围；只保存并编译 ingress 接受的精确 Replay 字节。
4. Source observer 与 Replay observer 独立取证；observer 不读取 `π_source`。normalizer 只依据冻结 witness 分别规范化两侧观测。
5. Gate1 只有在规范化观测完全相等时闭合；任何 ingress、编译、观察、规范化或比较失败都禁止进入 Gate2。
6. Gate2 在同一 Harness 估值上执行 Source、Gate1 接受的精确 Replay 和 `RefJ`，保存 Source–Replay、Source–J、Replay–J 三条比较。
7. 只有 Gate1 闭合、Gate2 三边全通过且 evidence/lineage 完整时，才记录 `scrc_certified` 并更新闭包清单。

批量入口支持清单验证、dry-run 和真实运行：

```bash
PYTHONPATH=python python tools/scrc_batch.py --validate-manifest --json

PYTHONPATH=python python tools/scrc_batch.py --dry-run \
  --case '<registered-case-id>' \
  --run-root '/path/to/evidence/scrc-batch'

PYTHONPATH=python python tools/scrc_batch.py --run \
  --case '<registered-case-id>' \
  --run-root '/path/to/evidence/scrc-batch'
```

TypeCheck、import、compile/build、协议检查、插件验证、mock/smoke 或任意两条比较边都不是 Kernel 闭包。

## 5. Backend plugin、验证与 Lower

生产后端固定 J 后再绑定目标插件：

```text
canonical J + E_C + target/capabilities
  -> generate/select complete C_target candidates
  -> ValidatePlugin
  -> deterministic target-neutral Intent MLIR materialization from J
  -> apply C_target to ephemeral lowering IR
  -> target lowering and structural candidate admission
  -> compile and execute
  -> correctness audit
  -> target timing/search report
```

`BindTarget` 只能依据 J 结构、强类型 `E_C`、明确 target identity 和 capability snapshot 生成或选择候选。当前 `C_target` 显式保存标量 binding、target/J/interface/capability digest，以及一个不透明 candidate key+digest；后端用它在可信内部 registry 中重新枚举、重新准入并唯一解析完整候选。完整目标仍是把 tile、vector width、memory/layout、parallel mapping、pipeline、fusion/split 和 lowering intrinsic 逐项迁入公开的闭合结构 grammar；在完成前不得声称插件 wire 已逐字段表达这些结构。

对于标记为必须具体化的 extension point，`C_target` 必须提供实际值或结构选择。对于明确允许保持参数化、且当前 target toolchain 不需要具体值的点，后端直接保留从 J 确定性生成的编译占位符，不得为凑齐 C 而合成假参数。一旦后续编译阶段需要实际值，必须回到 BindTarget 产生新的完整插件并重新验证，不能在 Lower 内部暗中补值。

`ValidatePlugin` 在任何 lowering 之前失败关闭，至少拒绝：

- wrong-target plugin 或 target identity 不一致；
- stale、缺失或不兼容的 capability snapshot digest；
- 错误的 J digest 或 `E_C` contract digest；
- 缺失、额外、错类型或重复 binding；
- 未知 subject/axis/authorization attachment；
- 候选要求的 capability 不在当前 target contract 中，或 key+digest 不在重建后的候选 universe；
- 新增或改写 operator、predicate、effect、result 的 C；
- 携带任意源码、自由文本策略或无法由闭合 grammar 验证的 C；
- Lower 需要但插件未记录的结构选择。

验证通过后，J 先被确定性物化为 target-neutral Intent MLIR。C 只作用于临时 lowering IR，既不能回写 J，也不能改变 `RefSemantics(J)`。Lower 完成后必须复核 J canonical bytes/digest 未变化。

## 6. CUDA/RVV 后端切换不变量

同一 Source 的 CUDA 与 RVV 流程必须从同一份 LiftSemantic 输出开始：

```text
canonical_bytes(J_cuda) == canonical_bytes(J_rvv)
digest(J_cuda) == digest(J_rvv)

C_cuda != C_rvv
Lower(J, C_cuda) != Lower(J, C_rvv)
RefSemantics(J) remains unchanged
```

跨后端复核至少保存 J bytes/digest、`E_C` contract digest、target/capability digest、plugin digest、验证结果和 lowered artifact digest。若绑定或 lowering 前后 J 的任何字节变化，整个后端结果无效。

CUDA 内部候选负责 NVIDIA memory/layout、thread/block/warp、pipeline 和 GPU lowering 选择；RVV 内部候选负责 vector-length policy、lane mapping、memory access、tail policy 和 LLVM/RVV lowering 选择。各自 plugin 以 target/capability/key+digest 锁定候选，不共享实现实例。CPU 标量路径主要服务 `RefJ`，不能替代 CUDA/RVV 的真实目标性能证据。

### 6.1 兼容 Source-prior 工具边界

`ProductionSnapshot` 的持久化字段 `c` 仍是 `ImplementationConfig`，只供现有 benchmark/export 工具通过 `attach_c()` 作为可擦除的 Source/search-order prior。它不是 `C_target`，不能进入 `LoweringRequest`。新 production Lower 的公共边界只接受 `TargetLayerCPlugin | None`；工具迁移完成前，文档和报告必须明确标注这条 legacy prior 路径。

## 7. 精确 artifact 的零模型能力复核

`scrc_gate2_recheck.py` 复用 parent 中已经接受的 artifact 与精确 Replay，不调用新的 Lift/Replay 模型。它先让历史字节经过当前 packet-aware Replay ingress 和 Gate1；只有当前 Gate1 闭合时才继续 Gate2：

```bash
PYTHONPATH=python python tools/scrc_gate2_recheck.py \
  --case '<registered-case-id>' \
  --artifact '/path/to/parent/attempt-N/artifact.json' \
  --parent-evidence '/path/to/parent/attempt-N' \
  --dataset-root '/path/to/Dataset' \
  --run-root '/path/to/evidence/recheck' \
  --json
```

Parent 必须包含 `gate1.json`、`replay.py` 与 `replay_receipt.json`；其本身或祖先运行目录还必须包含精确 `request.json`、`source_projection.json` 和 `source.py`。lineage 不一致时失败关闭。

该流程只是历史 exact-byte capability diagnostic。即便当前 Gate1/Gate2 通过，也只能记录非认证性的 `capability_execution_pass`；它不证明 fresh Replay agent 消费了当前 9/10 文件 episode，不改变 parent verdict，也不单独增加闭包计数。未通过当前 ingress 的旧字节记录为 `gate1_no_closure`。`scrc_gate1_recheck.py` 会创建新 Replay 并调用模型，但仍是诊断工具，不进入 Gate2，也不改变 parent verdict。

## 8. 多前端、证据与文档

统一入口 `tools/scrc_frontend.py` 根据登记项选择 Triton、TileLang 或 CUDA 路径。前端只负责真实 Source/Harness/compiler-view 投影、LiftSemantic 和 `C_source`；J 的所有权、Replay 可见性、Gate 顺序和 Gate2 三边规则不随前端变化。

```bash
PYTHONPATH=python python tools/scrc_frontend.py --list --json
```

列表输出是当前可运行能力的权威视图。Dataset 缺失或本机前端依赖不可用时应报告 unavailable，不能用模拟 Source 替代。

- 公共 `docs/` 只描述当前系统合同、可复现入口和唯一最终任务。
- `docs/final-task.md` 是唯一活动执行记录；旧状态应由当前事实替换，不追加成实现日记。
- `local/internal/scrc/`、`local/internal/frontend-extension/`、`local/internal/backend/` 保存不提交 Git 的内部理解和重要真实结果。
- `local/archive/` 保存旧文档、合并前快照和历史材料；这些内容不得重新成为活动流程。
- 大型 evidence 建议保存在仓库外，并在最终任务中记录绝对路径、case、Gate 状态、J/plugin digest 和 parent lineage。
