# 最终任务：单 Kernel Lift、A/B–C 解耦与真实闭包

本文是 `/home/wwq/TianchenIR` 唯一活动执行记录，只保留当前合同、完整流程、真实 Kernel 证据、未解决阻塞和下一步。旧实现过程、逐次变更日记、机械检查结果和过期结论不在此累积。

当前权威状态保持不变：

```text
活动 Kernel：28
已闭包：24/28
未闭包：4/28
```

这里的 `24/28` 是本文件维护的权威声明。`spec/lowering/high-performance-kernels.json` 定义 28 个 Kernel 的 workload universe，但它不是证书汇总台账，不能单独证明 24 张闭包证书。当前修改没有形成 fresh Lift、fresh Replay、Gate1 closure、Gate2 三边和完整 lineage，因此 closed-Kernel 数不能增加。

## 1. 最终合同

Lift 必须把算法语义和硬件 realization 分开：

```text
Layer A + Layer B = canonical J
Layer C           = 可替换的后端 realization 插件
```

### 1.1 Layer A

Layer A 保存源代码的算法结构：

- logical tensor、dtype、shape 和结果；
- 数学 operator、常量、def-use；
- 算法本身需要的 control、recurrence、reduction 和 state transition；
- 不因 CUDA、cuTile、TileLang、RISC-V 或其他后端改变的计算含义。

Layer A 禁止保存：

- block、thread、warp、lane、worker、program id；
- tile size、vector width、layout、memory space；
- launch geometry、pipeline stages、double buffering；
- 目标 API、目标源码、compiler command 或按 kernel 名选择的 recipe。

### 1.2 Layer B

Layer B 保存后端无关的可观测边界：

- logical domain、coordinate、index expression；
- logical access、predicate、fill；
- effect、result relation、update、alias 和 semantic storage identity；
- logical ABI、shape/dimension relation；
- 对 realization 的强类型授权，但不保存具体 realization。

Layer B 中任何可能被硬件改变的结构必须拆成稳定的 typed attachment handle，由 Layer C 绑定，不能继续以开放字符串、源码片段或隐藏物理字段留在 J。

### 1.3 `E_C`、`C_source` 与 `C_target`

`E_C` 是 value-free Layer-C extension interface。每个扩展点最终应包含：

- 稳定 id；
- 闭合 kind 和类型/结构 grammar；
- required、optional 或 target-conditioned binding mode；
- 指向 J subject、axis、access、effect 或 authorization 的 attachment；
- 可机械检查的约束和 capability requirement。

`C_source` 与 `C_target` 共享 typed extension 原语，但所有权不同：

| 对象 | 所有权 | 禁止事项 |
| --- | --- | --- |
| `C_source` | 当前 Source 编译确实需要的真实 constexpr/compiler realization | 不得充当生产后端插件，不得修改 J，不得为不活动参数猜值 |
| `C_target` | 当前 target/capability 对同一 J 的后端 realization 插件 | 不得新增数学 operator、predicate、effect 或 result，不得携带任意源码/命令 |

切换后端时必须满足：

```text
J 保持不变
E_C 保持稳定或做兼容扩展
C_target 随 target/capabilities 替换
Lower(J, C_target) 产生不同后端实现
RefSemantics(J) 不变
```

不得为了新后端重新 Lift、按文件名或算法名套模板，或修改 A/B 来迁就一个后端。

### 1.4 占位符和实际参数

规则只有两条：

```text
编译/执行确实需要实际 realization 参数
  -> 必须由可信宿主绑定真实 C_source 或 C_target
  -> 完整检查类型、attachment、required 值和 capability

不需要实际 realization 参数
  -> 不创建假 C 值
  -> 从 canonical J 确定性生成参数化 Kernel IR/源码占位符
  -> 占位符仍属于 J ABI 或编译接口，不得冒充 C
```

任何需要 C 才能决定数学结果的 J 都是不合格 J。C 只能改变“如何实现”，不能补全“计算什么”。

### 1.5 `π_source`

`π_source` 只属于 Gate1 宿主侧的 unary observation normalization：

```text
Source observation --N^(π_source)--> normalized Source observation
Replay observation --N^(π_source)--> normalized Replay observation
```

它不得进入：

- Replay Agent 或 Replay packet；
- Replay episode workspace；
- Intent Kernel MLIR；
- 任何生产 Lower、compile consumer 或目标 realization 请求；
- `C_target` 或生产 Lower；
- 后端 recipe 或目标源码。

每一臂必须独立把物理 occurrence 唯一映射回具体 J anchor。没有唯一 provenance 时，该 relation 保持 `hard`，normalizer不得做 module-wide 文本放宽。

## 2. 完整端到端流程

### 2.1 Source 选择和 Lift

```text
Source file + explicit source_symbol + Harness
  -> frontend source slicer
       对 Triton/CUDA/TileLang 等前端先验证真实 device entry
  -> single-kernel Intent projection
       精确选择一个顶层 runtime kernel
       保留所需 import、constant、class、enum、普通 helper、@intent.fn
       删除未选 sibling @intent.kernel
       对 sibling 引用或无法证明的 decorator 绑定失败关闭
  -> anonymize
       唯一 runtime entry = scrc_k0
       tensor/value/domain/access/effect/parameter/C slot 使用匿名 handle
  -> model-visible semantic projection
       anonymous Source
       semantic_lift_view
       value-free layer_c_declarations
  -> host-private Source realization
       actual constexpr/compiler binding
       concrete valuation、grid、schedule、compiler view
       不向 Lift Agent暴露
  -> LiftSemantic
       canonical J = Layer A + Layer B
       E_C declarations，candidate c.values 必须为空
       π_source
  -> CandidateCheck
       只检查 J/E_C/semantic ABI
  -> BindSource
       需要实际参数：完整真实 C_source
       不需要实际参数：C_source = ∅
  -> persisted artifact + value-free Layer-C interface
```

一个外部输入文件即使含有 `gemm`、`bf16_gemm`、`quantized_gemm` 等多个入口，也只能处理显式 `source_symbol` 选中的一个 kernel。不能把同文件所有变体合并到一个 J，也不能把任何外部实现复制为 Lift 结果。

### 2.2 Source Gate 分支

```text
J + optional complete C_source + public U_f
  -> conditional-C Replay packet
  -> fresh source-blind Replay Agent
  -> final ingress
  -> compile exact accepted Replay bytes
  -> Source/Replay 独立 observation
  -> 两臂各自执行 N^(π_source)
  -> Gate1 exact hard comparison
  -> Gate2 Source–Replay
  -> Gate2 Source–J
  -> Gate2 Replay–J
  -> complete request/source/projection/adapter/Lift/Replay/Gate lineage
```

唯一闭包判定是：

```text
Gate1 closure
AND Source–Replay passed
AND Source–J passed
AND Replay–J passed
AND complete exact-byte lineage
```

TypeCheck、构建成功、schema、文档、源码生成、capability diagnostic 或两条比较边均不能替代闭包。

每个 candidate branch 的生命周期暂时允许最多 `3` 次逻辑 Lift invocation；该上限由
`SCRCRunRequest.max_lift_calls` 统一执行，并由 batch、单 case frontend 和 continuation
入口显式传递。一次逻辑 Lift 内部的 provider/transport 重试只计入 physical-call telemetry，
不增加逻辑 Lift 次数；固定 artifact 的 Replay/Gate1 recheck 不调用 Lift，也不消耗该上限。

### 2.3 Production Lower 分支

```text
immutable canonical J + E_C
  -> deterministic single TianchenIR-owned neutral Kernel MLIR artifact
       exactly one intent.kind="kernel"
       无 intent_plan
       无 target identity
       无 actual C value
       无 π_source
       无原 Source 名称和路径
  -> BindTarget(target, capabilities)
       形成后端无关的 C_target plugin artifact
  -> ValidatePlugin(J, E_C, C_target)
  -> 通过公开文件或 CLI 边界交给可选、独立安装的 Lower consumer
  -> independent source-output + explicit device_entry_symbol
  -> Triton / cuTile / TileLang target source
  -> target compile and numerical correctness
  -> target source slice + anonymize
  -> 新的单 Source kernel
  -> 新一轮 Lift / Replay / Gate closure
```

当前已经落地这个分支的基础交接层：窄范围 J-only exporter 和 source slicing、backend-neutral 的
`BindTarget -> ValidatePlugin -> TargetPluginArtifact` bridge、公开序列化/反序列化边界、
CUDA/RISC-V consumer ingress，以及正式 generated-source manifest/adapter/lifecycle。neutral
exporter 当前有两个明确的空 `E_C` profile：dense pointwise，以及
`ragged -> ragged_outer/ragged_member` direct Ragged query；Ragged descriptor 只作为函数内部
tuple SSA 存在，不进入 function parameter/result 或 target ABI，common CPU/RISC-V 和 CUDA
consumer 使用同一张中立 graph，CUDA ingress 再将其转换为 destination-passing memref ABI。
仍未完成的是把这两个窄 profile 扩展到完整生产 grammar、所有独立 target consumer 的真实
compile/execute correctness、目标源码 round-trip closure，以及由这些结果组成的 fresh Gate
closure。

## 3. conditional-C Replay 当前合同

当前真实 packet 规则为：

```text
实际 C_source 非空
  packet = {j, c, u_f}
  c.json 必须存在
  c.json == packet.c == artifact.c
  required slot 必须完整且类型正确

实际 C_source 为空
  packet = {j, u_f}
  不创建 c.json
  编译占位参数由 J.signature 确定性生成
  不把占位值写入 C
```

`π_source` 在两种 packet 中都不存在。

真实 artifact 复核结果：

- FlashAttention forward 的非空 C artifact：6 个 slots、6 个真实 values，packet keys 为 `{j,c,u_f}`；
- split-K backward 的空 C artifact：0 个 slots、0 个 values，packet keys 为 `{j,u_f}`；
- 两者 packet 均无 `pi`。

因此不再需要“给空 C 填 mask/假值”的路径。若某个 constexpr 只是从 J shape 确定性派生，它应作为 semantic ABI/编译占位符处理；只有后端选择或编译确实需要的 realization 值才进入 C。

## 4. 单 Kernel 投影当前状态

实现位置：

- `python/intentir/extraction/lifters/scrc.py`
- `python/intentir/verify/frontend_workloads.py`

当前行为：

- 只识别精确的 `@intent.kernel`、`@alias.kernel` 或 `from intent import kernel` alias；
- 不按通用 `.kernel` 后缀误删 Triton 等其他前端装饰器；
- `from intent import *` 无法证明 decorator 来源时失败关闭；
- Intent decorator alias发生 module-level rebinding 时失败关闭；
- 模块存在 Intent runtime kernels 时，`source_symbol` 必须选中其中一个，不能选 `@intent.fn` helper；
- 删除 sibling kernel 前，若保留的顶层代码仍引用 sibling，失败关闭；
- sibling 删除发生在函数编号、局部扫描和 alpha rename 前；
- Lift prompt 明确 `scrc_k0` 是唯一 runtime kernel，禁止枚举、合并或重建 sibling variants。

这些 decorator 和 import 形式只作为输入源码的 AST 语法被识别；TianchenIR frontend不得导入、执行或依赖对应的 Python package。

实际多入口输入检查确认：显式 `source_symbol` 只保留一个 runtime kernel及其必要 helper；若 `source_symbol` 指向 helper，则失败关闭。该检查只说明当前 source-slicing capability，不构成 Gate闭包证据。

仍未完成的是完整 selected-kernel dependency closure。当前会拒绝保留代码中的显式 sibling name 引用，但没有机械删除所有只属于 sibling 的字符串注册表或动态反射结构；遇到无法静态证明的情况必须继续失败关闭。

## 5. canonical J 到 Intent Kernel MLIR

实现位置：

- `python/intentir/frontend/intent_kernel_mlir.py`
- `tools/export_intent_kernel.py`

exporter 使用 TianchenIR自有、私有、封闭且确定性的 Kernel IR emitter和MLIR serializer。它只接受已通过 representability检查的 canonical J projection；operation词汇、type、attribute、region形状以及SSA/operation ID allocator都在 TianchenIR内穷举定义。它不导入或调用任何外部 Python package、builder、源码树或内部API，也不复制外部实现，不按kernel、文件或算法名分派。当前有两个故意很窄且都要求空 `E_C` 的 representability profile：

- dense pointwise：两个 `f32` 输入、一个 `f32` 输出；一条 flat `map_binary(add|multiply)`；三个 tensor shape 完全相同且非标量；direct dense load/load/store；一个 exact result relation；无 alias、broadcast、reduction、update、predicate、fill、storage、nested control；至少一个静态 axis。
- direct Ragged query：一个 `ragged(offsets, indices)` producer，后接一个或两个 direct `ragged_outer`/`ragged_member` query，query 结果直接作为 function outputs；carrier 使用闭合的 Ragged 整数 dtype，descriptor 只在函数内部以 tuple SSA 存在；无 nested/conditional/non-local graph、runtime dimension 或 semantic storage。

超出范围时抛出 capability error，不猜模板、不回退到 example实现。非空 Layer-C interface 或 actual extension values 当前也直接拒绝；target-specific extension 只能由独立的 `BindTarget -> ValidatePlugin` bridge 消费，因此不能把 exporter 描述成通用 `C_target` bridge。

`intent.*` operation/type spelling在这里是 TianchenIR本地生成的中立落盘交换词汇，不代表Python import、共享进程内对象或实现依赖。其他工具只能通过公开文件或CLI边界读取该文本。

输出 Kernel MLIR 保证：

- 精确一个 `intent.kind="kernel"`；
- 无 `intent_plan`；
- 无 target identity；
- 无 actual C；
- 无 `π_source`；
- location 使用 `<canonical-j>`，不带原 Source 名/path；
- module handoff 明确给出 J digest、`E_C` interface digest 和固定入口符号；
- 对每个运行时 shape 轴，显式给出 `scalar_equals_view_dimension` ABI relation。

### 5.1 runtime dimension 的当前处理

历史 add2d J 同时包含：

```text
tensor shape symbol x0
runtime scalar p3 with logical_ref = x0
```

旧投影曾直接用自由 `p3` 作为 domain upper，同时下游交换链路又从 view shape独立派生 `x0`，却没有公开 relation 要求二者相等；这会在错误调用下产生少写或越界风险。

当前 exporter 已改为：

```text
dynamic domain upper = intent.dim(first_equal_shape_input, axis)
```

`p3` 仍保留在 J callable ABI 中，但不再作为自由内存访问边界。对满足 J precondition 的合法 valuation，`p3 == x0 == view.shape[axis]`，从 view dimension 取 representative 与 J 等价；外部误传不一致的 `p3` 也不会改变内存覆盖边界。当前窄 profile 的 Kernel-IR module 同时落盘 `intent.abi_shape_relations`，把 `p3`、`x0`、代表 view 和 axis 机械关联起来；独立 Lower consumer 不必依赖注释或外部 Source 名称才能读取这条 relation。

限制仍然存在：当前 handoff relation 已覆盖窄 profile 中由 J.signature 声明的 runtime dimension
parameter，但 `value_dimension_relations`、派生 dimension relation 以及更完整的 stride/layout
relation 仍未全部进入统一 Kernel-IR grammar。generated-source manifest/adapter 已经独立承载
shape coordinate、scalar relation、stride、grid 和 output binding；这些事实仍需由各独立 consumer
的 verifier 实际消费。动态 add2d 目前只能算窄范围编译/source-generation capability，不能算完整
目标 correctness 或闭包证据；这些 relation 不属于 C。

对 dense pointwise profile，全 runtime shape 仍失败关闭；该 profile 要求至少一个静态 axis，以形成确定性的 outer domain。静态 axis 形成 deterministic `intent.parallel` placeholder，runtime axis 使用 `intent.for`；这是空 C 情况下由 J 生成的编译结构，不是实际 C value，也不是性能结论。direct Ragged query profile 不依赖这个 dense outer-domain 条件，而是要求 carrier shape 为正的 concrete extent，并将动态 `ragged_member` 长度交给后端的显式运行时 ABI 处理。

## 6. 可选外部 Lower 消费者兼容性证据

历史真实 closed J 来源：

```text
/mnt/hdd/tmp/intentir-scrc-frontend-extension-20260728/
  cuda_tilelang_add2d/run-002/attempt-02/artifact.json
```

该 artifact 的 J 为：

```text
t0: f32[x0,16]
t1: f32[x0,16]
t2: f32[x0,16]
t2 = map_binary(add, t0, t1)
C_source = ∅
```

改写前留下的历史兼容性参考位于：

```text
/mnt/hdd/tmp/intentir-tianchen-final-audit-P1wV0lmO/
```

关键文件：

```text
kernel-dim.mlir
realized-dim-triton.mlir
realized-dim-cutile.mlir
realized-dim-tilelang.mlir
generated-dim-triton.py
generated-dim-cutile.py
generated-dim-tilelang.py
```

对当前TianchenIR exporter的独立导出复核仅确认：它从上述真实Gate artifact生成的neutral kernel input与历史`kernel-dim.mlir`字节一致。这是一项非认证的格式兼容性检查，不是Gate进展。该过程不读取Lower消费者源码树、不导入其Python package，也不调用其builder或内部API。目录中的realized/source结果均为改写前的历史外部兼容性参考，不能冒充当前代码新生成的产物；外部消费者的位置、构建方式和内部目录布局不属于TianchenIR合同。

上述改写前外部兼容性检查的历史执行结果：

| Target | 进程 | stderr | 当前判定 |
| --- | --- | --- | --- |
| Triton | exit 0 | 空 | source-generation capability通过；未执行数值 |
| TileLang | exit 0 | 空 | source-generation capability通过；未执行数值 |
| cuTile | exit 0 | 三条 axis-binding error | capability失败；文件存在和 exit 0 不能算成功 |

cuTile 三条错误分别对应两次 `intent.view_load` 和一次 `intent.view_store`：

```text
indexes a domain without a cuTile axis binding
```

因此调用侧必须把 MLIR `error:` 诊断视为失败，不能只检查 exit code。

这些历史结果没有 GPU execution、numerical comparison、Gate1 或 Gate2，不能增加 closed-Kernel 数。

## 7. 目标源码回灌能力

当前 Triton `source-output` 的显式设备入口为：

```text
scrc_j0_kernel
```

正式 source slicer `build_triton_source_slice` 能从独立 source-output中只保留：

```text
imports: triton, triton.language as tl
function: @triton.jit scrc_j0_kernel
host launch: absent
host run: absent
```

设备 ABI 为：

```text
p0_ptr
p1_ptr
p2_ptr
p3
x0: tl.constexpr
p0_stride_0
p0_stride_1
p1_stride_0
p1_stride_1
p2_stride_0
p2_stride_1
```

匿名化后得到精确一个：

```text
@triton.jit
def scrc_k0(p0,p1,p2,p3,c0,p4,p5,p6,p7,p8,p9): ...
```

这只证明：

```text
independent target source-output
+ explicit device_entry_symbol
-> single device-kernel slice
-> TianchenIR anonymous Source
```

这已经进入正式 generated-source adapter 边界，但 adapter 本身仍不等于 SCRC admission 或 closure。
当前 `GeneratedTargetSourceManifest`、`TritonWorkloadAdapter` 及其 lifecycle 记录已经明确提供：

- buffers/opaque inputs、dtype、rank、共同 shape coordinate 和 output binding；
- `p3` 与 `x0` 的 logical dimension relation，以及 value/shape/layout relation记录；
- `x0` 是由 J shape 派生的 semantic constexpr placeholder，不是 C slot；
- strides、grid、runtime valuation、显式 device entry 和 J/E_C/contract digest；
- source compile、execute、output capture 和 target-source re-ingress 的 lifecycle记录。

当前直接匿名化仍可能把 `x0: tl.constexpr` 表面编号成 `c0`；正式 adapter 已将它声明为
semantic bound constexpr，不能把这个名字编号误当成真实 `C_source`。中立 Kernel-IR handoff
和 generated-source manifest 都已有 machine-readable entry/digest/relation metadata。尚未完成的
是所有独立 target consumer 对这些 manifest 的真实数值编译/执行、完整输出校验和 round-trip
re-ingress closure；这些工作也不能用 schema 或 source generation capability 代替。

## 8. Replay/π 当前状态

实现位置：

- `python/intentir/verify/gate1/normalize.py`
- `python/intentir/verify/typecheck.py`

当前 TTIR unary normalizer不再执行任何没有逐 occurrence provenance 的文本变换：

- 删除 module-wide `tt.addptr` exemption；
- 删除 module-wide lower-bound guard删除；
- 删除全局 `!tt.ptr<i8> -> !tt.ptr<i1>` 替换；
- 删除“J 中任意 reduction允许 reorder就删除所有 identity-reduce-reshape”的全局规则；
- program、iteration、control、materialization、access representation均保持原始结构。

每条 π row 单独记录：

```text
path
family
relation
anchor_refs
disposition
reason
```

当前所有尚无 observation matcher 的 row均为 `hard`：

| relation | 当前 reason |
| --- | --- |
| `address_recomputed` / `pointer_advanced` | `per_access_address_occurrence_mapping_unavailable` |
| `guard_lower_bound_elided` | `per_access_guard_occurrence_mapping_unavailable` |
| `boolean_storage_byte` | `per_access_pointer_chain_mapping_unavailable` |
| program/iteration/control/materialization | `observation_anchor_mapping_unavailable` |

当前 normalizer输出文本与输入逐字节相同，`active_rules=[]`、`representation_anchor_exemptions=[]`。TypeCheck只验证 authored π row是否有合法 J资格：lower-bound relation必须来自该 access的精确 domain-coordinate下界，boolean storage relation必须指向 logical bool/i1 tensor；TypeCheck不发现、补全或应用物理对应。

未来恢复某条 normalization必须在 Source臂和Replay臂分别独立证明：

```text
one reachable physical occurrence
  -> exactly one J anchor
```

access至少联合验证 tensor root、access kind、logical index、predicate/fill、effect occurrence和multiplicity；reduction还必须验证 dtype/shape、axes、combiner、control ownership和downstream dataflow。多解、缺失或额外物理 occurrence都继续 hard。

该收紧会让依赖旧 module-wide exemption的历史闭包重新暴露结构差异。保存的 Rope TTIR离线审计已经显示旧闭包存在 op multiset和dataflow风险，但当前没有 fresh GPU Gate artifact，因此不能直接改写历史父 verdict，也不能把离线比较当成新闭包或新失败证书。

## 9. 权威真实 Kernel 证据

### 9.1 历史真实 add2d closure

完整父 evidence：

```text
/mnt/hdd/tmp/intentir-scrc-frontend-extension-20260728/
  cuda_tilelang_add2d/run-002/
```

权威历史事实：

- `main_status = SCRC_CERTIFIED`；
- `accepted_attempt = 2`；
- `j_necessity_status = proved`；
- Gate1 status为 `replay_closed`；
- Gate2 `passed = true`；
- Source–Replay、Source–J、Replay–J 三边全部通过；
- 输出为 `f32[4,16]`；
- 三边最大绝对误差均为 0；
- request、source、projection、adapter、Lift、Replay、TypeCheck、Gate1、Gate2 lineage完整。

这是历史真实 closure fixture。它没有按当前 exporter、runtime-dimension处理和严格 π合同重新执行 fresh closure，因此不能重新认证或增加计数。

### 9.2 当前 conditional-C capability diagnostics

非空 C FlashAttention forward：

```text
/mnt/hdd/tmp/intentir-maskc-current-gate2/
  flash_attn_flash_attn_triton_fwd_kernel/gate2-recheck-007/
```

当前事实：

- status为 `gate1_no_closure`；
- Replay ingress因12个 `unowned_value_carrier_use_forbidden` 停止；
- 涉及 `p8`–`p16`、`p20`–`p22`；
- Replay compile、normalization、comparison均未发生；
- Gate2未运行。

空 C split-K backward：

```text
/mnt/hdd/tmp/intentir-maskc-current-gate2/
  xformers_ops_fmha_triton_splitk_kernels_splitk_reduce_varargs_backward/
  gate2-recheck-004/
```

当前事实：

- status为 `gate1_no_closure`；
- ingress达到20条同类 carrier诊断上限，首个为 `p40`；
- Replay compile、normalization、comparison均未发生；
- Gate2未运行。

这两组是 exact-parent、零模型调用的 non-certifying capability diagnostics，只说明当前 ingress边界；它们不改变父 verdict，也不能算 current-contract closure。

### 9.3 当前 24/28

未闭包的四个精确 case：

1. `high_performance:flash_attn.flash_attn_triton._bwd_kernel`
2. `high_performance:xformers.ops.fmha._triton.splitk_kernels._fwd_kernel_splitK`
3. `high_performance:flash_attn.flash_attn_triton._bwd_kernel_one_col_block`
4. `high_performance:flash_attn.ops.triton.linear.kernel_fwd`

每个 case都必须重新具备 Gate1 closure、Gate2三边和完整 lineage后才能更新计数。不得为单个 kernel增加全局 π/Gate放宽，也不得以手写 Replay、mock、构建或旧 exact-parent bytes结案。

## 10. 当前阻塞

### 10.1 Fresh Agent

合法 provider 配置已经被现有工作区恢复并被真实 `/responses` 请求使用；配置内容、凭据和
endpoint 不进入本执行记录。一次使用配置当前 `reasoning_effort=max` 的真实请求选择了
`gpt-5.6-luna`，等待约 15 分钟后收到 HTTP 500；代码将其保持为
`provider_http_error`/基础设施失败，没有误分类为“provider 没有返回候选”。同一 provider、
同一 workload 显式使用 `medium` 后能够返回候选，因此当前不是 Python、GPU、认证或 provider
无候选问题；若要使用默认配置，应由宿主将其 effort 设为 relay 实际支持的值，TianchenIR 不会
改写用户的私有 provider 文件。

source-aware Lift 的 framework timeout 当前为无界值；旧的 `ulimit -t 60` 也已移除。
provider/transport timeout 若由调用方显式传入仍按显式实验策略处理，默认 Lift 不截断；
可续接的 timeout/no-candidate 会保留同一 value-free draft，并受每 branch 最多 3 次逻辑
Lift 约束。当前 CLI 的基础设施失败路径也会返回 JSON、evidence path 和已知阶段耗时，失败
证据中的 provider 诊断保持无密钥。

最近一次真实 `high_performance:liger_cross_entropy` full workflow（显式 `medium`）证据位于：

```text
/mnt/hdd/tmp/intentir-continue-20260903-cross-entropy-medium/
  liger_cross_entropy/run-001/
```

```text
prepare_registered_workload : 99.918 ms
provider_setup               : 0.143 ms
Lift（3 次逻辑/3 次 physical）: 331,189.036 ms
search_total                : 331,192.633 ms
frontend_total              : 331,318.784 ms
结果                         : 3 次 CandidateCheck 均拒绝
```

三次 provider 都返回了候选并进入正式 CandidateCheck；主要诊断是 `c`/`pi` 被嵌套到 `j`
下、因此根级 J 必填字段缺失，部分候选还把 `p*` 当作常量值、在 C slot 中保留不允许的
空附件，或产生 malformed `node_ids`/额外 `operands`。这证明当前剩余问题是复杂
Source-to-J authoring 输出没有遵守已经统一的闭合 grammar，不是 provider 无候选、环境或
后端执行失败。该 run 没有进入 BindSource、TypeCheck、fresh Replay、Gate1 或 Gate2，也不
构成闭包证据；该 case 的 3 次逻辑 Lift 已耗尽，不能再次 fresh Lift。

当前 blocker 是 Source-to-J authoring 的结构完整性：固定 ABI lock、CandidateCheck 和
最终 Decode 已统一为“固定 tensor 输入 → optional rank-0 scalar t* → output-only tensor”
的 dense 布局，并统一了 `value_ref=t* / dimension_ref=x* / relation=eq` 与 C slot 附件
规则；loop 闭合 grammar、根级 `c`/`pi` 检查和 oneOf 深层诊断均已落地。现有固定消费者
正确拒绝了这些不完整候选；在不放宽验收合同的前提下，还需要后续未耗尽 case 的最小真实
Lift 来验证 authoring 改善，再进入 BindSource 和 Gate 流程。

`pi.program_bindings` 的 relation 形式现在在 schema、解析、normalization、Replay ABI
投影和编译 obligation 中都带有同一 physical axis；但当前可读取的 28 个历史
`artifact.json` 都没有非空的该字段。因此它尚缺真实 Kernel Gate 工件，不能作为 closure
证据或计数依据。后续只能在真实 Source observation 产生该 relation 后走完整 Gate；不得
人工构造 π artifact 来替代该证据。

另一个确定性流程漏洞已经修复：共享 Gate2 输入物化器此前会对没有 Ragged op 的 dense J
抛出“没有 descriptor producer”，导致合法 dense candidate 在 Gate2 前被错误阻塞。现在
dense 图保持原始输入，只有检测到 `ragged`/`ragged_outer`/`ragged_member` 才进入严格的
descriptor 校验。对一个符合当前 envelope、空 C、完整 Replay receipt 的历史
`xformers.ops.fmha._splitK_reduce_varargs_backward` parent 做的零模型调用当前合同
recheck 已通过 Gate1 和 Gate2（`gate2_pass=1`）；该结果仍是 capability diagnostic，不能
改变历史 parent verdict 或 `24/28`。

### 10.2 GPU

当前 NVIDIA driver 已恢复，不再是环境阻塞：

```text
nvidia-smi: NVIDIA-SMI 580.95.05
device: NVIDIA GeForce RTX 5090 D
compute capability: 12.0
```

现有 `/usr/bin/python` 已切换到支持当前硬件的官方 CUDA wheel：

```text
torch: 2.7.1+cu128
torch.version.cuda: 12.8
triton: 3.3.1
torch.cuda.get_arch_list(): ... sm_120, compute_120
```

RTX 5090D 上的真实 CUDA `16x16 f32` 加法已返回 `3.0`。这只证明运行环境和最小 device visibility 已恢复，不构成任何 Kernel Gate1/Gate2 或性能闭包证据；目标源码的完整 correctness 和剩余四个 Kernel 的闭包仍待真实流程执行。

### 10.3 Dataset

真实外部 Source 已接入现有 resolver。当前外部根为：

```text
/home/wwq/intent-dsl/source/triton
```

该根由 resolver 自动发现，也可以通过 `INTENTIR_EXTERNAL_SOURCE_ROOTS` 显式追加；不再要求每次命令临时导出变量。现有 normalized FlashAttention/xFormers source 和官方 upstream source 均保持在 active tree 外部，TianchenIR 不复制或导入第三方包。

当前 manifest validation：

```text
total: 28
resolved: 28
unresolved: 0
passed: true
```

Liger resolver 已排除同名 `_ascend`/cuTile 实现，只接受当前 CUDA/Triton 目标布局。上述
manifest 是 Triton 外部 Source resolver 的结果，不等于 native multi-frontend Dataset
projection。当前 `tools/scrc_frontend.py --list --json` 仍报告 native Dataset 不可用：
缺少至少 30 个 TileLang/CUDA kernel、manifest 无效或缺失，且 TileLang 版本约束未满足。
这只阻塞 native Dataset case，不阻塞已经解析的 Triton workload；source resolution 本身
也不等于 fresh Lift、Gate1 或 Gate2 闭包。

### 10.4 Generated-source admission

正式 generated Triton workload adapter 已存在于 `python/intentir/verify/workload_adapters.py`，
并由 `frontend_workloads.py`/`scrc_workload_run.py` 交接。它覆盖 target-source device-entry
manifest、runtime buffers/opaque inputs、scalars、semantic constexpr、C slots、shape
coordinates、strides、grid、output binding、J/E_C/contract digest，以及 compile/execute/output/
re-ingress lifecycle。当前阻塞不再是缺少 adapter，而是还没有对所有独立 target consumer 完成
真实数值 correctness、完整 target-source round-trip 和对应 fresh Gate lineage。

### 10.5 Layer-C production bridge

exporter 当前接受两个窄范围的空 `E_C` profile：dense pointwise 和 direct Ragged query；
这是 exporter 的能力边界，不等于 Layer-C bridge 缺失。backend-neutral Layer-C grammar、
`BindTarget`、`ValidatePlugin`、`TargetPluginArtifact` 及其 serialize/deserialize 已存在，
CUDA/RISC-V consumer 也会在 ingress 重新验证 producer artifact，CPU 保持显式 empty-`E_C`
handoff。尚未完成的是将 neutral exporter 扩展到完整的 memory/layout/parallel/pipeline/
fusion/split/lowering grammar，并让每个独立 consumer 对非空 `E_C`/`C_target` 完成真实
compile/execute correctness。特定 Lower consumer 的对象映射仍必须由公开文件/CLI 边界上的
独立 adapter 完成，不得进入 TianchenIR exporter 或运行时依赖。

### 10.6 Replay provenance

当前缺少两类稳定映射：

- representation-only/value carrier 对 J semantic ABI或 C attachment的明确 ownership；
- Source/Replay physical access、reduction、control occurrence到唯一 J anchor的独立 provenance。

在这些映射完成前，carrier ingress会拒绝部分历史 Replay，π relation必须保持 hard。

### 10.7 cuTile consumer路径

可选cuTile Lower consumer可能在发出MLIR error后仍返回成功。修复前必须以stderr中的`error:`判capability failure，不能接受生成文件。

### 10.8 TianchenIR内部 import closure与Python运行环境

`intentir.frontend`、`intentir.verify`、`intentir.extraction`以及CUDA/RISC-V package boundary 已保持 import-light。当前项目运行入口统一使用现有 `/usr/bin/python` 用户 site，已永久补齐并验证：

```text
numpy: 2.2.6
torch: 2.7.1+cu128
triton: 3.3.1
torchvision: 0.22.1+cu128
torchaudio: 2.7.1+cu128
```

旧 venv 没有被重建或删除；其已有 `pyvenv.cfg` 已改为复用系统及用户 site，因此激活该 venv 也会看到同一份 CUDA/PyTorch/Triton 运行时。完整 Gate2、CPU native execution 和目标运行时仍需要各自的真实工具链；这不是当前 Python/GPU/Dataset 环境缺包阻塞。

## 11. 收束后的下一步

以下是恢复工作时的唯一顺序，不在当前收束中继续执行：

1. 将 `value_dimension_relations`、派生 dimension relation 和其余 runtime scalar-to-shape-symbol
   relation统一作为 Layer B typed ABI relation传入 Kernel IR；由 verifier 和独立
   Triton/cuTile/TileLang wrapper 检查，而不是放入 C。
2. 将当前 exporter 的两个窄 `empty_E_C` profile 扩展为闭合的
   memory/layout/parallel/pipeline/fusion/split/lowering grammar；继续通过已经存在的
   `BindTarget -> C_target -> ValidatePlugin -> backend-neutral target-plugin artifact` 交接，
   特定消费者的格式转换由公开边界 adapter 负责。
3. 修正 representation-only carrier ownership：逻辑 size/stride/value carrier必须明确归属 J ABI、
   J-derived placeholder或 C attachment；无法归属时继续拒绝。
4. 为每种 π relation实现单臂、逐 occurrence matcher；先建立唯一 provenance，再允许局部
   normalization，不恢复 module-wide 放宽。
5. 修复或隔离可选 cuTile consumer 的非致命 error 路径；任何 error 必须导致 target capability
   失败。
6. GPU、Python依赖、Triton Dataset source、provider 的基础 `/models` 和真实 `/responses`
   请求已恢复；native multi-frontend Dataset projection 仍按 manifest/版本约束失败关闭。
   source-aware Lift 默认无 framework wall-time 上限，continuation 使用 value-free draft
   和本地 stateless history。继续执行时必须保持每 branch 最多 3 次逻辑 Lift，保留同一
   persistent owner，并增加可核对的阶段性 artifact 读取；不得盲目并行启动或用 mock 代替
   真实流程。
7. 对空 C与非空 C各执行一次完整 fresh流程：fresh Lift、CandidateCheck、BindSource、fresh
   source-blind Replay、ingress、Gate1、Gate2三边和完整 lineage。
8. 对四个未闭包 case逐一执行相同真实流程；只有全部闭包条件成立时更新 `24/28`。
9. 对独立 Lower consumer 经公开交换接口生成的目标源码执行真实数值正确性，再使用已经存在的
   source manifest/adapter 将目标源码作为新的单 Kernel Source 回灌 TianchenIR，完成 round-trip
   closure。

## 12. 最终完成条件

整个任务只有同时满足以下条件才算完成：

- A/B 中不再保留已知 target identity、launch、layout、memory hierarchy、physical parallel mapping、pipeline或lowering选择；
- 所有硬件相关参数和结构都通过 typed、value-free `E_C` attachment表达；
- 需要实际参数时使用完整真实 `C_source/C_target`，不需要时使用 J-derived placeholder且不伪造 C；
- 更换后端只替换 `C_target`，canonical J不变；
- 单 example只处理显式选中的一个 Source kernel；
- `π_source`完全隔离在Gate1宿主侧，且只应用逐 occurrence已证明的 unary normalization；
- TianchenIR输出采用自身定义的中立交换格式，可被可选、完全独立安装的Lower consumer读取；consumer生成的独立目标源码能通过公开adapter重新作为TianchenIR输入；
- Triton、cuTile、TileLang目标源码先通过真实数值正确性；
- 四个剩余 Kernel各自取得Gate1 closure、Gate2三边和完整lineage；
- 权威状态从 `24/28` 更新为 `28/28`。

在这些条件未满足前，当前准确结论是：单 Kernel 投影、conditional-C packet、严格 hard-only π、
窄范围空-C J-to-Kernel exporter（dense pointwise 与 direct Ragged query）、backend-neutral
`C_target` plugin bridge、三后端 ingress 校验，以及正式 generated-source manifest/adapter/
lifecycle 已经落地；完整 dynamic ABI grammar、
所有 target consumer 的真实 correctness、目标源码 round-trip 和新的真实闭包仍未完成，
closed-Kernel 状态保持 `24/28`。
