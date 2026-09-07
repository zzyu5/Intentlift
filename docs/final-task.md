# 最终任务：真实 Kernel 的语义闭包与生产 Lower

本文是 `/home/wwq/TianchenIR` 唯一活动执行记录。它只保留当前合同、未解决阻塞、下一步和真实 Kernel 的闭包事实；实现日记、合成 canary、历史 artifact 复核和机械诊断不属于活动记录。

## 当前合同

```text
真实 Source + Harness
  -> LiftSemantic
  -> canonical J = Layer A + Layer B, E_C, π_source
  -> BindSource
  -> fresh source-blind Replay
  -> Gate1
  -> Gate2: Source–Replay ∧ Source–J ∧ Replay–J

canonical J + E_C + target/capabilities
  -> BindTarget
  -> ValidatePlugin
  -> Lower
```

- `J=A+B` 完整保存算法的可观测语义：logical tensor、operator、def-use、control/recurrence、domain、access、predicate、effect、alias、结果和 typed dimension relation。它不保存目标名称、Source 路径、launch、layout、memory hierarchy、program/lane 坐标或后端 recipe。
- `E_C` 只声明强类型 realization extension point。`C_source` 是本次 Source 编译实际需要的绑定；`C_target` 是按 target/capabilities 重建并验证的后端插件。两者都不能改变 J 的语义，也不能相互冒充。
- `π_source` 只供 Gate1 两个独立 observation arm 做已证明的逐 occurrence normalization。它不得进入 Lift 的 Replay packet、生产 Lower 或 target plugin。
- 活动流程只接受当前 J、当前 Layer-C interface 和当前 source-isolated packet。历史 J、历史 packet 和旧 evidence artifact 不会进入 Lift、BindSource、Replay 或 Gate，也不影响闭包计数。
- 空 `C_source` 时 packet 是 `{j,u_f}`，编译占位符只从 `J.signature` 确定性生成；非空 `C_source` 时 packet 是 `{j,c,u_f}`，每个活动 slot 都必须有真实、完整、类型匹配的绑定。

## 真实闭包事实

活动 workload universe 由 [high-performance-kernels.json](../spec/lowering/high-performance-kernels.json) 定义，共 28 个真实 Kernel。当前权威汇总为：

```text
活动 Kernel：28
已闭包：24
待闭包：4
```

待闭包的 case：

1. `high_performance:flash_attn.flash_attn_triton._bwd_kernel`
2. `high_performance:xformers.ops.fmha._triton.splitk_kernels._fwd_kernel_splitK`
3. `high_performance:flash_attn.flash_attn_triton._bwd_kernel_one_col_block`
4. `high_performance:flash_attn.ops.triton.linear.kernel_fwd`

每个“已闭包”计数都必须有同一真实 case 的完整 source、Harness、Lift/BindSource、fresh Replay、Gate1 和 Gate2 三边 evidence/lineage。清理活动树、Python import、schema 解析、TypeCheck、构建或 target capability 都不是闭包，不能增加上述计数。本轮清理没有调用 provider、fresh Lift、Gate1 或 Gate2，因此计数不变。

## 当前阻塞

- 复杂真实 Source 的 Lift 候选仍会在 CandidateCheck 前后暴露结构不一致：根级 `j/c/pi` 边界、逻辑 tensor 与物理 carrier 的归属、C slot attachment、value/dimension reference、optional scalar 顺序，以及 loop carry/body/yield 必须共同满足同一闭合 grammar。检查器必须保持失败关闭，不能为通过候选放宽合同。
- Replay ingress 仍要求每个 representation-only carrier 都能明确归属为 J ABI、J-derived placeholder 或合法 C attachment；无法证明所有权的 carrier 必须继续拒绝。
- `π_source` 的 access、reduction、control 和 materialization relation 尚需两臂各自证明到唯一 J anchor 的 physical occurrence mapping。在此前，未证明 relation 保持 hard，不恢复模块级文本放宽。
- native 多前端 Dataset 的完整来源和对应工具链仍需满足 manifest 与前端版本约束；不可用时必须报告 unavailable，不得用仓库内合成 Source 代替真实 workload。
- production Lower 已有 J/E_C/C_target 边界，但完整 memory/layout/parallel/pipeline/fusion/split grammar 及每个独立 target consumer 的真实 compile、execute、数值正确性和 source round-trip closure 尚未全部完成。

## 下一步

1. 先针对一个未耗尽真实 workload 修复 CandidateCheck 所报告的最小 authoring/contract 断层，保持每个 candidate branch 最多三次逻辑 Lift，并记录各阶段实际耗时。
2. 只有 Lift、BindSource 和 fresh source-blind Replay 都成功后，执行该 case 的 Gate1；Gate1 闭合后才执行 Gate2 三条比较，并保存完整 evidence/lineage。
3. 对空 `C_source` 与非空 `C_source` 各完成一条真实闭包路径，再逐一处理四个待闭包 case。满足 Gate1 与 Gate2 三边后，才可以更新 closed-Kernel count。
4. 对 CUDA、RISC-V 和其他独立 Lower consumer，以相同 canonical J 为输入完成 target plugin 验证、真实执行与数值正确性；将生成的独立 target source 通过正式 adapter 作为新的单 Kernel Source 回灌，再完成 fresh closure。

## 完成条件

任务完成仅当下列条件同时成立：

- J 与 E_C 的语义/realization 边界完整，任何 target 切换都不改变 canonical J；
- Source 与 target 的实际 C binding 均完整、强类型并经过 attachment/capability 验证；
- `π_source` 只含逐 occurrence 已证明的 Gate1 relation；
- 每个活动 Kernel 都拥有 Gate1 closure、Gate2 三边通过和完整真实 lineage；
- 目标 consumer 的生产 Lower、真实执行和 source round-trip 都不依赖合成 canary、mock 或历史 artifact 复核。
