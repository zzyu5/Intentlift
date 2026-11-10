# 工作流

## 1. 新 Kernel 的 SCRC 闭包

1. 在静态清单或对应前端 Dataset 中登记真实 Source 与 Harness；Harness 只描述实际 ABI、输入、grid 和初始 `C`。
2. Lift 从完整 Source/compiler view 生成原有 `(J,C,π)`，TypeCheck 对 artifact 失败关闭。
3. 为每个正式轮次创建全新的、Source-blind Replay；Replay 只消费规范化 `J`、与 Source/Harness 已绑定且已由 Gate1 校验的同一份 `C`，以及固定工具环境。Replay 不另行生成 `C_target`。
4. Gate1 独立编译 Source 与 Replay，observer 分别取证，再用冻结 `π` 规范化和比较。
5. 只有 Gate1 闭合的候选才能进入隐藏 Gate2。
6. Gate2 在同一估值上执行 Source、精确 Replay 和 `RefJ`，保存 Source–Replay、Source–J、Replay–J 三条比较。
7. 三边全部通过才记录 `scrc_certified`；其他状态保留为失败证据，不得以修改 Gate、手写 per-kernel 规则或只通过 TypeCheck 替代。

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

## 2. 精确 artifact 的复核

复核不得从历史 parent artifact、rejected candidate、draft、trajectory、session 或 repair base 继续 Gate1。需要重跑时必须建立新的独立 run root，从新的合法 Lift lifecycle 开始，并按正式 provider/Lift 预算依次通过 CandidateCheck、TypeCheck、Replay、Gate1 和 Gate2；不得调用模拟 Source、绕过当前 Gate，或把 TypeCheck 和单边比较当作闭包证据。

唯一允许的精确 artifact Replay 重试发生在尚未落盘的同一 Kernel 生命周期内：当前 runner 已取得 Gate1 闭包、Gate2 明确显示 Source–J 通过而两个 Replay 边同时失败、且仍有正式 Gate1 轮次时，runner 可对自己刚闭合的 TypeCheck-valid artifact 创建新的 Source-blind Replay。该入口不接受 artifact 路径、历史证据或外部 feedback，不调用 Lift，也不能替换候选；跨 root 重跑仍必须从 fresh Lift 开始。

## 3. 多前端工作流

多前端共享 `(J,C,π)` 的基本合同，但保留各自的 Source、compiler、observer 和 Gate 运行逻辑：

- `tools/scrc_frontend.py` 只负责 Triton/TTIR 的共享 SCRC 清单和完整生命周期，并保留 Triton 自己的 TTIR/source observer；
- CUDA 使用 CUDA Dataset、CUDA compiler、CUDA observer/Gate1 runner 和 native Gate2；
- CuTe 使用独立的 CuTe Dataset、compiler、observer/Gate1 runner；
- TileLang 使用独立的 TileLang Source/compiler/observer/Gate1 runner 和 native Gate2。

native 前端不能转成 Triton/TTIR，也不能回退到共享 SCRC runtime。Dataset 缺失或本机前端依赖不可用时，入口应明确报告 unavailable，而不是以模拟 Source 代替：

```bash
PYTHONPATH=python python tools/scrc_frontend.py --list --json
PYTHONPATH=python python tools/frontend_dataset.py --validate
```

## 4. 生产后端工作流

生产后端严格按以下顺序工作：

```text
accepted J
  -> deterministic Intent MLIR materialization
  -> attach the current backend C/typed contract
  -> target lowering
  -> structural candidate admission
  -> compile and execute
  -> correctness audit
  -> target timing/search report
```

CUDA 和 RVV 可以消费同一份规范化 `J`，但各自拥有目标 lowering、候选空间、compiler 和执行器。CUDA/CuTe/TileLang 的 native compiler/source evidence 不得混入 Triton 的 observer 或 Gate。性能报告必须保留 workload profile、目标信息、候选空间、正确性结果和测量设置；只有真实目标执行结果可以支撑性能结论。

## 5. 证据与文档

- 公共 `docs/` 只描述当前系统合同、可复现入口和唯一最终任务。
- `docs/final-task.md` 是唯一活动执行记录；旧状态不追加成日志，而是被当前事实替换。
- `local/internal/scrc/`、`local/internal/frontend-extension/`、`local/internal/backend/` 保存不提交 Git 的内部理解和重要真实结果。
- `local/archive/` 保存合并前快照、旧文档、Trellis 与历史测试；这些内容不得重新成为活动流程。
- 大型 evidence 建议保存在仓库外，并在最终任务中记录绝对路径、case、Gate 状态和 parent lineage。
