# 系统架构

## 1. 系统边界

IntentIR 把“理解一个源 Kernel”和“为目标硬件实现它”拆成两个可核验问题。Lift 读取完整 Source 与当前前端的 Harness/compiler view，输出原有的 `(J,C,π)`；Replay 在看不到 Source 的条件下消费 `J` 与遮蔽后的 `C`，产生目标前端程序；Gate1 和 Gate2 分别检查结构闭包与执行语义闭包。

```text
完整 Source + Harness/compiler view
                │
                ▼
               Lift ───────────────► J + C + π
                                      │   │
                    J + source-bound C  │ π
                                      │   │
                                      ▼   ▼
                                  Replay Gate1 ◄── Source observer
                                      │
                       Gate1 closed ─┴────────────► Gate2
                                                     ├─ Source–Replay
                                                     ├─ Source–J
                                                     └─ Replay–J
```

任何单边通过都不足以认证；只有 Gate1 闭合且 Gate2 三边全部通过，Kernel 才进入闭包计数。Lift、Replay、Gate1 和 Gate2 使用同一份规范化 `(J,C,π)` 合同，不增加另一条中间 ABI 输入路径。

## 2. `(J,C,π)` 的职责

| 对象 | 拥有的事实 | 不得承担的职责 |
| --- | --- | --- |
| `J` | 张量签名、shape、访问、谓词、fill、归约、效果、输出语义 | 目标 tile、launch 参数、源文件路径、算法名称驱动的后端选择 |
| `C` | 当前实现槽位及其具体估值，例如 tile、warp、stage、编译选项 | 修改 `J` 的数学语义，或为缺失语义提供补丁 |
| `π` | Source 与 Replay 观测之间冻结、带类型的对应关系 | 读取后改写观测、放宽 Gate、携带新的语义或目标实现配方 |
| Harness/compiler view | 实际 ABI、输入、shape 估值、grid、Source compiler 所需信息和初始 `C` | 充当 `J` 候选集合或证明语义正确 |

`J` 必须完整表达算法可观测语义；`C` 只表达当前流程实际使用的实现选择。`π` 不只覆盖程序轴，也可以表达访问、guard、布尔和归约关系。所有关系都必须冻结、可类型检查，并由两侧独立观测验证。

### 生产 C 与 Replay C 的边界

这两个场景使用的 C 名称相近，但所有权不同：

| 场景 | C 的来源 | 用途 | 是否生成新的 C |
| --- | --- | --- | --- |
| 真实生产 | 当前 CUDA/CPU/RISC-V/native backend 的 `target_C` | lowering、编译、执行、性能搜索 | 按生产后端流程生成 |
| Replay/Gate1/Gate2 验证 | Source/Harness 已绑定并校验过的 `artifact.c` | 固定验证 ABI、schedule 和隐藏估值 | 不重新生成独立 `target_C` |

Replay 使用 `artifact.c` 是为了让 Source 与 Replay 在同一个 Harness 估值上比较，
不是把验证 C 变成生产 C，也不允许 Replay C 反向修改后端 target artifact。

## 3. 可见性与所有权

- Lift 可以读取完整 Source 和当前前端的 Harness/compiler view，并负责产出整个 `(J,C,π)`。
- Replay 只接收规范化 `J`、与 Source/Harness 一致的已接受 `C` 和固定工具环境 `U_f`。它不能读取 Source、`π`、未遮蔽的 C lineage、Gate 差异或以往失败历史；Replay 不单独生成 `C_target`。
- Source observer 和 Replay observer 独立产生原始结构观测；observer 不读取 `π`。
- normalizer 依据同一个冻结 `π` 对两侧观测做单边规范化，不能通过联合查看两侧来制造相等。
- TypeCheck 在 Gate 前失败关闭，检查 artifact 的结构与类型一致性；它不证明 Source/J 等价，也不计作闭包。

## 4. Gate1 与 Gate2

Gate1 比较 Source 与 Replay 的规范化编译器观测。正常臂必须闭合；可选的 masked-J 臂只用于诊断 `J` 是否提供必要信息。Gate1 先于任何隐藏 Gate2 输入执行。

Gate2 在冻结 Harness 上运行三个执行腿：

1. 原始 Source；
2. Gate1 已接受的精确 Replay；
3. 从同一 `J` 构造的参考执行 `RefJ`。

比较策略由 workload 的外部 `Q` 策略决定，不写入 `J`。Source–Replay、Source–J、Replay–J 必须全部通过；运行失败、reference unsupported 或任一数值比较失败都不能认证。

### 共享 guarded load 的闭合例子

当一个物理 masked load 被多个逻辑消费者复用时，J 不复制物理访问，而是保留
一个 access 和多个显式消费者关系：

```text
a0: load t0, predicate=(k1 < x1), fill=-inf
  ├─ n2: reduce(max, t0), outside=-inf
  └─ n5: reduce(sum, v4), outside=0
```

这里 `n5` 的 operand 是经过 `sub`/`exp` 后的 `v4`，并不是物理输入名 `t0`。
因此边界投影必须同时保存物理 access 的 `fill_value` 和消费者自己的 outside
identity；把它们合并会让 J reference 或 production projection 产生错误结果。

## 5. 前端

前端保持各自的真实 Source、compiler 和 observer 逻辑：

- Triton/TTIR：由共享 SCRC 生命周期驱动，并保留 Triton 自己的 TTIR/source observer；
- CUDA：使用原生 CUDA Source、CUDA compiler 和 CUDA Gate runner；
- CuTe：使用独立的 CuTe Source/compiler/observer/Gate runner；
- TileLang：使用独立的 TileLang Source/compiler/observer/Gate runner。

`tools/scrc_frontend.py` 的共享清单只负责 Triton/TTIR。CUDA、CuTe、TileLang 不转成 Triton/TTIR，也不回退到共享 SCRC runtime。native Dataset 通过 `tools/frontend_dataset.py --validate` 和对应前端 runner 单独检查。

## 6. 后端

生产后端从规范化 `J` 开始确定性物化 Intent MLIR；随后由各自 backend 的 typed contract、候选搜索、编译和执行路径消费当前后端需要的 `C`。后端合同不能新增 `J` 中不存在的 operator、predicate、effect 或 result，也不能由 Source 名称或历史手工标签恢复被遮蔽的信息。

- CUDA 后端：Intent MLIR 到 NVIDIA GPU/PTX/cubin 的 lowering、Driver 执行、结构化候选搜索和基准导出。
- RISC-V 后端：Intent MLIR 到 LLVM/RVV 的 lowering、目标对象导出和目标机搜索/计时。
- CPU 路径：标量 MLIR/LLVM 参考执行，主要服务 Gate2 的 `RefJ`；它不是 CUDA/RVV 生产性能的替代实现。

后端之间可以共享 `J/C/π` 的基本类型合同，但每个 target 的 compiler、lowering 和执行证据保持独立。后端接口不反向修改 `J`，也不改变 Source/Replay/Gate1 的输入和 observer 语义。

## 7. 可移植性边界

Git 跟踪的仓库包含构建系统、Python 包、生产 Source、机器可读规格和公共文档。Dataset、大模型 provider 配置、GPU/RVV 运行证据、内部研究笔记和历史归档均位于仓库之外或 `local/`，通过显式路径注入。任何新机器都应能在没有这些私有文件时构建核心系统并列出静态能力；真实闭包和目标基准则需要相应 Dataset、工具链和硬件。
