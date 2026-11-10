# 真实 Kernel 语义示例与闭包证据

本页使用 `high_performance:liger_softmax` 说明 `J/C/π` 的语义边界；它是真实
Dataset Kernel，不是 mock、单元测试或合成 canary。但该 softmax 示例不是当前
`docs/project-guide/real-evidence` 保留的闭包证据，也不能用来替代当前 fresh 重测结果。

## Source

工作负载的 Source 位置是：

```text
dataset://Liger-Kernel/src/liger_kernel/ops/softmax.py
```

设备入口是 `_softmax_single_block_forward_kernel`。本例的核心计算可以概括为：

```python
mask = offs < n_cols
x = tl.load(
    X_ptr + row_id * X_row_stride + offs,
    mask=mask,
    other=-float("inf"),
)
m = tl.max(x, axis=0)
e = tl.exp(x - m)
d = tl.sum(e, axis=0)
y = e / d
tl.store(Y_ptr + row_id * Y_row_stride + offs, y, mask=mask)
```

这段 Source 同时包含：

- 一个带 `offs < n_cols` 谓词的 masked load；
- max reduction 和 sum reduction 两个不同的归约消费者；
- 对应的 masked store；
- 输入、输出形状 `(2, 127)`，以及 `BLOCK_SIZE=128` 的 C 绑定。

## J 的语义边界

规范化后的 J 保留逻辑数据流，不把 Triton 的物理指针、launch 或布局当作新的
逻辑 tensor：

```text
a0: load t0, predicate = (k1 < x1), fill = -inf
v1: i1 guard derived from iota/lt
n2: reduce(max, t0), outside = -inf
n5: reduce(sum, v4), outside = 0
a1: predicated store t2
```

其中 `a0` 是一个共享的 node-less load，但它有两个显式
`consumer_extensions`：

```text
a0 -> n2: max reduction, outside = -inf
a0 -> n5: sum reduction, outside = 0
```

这里必须区分两种语义：

- `fill=-inf` 是 Source 物理 load 在 predicate 为 false 时产生的填充值；
- `outside=-inf` 和 `outside=0` 分别是 max、sum 消费者自己的归约恒等元。

不能把两个消费者合并为一个 outside，也不能复制成两个物理 load。前一种做法会
改变 sum 的语义，后一种做法会破坏 Source 中共享读取的结构。

## C、π 与后端边界

本例的规范化 C 绑定如下：

```text
normalized C slot: c0 / constexpr / i32
normalized C value: c0 = 128
source constexpr: BLOCK_SIZE = 128
```

`π` 只服务于 Gate1 的已证明 occurrence-level 对应关系，例如访问、谓词、归约
和程序轴。它不修改 J，也不参与 Replay 的数值结果生成。

本例的地址表示绑定如下：

```text
a0 <- p3, axis_ref = x0, relation = address_recomputed
a1 <- p1, axis_ref = x0, relation = address_recomputed
```

这里 `p3`/`p1` 是 row stride carrier，`p4` 是逻辑 `n_cols` 标量。Replay 的
二维向量地址必须保持 `row * stride + lane`；`p4` 只参与 `lane < p4` 的逻辑
谓词，不能被当作 stride 或额外地址乘数。

生产使用和 Replay 验证的 C 所有权不同：

| 场景 | 使用的 C | 说明 |
| --- | --- | --- |
| CUDA/CPU/RISC-V 生产后端 | 各自生成的 `target_C` | 用于真实 lowering、编译和执行 |
| 本例 Replay/Gate1/Gate2 | Source/Harness 已绑定的 `artifact.c` | 与 Source 的真实编译接口一致，不重新生成独立 `target_C` |

因此，Replay 的目标是复现已绑定 Source 的可观察行为，而不是另外搜索一份 C 再
与 Source 比较。生产后端仍保持各自的 compiler、observer、lowering 和执行路径。

## 闭环关系

```text
真实 Source
    │
    ├── Lift / canonicalize ──> J, C, π
    │
    ├── Source compiler ──────> source artifact
    │
    └── BindSource ───────────> source-bound artifact.c
                                      │
                                      ▼
                              source-blind Replay
                                      │
                                      ▼
                                    Gate1
                                      │
                                      ▼
                                    Gate2
                         Source–Replay ∧ Source–J ∧ Replay–J
```

按当前证据保留约束，仓库内只保留一个真实 Kernel 闭包证据：

```text
real-evidence/liger_swiglu/closure/
```

该目录对应 `high_performance:liger_swiglu`，权威摘要为
`classification=scrc_certified`、`main_status=SCRC_CERTIFIED`、
`j_necessity_status=proved`、TypeCheck 实际调用 `1` 次、Gate2 `gate2_pass=1`，且
Source–Replay、Source–J、Replay–J 三条比较边已完成。这些是 swiglu 的证据事实，
不应倒灌为上文 softmax 语义示例的当前闭包结论。

当前全局闭包数、阶段结果与后续重测状态仅在
[唯一最终任务](../final-task.md) 维护；未经用户再次明确要求，后续 Kernel 结果不再
写入 `real-evidence` 目录。
