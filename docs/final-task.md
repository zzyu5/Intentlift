# 最终任务：真实 Kernel 语义闭包

这是仓库唯一的活动执行记录，只保留当前合同、当前状态、未解决阻断、下一步和权威闭包事实。不保存 Source、TTIR、candidate、Replay Source、provider response、拒绝草稿、trajectory、凭据或临时 root。

## 当前合同

- 工作负载全集由 `spec/lowering/high-performance-kernels.json` 定义，共 28 个真实 Kernel。
- 只有 `Gate1=replay_closed` 且 Gate2 的 Source–Replay、Source–J、Replay–J 三边全部通过，才计为闭包。CandidateCheck、TypeCheck、局部投影通过和单边比较都不计闭包。
- Gate1 只判定已接受 `(J,C,π)` 的算法结构、ABI、J-level dataflow、reduction、predicate/effect 和 output topology；物理 TTIR 指令数量、临时 SSA、地址 carrier、PTX 布局、寄存器、共享内存、调度和性能不作为 Gate1 的额外硬阻断。
- Source owner、padding、完整 physical-store/logical-output partition 仍由 Source/functional admission 证明；不能删除检查、猜测语义或把拒绝草稿作为修补基础。
- 所有后续修复必须优先扩展共享 canonical projection、relation builder、validator 和安全诊断 envelope；禁止为单个 Kernel、operator、provider 失败或生命周期阶段追加平行字段、平行协议或专用引导分支。适配层只允许做既有契约定义的确定性表示转换，并同步所有消费者。
- masked-J 是独立诊断。normal Replay 结构闭合才可形成 Gate1；masked-J 的 `proved`、`not_proved`、`inconclusive` 不得伪装成 normal Gate1 结果。
- 正式 Lift 只使用 MCP `miniswe` 路径；Replay 可使用 Responses transport。真实运行必须单 Kernel、单分支、单 worker、严格串行、全新 external root、全新 lifecycle，不复用旧 root、candidate、draft、trajectory、session、Replay Source 或 repair base。
- 真实测试最多三次彼此独立的 fresh Lift。产生真实 candidate 后才允许进入 CandidateCheck、TypeCheck、Replay、Gate1 和 Gate2；遇到 provider overload、rate、usage、HTTP、认证、transport 或连续 SSE failure，立即停止当前 lifecycle。
- `docs/project-guide/real-evidence` 不新增证据；顶层只保留既有 `liger_swiglu` 闭包证据。统一测试重新启动时闭包计数从 `0/28` 开始，历史结果不结转。
- 不创建或恢复测试目录、mock、synthetic、smoke 或 canary。离线检查只能验证契约，不得写成闭包进度。

## 当前状态

- 统一 28 Kernel 测试：`未启动`；最近一次仅对 `high_performance:liger_poly_norm` 做定向 fresh lifecycle 验证，因 MCP semantic-transfer 无进展已停止。
- 当前闭包计数：`0/28`（本次未达到闭包条件）。
- 当前总状态：`NO_CLOSURE`。
- 当前没有运行中的 provider 请求。最近一次 fresh root 保留在 `/tmp/intentir-real-mcp-TbdpU9/liger_poly_norm/run-001`，未写入 `real-evidence`。

## 本轮离线验证

- 28 个注册 workload 清单可解析，manifest validation 通过。
- 目标 Python 文件的 `compileall`、`pyflakes` 和 `git diff --check` 通过。
- 使用真实 `liger_poly_norm` Source、真实 host launch contract 和真实 TTIR observer 进行无 provider 复核：`Source observation=pass`、`SourceTransferLedger=pass`、9 个 occurrence、4 个 Source owner outline、Source authoring blockers 为 0。
- `t5` 有一个完整二维 store owner；`t6` 有三个带坐标的物理 store owner，三个 owner DAG 根的类型是 `[dim0]` slice，而最终输出类型是 `[dim0,3]`。
- 已修正 Source→MCP→Agent graph projection：owner producer requirement 现在区分 `logical_output_value` 与 `physical_store_slice`；多物理 store output 额外携带由既有 Source store ordinal/coordinate 事实构成的 `source_store_partition_requirement`。
- 该修复只改善角色、坐标和 output-binding 义务的传递；没有创建 producer、scatter、update、partition proof 或新的 Source 语义，也没有放宽 canonical/functional admission。
- 离线 bundle 与 mounted MCP projection 已确认 partition requirement 保持一致；错误 relation 不属于 canonical contract；Agent 引导已同步 slice 与完整 logical output binding 的区别。
- 新增 locator 使真实 `source-brief` 超过固定 MCP 输出预算的内部问题已修复：brief 压缩阶段现在同时延后重复的 `source_store_requirements` 与 `source_store_partition_requirement`，二者仍由 mounted bundle 和 `begin_candidate` 重新校验并传递。离线真实 Source 复核得到 `source-brief returncode=0`、约 9,805 字符，低于 10,000 字符上限。
- MCP context reader 已补齐 store projection 的 fail-closed 同步校验：`source_store_requirements` 必须逐行绑定同一份 Source access ordinal、owner operation family、owner store coordinates 和 output；多 store 时的 partition locator 必须逐项等于这些 rows，错误 relation 或坐标会在 `source-brief` 前拒绝。真实 `liger_poly_norm` Source 的合法 bundle 通过，relation/coordinate 负例均被拒绝。
- FunctionalBoundary 已补齐物理 slice producer 与 logical concat owner 的分层投影：每个 store 的 `access.node_ref` 必须唯一产生对应 concat operand，而 `output_mapping.node_ref` 必须唯一产生最终 concat result；完整 partition 仍单独检查 operand 顺序、坐标无重复无缺口、predicate、dtype/shape 和最终 ABI value。单个 slice 不能成为 logical output owner。
- MCP context/environment 的 assembly 校验均按 Source 显式 `ordinal` 建立 access 索引，不再以 rows 的列表位置代替 occurrence identity；真实 poly_norm bundle 的 assembly 正例以及错误轴、缺片、重复坐标、错误 ordinal、错误 result type 五个负例均按预期拒绝。
- Source wire construction、host MCP environment 与独立 MCP context tool 之前各自维护 assembly 判据；现将物理 store assembly 校验收敛到单一共享实现，并把同一实现挂载到 context tool。严格 Source ordinal 判据也移到依赖无关的共享 contract，由 canonical construction、assembly validator 与独立 context tool 共用。共享判据保留唯一 occurrence、坐标完整性、operand/result 类型和 Source store owner-root 检查；没有改变为通过而放宽拒绝条件。改动文件 `compileall`、`pyflakes`、`git diff --check` 通过。该合并尚未被 context tool 实际调用或真实 candidate 生命周期验证。
- 另外补齐了 canonical seed output → provider `output_construction` → MCP begin typed-output 的正向绑定；`xN` 只按唯一规则映射为 `dimN`，错误 output name、dtype 或 shape 在 context reader 和 MCP environment 两条入口都会拒绝。真实合法 bundle、错误 partition relation、错误 output shape 的离线正/负复核均通过预期判定。
- `provider_dimension_aliases` 现在还必须与 canonical allocator 的完整 `x0...xN` 集合一一对应；额外、缺失或错序 alias 不再只因自身看似 dense 就被接受。28 个 active workload 的 manifest/resolution 复核仍为 `resolved=28/28`、`passed=true`。
- Relift 的 `open_guidance`/`resolved_guidance` 现在复用统一的 provider diagnostics projection，不再维护只保留 `rule_code/path/status` 的第二套字段白名单；安全的 typed relation 会随反馈保留，同时 candidate、Source、operation row 和递归 witness 仍由共享过滤器移除。离线正例确认 `expected_shape`、`expected_dtype`、`type_repair_rule` 可传递，敏感 payload 仍被丢弃。
- preparation receipt、CandidateCheck、SourceReconcile 和 context-query prior feedback 现在共同复用同一 reject-only diagnostics envelope；不再由各阶段分别重建 rule/path 字段集合。predicate target shape、predicate relation、expected dtype/type repair 等安全字段在跨生命周期投影中保持一致，candidate、Source text、operation row 和 trajectory 仍被过滤。离线正例/负例均通过。
- MCP graph admission、predicate 缺失定位和后续 computed-guard 投影现在共同消费一个 Source access typed-relation builder；`load`、`store`、`update` 不再因阶段不同而使用不同的 predicate dtype/shape 判据。该修改只统一既有 canonical `i1`、target shape 和唯一 rank-insertion 关系，未生成 predicate、未放宽 fail-closed 检查，离线 compile、pyflakes 和 diff 检查通过。
- 最近 lifecycle 的 predicate 拒绝回执只给出 compatible count=0 和 target shape，未真正兑现代码注释所称的“区分没有 i1 producer 与 i1 shape 不兼容”。MCP graph validator 与安全反馈现共用每个 Source access slot 的候选计数，回执仅增加 boolean-result 数、shape-incompatible 数及固定分类；provider-facing guidance 据此要求只按当前 Source/owner 事实修复。离线确认合法 i1 rank-insertion 通过，无 i1 与错误 shape 仍拒绝且诊断不同；仍待 fresh lifecycle 验证。
- 已继续收紧并统一 computed-guard 的 predicate dtype 判据：MCP graph preflight、canonical materializer、FunctionalBoundary、TypeCheck 和 ContractMaterialization 现在共同消费既有 canonical `is_closed_boolean_dtype`；该 canonical 表示只接受规范拼写 `i1`，不再由前置阶段额外接受后续阶段必然拒绝的 `bool`、大小写变体或缺失值。规范 `i1` 正例以及 `bool`、大小写变体和缺失值负例均按预期通过；没有放宽任何 Gate 或生成 predicate。
- Source-order ordinal 现在统一使用共享的 `is_canonical_ordinal`：要求严格的非负 Python `int`，显式拒绝 `bool`、负数、浮点数和字符串。Source semantic wire、provider dimension alias、compiler observation、MCP environment 和独立 context-query 的 ordinal 投影均保持该边界；合法 ordinal 正例与 `True`/负数等负例的离线检查均通过，没有改变 occurrence 顺序或生成新的 identity。
- Source operation requirements 的生成顺序与 Agent 引导曾相互矛盾：投影按 AST 外层优先产生行，嵌套 producer 可能排在 consumer 后；同时行内没有显式 Source-ref 依赖，Agent 只能自行反查 operand。现将现有唯一 `sx*` expression result/operand 关系投影为稳定 producer order；首轮 brief 用紧凑 `[consumer_index, producer_index]` edges 表示依赖，MCP 校验 edge 集合精确等于 Source rows 中可推导的直接依赖，再将其还原为 `begin_candidate` 每行的 `source_dependencies`。真实 `liger_poly_norm` 的离线 Source→TTIR observation→mounted bundle→host MCP 路径通过：brief 9,952/10,000 字符；30 个 operation requirement rows、15 条依赖边均为 producer-first；family counts 与 `required_operation_families` 完全相等，attributes counts 也完全相等；MCP begin graph receipt 为 6,902 字符，包含 4 个 Source owner producer requirements 和 1 个 logical output assembly requirement。无 provider 请求、无 candidate、未读 rejected graph；未减少 required counts 或放宽 graph admission。静态检查通过，fresh lifecycle 尚未验证。
- Source masked-load fill 的 JSON 投影现在复用 canonical `source_float_literal`，不再由 MCP 单独接受后续 owner 会拒绝的布尔值或不同的非有限值拼写；有限数以及既有 `-inf`/`inf` spelling 的正例，与 `True`、`NaN` 和任意表达式的负例保持同一 fail-closed 接受集合。该修复只同步 Source owner 与 semantic wire 的接受集合，没有替 Source 生成 fill。
- Source wire 中已有的 `provider_owned_relations` 现在完整进入 MCP access slot、access admission、失败模板和 compact repair context；这些消费者统一读取同一关系映射，不再由 environment 另写 `node_ref/effect_node_ref` 平行判断。未知关系、缺失关系或不相等关系继续 fail-closed，离线 compile、pyflakes 和 diff 检查通过。
- reduce/scan 的 consumer outside identity 曾在 FunctionalBoundary 与 TypeCheck 重复实现；现已收敛到共享 IntentOp contract 判据，两处仍分别执行 operand dtype/provenance 校验。23 个已有 identity 正负组合和三模块静态检查通过；未改变接受集合，也未覆盖真实 lifecycle 验证。
- 维度投影现在共用单一 `provider_dimension_alias_map` 与 `provider_shape_from_canonical`：Source wire 输入/transfer shape、owner typed shape、physical-store partition owner shape、output construction、MCP context allocator 绑定、Lift 环境 Source tensor types 和 provider `type.shape` admission、最终 `dimN → xN` materialization 都消费同一 Source ordinal 映射。provider operation 的符号 extent（含 `iota.attrs.n`）只接受当前 Source aliases，或非负整数；`argN`/`rN`/未声明 `xN` 不再作为 shape extent 参与后续类型推导。移除了 iota 诊断把非法 `xN` 暂时改写为 `dimN` 的路径。离线正负核验覆盖 `x0`、`x10`、literal、参数/SSA 标签、越界维度、布尔和负 extent，并覆盖 partition owner 的合法/未声明维度；`compileall`、`pyflakes`、`git diff --check` 通过。另已补充 relay HTTP 500 envelope 对 TLS/SSL EOF 的安全分类；未放宽任何语义准入。
- 已将远端两套 CPA 的全部 `proxy-url` 统一切换到新代理并重启对应 user-systemd 服务。只读 HTTPS CONNECT 预检成功：CONNECT 返回 200、TLS 校验成功、普通 HTTPS 返回 200；这只证明代理隧道具备基本连通性，尚未证明模型 Responses lifecycle 已恢复。

## 最近一次真实验证

- 使用全新 root 串行运行 `high_performance:liger_poly_norm`；显式使用 `gpt`、默认 MCP/miniswe Lift、Responses Replay、masked-J、单 Kernel/单分支/单 worker，未传 provider 配置。
- 新代理上的真实 Responses 调用已恢复：第 1 次 Lift 完成 9 次物理 provider call，`transport_retries=0`，没有 TLS、HTTP、SSE、认证、rate、usage、overload 或其他 provider failure。运行约 58.8 秒后由本地 MCP preparation fail-closed 停止。
- 安全终态为 `lift_mcp_no_progress`：阶段停在 `resource:semantic_transfer`，`mcp_phase_failure_count=2`、`last_failure_class=mcp_protocol_rejected`、`completed_step_count=0`、`transfer_read_generation=0`；终态却给出 `next_legal_action=begin_candidate`。没有 candidate，CandidateCheck、TypeCheck、Replay、Gate1、Gate2 均未启动，未计闭包。
- 代码审计确认动态 MCP resource schema 只绑定当前 URI 和 count 上限，没有把 host 当前唯一合法的 transfer `start` cursor 固定为 schema const；host dispatch 又严格拒绝不等于 `_semantic_transfer_expected_start` 的调用。同时 preconstruction `next_legal_action` 和 phase digest 均未投影该 cursor。这是分页状态在 provider schema、host admission 和安全诊断之间未同步的通性接口缺口；现有安全证据不足以断言具体被拒绝的页码。
- 运行中的 Pydantic `ResponseAPIUsage` serializer warning 未升级为 provider failure，与本次 MCP 无进展分开处理。当前 `NO_CLOSURE`、闭包数 `0/28`；root 保留，未读取 provider response、trajectory、candidate 或拒绝内容，未新增 `real-evidence`。

## 当前未解决阻断

1. semantic-transfer 分页 cursor 没有由 host 状态统一投影到 provider-facing tool schema、dispatch admission、phase identity、material-progress 记录和 `next_legal_action`。当前宽 schema 允许 Agent 生成 host 必然拒绝的 stale/default `start`，而安全终态又把下一动作误报成 `begin_candidate`。
2. MCP 失败 envelope 只保留 `application_error`，没有安全区分 cursor mismatch、reader cardinality/identity failure 与其他 resource application error；本轮不能仅靠终态证明具体页码或底层规则。应增加闭合集合内的规则分类和分页位置，不保存资源正文。
3. Source owner broadcast / computed-guard 等语义链仍保持 fail-closed；由于最近 lifecycle 没有 candidate，不能声称这些修复已经通过 CandidateCheck 验证。
4. 既有投影 family（load occurrence/domain/predicate/fill、reduce typed dataflow、program-domain/ABI、logical update/output mapping）仍需在 MCP 分页契约修复后的真实 candidate lifecycle 中验证；离线通过不计闭包。

## 下一步

1. 用同一 MCP phase-state 投影函数生成 semantic-transfer 的唯一合法 URI、必填 exact `start`、count 上限和下一动作；provider schema 与 host dispatch 必须消费同一结果，不能靠 prompt 要求模型记忆 cursor。
2. 将 `semantic_transfer_expected_start`、是否完成初始 ledger、是否处于 post-graph reread 纳入安全 phase identity/progress receipt；preconstruction `next_legal_action` 必须准确返回 `read_next_transfer_page` 或 `begin_candidate`。
3. 离线验证多页 ledger 可以从 start=1 连续读到 `next_start=null`，stale/跳页 cursor 仍 fail-closed，并让安全错误明确区分 cursor、identity、cardinality 和 reader failure；不创建测试目录。
4. 修复并完成静态/离线检查后，下一次真实验证仍使用全新 root、默认 MCP/miniswe、单 Kernel、单 worker、严格串行的 `liger_poly_norm` lifecycle。任何 provider failure 立即停止；只有真实 candidate 到达 CandidateCheck 后，才继续 TypeCheck、Replay、Gate1 和 Gate2。
