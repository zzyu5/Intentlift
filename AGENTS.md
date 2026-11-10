# Repository execution constraint

- `python/tests` was removed from the tracked active tree; its historical contents
  are retained only in the ignored workspace archive at
  `local/archive/legacy-trees/scrc-legacy-docs-tests.tgz` when local archival
  assets are present.
- Do not add, regenerate, or restore test files or test directories in the active tree.
- Do not replace real SCRC closure work with mocked unit, regression, smoke, or protocol tests.
- Validate changes through the real Kernel closure workflow and its actual Gate/closure artifacts.
- The archived test suite is historical material only; it is not part of the active workflow.

# Final-task-only development

- Do not create, bump, rename, synchronize, or discuss `vN`-style versions for
  internal implementations, Agent backends, prompts, tools, protocols, observer
  profiles, or intermediate specifications. Existing compatibility tags may stay
  only where current parsers require them; ordinary changes must not introduce a
  new tag or turn version bookkeeping into work.
- Do not create a new Markdown design/status/changelog document after each code
  change. Do not append per-commit or per-version implementation diaries to the
  active task.
- Maintain one active final task as the sole execution record. It should contain
  only the current contract, unresolved blockers, next actions, and authoritative
  real-Kernel closure evidence. Replace stale current-state text instead of
  accumulating historical narratives; archive genuinely superseded material.
- Measure progress only by real Gate1 closure, all three Gate2 comparisons,
  closed-Kernel count, and the final task's deliverables. TypeCheck is only a
  mechanical prerequisite/diagnostic and must never be reported as progress.
  Likewise, a new protocol label, document, receipt format, mock check, tool
  call, or implementation milestone is not progress.

# Responses API relay integration

When calling the new-api relay for Codex-backed models, follow these rules to avoid SSE errors like "stream closed before response.completed":

## Calling code

- Always set `stream: true`. Codex-family models on this relay only accept Responses API in streaming mode; non-streaming requests return 400.
- Stateless requests only. Never use `previous_response_id` or `store: true`. The relay is a stateless forwarder and rejects stateful chaining with 400.
- Re-send the full conversation history in every request. Each `input` array must contain the complete sequence of items (user messages, assistant messages, tool calls, tool results) from the start of the turn.
- Use correct content types per role: `user` messages must use `input_text`; `assistant` messages must use `output_text`. Mixing these up causes 400 errors.
- Implement client-side retry with exponential backoff. The relay intermittently closes streams before emitting the terminal `response.completed` frame. Tolerate premature close: if you received the final output text, treat the request as successful even without the completion event.

## MCP tool integration

- The MCP protocol logic itself does not change. MCP servers communicate with your agent over stdio/SSE locally, independent of the relay.
- For tool round-trips through the model: each request that continues a conversation must include both the `function_call` item (the model's tool invocation) and the `function_call_output` item (your tool's result) in the `input` array.
- Do not rely on the server remembering previous tool calls. Re-send the full history of function calls and outputs every time.
- This is the same stateless principle as above, applied to tool interactions.