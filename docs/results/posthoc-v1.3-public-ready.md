# benchmark-v1.3 post-hoc：公开 ready greedy 审计

本审计是对已发布 benchmark-v1.3.0 的追加解释，不修改 v1.3 tag、release asset、
canonical 数字或历史 verifier。它验证的是一个更窄的问题：只读取策略公开输入的确定性规则，
能否完成冻结测试集。

## 结果

在 1,000 个 canonical test cases 上，`R1-public-ready-greedy-v1` 得到：

| 指标 | 结果 |
|---|---:|
| 成功题数 | 1,000 / 1,000 |
| TSR | 1.0000 |
| 平均步数 | 10.12 |
| forbidden side effects | 0 |

规则只接收 `PolicyInput`，逐候选选择第一个同时满足以下条件的动作：

```text
action_mask
and tool_schema.policy_allowed
and tool_schema.mutating
and tool_schema.operation_id not in visible_state.completed_nodes
```

这说明 v1.3 的公开 schema、candidate builder 与 action mask 已把 DAG readiness 暴露给策略；
因此 v1.3 的 TSR 不能单独支持“模型独立规划或发现 DAG”的结论。它仍可用于验证结构化动作执行、
约束遵循和可复现实验工程。

## 复现与独立重放

```bash
uv run python scripts/audit_v13_public_ready.py \
  --output artifacts/audits/v1.3-public-ready
```

命令默认以冻结参数 `benchmark-v1.3.0`、canonical base seed `2006549735` 重新生成 1,000 个 test
cases，也可通过 `--benchmark-test path/to/test.jsonl` 读取发布快照。它会写出：

- `v1.3-public-ready-audit.jsonl`：逐题、逐步 trace，包含 `PolicyInput` SHA256、选择理由、
  去标识化动作和环境结果；
- `v1.3-public-ready-audit.manifest.json`：1,000 个逐题 row hash、输入序列 hash 与聚合值；
- `v1.3-public-ready-audit.verification.json`：从任务重新构造每个 `PolicyInput`、重新决策、
  重新执行环境后的逐字节比较结果。

对已有证据仅执行独立重放时，追加 `--verify-only`；该模式不会先覆盖待验证文件。

当前实现的确定性复现值为：

```text
trace_sha256    2fcc1f2bf0492e241071b8a4f30714d8b86fb25731afb6930cd62dd1b15fd79c
manifest_sha256 38f697c8aed21b2ceea240e1fc58ca036631c7acf15f363c6d645eab7d49abc8
verification    passed=true, replayed_case_count=1000
```

`PolicyInput` 不包含 case/task ID、原始 entity identity、candidate `call_id`、oracle、hidden
goal、validity label 或隐藏 ledger；其 canonical bytes 使用 UTF-8、排序 key、无多余空白的 JSON。
