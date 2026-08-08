# Benchmark 说明

本文定义 benchmark-v1.3.0 的任务分布、held-out 规则、action-validity-v2、模拟 service-time 成本和指标。修改生成器后必须升版本、重新生成 manifest，并丢弃旧版本的横向可比性。

## 1. Canonical 配置

configs/cpu_full.yaml：

| Split | 数量 | 用途 |
|---|---:|---|
| train | 2000 | BC、Progress 和 PPO rollout |
| dev | 200 | Progress 校准与开发期 floor/ceiling 检查 |
| test | 1000 | 一次性策略评测与动作有效性来源 |

canonical base seed 在预注册提交后按[锁定协议](protocol/preregistration-v1.3.md#4-final-seed-的一次性派生)一次性派生，并以 configs/cpu_full.yaml 与锁定表为准；split 使用独立 offset，case、entity 和 task seed 互不重叠。完整评测使用 5 个预注册训练 seed 和 1000 次 bootstrap。smoke 使用独立的固定开发 seed、32/12/32、一个训练 seed 和 50 次 bootstrap，只验证 B-BC-Mask / E-PPO-Progress-Mask 工程闭环。

## 2. 十个业务族与长度

calendar_coordination、ecommerce_returns、travel_booking、it_permissions、customer_support、expense_reimbursement、subscription_changes、cloud_incident_response、recruitment_interviews、order_fulfillment 各覆盖 100 个 canonical test cases。

| 难度 | 最优步骤数 | 每族 | 全 test |
|---|---:|---:|---:|
| short | 6–8 | 30 | 300 |
| medium | 9–11 | 40 | 400 |
| long | 12–15 | 30 | 300 |

每题 max_steps = optimal_steps + 4，允许有限错误或合法无用调用，但不允许无限循环。

## 3. 四拓扑与 held-out

v1.3 支持：

| Topology | 结构要点 |
|---|---|
| diamond_tail | 单根分叉、汇合后进入尾链 |
| fanout_gate | 单根 fan-out，后续节点受多前置 gate 约束 |
| dual_lane | 两条交错 lane 推进 |
| dual_root_mesh | 双根与多前置 mesh |

生成器再把审批 barrier 和最终动作接到主体 sinks 后。每题确定性保存两条不同的合法 topological orders；成功按隐藏终态谓词判定，不要求复刻第一条 oracle 文本。

对每个 family，WORKFLOW_TOPOLOGIES[family_index mod 4] 是专属 held-out topology：

- test 只生成这个 family×topology 组合；
- train/dev 只生成该 family 的其余三种 topology；
- 因此 test 组合与 train/dev 的 family×topology pair 集合严格不相交。

这是结构组合泛化，不只是 entity 或 seed 换名。

## 4. Opaque 标识与候选构造

### 4.1 Operation ID

每个 operation ID 由 generator version、task seed 和操作标签经 BLAKE2s 生成：

~~~text
op_<20 hexadecimal characters>
~~~

不同 task seed 的 ID 不重叠，也不编码 step-XX 序号。FeatureEncoder 对 completed IDs 与候选 ID 使用共享哈希桶保留匹配信号，但不恢复 ordinal。

### 4.2 确定性候选乱序

Candidate Builder 的局部 seed 只来自公开 ToolSchema 与 visible state，包括 entity、version、status、priority、region、channel、completed operations 和 approval 是否可见。每个状态的候选用该 seed 确定性 shuffle：

- 相同输入可字节级重现；
- 固定 candidate index 不再是动作标签；
- call_id 只用于证据，不进入策略特征。

### 4.3 候选类型

候选包含：

1. 满足公开前置条件的可推进 action；
2. 合法 inspect_status；
3. 从 priority/region/channel 匹配的合法 contextual read-only query；
4. 已完成 operation 的合法 idempotent replay；
5. Schema、Grounding、Precondition、Safety 四类非法 action。

contextual query 与 replay 可执行但不推进目标。ActionMask 应放行它们；策略必须学习相关性，不能依赖“mask 后任选”。

## 5. 事务真值与 oracle

环境 validator 按顺序检查：

- Tool Schema、字段、类型和额外参数；
- entity/tool/opaque operation grounding；
- expected_version 与公开 predecessor；
- approval、idempotency 和禁止副作用。

通过后 ToolRegistry 原子提交 effects、version、audit log 和 idempotency ledger；拒绝调用只能增加一步与模拟 service time，业务状态保持不变。

生成 split 前，oracle 在真实 TransactionalWorkflowEnv 中回放合法 plan。manifest 保存 split 文件 SHA256、family/difficulty 统计、seed/entity digest、oracle solvability、base seed 和 generator version。verify-run 还会按配置重新生成并逐项比对，而不只信任 manifest 自报。

## 6. action-validity-v2

canonical 分类集：

~~~text
1000 tasks × 5 states/task × 4 candidates/state = 20,000
~~~

每个状态有 2 valid、2 invalid，总计 valid/invalid 各 10,000。四类 invalid 各 2,500。

| Challenge source | 数量 | 判定信息 |
|---|---:|---|
| standard_candidate | 19,500 | 公开 ToolSchema/Observation 足以判定 |
| hidden_ledger_collision | 500 | 只有环境隐藏 idempotency ledger 暴露冲突 |

标签由环境 dry-run 生成；该模块不导入 ActionMask。评分时读取冻结标签，仅运行策略侧独立规则 ActionMask。样本保存 observation、ToolCall、label、invalid kind、challenge source 和 state SHA256。

500 个 hidden-ledger negatives 的公开输入与合法调用一致，规则型 Mask 按可见信息会放行。这让数据显式包含部分可观测边界。应把指标准确写为“规则型 ActionMask 在 action-validity-v2 合成约束集上的 Balanced Accuracy”，不能写成策略智能、真实业务规则发现率或全状态安全率。

动作 BAcc 与策略执行指标分开，不能按 variant 复制：

- BAcc/recall/macro-F1：冻结候选分类能力，只写入全局 action_validity.metrics.json；
- invalid_action_rate：策略实际执行动作中的非法比例；
- TSR：终态任务成功率。

## 7. 策略 run 与全局动作有效性指标

每个策略 run 的 metrics.json 只发布 TSR（含 family macro 聚合）、实际执行/安全、步骤和两种模拟 service-time/cost。规则型 ActionMask 分类指标属于 benchmark 级全局证据，不属于 A–F 任一变体；canonical claim 引用该全局 BAcc。

### 7.1 Task Success Rate

~~~text
TSR = successful_cases / all_cases
~~~

成功要求隐藏目标全部满足且禁止副作用为 0。macro_tsr 先按 family 计算，再对 10 个 family 等权平均。

### 7.2 全局 Action-validity Balanced Accuracy

~~~text
valid_recall   = TP / (TP + FN)
invalid_recall = TN / (TN + FP)
BAcc           = (valid_recall + invalid_recall) / 2
~~~

同时在 benchmark/action_validity.metrics.json 发布 recall、macro-F1、混淆矩阵和四类 invalid recall；这些字段不进入策略 run 的 metrics.json。

### 7.3 执行与安全

~~~text
invalid_action_rate = 1 - mean(executed_action_is_valid)
forbidden_side_effect_rate = affected_cases / all_cases
~~~

前者只统计实际执行动作，不把未被选择的 candidate 放进分母。

### 7.4 Step efficiency

成功题：

~~~text
min(1, optimal_steps / max(actual_steps, optimal_steps))
~~~

失败题为 0；steps_efficiency 对全部题取均值。

## 8. Service-time 来源与指标

### 8.1 公开来源

冻结档位派生自 [Microsoft Azure Functions Trace 2019](https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsDataset2019.md) 的 14 天 function-duration 文件。处理规则：

- 保留 finite Average >= 0 且 Count > 0；
- 662,922 行；
- 按 Count 对 12,481,740,344 次 invocation 加权；
- Average 从毫秒转换为秒；
- 数据许可为 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)；
- 官方 archive SHA256 为 aff8b3ca7240a41a109e4ee598e0a96e45fcb92e7b8395ac19cb3748cd260d89。

冻结记录见 [azure-functions-2019-latency-profile.json](data/azure-functions-2019-latency-profile.json)。数据集要求引用 Shahrad et al., [Serverless in the Wild](https://www.microsoft.com/en-us/research/uploads/prod/2020/05/serverless-ATC20.pdf), USENIX ATC 2020。

下载官方 archive 后可复现推导：

~~~bash
uv run python scripts/derive_azure_latency_profile.py \
  /path/to/azurefunctions_dataset2019_azurefunctions-dataset2019.tar.xz
~~~

[推导脚本](../scripts/derive_azure_latency_profile.py) 仅依赖标准库、不主动联网，读取 archive 或含 14 个 CSV 的目录；archive 输入默认先校验官方 SHA256，再输出行数、invocation 数和 p50–p97.5 分位数。

### 8.2 冻结工具成本

| 调用 | 模拟 service time |
|---|---:|
| 首次合法 mutating operation | 等概率取 p85/p87.5/p90/p92.5/p95：1.592/1.633/1.658/1.686/1.734 s |
| read-only query | p80：0.902 s |
| idempotent replay | p50：0.140 s |
| rejected validation | p75：0.334 s |

每题 mutating 档位由 task-local RNG 确定并写入 latency_trace。它们是模拟工具执行成本，不包含 Agent 推理、网络、队列、cold start 或真实系统端到端时间。

### 8.3 发布指标

**successful_conditional_simulated_service_time_s**

~~~text
mean(raw simulated service time | success)
~~~

只对成功任务计算、不截断；若没有成功任务则为 null。它不对应 wall-clock 或线上端到端时延。

**timeout_penalized_simulated_cost_s**

~~~text
successful case: min(raw simulated service time, timeout)
failed case:     timeout
aggregate:       mean over all cases
~~~

完整配置 timeout=80 s。该量用于让快速失败不能获得虚假成本优势；它是惩罚性模拟成本，也不是完成耗时。

## 9. 置信区间

单 run 的可重采样标量使用 case-cluster percentile bootstrap。完整配置为 1000 samples、95% confidence、固定 seed；smoke 为 50 samples。

successful_conditional_simulated_service_time_s 因条件样本集合随成功变化，只作为点估计，不在 scalar bootstrap 列表中。timeout_penalized_simulated_cost_s 进入 case-cluster CI。

canonical 主比较另使用 paired seed×case bootstrap：

- treatment 为 E-PPO-Progress-Mask；
- baseline 为 B-BC-Mask；
- 先按完全相同的 (seed, case_id) 配对布尔 success；
- 每个 replicate 分别有放回采样 seed 和完整 case clusters；
- E/B 配对始终保留；
- 预注册统计量为 E−B TSR difference，完整配置 1000 replicates、95% CI。

CI 下界是否大于 0 是假设结果，不是 canonical 结构门槛。下界不大于 0 的报告仍是有效 canonical 反例。

## 10. 生成与验真

~~~bash
uv run agentic-tool-rl generate \
  --config configs/cpu_full.yaml \
  --output artifacts/benchmark-v1
~~~

关键文件：

~~~text
manifest.json
train.jsonl
dev.jsonl
test.jsonl
action_validity.jsonl
action_validity.manifest.json
action_validity.metrics.json
~~~

verify-run 会重生成三个 split 和动作集、校验 family×topology held-out、opaque IDs、challenge counts、文件哈希与 case manifest，再逐步重放策略 trace。

## 11. 版本边界

benchmark-v1.2.0 的首个 pilot 暴露 BC ceiling、ordinal operation ID、固定拓扑/候选顺序和不公平主比较，因此主动停止并将已消费 test 降级为开发证据。详情见 [pilot-v1.2 归档](results/pilot-v1.2.md)。

v1.3 final 尚未产生；本文只定义协议和口径，不发布推测数字。
