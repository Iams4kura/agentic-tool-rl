# 方法与复现

本文定义 benchmark-v1.3.0 的公平比较、统计报告、证据恢复和结果发布规则。目标是让正结果、零结果和负结果都能被同一协议接受，而不是追逐一个固定分数。

## 1. 实验问题

1. 在相同共享 BC 起点与推理 Action Mask 下，动作级 PPO + Progress Reward（E）相对 BC+Mask（B）是否提升 TSR？
2. inference mask、sparse action PPO、Progress Reward 和信用分配粒度分别产生什么变化？
3. 策略运行的成功率、实际非法执行、安全、步骤效率和两种模拟 service-time 指标，与独立的全局规则型 Mask 有效性证据分别表现如何？
4. 结论能否从冻结 trace、checkpoint 和 manifest 重建？

工程闭环与统计结果分开：

- **工程有效**：生成、训练、评测、恢复、重放和一致性复算正确。
- **canonical**：完整结构证据符合预注册协议。
- **hypothesis_passed**：E−B paired TSR difference 的 95% CI 下界大于 0。

canonical 可以为 true 而 hypothesis_passed 为 false；后者是应保留的有效反例，不是工程失败。

## 2. 环境与入口

默认支持 macOS/Linux、Python 3.11，依赖由 uv.lock 固定。

~~~bash
make sync
make check
make verify-smoke
~~~

主 CI 运行 lint、strict mypy、pytest 和 B/E smoke。Qwen 契约位于独立手动 Optional Qwen adapter workflow，不随 push/PR 运行。

完整实验：

~~~bash
make benchmark
~~~

等价于：

~~~bash
uv run agentic-tool-rl ablate \
  --config configs/cpu_full.yaml \
  --ablation configs/ablation.yaml \
  --benchmark-dir artifacts/benchmark-v1 \
  --output artifacts/runs/full
~~~

## 3. 数据使用与 held-out

| 数据 | BC | Progress | PPO rollout | 最终评测 |
|---|---:|---:|---:|---:|
| train | 是 | 是 | 是 | 否 |
| dev | 否 | 校准/开发检查 | 否 | 否 |
| frozen test | 否 | 否 | 否 | 是 |

每个 family 的 test topology 与该 family 在 train/dev 的三个 topology 严格分离。operation ID 是 task-seeded opaque hash；candidate order 由公开 schema/state 确定性 shuffle。FeatureEncoder 不读取 call_id、标签、oracle 或隐藏目标，并去除 task/entity/approval 身份。

v1.2 已观察的 test 不再作为 final holdout。停止理由和 ceiling 证据见 [pilot-v1.2 归档](results/pilot-v1.2.md)。

## 4. 共享训练起点

对每个预注册 seed：

1. 从 train oracle demonstration 训练一个 BC Actor-Critic；
2. 从 train/dev noisy continuation 训练并冻结 Progress Estimator；
3. 保存 shared/seed-<seed>/bc-checkpoint.pt；
4. A/B/C/D/E/F 都复制同一 Actor-Critic 与 FeatureEncoder 起点。

A/B 必须满足 PPO updates=0、rollout_steps=0、parameter_l2_delta=0 且参数摘要等于共享 BC。C/D/E/F 必须有正数 PPO updates、正数 rollout_steps 和非零 L2 delta，否则 verifier 拒绝。

## 5. 六变体与可解释对照

| 名称 | 算法 | Progress | Mask |
|---|---|---:|---:|
| A-BC-Unmasked | BC | 否 | 否 |
| B-BC-Mask | BC | 否 | 是 |
| C-PPO-Sparse-Unmasked | action PPO | 否 | 否 |
| D-PPO-Sparse-Mask | action PPO | 否 | 是 |
| E-PPO-Progress-Mask | action PPO | 是 | 是 |
| F-Sequence-PPO-Progress-Mask | sequence PPO | 是 | 是 |

解释顺序：

- B vs A：BC 策略下推理 mask 的系统影响；
- D vs C：sparse PPO 下推理 mask 的系统影响；
- D vs B：同为 mask、无 Progress，隔离 sparse action PPO；
- E vs D：同为 action PPO+mask，隔离 Progress；
- E vs B：预注册主比较；相同 BC 起点、mask、seed/case 和推理环境下，比较 PPO+Progress 联合训练增量；
- E vs F：只改变动作级/序列级信用分配。

主比较不能写成 E vs A，因为两者的 inference mask 不同，会把硬约束收益混入 RL 训练收益。

## 6. 训练协议

完整 PPO 配置：

| 参数 | 值 |
|---|---:|
| iterations | 8 |
| rollout episodes / iteration | 128 |
| epochs / iteration | 4 |
| minibatch size | 512 |
| learning rate | 3e-4 |
| gamma | 0.99 |
| GAE lambda | 0.95 |
| clip ratio | 0.2 |
| value coefficient | 0.5 |
| entropy coefficient | 0.01 |
| max gradient norm | 0.5 |

rollout 使用 seeded family-stratified schedule。动作级 PPO 按 trajectory 计算 GAE；sequence PPO 对一个 episode 只计算一个 sequence ratio、discounted return 和初始 value。

Progress：

~~~text
Phi(s) = P(success | visible state, user_goal)
F(s,s') = beta * (gamma * Phi(s') - Phi(s))
~~~

Estimator 在 PPO 前冻结，输入含去标识 user_goal、visible state 和 step budget；不含 test、oracle 或隐藏 goal predicates。task、progress、step、invalid、forbidden 奖励分量分别写入 trace。

## 7. Smoke 与完整矩阵

### 7.1 Smoke

~~~bash
make verify-smoke
~~~

smoke 使用一个 seed，运行 B-BC-Mask 与 E-PPO-Progress-Mask，覆盖同一生产代码路径。它的 run manifest 为 claim_mode=none，不生成 claim-check.json；同目录中的陈旧报告会被删除。

### 7.2 Canonical ablation

ablate 在训练前强制：

- seed 序列与 configs/ablation.yaml 完全一致且五个唯一；
- 六变体名称与 algorithm/progress/mask 定义完全一致；
- test 恰为 1000 个唯一 cases；
- action-validity-v2 恰为 20,000 条且正负各 10,000；
- bootstrap samples 至少 2；完整配置实际为 1000，confidence=0.95。

完整矩阵产生 30 runs 和 30,000 个策略评测单元。

## 8. 指标

每个 run 从 trace 生成 metrics.json：

- TSR 与 family macro TSR；
- invalid_action_rate、forbidden_side_effect_rate；
- steps_efficiency、mean_steps；
- successful_conditional_simulated_service_time_s；
- timeout_penalized_simulated_cost_s。

规则型 ActionMask 的 valid/invalid recall、BAcc、macro-F1、混淆矩阵和分层 recall 不从策略 trace 计算，也不进入任何 A–F 变体的 metrics.json。它们只对冻结 action-validity-v2 全局计算一次，直接写入当前 run bundle 的 benchmark/action_validity.metrics.json，不修改共享 benchmark；canonical claim 引用其中的全局 BAcc。

两个 service-time 字段必须全名展示：

- successful_conditional_simulated_service_time_s：成功任务的未截断模拟工具 service time 均值；无成功时为 null。
- timeout_penalized_simulated_cost_s：成功任务截断到 timeout，失败任务计 timeout 的全样本均值。

它们都不对应 Agent wall-clock 或线上端到端时延。禁止把第二项简写成 latency，也禁止把第一项解释为线上耗时。

完整配置对 scalar metrics 做 1000 次 case-cluster percentile bootstrap。successful-conditional 项只作为点估计；timeout-penalized cost 有 case-cluster CI。

## 9. Canonical statistical report

输出文件仍名为 claim-check.json，但语义是 schema_version=3.0 的统计报告，不是固定分数门槛。

### 9.1 结构检查

报告检查：

- 五 seed、六冻结变体、1000 unique test cases；
- A–F 每个 seed 精确覆盖相同 case 集；统计摘要和比较仍只使用 A/B/E；
- 每个 (variant, seed, case) 有一个布尔 success；
- 聚合 TSR 可从这些 outcomes 重算；
- 20k 动作集及 10k/10k 平衡；
- paired bootstrap 配置和 95% confidence。

### 9.2 主比较

预注册项来自 configs/ablation.yaml：

~~~text
treatment:       E-PPO-Progress-Mask
matched baseline: B-BC-Mask
primary metric:   TSR
hypothesis:       paired TSR-difference CI lower bound > 0
~~~

对每个 seed 和 case 先形成 success_E − success_B。每个 bootstrap replicate 分别有放回重采样 seed 与完整 case clusters，并保持 E/B 配对。报告写入 estimate、low、high、seed_count、case_cluster_count 和 paired_observations。

### 9.3 失败假设也是有效结果

- canonical=true：结构证据完整；
- hypothesis_passed/passed=true：CI lower > 0；
- canonical=true 且 passed=false：完整、可信但未支持假设的实验。

CLI ablate 只因 canonical=false 失败，不因 hypothesis_passed=false 失败。仓库没有 metrics-only 的 canonical 旁路，也没有固定 TSR/BAcc/service-time 通过阈值。

## 10. 证据目录

~~~text
artifacts/benchmark-v1/
├── manifest.json
├── train.jsonl
├── dev.jsonl
├── test.jsonl
├── action_validity.jsonl
└── action_validity.manifest.json

artifacts/runs/full/<run-id>/
├── run-manifest.json
├── case-id-manifest.json
├── claim-check.json
├── benchmark/
│   ├── manifest.json
│   ├── train.jsonl
│   ├── dev.jsonl
│   ├── test.jsonl
│   ├── action_validity.jsonl
│   ├── action_validity.manifest.json
│   └── action_validity.metrics.json
├── shared/seed-<seed>/bc-checkpoint.pt
└── <variant>/seed-<seed>/
    ├── checkpoint.pt
    ├── training.json
    ├── traces.jsonl
    ├── metrics.json
    ├── recompute.json
    └── run-integrity.json
~~~

run ID 由 experiment config、benchmark manifest、seed、变体和 source fingerprint 决定。source fingerprint 覆盖 src、configs、pyproject.toml 与 uv.lock。`run-manifest.json` 以自身父目录为 `path_base`，所有 artifact 路径均为规范相对路径；整个 `<run-id>/` 是可搬迁、可重命名且不依赖 cwd 的自包含证据包。verifier 拒绝绝对路径、`..` 和逃逸 bundle 的符号链接。

## 11. Trace 复算与 verifier

单 run 一致性复算：

~~~bash
uv run agentic-tool-rl recompute \
  --traces artifacts/runs/full/<run-id>/E-PPO-Progress-Mask/seed-17/traces.jsonl \
  --metrics artifacts/runs/full/<run-id>/E-PPO-Progress-Mask/seed-17/metrics.json \
  --output artifacts/recompute.json
~~~

recompute 从原始 trace 重新调用同一 compute_metrics 实现并比较，容差为 1e-9。它证明 trace 与发布 metrics 一致，但不是第二套独立指标定义。

整个 run：

~~~bash
uv run agentic-tool-rl verify-run \
  --manifest artifacts/runs/full/<run-id>/run-manifest.json
~~~

verifier：

1. 校验 source/runtime identity；
2. 重生成 benchmark-v1.3 与 action-validity-v2，核对 held-out、20k 分层和 SHA256；
3. 校验共享 BC、变体 checkpoint、training metadata、PPO update、rollout 和实际 L2 delta；
4. 校验 Resume Guard 的 benchmark/config/source/checkpoint 哈希；
5. 逐步重放 candidate、ToolCall、dry-run、事务结果、终态与模拟 service time；
6. 使用声明的 checkpoint、FeatureEncoder、Progress Estimator 与变体开关确定性重跑每个 case，逐字段比对策略 trace；
7. 使用同一 compute_metrics 做 trace 一致性复算；
8. 验证 A–F 完整 outcomes，并重建 paired A/B/E bootstrap CI 与 claim-check.json。

`verify-run` 强制 source/runtime identity，适用于当前源码产生的 run。验证已经发布、随后源码继续演进的历史证据时，应使用 tag-bound 入口，例如 `uv run --locked python scripts/verify_release.py --release v0.1.0`；该入口会先固定 Release 资产、tag object、commit、源码、运行时和 uv 版本，再调用对应 tag 内的 `verify-run`。

## 12. 恢复语义

TraceStore 以换行完成一条 JSONL 提交；崩溃时只截断末尾不完整行。相同 case/相同内容追加是 no-op，相同 case/不同内容冲突。Resume Guard 在读取非空 trace 前验证四类输入；缺 guard 的历史 trace 不可追认，任一输入变化都必须进入新 run。

## 13. Azure 推导

延迟档位的公开来源与口径见 [Benchmark Service-time](benchmark.md#8-service-time-来源与指标)。复现：

~~~bash
uv run python scripts/derive_azure_latency_profile.py <official-archive.tar.xz>
~~~

脚本默认校验 archive SHA256，只读 14 个 duration CSV，并按 Count 计算 invocation-weighted quantiles。它不下载原始数据。

## 14. Qwen 扩展

~~~bash
make qwen-check
~~~

该命令只在手动 workflow 或本地运行。它验证固定 Qwen revision、strict YAML、LoRA 参数、设备策略、结构化调用和 finite log-prob/value；dry-run 明确 would_download_weights=false。当前没有 GPU trainer 或 Qwen final 结果。

## 15. 发布规则与正式结果

v1.3 final 已完成并通过内嵌及独立 canonical verifier。实际六变体、五 seed、case-cluster CI、E−B 配对 CI 与全局 ActionMask 结果见 [canonical final report](results/canonical-v1.3.md)。发布仍遵循：

1. 先运行 verify-run；
2. 报告全部六变体的五 seed 分布和 CI，不只挑最佳 seed；
3. 重点报告 E vs B paired CI，同时区分 E vs A total-system comparison；
4. service-time 字段使用全名并声明模拟口径；
5. 将全局 action-validity 表与策略变体表分开，明确 BAcc 是规则型 Mask 在 action-validity-v2 上的单次 benchmark 级结果；
6. 无论 hypothesis_passed 为 true 或 false 都发布，不按结果修改 holdout。

主比较结果为：B TSR `0.4636`，E TSR `0.5288`，E−B=`0.0652`；配对 seed×case bootstrap 95% CI 为 `[0.034195, 0.09782]`。`canonical=true`、`hypothesis_passed=true`。Run ID 为 `886622c5cb6cffc3`。

## 16. 限制

- 四拓扑与 held-out 仍由同一合成生成器定义。
- Mask/oracle 共享公开规范；500 hidden-ledger cases 只展示一种部分可观测性。
- 模拟 service time 不含模型推理、网络、排队或真实故障。
- 五 seed 与 1000 bootstrap 提供不确定性估计，但不消除 synthetic-to-real gap。
- 轻量结果不能外推为 Qwen 效果。
