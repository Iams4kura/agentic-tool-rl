# benchmark-v1.3.0 canonical final report

状态：**canonical=true；hypothesis_passed=true；一次性 final holdout 已完成。**

本页只报告 seed-lock 后生成的 benchmark-v1.3.0 final holdout。开发集选择、v1.2 pilot 和停止理由不混入本页结果。实验协议见[预注册文档](../protocol/preregistration-v1.3.md)，指标定义见[方法与复现](../methodology.md)。

## 1. 结论

在相同共享 BC checkpoint、相同冻结 test cases、相同候选生成器和相同推理 Action Mask 下：

- `B-BC-Mask` 的聚合 TSR 为 `0.4636`；
- `E-PPO-Progress-Mask` 的聚合 TSR 为 `0.5288`；
- E−B 配对差值为 `0.0652`，即 `+6.52` 个百分点；
- 配对 seed×case bootstrap 95% CI 为 `[0.034195, 0.09782]`；
- CI 下界大于 0，因此预注册主要假设获支持；
- E 在五个预注册 seed 上的 TSR 均高于 B。

这里支持的是“动作级 PPO + Progress Reward 联合训练相对 BC+Mask matched baseline 的增量”，不能改写为“Progress Reward 单独带来 6.52 个百分点”。

## 2. 实验身份

| 字段 | 值 |
|---|---|
| Run ID | `886622c5cb6cffc3` |
| Generator | `benchmark-v1.3.0` |
| Final base seed | `2006549735` |
| Train / dev / test | `2000 / 200 / 1000` |
| 业务族 | 10 |
| 训练 seed | `17, 29, 43, 71, 101` |
| 变体 | 6 |
| 策略评测单元 | 30,000 |
| Action-validity 样本 | 20,000，valid/invalid 各 10,000 |
| Bootstrap | 1000 次，95% percentile CI |
| Bootstrap seed | `20260808` |
| Timeout-penalized cost timeout | `80.0 s` |
| Preregistration commit | `356c8b55b0f088dc613ecf00a8d832ca311076ff` |
| Seed-lock commit | `46cd948b39981f8a9ef6a8283fc7f2ece2d3ce40` |
| Source fingerprint | `75eb32b7a94178616eaa977f0c42f147143af904fd09d69928cde000bbfe40d3` |
| Benchmark manifest SHA256 | `86ca5375c648ab54e7cbd54a13f9a4f978025c30d5ae6f90a3a23357bb4a1275` |
| Action-validity metrics SHA256 | `d5f871d43123d7217f1ebf6be77db9862d6af019cbf224c2bcb5ccd719b83a6b` |
| Run manifest 文件 SHA256 | `5153952e0e7cef78da1338a37b0b57b53b75e7b38174d4d5584cafabeca38b19` |
| Claim-check 文件 SHA256 | `f6e206ff4ceb2a0040e019ffc01edb9ac783b218ee3637819dc8bf116ef900f2` |
| Case-ID manifest SHA256 | `b9a81dceb092831df1abfb77b68ec661ec7058e0e69dba636bc71c1867867434` |
| 规范化证据包 SHA256 | `0de48d58521aebadef8abdf990616c5dc5787de56c0f27a4d9c9bfa9033ff56f` |
| Runtime | CPython `3.11.15`、PyTorch `2.13.0`、CUDA `null` |

`source fingerprint` 是对 `src/`、`configs/`、`pyproject.toml` 与 `uv.lock` 的内容指纹，不是 Git commit SHA。

## 3. 证据完整性

- 六变体 × 五 seed 的 30 个运行目录均存在；
- 每个运行包含 checkpoint、training metadata、1000 条 trace、metrics、recompute 和 Resume Guard；
- 30 个运行共 30,000 条 trace；
- 30/30 份 `recompute.json` 均为 `matches=true`，比较容差为 `1e-9`；
- 30 个 checkpoint SHA256 和 30 个 trace SHA256 均记录在 `run-manifest.json`；
- A/B 的 PPO updates、rollout steps 和参数 L2 delta 均为 0；
- C/D/E/F 均有真实 PPO updates、正数 rollout steps 和非零参数 L2 delta；
- `ablate` 结束前执行的 checkpoint-bound verifier 通过全部 30 runs / 30,000 evaluation units；
- 独立调用 `verify-run` 再次得到 `passed=true`、`runs=30`、`evaluation_units=30000`；
- 规范化证据包解压到随机临时目录后第三次得到相同 verifier 结果，证明 manifest 不依赖原始绝对路径；
- `claim-check.json` 为 `schema_version=3.0`，且 `canonical=true`、`passed=true`。

## 4. 六变体聚合结果

每个变体由五个 seed、每 seed 1000 题组成，共 5000 条 trace。表中的变体 CI 使用 `compute_metrics` 的 case-cluster percentile bootstrap：以 1000 个 `case_id` 为 cluster，同一 case 的五个 seed 始终一起重采样。百分比仅在展示时四舍五入。

### 4.1 成功与安全结果

| Variant | 成功数 / 5000 | TSR，95% CI | Macro TSR，95% CI | Invalid action rate，95% CI | Forbidden side-effect rate，95% CI |
|---|---:|---:|---:|---:|---:|
| A-BC-Unmasked | 370 | 7.400% [6.400%, 8.421%] | 7.400% [6.383%, 8.422%] | 65.266% [64.195%, 66.429%] | 0% [0%, 0%] |
| B-BC-Mask | 2318 | 46.360% [44.000%, 48.620%] | 46.360% [44.059%, 48.591%] | 0% [0%, 0%] | 0% [0%, 0%] |
| C-PPO-Sparse-Unmasked | 388 | 7.760% [6.700%, 8.900%] | 7.760% [6.714%, 8.891%] | 54.884% [53.460%, 56.289%] | 0% [0%, 0%] |
| D-PPO-Sparse-Mask | 2616 | 52.320% [50.079%, 54.441%] | 52.320% [50.078%, 54.448%] | 0% [0%, 0%] | 0% [0%, 0%] |
| E-PPO-Progress-Mask | 2644 | 52.880% [50.680%, 55.001%] | 52.880% [50.777%, 54.905%] | 0% [0%, 0%] | 0% [0%, 0%] |
| F-Sequence-PPO-Progress-Mask | 2597 | 51.940% [49.640%, 54.200%] | 51.940% [49.675%, 54.162%] | 0% [0%, 0%] | 0% [0%, 0%] |

由于每个 family 在五个 seed 中均恰有 500 条结果，聚合点估计中的 macro TSR 与 micro TSR 相同；bootstrap replicate 可能改变 family 权重，因此两者的区间略有差异。

### 4.2 步骤与模拟成本结果

| Variant | Executed actions | Step efficiency，95% CI | Mean steps，95% CI | successful_conditional_simulated_service_time_s | timeout_penalized_simulated_cost_s，95% CI |
|---|---:|---:|---:|---:|---:|
| A-BC-Unmasked | 69,410 | 6.827% [5.933%, 7.764%] | 13.8820 [13.6978, 14.0686] | 12.5289 s | 75.0071 s [74.3064, 75.6849] |
| B-BC-Mask | 62,281 | 44.703% [42.498%, 46.913%] | 12.4562 [12.2320, 12.6977] | 14.6532 s | 49.7052 s [48.1827, 51.3597] |
| C-PPO-Sparse-Unmasked | 69,419 | 7.031% [6.053%, 8.078%] | 13.8838 [13.7008, 14.0726] | 12.3655 s | 74.7516 s [73.9789, 75.4817] |
| D-PPO-Sparse-Mask | 61,203 | 50.532% [48.283%, 52.601%] | 12.2406 [12.0238, 12.4869] | 15.0826 s | 46.0352 s [44.6145, 47.5220] |
| E-PPO-Progress-Mask | 61,020 | 51.186% [49.019%, 53.224%] | 12.2040 [11.9778, 12.4488] | 15.0507 s | 45.6548 s [44.2286, 47.1213] |
| F-Sequence-PPO-Progress-Mask | 61,247 | 50.168% [47.925%, 52.387%] | 12.2494 [12.0234, 12.4870] | 14.9399 s | 46.2078 s [44.7352, 47.6964] |

`successful_conditional_simulated_service_time_s` 按预注册规则只提供点估计。它只统计成功任务，因此不同成功集合之间不能直接解释为系统变快或变慢。

`timeout_penalized_simulated_cost_s` 对失败任务计 80 秒，对成功任务计 `min(service_time, 80)`。两个字段均为模拟工具 service-time/cost，不是模型推理延迟、网络 RTT、wall-clock latency 或线上 SLO。

## 5. 每 seed TSR

表内单位为百分比。

| Variant | Seed 17 | Seed 29 | Seed 43 | Seed 71 | Seed 101 | 聚合 |
|---|---:|---:|---:|---:|---:|---:|
| A-BC-Unmasked | 5.5% | 8.7% | 9.4% | 6.6% | 6.8% | 7.40% |
| B-BC-Mask | 47.6% | 43.0% | 43.9% | 48.2% | 49.1% | 46.36% |
| C-PPO-Sparse-Unmasked | 5.7% | 8.6% | 9.1% | 7.9% | 7.5% | 7.76% |
| D-PPO-Sparse-Mask | 52.9% | 53.0% | 52.1% | 51.6% | 52.0% | 52.32% |
| E-PPO-Progress-Mask | 52.8% | 53.3% | 53.6% | 52.5% | 52.2% | 52.88% |
| F-Sequence-PPO-Progress-Mask | 55.4% | 51.6% | 49.6% | 52.5% | 50.6% | 51.94% |
| E−B | +5.2 pp | +10.3 pp | +9.7 pp | +4.3 pp | +3.1 pp | +6.52 pp |

## 6. 预注册主比较

主比较严格按相同 `(seed, case_id)` 对齐 E 与 B，再分别有放回重采样 seed 和完整 case cluster。

| 字段 | 值 |
|---|---:|
| Treatment | `E-PPO-Progress-Mask` |
| Matched baseline | `B-BC-Mask` |
| Treatment TSR | `0.5288` |
| Baseline TSR | `0.4636` |
| E−B estimate | `0.0652` |
| 95% CI | `[0.034195, 0.09782]` |
| Bootstrap samples | `1000` |
| Seed count | `5` |
| Case-cluster count | `1000` |
| Paired observations | `5000` |
| Hypothesis | CI lower bound > 0 |
| Result | `hypothesis_passed=true` |

E 与 B 的 `successful_conditional_simulated_service_time_s` 点估计差值为 `+0.3975167283 s`；`timeout_penalized_simulated_cost_s` 点估计差值为 `-4.0504074 s`。协议没有为这两个差值计算配对 CI，因此不对其作显著性声明。

补充 total-system comparison 中，E 相对 A 的 TSR 点估计差值为 `+0.4548`，但 E 与 A 的 inference mask 不同，这个差值不能归因于 PPO。

## 7. 消融点估计

以下仅为聚合 TSR 点估计差值；除预注册 E−B 外，没有计算对应的配对差值 CI。

| 对比 | TSR 差值 | 解释边界 |
|---|---:|---|
| B−A | +38.96 pp | BC 下加入推理 Action Mask 的总系统变化 |
| C−A | +0.36 pp | 无 Mask 条件下 sparse action PPO 的点估计变化 |
| D−C | +44.56 pp | Sparse PPO 下加入推理 Action Mask 的总系统变化 |
| D−B | +5.96 pp | 相同 Mask、无 Progress 下 sparse action PPO 的点估计变化 |
| E−D | +0.56 pp | 相同 action PPO+Mask 下 Progress Reward 的点估计变化 |
| E−F | +0.94 pp | 动作级相对序列级信用分配的点估计变化 |

实验支持 E−B 的联合训练结论，但 E−D 只有 `+0.56 pp` 点估计，不能单独声称 Progress Reward 获得统计显著提升。

## 8. Family 聚合 TSR

每格为五 seed、每 family 500 条 trace 的 TSR 点估计；未计算 family-specific CI。

| Family | A | B | C | D | E | F |
|---|---:|---:|---:|---:|---:|---:|
| calendar_coordination | 4.0% | 45.2% | 2.8% | 54.4% | 55.2% | 55.6% |
| cloud_incident_response | 8.2% | 51.8% | 12.8% | 53.4% | 54.0% | 55.6% |
| customer_support | 8.2% | 42.2% | 5.4% | 54.0% | 54.8% | 51.4% |
| ecommerce_returns | 9.6% | 49.4% | 5.4% | 52.0% | 56.0% | 53.6% |
| expense_reimbursement | 6.4% | 46.2% | 6.2% | 53.6% | 51.8% | 54.6% |
| it_permissions | 6.2% | 41.0% | 8.0% | 49.8% | 48.8% | 46.8% |
| order_fulfillment | 10.8% | 48.6% | 10.6% | 57.6% | 55.0% | 53.6% |
| recruitment_interviews | 9.0% | 50.6% | 9.6% | 59.0% | 60.2% | 56.0% |
| subscription_changes | 9.8% | 47.6% | 13.8% | 47.6% | 49.4% | 47.8% |
| travel_booking | 1.8% | 41.0% | 3.0% | 41.8% | 43.6% | 44.4% |

E 的 family 聚合点估计在全部 10 个 family 上均高于 B；该描述不替代整体配对 bootstrap 结论。

## 9. 全局 ActionMask 证据

以下指标来自冻结的 `action-validity-v2`，只对规则型 ActionMask 全局计算一次，不属于 A–F 任一策略变体。

| 指标 | 点估计 | 95% case-cluster CI |
|---|---:|---:|
| Balanced Accuracy | 97.5000% | [97.3450%, 97.6701%] |
| Valid recall | 100.0000% | [100.0000%, 100.0000%] |
| Invalid recall | 95.0000% | [94.6900%, 95.3403%] |
| Macro-F1 | 97.4984% | [97.3431%, 97.6689%] |

混淆矩阵以 valid 为正类：

| TP | FN | TN | FP |
|---:|---:|---:|---:|
| 10,000 | 0 | 9,500 | 500 |

Invalid-kind recall：

| Kind | Recall | 95% CI |
|---|---:|---:|
| grounding | 100% | [100%, 100%] |
| precondition | 100% | [100%, 100%] |
| safety | 80% | [79.0202%, 81.1038%] |
| schema | 100% | [100%, 100%] |

该结果必须准确表述为“规则型 ActionMask 在 action-validity-v2 合成约束集上的 Balanced Accuracy”。500 个 false positive 体现公开观察无法识别全部隐藏状态冲突；它不是策略智能、真实业务规则发现率或全状态安全保证。

## 10. 复现

自包含证据目录：

```text
artifacts/runs/full/886622c5cb6cffc3/
```

发布资产为 `agentic-tool-rl-benchmark-v1.3-886622c5cb6cffc3.tar.zst`（约 46 MiB），同时附带归档 SHA256 与 195 个内部文件的逐文件 SHA256 清单。归档固定路径顺序、mtime、uid/gid 与权限；解压后的 manifest 仍使用相对路径，可从任意目录验证。

主要入口：

```text
run-manifest.json
claim-check.json
case-id-manifest.json
benchmark/action_validity.metrics.json
<variant>/seed-<seed>/training.json
<variant>/seed-<seed>/traces.jsonl
<variant>/seed-<seed>/metrics.json
<variant>/seed-<seed>/recompute.json
<variant>/seed-<seed>/run-integrity.json
```

完整校验：

```bash
uv run --locked python scripts/verify_release.py --release v0.1.0
```

Release verifier 先校验固定 annotated tag object、peeled commit、source/runtime/toolchain、归档和逐文件摘要，再在固定 Darwin/arm64 平台、v0.1.0 tag 与 CPython 3.11.15 / Torch 2.13.0 / uv 0.11.29 环境中调用原始 verifier；其他平台会在下载证据前 fail-closed。内层 verifier 会重新生成 benchmark 与候选、重放事务环境，并使用每个运行声明的 checkpoint 确定性重跑全部题目，逐字段核对策略轨迹、训练元数据、输入哈希、指标和 canonical claim。演进后的 `main` source fingerprint 与本次发布不同，不能直接替代 v0.1.0 verifier。

## 11. 限制

- 结果来自 10 类合成事务 DAG，不代表真实网页或企业系统成功率；
- 五 seed 和 bootstrap CI 量化不确定性，但不消除 synthetic-to-real gap；
- Action Mask 使用公开规则，不能识别全部隐藏状态约束；
- 所有 service-time/cost 均为冻结配置下的模拟工具成本；
- 轻量 PyTorch 核心结果不能外推为 Qwen/LLM、LoRA 或 GPU trainer 结果；
- E−B 支持的是 PPO+Progress 联合训练，不能拆解为 Progress Reward 的单独显著效果。
