# Agentic Tool RL

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

面向长链路结构化工具调用 Agent 的可复现强化学习框架。一次结构化 ToolCall 对应一个 RL Step；轻量 PyTorch 核心实现行为克隆（BC）、动作级 PPO、Progress Estimator 势函数奖励、Action Mask、序列级 PPO 对照，以及从逐题 trace 到统计报告的完整证据链。

仓库采用两层方案：

- **轻量核心层**：在 CPU/Apple Silicon 上负责默认 CI 和完整实验验收。
- **可选 Qwen 层**：提供固定 revision 的 Qwen Adapter、LoRA/FSDP 配置和 fake-runtime 契约；独立手动 workflow 只做接口检查，不下载权重，也不是已完成的 GPU PPO trainer。

> 本仓库没有预设目标分数。canonical 结果是一个结构完整、可重算的统计报告；预注册假设未获支持仍然是有效实验，不会被改写为“通过”。

## 设计与证据

| 能力 | 当前实现 | 可验证证据 |
|---|---|---|
| 动作级决策 | 动态候选 Actor-Critic；每个工具调用一个 Step | observation、candidate hash、mask、action、log-prob、value、reward、done |
| 动作级 PPO | 轨迹边界安全 GAE、clipped policy/value objective、entropy regularization | update 数、loss/KL/clip fraction、参数摘要与 L2 delta |
| Progress Reward | Phi(s)=P(success given visible state and user_goal) 的势函数差分 | 校准指标、冻结参数、逐步奖励分解 |
| Action Mask | 只读取公开 ToolSchema 与 Observation | 独立规则实现、冻结 action-validity-v2、全局 20k 分类证据 |
| 事务环境 | 验证通过后原子提交；拒绝动作不产生部分写入 | dry-run、状态快照、审计日志、禁止副作用计数 |
| Benchmark v1.3 | 四种 DAG、opaque operation ID、确定性候选乱序、family×topology held-out | 可确定性重生成的 JSONL、oracle 回放和 SHA256 |
| 统计报告 | 六变体、五种子、1000 题；主比较 E vs B | 精确配对的 seed/case outcomes 与 1000 次 bootstrap 95% CI |
| 完整性 | 数据重生成、环境重放、checkpoint 策略重执行、BC/PPO/参数/恢复校验 | verify-run、manifest、checkpoint、recompute、Resume Guard |

~~~mermaid
flowchart LR
    O["Public Observation + user_goal"] --> C["Dynamic candidates"]
    C --> M["Public-rule ActionMask"]
    O --> P["Actor-Critic"]
    M --> P
    P --> R["Transactional ToolRegistry"]
    R --> N["Next observation"]
    O --> E["Frozen Progress Estimator"]
    N --> E
    E --> W["Potential reward"]
    R --> W
    W --> B["Trajectory-safe buffer"]
    B --> A["Action PPO / Sequence PPO"]
    A --> P
~~~

详见 [架构说明](docs/architecture.md)、[Benchmark 说明](docs/benchmark.md) 和 [方法与复现](docs/methodology.md)。

## 快速开始

需要 Git；项目固定 CPython 3.11.15 与 [uv 0.11.29](https://docs.astral.sh/uv/)，`uv.toml` 会拒绝漂移的 uv 版本。

~~~bash
git clone https://github.com/Iams4kura/agentic-tool-rl.git
cd agentic-tool-rl
make sync
make ci
~~~

**make ci** 与主 GitHub Actions 一致：先检查 lockfile，再运行 Ruff、strict mypy、pytest、轻量 smoke，最后两次隔离构建 wheel/sdist 并校验逐字节可复现性、归档路径和敏感文件。smoke 使用 32/12/32 的 train/dev/test 和一个 seed，只比较：

- **B-BC-Mask**
- **E-PPO-Progress-Mask**

它覆盖生成 → BC/Progress → PPO → 评测 → trace 一致性复算 → run verifier，但 claim_mode=none，不生成 canonical 统计报告。Qwen 检查不属于主 CI。

## 六变体公平矩阵

完整配置位于 configs/cpu_full.yaml，矩阵位于 configs/ablation.yaml。

| 变体 | 算法 | Progress | 推理 Action Mask | 对照意义 |
|---|---|---:|---:|---|
| **A-BC-Unmasked** | BC | 否 | 否 | 无 RL 的总系统基线 |
| **B-BC-Mask** | BC | 否 | 是 | 主比较的 matched baseline |
| **C-PPO-Sparse-Unmasked** | Action PPO | 否 | 否 | 稀疏奖励、无 mask |
| **D-PPO-Sparse-Mask** | Action PPO | 否 | 是 | 在相同 mask 下隔离 sparse PPO |
| **E-PPO-Progress-Mask** | Action PPO | 是 | 是 | 预注册 treatment |
| **F-Sequence-PPO-Progress-Mask** | Sequence PPO | 是 | 是 | 只改变信用分配粒度 |

主比较是 **E vs B**：同一 seed 使用同一共享 BC checkpoint，并在相同冻结 test cases、候选生成器和推理 Action Mask 下评测；差异是 E 在 BC 之后接受动作级 PPO 与 Progress Reward 训练。A 用于补充说明包含 mask 在内的总系统变化，不替代 E vs B 的预注册结论。

~~~bash
make benchmark
~~~

该命令执行 6 variants × 5 seeds × 1000 test cases；完整配置使用 1000 次 bootstrap。也可手动触发 Full benchmark workflow。

## Canonical 统计报告

ablate 只在下列结构条件成立时生成 claim-check.json：

- 五个预注册且唯一的 seed；
- 六个冻结变体定义；
- 1000 个唯一 test cases；
- A–F 每个 seed 都覆盖完全相同的 case 集合和布尔成功结果；统计比较仍只汇总 A/B/E；
- action-validity-v2 恰为 20,000 条，valid/invalid 各 10,000；
- 95% 配对 seed×case bootstrap CI 可计算。

预注册假设是：

~~~text
paired TSR difference CI lower bound for E - B > 0
~~~

报告中的 canonical 表示证据结构有效；hypothesis_passed/passed 表示预注册假设是否获支持。后者为 false 时，ablate 仍正常完成并保留结果。仓库没有从两份汇总 metrics 生成 canonical 结论的旁路。

完成后强校验证据：

~~~bash
uv run --locked agentic-tool-rl verify-run \
  --manifest artifacts/runs/full/<run-id>/run-manifest.json
~~~

verify-run 会确定性重生成 benchmark 与动作集、逐步重放环境，并用声明的 checkpoint、FeatureEncoder 和 Progress Estimator 确定性重跑每个 case，逐字段核对动作、mask、log-prob、value、奖励和终态；同时校验 training metadata、BC 无漂移、PPO 有真实更新与非零 L2 delta、Resume Guard 四类哈希、case 精确覆盖、指标一致性复算和 canonical 报告复算。因此仅替换 trace、metrics 与哈希不能伪造更高结果。recompute 与 verifier 都从原始 trace 重新计算，但复用同一 compute_metrics 定义，不声称存在第二套独立指标实现。

每个 `<run-id>/` 是路径自包含的证据包：内部 `benchmark/` 保存该次运行使用的完整冻结快照，`run-manifest.json` 的所有 artifact 路径都相对 manifest 所在目录。整个目录下载、搬迁或重命名后仍可验证，不依赖原始工作目录；绝对路径、`..` 和逃逸 bundle 的符号链接都会被拒绝。验证器同时强制 source/runtime identity，因此历史证据必须配合产生它的 tag 与精确运行时，不能直接用演进后的 `main` 绕过身份检查。

已发布的 v0.1.0 canonical 证据使用固定 Release 入口验证：

~~~bash
uv run --locked python scripts/verify_release.py --release v0.1.0
~~~

该命令额外要求 Darwin/arm64（发布证据的固定重放平台）且系统 `PATH` 中可用 `zstd`；其他平台会在下载证据前 fail-closed。它只接受仓库内 allowlist 的 Release：校验 annotated tag object、peeled commit、source fingerprint、平台、CPython/Torch/uv 版本、46 MiB 归档与 195 个文件摘要，然后用 v0.1.0 tag 中的 checkpoint-bound verifier 重放 30 runs / 30,000 evaluation units。它没有任意 ref、URL 或 `ignore/skip` 参数。

仅检查 Python 分发包可复现性与归档安全时运行：

~~~bash
make package-check
~~~

## 指标口径

每个策略 run 的 metrics.json 只包含策略执行结果：TSR（含 family macro 聚合）、invalid_action_rate、forbidden_side_effect_rate、steps_efficiency/mean_steps，以及下面两种模拟 service-time/cost。规则型 ActionMask 的 recall、BAcc、macro-F1 和混淆矩阵不属于任何策略变体；它们对冻结 action-validity-v2 全局计算一次，写入 artifacts/benchmark-v1/action_validity.metrics.json，并快照到 run bundle 的 benchmark/action_validity.metrics.json，由 canonical report 引用其中的全局 BAcc 证据。

发布的 service-time 指标只有两个：

- **successful_conditional_simulated_service_time_s**：只对成功任务计算，取未截断的模拟工具 service execution time 均值；没有成功样本时为 null。
- **timeout_penalized_simulated_cost_s**：成功任务计 min(service_time, timeout)，失败任务计 timeout，再对全部任务取均值。

二者都不对应 Agent wall-clock latency、网络 RTT 或线上 SLO。前者是成功条件下的模拟服务时间，后者是带失败惩罚的模拟成本。

工具成本来自 Microsoft Azure Functions 2019 公开 trace 的调用加权分位数。来源、CC BY 4.0、archive SHA256 和冻结值见 [延迟档案](docs/data/azure-functions-2019-latency-profile.json)；复现推导可运行：

~~~bash
uv run python scripts/derive_azure_latency_profile.py \
  /path/to/azurefunctions_dataset2019_azurefunctions-dataset2019.tar.xz
~~~

脚本只使用标准库，不主动下载数据，并默认校验官方 archive SHA256。

## Benchmark v1.3 摘要

- 10 个事务型业务族，任务长度 6–15 步；
- diamond_tail、fanout_gate、dual_lane、dual_root_mesh 四种拓扑；
- 每个 family 的一个 topology 只出现在 test，train/dev 仅使用该 family 的另外三种；
- operation ID 为 task-seeded opaque op_<20 hex>，不暴露 step 序号；
- 候选顺序由公开 schema/state 确定性乱序，不提供固定位置捷径；
- 候选包含合法 contextual read-only query 和合法 idempotent replay；mask 不能代替策略学习相关性；
- canonical test 为 1000 题，每族 100，short/medium/long 为 30/40/30；
- action-validity-v2 为 19,500 条 standard + 500 条 hidden-ledger collision，正负各半。

规则型 ActionMask 在这个合成约束集上的 BAcc 必须准确命名，不能外推为策略智能或真实业务安全能力。

v1.2 暴露顺序与拓扑捷径并出现 ceiling，已停止且不作为最终结果；审计与停止理由见 [benchmark-v1.2 pilot 归档](docs/results/pilot-v1.2.md)。

v1.3 final 已完成：在 5 seeds × 1000 frozen test cases 上，E-PPO-Progress-Mask 相对相同 Action Mask 与共享 BC 起点的 B-BC-Mask，将 TSR 从 46.36% 提升至 52.88%，绝对提升 6.52 个百分点；配对 seed×case bootstrap 95% CI 为 [3.4195, 9.7820] 个百分点，`canonical=true`、`hypothesis_passed=true`。完整六变体结果、ActionMask 全局证据与限制见 [canonical report](docs/results/canonical-v1.3.md)。

## 常用 CLI

| 命令 | 用途 |
|---|---|
| **generate** | 生成 splits、oracle 证据、动作标签与 manifest |
| **train** | 训练并评测一个变体与一个 seed |
| **smoke** | 运行 B/E 的 CI 尺寸工程闭环 |
| **ablate** | 运行完整六变体、五种子并生成 canonical 统计报告 |
| **run-episode** | 使用显式 `--variant`/`--ablation` 展开匹配 checkpoint 的完整可审计轨迹 |
| **evaluate** | 使用显式变体语义在冻结 test split 评测，并用四项输入哈希保护断点续跑 |
| **recompute** | 使用同一指标定义从原始 trace 做一致性复算 |
| **verify-run** | 重生成、重放并强校验整个 run |
| **qwen-dry-run** | 校验可选 Qwen/LoRA/GPU 配置，不下载权重 |

单 checkpoint 命令不接受独立的 mask/reward 布尔开关；动作约束、进度奖励和
credit assignment 均来自冻结消融矩阵，并与 checkpoint 中的变体身份交叉校验：

~~~bash
uv run agentic-tool-rl run-episode \
  --checkpoint artifacts/runs/<run-id>/E-PPO-Progress-Mask/seed-17/checkpoint.pt \
  --variant E-PPO-Progress-Mask --ablation configs/ablation.yaml \
  --benchmark-dir artifacts/benchmark-v1 --output artifacts/demo-episode.json

uv run agentic-tool-rl evaluate \
  --checkpoint artifacts/runs/<run-id>/E-PPO-Progress-Mask/seed-17/checkpoint.pt \
  --variant E-PPO-Progress-Mask --ablation configs/ablation.yaml \
  --config configs/cpu_full.yaml --benchmark-dir artifacts/benchmark-v1 \
  --output artifacts/evaluation/e-seed-17
~~~

## 可选 Qwen Adapter

主 CI 不运行 Qwen。需要时手动执行 Optional Qwen adapter workflow，或：

~~~bash
make qwen-check
~~~

Qwen/Qwen3-4B 固定到 revision 1cfa9a7208912126459214e8b04321603b3df60c。当前层提供结构化调用编码/解析、候选 grounding、finite log-prob/value 契约、LoRA 和 FSDP 配置；它不是轻量 trainer 的 drop-in GPU 后端，也没有被当作已完成的 Qwen PPO 结果。

## 边界

- 合成 DAG 不等同于真实网页、企业系统或线上流量。
- Action Mask 是独立公开规则实现；500 个 hidden-ledger collision 明确展示隐藏状态不可观测边界，因此 BAcc 不是全状态安全保证。
- Azure 数据只用于冻结模拟 service execution time 成本，不代表完整 Agent 耗时。
- 轻量核心的结论不能外推为 Qwen/LLM 效果。
- final 结果无论支持或不支持假设都应按 artifacts 发布，不能因结果不理想改 test 或阈值。

端到端案例见 [E2E cases](docs/idea-to-deliverable/2026-08-08-agentic-tool-rl/e2e/cases.md)。

## License

[MIT](LICENSE)
