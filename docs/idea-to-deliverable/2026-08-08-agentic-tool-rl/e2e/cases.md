# Agentic Tool RL 端到端验收案例

## 1. 范围

- 轻量核心在 Python 3.11/PyTorch CPU 路径完成 benchmark-v1.3 生成、BC、Progress、动作/序列 PPO、评测、恢复、trace 重放和统计报告。
- 主 CI 运行 B-BC-Mask / E-PPO-Progress-Mask 的 32/12/32、单 seed smoke。
- 完整 canonical 运行覆盖 6 variants × 5 seeds × 1000 test cases 和 action-validity-v2 20k 动作集。
- Qwen Adapter/LoRA/FSDP 只在独立手动 workflow 做契约与 dry-run，不下载权重、不声称 GPU PPO 已完成。
- canonical 只表示结构证据有效；预注册 E vs B 假设失败仍须作为有效实验保留。

## 2. 非目标

- 不连接真实支付、邮件、网页或其他外部副作用系统。
- 不把模拟工具 service execution time 表述为 Agent wall-clock 或线上端到端时延。
- 不把规则型 Mask BAcc 称为策略智能或真实安全率。
- 不按测试结果修改 holdout、变体定义或统计假设。
- 不把 v1.2 pilot 当作 v1.3 final；归档见 [pilot-v1.2](../../../results/pilot-v1.2.md)。

## 3. 案例索引

| Case | 验收目标 | 关键证据 |
|---|---|---|
| E2E-001 | 确定性生成 v1.3 与 held-out | manifests、双次生成 SHA、family×topology 检查 |
| E2E-002 | 真实长链路 ToolCall 执行 | full trace、事务终态 |
| E2E-003 | 独立 Mask、环境真值与部分可观测反例 | action-validity-v2、状态不变断言 |
| E2E-004 | 四拓扑、替代合法顺序与合法无用动作 | oracle replay、query/replay traces |
| E2E-005 | 动作级/序列级 PPO 数据链路 | rollout、GAE/sequence batch、L2 delta |
| E2E-006 | Progress Estimator 无泄漏 | 校准指标、冻结参数、奖励分解 |
| E2E-007 | 六变体五 seed 完整矩阵 | run manifest、30 checkpoints、30k units |
| E2E-008 | 从原始 trace 一致性复算 | recompute.json、metrics diff |
| E2E-009 | Canonical statistical report 与反例验真 | paired CI、canonical=true/false-hypothesis fixture |
| E2E-010 | 中断恢复、输入与策略证据绑定 | Resume Guard、checkpoint 重执行、篡改负测 |
| E2E-011 | Qwen 两层边界 | fake runtime、fixed revision、GPU config dry-run |
| E2E-012 | 干净环境与主 CI | make ci、B/E smoke、远端 workflow |

## 4. 可执行案例

### E2E-001：benchmark-v1.3 确定性与 held-out

- 设置：空 artifacts，固定配置与 base seed。
- 操作：生成 train/dev/test 两次。
- 预期：test 为 1000 unique cases，10 family 各 100，short/medium/long 各族 30/40/30；四 topology 均出现；每个 family 的 test topology 不出现在该 family 的 train/dev；operation ID 为 task-seeded opaque op_<20 hex>；候选乱序可重现。
- 通过：两次 JSONL SHA256 相同，split 身份互斥，oracle 全部可解，manifest 统计一致。

### E2E-002：真实长链路执行

- 设置：选择 12–15 步冻结 case 与实际 checkpoint。
- 操作：通过 policy → candidate builder → ToolRegistry 执行，不直接写终态。
- 预期：每步记录 public observation、candidate hash/mask、action、value、奖励、done 和事务结果；终态满足隐藏 goal，无禁止副作用。
- 通过：verify-run 可逐步重放并得到相同终态与模拟 service time。

### E2E-003：Mask、环境与隐藏 ledger

- 设置：standard Schema/Grounding/Precondition/Safety negatives，以及 hidden_ledger_collision。
- 操作：由独立规则 ActionMask 预测，再绕过 mask 交给环境 dry-run/execute。
- 预期：standard 四类由 Mask 与环境一致拒绝，拒绝前后业务状态相同；hidden ledger 候选因公开输入不可区分而被 Mask 放行，但环境依隐藏 ledger 拒绝。
- 通过：action-validity-v2 恰为 19,500 standard + 500 hidden、valid/invalid 各 10k；recall/BAcc/macro-F1 只写入全局 action_validity.metrics.json，报告准确命名为规则型 Mask 在合成约束集上的结果，不进入任一策略变体 metrics。

### E2E-004：拓扑、替代顺序与合法无用动作

- 设置：四种 topology 各选 case。
- 操作：分别回放两条不同合法 oracle plan；插入 contextual read-only query 与合法 idempotent replay；另尝试有害 shortcut。
- 预期：两条合法顺序均成功；query/replay 被 Mask 和环境接受但不推进；shortcut 被环境拒绝且无副作用。
- 通过：终态判分不依赖 gold trajectory 文本或 candidate 固定位置。

### E2E-005：PPO 信用分配

- 设置：C/D/E/F 从同 seed 的共享 BC checkpoint 开始。
- 操作：采样完整轨迹，执行 action PPO 或 sequence PPO 更新。
- 预期：动作级 batch 的 trajectory/step/log-prob/value/reward/done 对齐，GAE 不跨 episode；序列级 batch 每 episode 一个 ratio/return；PPO updates、rollout_steps 和 L2 delta 均为正。
- 通过：checkpoint metadata 与 training.json 一致，verifier 重新计算实际参数 delta。

### E2E-006：Progress Estimator

- 设置：只使用 train/dev noisy continuation。
- 操作：训练 Phi(s)=P(success | visible state,user_goal)，冻结后计算 beta*(gamma*Phi(next)-Phi(current))。
- 预期：输入包含去标识 user_goal，不含 test/oracle/隐藏 goal；输出 BCE、Brier、ECE、AUROC；task/progress/step/invalid/forbidden 分量分开。
- 通过：冻结、无泄漏和 telescoping 单测通过。

### E2E-007：六变体完整矩阵

- 设置：A-BC-Unmasked、B-BC-Mask、C-PPO-Sparse-Unmasked、D-PPO-Sparse-Mask、E-PPO-Progress-Mask、F-Sequence-PPO-Progress-Mask；seed 17/29/43/71/101。
- 操作：对同一冻结 1000 题运行全部变体。
- 预期：主比较 E vs B 使用同 BC 起点、Action Mask、seed/case 和推理环境；保存 30 runs、30,000 evaluation units、checkpoint、trace、metrics、recompute。
- 通过：每个变体只报告 TSR、实际非法率、安全、步骤效率、successful_conditional_simulated_service_time_s、timeout_penalized_simulated_cost_s 和相应 CI；规则型 Mask BAcc 仅来自全局 action_validity.metrics.json 并由 canonical claim 引用；不填写尚未运行出的数值。

### E2E-008：Trace 一致性复算

- 设置：给定 traces.jsonl 与 metrics.json。
- 操作：运行 recompute。
- 预期：从原始 trace 重新调用同一 compute_metrics 定义，输出与发布 metrics 的逐字段差异。
- 通过：差异 <=1e-9。该案例证明 trace/metrics 一致，不宣称存在第二套独立指标实现。

### E2E-009：Canonical statistical report 反例验真

- 设置：一套结构完整且 E−B paired TSR CI lower > 0 的 fixture；另一套结构同样完整但 E 与 B outcomes 相同或 CI lower <= 0 的 fixture；另准备 case/seed 缺失 fixture。
- 操作：生成 schema_version=3.0 的 claim-check.json，并通过 run verifier 验证 A–F 完整 outcomes、重建 paired seed×case bootstrap。
- 预期：
  - 正例：canonical=true、hypothesis_passed=true；
  - 零/负结果：canonical=true、hypothesis_passed=false，ablate 仍正常完成并保留报告；
  - 结构缺失：canonical=false，CLI/verifier 拒绝。
- 通过：报告只检验预注册 E vs B CI 假设，不含固定 TSR、BAcc 或 service-time 目标；反例结果不会被伪造成成功，也不会被当作无效实验删除。

### E2E-010：恢复、输入绑定与策略 trace 绑定

- 设置：中断部分评测，记录已有完整 JSONL 行和 checksum；另准备一条 oracle 成功轨迹及 log-prob/value/reward 数值篡改。
- 操作：相同输入恢复；分别改变 benchmark/config/source/checkpoint；对每种 trace 攻击同步刷新 trace hash、metrics、recompute 与 run manifest 中所有自报派生字段；再搬迁并重命名完整 run bundle，分别注入绝对路径、`..` 与逃逸符号链接。
- 预期：同输入只补缺失 cases；不同输入均被 Resume Guard 拒绝；声明 checkpoint 会确定性重跑每个 case 并递归比对完整策略证据；合法搬迁不依赖 cwd，路径逃逸全部 fail closed。
- 通过：最终 case 集精确、无重复，旧完整证据不变；搬迁后的 bundle 仍通过 verify-run；oracle 路径替换与三类数值攻击均以 checkpoint policy trace mismatch 被拒绝，三类路径攻击均被拒绝。

### E2E-011：Qwen 扩展层

- 设置：默认轻量安装与 fixed-revision Qwen GPU config。
- 操作：用 fake runtime 检查 observation/candidate 编码、JSON ToolCall grounding、finite log-prob/value；运行 make qwen-check。
- 预期：核心导入不依赖 transformers；dry-run 不下载权重；revision/LoRA/FSDP schema 可解析。
- 通过：明确 Qwen 不是 drop-in GPU trainer，且没有把 dry-run 写成性能结果。

### E2E-012：全新环境与 CI

- 设置：干净 checkout。
- 操作：按 README 运行 make sync、make ci；观察主 GitHub Actions。
- 预期：lint、mypy、pytest、B/E smoke 和 verify-run 成功；主 CI 不执行 Qwen；完整矩阵仅由 make benchmark 或手动 Full benchmark workflow 触发。
- 通过：公开命令可复制，链接有效，smoke 不生成 canonical 报告。

## 5. 总体验收

- E2E-001～E2E-012 必须逐项有强证据。
- 工程正确、canonical 结构和假设结果是三个不同状态。
- final 假设无论支持与否都必须如实发布。
- v1.3 final 尚未产生前，文档不得放推测跑分；现已完成的唯一正式数字入口为 [canonical final report](../../../results/canonical-v1.3.md)。
