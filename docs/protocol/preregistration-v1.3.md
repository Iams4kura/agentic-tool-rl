# benchmark-v1.3.0 预注册协议

状态：**协议、final seed 与一次性 canonical run 均已锁定；结果已按预注册规则发布。**

本协议的目的不是让实验命中某组简历数字，而是在读取 final holdout 之前固定研究问题、数据边界、比较方法和发布规则。`benchmark-v1.2.0` 已因 BC ceiling 被消费并降级为开发证据，详见 [pilot 停止记录](../results/pilot-v1.2.md)。

## 1. 研究问题

主要问题：在使用完全相同的公开 Action Mask 推理约束时，动作级 PPO + Progress Reward 是否相对行为克隆提高长链路任务成功率？

- Treatment：`E-PPO-Progress-Mask`
- Matched baseline：`B-BC-Mask`
- Primary metric：TSR
- Primary hypothesis：配对 TSR 差值的 95% bootstrap CI 下界大于 0
- Total-system secondary comparison：`E-PPO-Progress-Mask` 对 `A-BC-Unmasked`

Action Mask 的 20k 动作样本 BAcc 是独立约束分类指标，不归因于 PPO，也不是 primary outcome。

## 2. 固定矩阵

训练种子固定为 `17, 29, 43, 71, 101`。六个变体及其算法、Progress Reward、Action Mask 开关以 [`configs/ablation.yaml`](../../configs/ablation.yaml) 为唯一机器可读定义：

1. `A-BC-Unmasked`
2. `B-BC-Mask`
3. `C-PPO-Sparse-Unmasked`
4. `D-PPO-Sparse-Mask`
5. `E-PPO-Progress-Mask`
6. `F-Sequence-PPO-Progress-Mask`

每个 seed 的六个变体共享同一 BC checkpoint、FeatureEncoder 和 Progress Estimator 初始化。任何 PPO 变体必须产生正的 update 数和非零参数 L2 差异。

## 3. 数据与 holdout

- Train：2000 个任务，仅用于 BC、Progress Estimator 训练和 PPO rollout；
- Dev：200 个任务，仅用于 final 前的 floor/ceiling 与实现检查；
- Final test：1000 个任务，仅在协议和 seed 锁定后生成一次；
- Action validity：final test 每题 5 个状态、每状态 4 个候选，共 20k 动作样本，正负各半；
- family × topology：每个 family 的 test topology 不出现在该 family 的 train/dev 中；
- operation ID：task-local opaque ID，不含步骤序号；
- candidate order：只由公开 schema/visible state 派生的局部确定性 seed 打乱。

## 4. Final seed 的一次性派生

在所有 `src/`、`configs/`、依赖锁文件、测试和本协议完成后，创建第一个本地预注册 commit，记为 `P`。然后只执行以下确定性规则：

```text
final_seed = int(SHA256(UTF8(P))[0:8], 16)
```

把 `configs/cpu_full.yaml` 的 `benchmark.seed` 更新为 `final_seed`，创建 seed-lock commit，并在本文件补记 `P` 与派生值。不得为了观察结果而重做、挑选或更换 `P`。

seed-lock 后，在 final run 完成前禁止修改 `src/**/*.py`、`configs/*.yaml`、`pyproject.toml` 或 `uv.lock`。若发现必须修复的实现问题，则本协议作废、版本升至 v1.4，并使用新的从未生成过的 holdout。

## 5. 指标与统计

每个策略运行必须从逐题 trace 计算，并通过同一 `compute_metrics` 定义从原始 trace 做一致性复算：

- TSR 与 macro TSR；
- invalid action rate 与 forbidden side-effect rate；
- mean steps 与 step efficiency；
- `successful_conditional_simulated_service_time_s`：仅成功题、未截断的模拟工具执行时间均值；
- `timeout_penalized_simulated_cost_s`：失败题按 80 秒计费、成功题最多截断为 80 秒的成本指标。

Action Mask 分类指标是独立的 benchmark 级全局证据：对冻结 `action-validity-v2` 只计算一次 BAcc、valid/invalid recall、macro-F1、混淆矩阵与分层 recall，并写入 `artifacts/benchmark-v1/action_validity.metrics.json`。这些字段不进入任何 A–F 策略运行的 `metrics.json`；canonical report 仅引用其中的全局 BAcc。

> 以上路径记录 v1.3 final 的历史执行协议。当前 main 将同一派生指标直接写入 run bundle 的 `benchmark/action_validity.metrics.json`，避免后续运行改写共享的冻结 benchmark 目录；历史 tag 与已发布证据不变。

Primary E-vs-B TSR 差值严格按相同 `(seed, case_id)` 配对，并同时对 seed 和完整 case cluster 有放回重采样；使用 1000 次重复、95% percentile CI 和固定 bootstrap seed。

这里的两个时间指标都不是 wall-clock、真实网络延迟或线上 SLO；延迟档位来自冻结的 Azure Functions 2019 trace 派生配置。

## 6. 发布与停止规则

1. `ablate` 完成 6 × 5 × 1000 个评测单元并通过 `verify-run`，即构成有效 canonical experiment；
2. `canonical=true` 只表示数据、覆盖、哈希、配对和复算结构成立；
3. `hypothesis_passed` 只表示 primary CI 下界是否大于 0；它为 false 不使实验无效；
4. 无论结果正、负或不显著，都发布实际均值、每 seed 结果、CI 和限制；
5. 不根据 final test 修改生成器、模型、超参数、门槛或措辞后再次测试；
6. 简历只能引用 canonical report 实际支持的数字，并明确“合成 benchmark / 模拟 service time”口径。

## 7. 锁定记录

| 字段 | 值 |
|---|---|
| Preregistration commit `P` | `356c8b55b0f088dc613ecf00a8d832ca311076ff` |
| `SHA256(UTF8(P))` | `779984e70092cc1fbf48dd6f67c1c064dabba5b75b963795f3a747582d880cdf` |
| Derived final seed | `int(0x779984e7) = 2006549735` |
| Seed-lock commit | `46cd948b39981f8a9ef6a8283fc7f2ece2d3ce40` |
| Canonical run ID | `886622c5cb6cffc3` |
| Final report | [benchmark-v1.3 canonical final report](../results/canonical-v1.3.md) |
