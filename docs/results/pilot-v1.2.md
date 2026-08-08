# benchmark-v1.2.0 pilot：停止结论

状态：**开发期失败实验，不是 canonical 结果，不用于性能声明。**

2026-08-08 启动完整矩阵后，第一个 `A-BC / seed-17` 已完成 1000 题评测。结果出现明显 ceiling，因此在后续变体完成前主动停止，避免继续消耗算力或用已观察测试集反向调参。

## 已观察结果

| 字段 | 值 |
|---|---:|
| Benchmark | `benchmark-v1.2.0` |
| Base seed | `20260808` |
| Run ID | `8d0383e89670dc0d` |
| Variant / seed | `A-BC / 17` |
| Cases | 1000 |
| TSR | 1.0000 |
| Macro TSR | 1.0000 |
| Mean steps | 10.1200 |
| Timeout-penalized simulated service cost（旧字段名 `simulated_latency_s`） | 16.808795 s |
| Action Mask BAcc（独立固定动作集；不归因于 BC） | 0.9750 |

独立 trace 复算为 `matches=true`。本地原始证据未提交到仓库；其完整性摘要如下：

| 文件 | SHA256 |
|---|---|
| `metrics.json` | `cd37bbc3aa3f6896e9d51d5a6a0b1cea0977057c676b1a9c8675b211cf33be88` |
| `traces.jsonl` | `bcc951c5ff9edd09440c335b0a62adb5c616e8dd53940aa725ed2ce793a4d60c` |
| `checkpoint.pt` | `4173fdd80367d9c4e03bc3e75c775f3ad8a446c30d7a78baff3e91b58cbf1282` |

## 为什么停止

BC 已达到 100% TSR，无法回答“动作级 PPO 是否在同一推理约束下提升长链路成功率”。同时旧 `simulated_latency_s` 把失败题按 80 秒 timeout 计入，实质是 timeout-penalized cost，而不是成功任务的平均完成耗时。

根因审计发现以下捷径：

- operation ID 直接暴露 `step-XX` 序号；
- DAG 拓扑几乎固定；
- 候选顺序固定；
- 原主比较把 PPO、Progress Reward 和推理时 Action Mask 同时加入，无法隔离 RL 增益。

## 后续规则

1. v1.2 test 从此降级为已消费的开发数据，禁止作为最终 holdout；
2. v1.3 在生成 final holdout 前移除序号/顺序捷径，增加多拓扑和 `BC+Mask` 公平对照；
3. 超参数和难度只在 train/dev 检查 floor/ceiling，不按 `85%/90%` 目标调参；
4. 协议先形成 Git commit，再由 commit 哈希确定一次性 holdout seed；
5. final 结果无论正负都按实测发布，不修改测试集追逐预设数字。
