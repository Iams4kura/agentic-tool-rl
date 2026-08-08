# benchmark-v1.3.0 开发集检查

状态：**train/dev 调试证据，不是 final holdout 结果。**

本页记录在预注册 commit 之前进行的唯一 PPO 样本预算选择。数据为 `base_seed=20260808` 的 200 题 dev split；没有读取 v1.3 final seed 或 final test。目标是排除 floor/ceiling 并选择覆盖更多训练任务的样本预算，不对齐任何简历目标值。

## 1. 当前信息充分性修复

FeatureEncoder v4 只加入一个公开关系特征：候选 `entity_id` 是否属于 observation 的 `available_entities`。它不编码实体原值，但避免让无 Mask 基线面对无法从输入区分的 grounding 正负例。修改后定向测试证明原始实体 ID 仍不进入 standalone action feature。

## 2. 四 iteration 开发结果

| Seed | Variant | TSR | Invalid action rate | Mean steps | Successful conditional simulated service time | Timeout-penalized simulated cost |
|---:|---|---:|---:|---:|---:|---:|
| 17 | A-BC-Unmasked | 0.085 | 0.6527 | 13.720 | 11.7842 s | 74.2017 s |
| 17 | B-BC-Mask | 0.465 | 0.0000 | 12.325 | 14.7046 s | 49.6376 s |
| 17 | D-PPO-Sparse-Mask | 0.525 | 0.0000 | 12.010 | 14.7212 s | 45.7286 s |
| 17 | E-PPO-Progress-Mask | 0.525 | 0.0000 | 12.100 | 15.0742 s | 45.9139 s |
| 29 | B-BC-Mask | 0.410 | 0.0000 | 12.440 | 14.3684 s | 53.0910 s |
| 29 | E-PPO-Progress-Mask | 0.425 | 0.0000 | 12.405 | 14.3966 s | 52.1185 s |

两 seed 的 matched baseline 平均 TSR 为 `0.4375`，E 平均 TSR 为 `0.4750`，开发差值为 `+0.0375`。seed 17 上 Sparse 与 Progress 版本相同，因此开发证据不支持单独声称 Progress Reward 有增益。

## 3. 八 iteration 样本预算检查

原配置为 `4 × 128 = 512` 个 rollout episodes，只覆盖 2000 个训练任务的约四分之一。唯一候选修改是把 iteration 加倍到 8，其余超参数保持不变：

| Seed | E-PPO-Progress-Mask TSR | Mean steps | Successful conditional simulated service time | Timeout-penalized simulated cost | Rollout steps |
|---:|---:|---:|---:|---:|---:|
| 17 | 0.500 | 12.200 | 14.7109 s | 47.3554 s | 12,626 |
| 29 | 0.515 | 12.065 | 14.4366 s | 46.2348 s | 12,936 |

八 iteration 下 E 的两 seed 平均 TSR 为 `0.5075`，相对同一 B 基线平均差值为 `+0.0700`。因此 final 配置固定为 8 iterations；这是基于 dev 平均和训练任务覆盖率作出的样本预算选择，不是基于 test 或目标分数。

## 4. 本地 trace 摘要

原始开发 trace 位于临时目录，不随仓库发布；以下 SHA256 用于证明本页不是事后改写：

| Trace | SHA256 |
|---|---|
| seed17 / 4-it / A | `379e7835437a2911f42fdba46f3986e66f8afe656874e2d2d3e5516f32392c4f` |
| seed17 / 4-it / B | `be8698e3ec5f5a1dbeef0fe98b1f8c59e93973e8f8f0e6b84962bae5c2a0bec9` |
| seed17 / 4-it / D | `f7789d49dc89ba8ce44c45f4d0e2f618ee6783b0448da9f11687e4da0e509cfa` |
| seed17 / 4-it / E | `1d1f2b81df5ea6b09f57fb0e4bf4130b5f5ab4bf1c5fe86d3f28a432bfd55049` |
| seed29 / 4-it / B | `5ac5c589f6e33fb233e0712a685e1119ffc7f941cbd193ec0d751318fd09c254` |
| seed29 / 4-it / E | `ec1b0b74eaa2f854a72869abbe08822b9773d74dcd8b4590795746aed6cfa2dd` |
| seed17 / 8-it / E | `6d76c234da663baae0420a16076b0bef591f4c0023f3b2de36122dc549e9291b` |
| seed29 / 8-it / E | `833fba6e9c3ba76e06ab3578caa90453f110fd1d9a82ea7eb16fa9e56a5a7939` |

这些开发值不得写进简历或 README 结果表。最终结论只来自 seed-lock 后的一次性 1000 题 holdout。
