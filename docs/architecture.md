# 架构说明

本文描述 benchmark-v1.3.0 的数据、运行时、学习和证据边界。运行入口见 [README](../README.md)，数据协议见 [Benchmark 说明](benchmark.md)，统计协议见 [方法与复现](methodology.md)。

## 1. 不变量

1. 一次结构化 ToolCall 就是一个 RL Step，状态、候选、选择、环境结果与奖励逐步对齐。
2. 策略不能直接写业务状态；唯一写路径是事务型 ToolRegistry，拒绝动作不产生部分更新。
3. ActionMask 与环境 validator 是两套规则实现。Mask 只读取公开 ToolSchema 与 Observation，环境拥有完整事务状态。
4. 指标来自 trace；报告必须绑定数据、配置、源码、checkpoint 和精确 case 集。
5. canonical 描述证据结构，不描述结果好坏；预注册假设失败仍是有效实验。

## 2. 两层方案

| 层 | 职责 | 验收边界 |
|---|---|---|
| 轻量核心 | v1.3 数据生成、事务环境、动态候选策略、BC、Progress、动作/序列 PPO、评测、复算、统计报告与完整性校验 | 主 CI 运行 B/E smoke；完整 CPU 实验可运行 |
| Qwen 扩展 | prompt 编码、结构化 JSON 调用解析、候选 grounding、finite log-prob/value、独立 value head、LoRA/FSDP 配置 | 仅手动 workflow 做 fake-runtime 契约和 dry-run |

Qwen/Qwen3-4B 固定到 revision 1cfa9a7208912126459214e8b04321603b3df60c。Qwen 层不是轻量 trainer 的 drop-in GPU backend；当前没有真实 GPU rollout/optimizer，也不产生 Qwen 性能结论。

## 3. 依赖方向

~~~mermaid
flowchart TB
    subgraph D["Deterministic data"]
        G["v1.3 generator"]
        O["Two legal oracle plans"]
        V["action-validity-v2"]
    end
    subgraph R["Transactional runtime"]
        OBS["Public Observation"]
        CAN["Deterministically shuffled candidates"]
        MASK["Public-rule ActionMask"]
        REG["Environment validator + ToolRegistry"]
        ST["Visible + hidden state"]
    end
    subgraph L["Learning"]
        FE["Leakage-resistant FeatureEncoder"]
        AC["Dynamic Actor-Critic"]
        PE["Frozen Progress Estimator"]
        BUF["Trajectory buffer"]
        PPO["Action PPO"]
        SPPO["Sequence PPO"]
    end
    subgraph E["Evidence"]
        TR["Replayable JSONL traces"]
        MET["compute_metrics"]
        AMET["Global action_validity.metrics.json"]
        REC["Trace consistency recompute"]
        STAT["Paired seed×case report"]
        VER["verify-run"]
    end
    G --> O
    G --> V
    G --> ST
    ST --> OBS
    OBS --> CAN
    OBS --> MASK
    CAN --> MASK
    OBS --> FE
    CAN --> FE
    MASK --> FE
    FE --> AC
    AC --> REG
    REG --> ST
    OBS --> PE
    REG --> BUF
    PE --> BUF
    BUF --> PPO
    BUF --> SPPO
    PPO --> AC
    SPPO --> AC
    REG --> TR
    AC --> TR
    TR --> MET
    V --> AMET
    TR --> REC
    TR --> STAT
    MET --> STAT
    AMET --> STAT
    REC --> VER
    STAT --> VER
~~~

环境不会调用 ActionMask；评分路径不会用环境重写冻结动作标签；核心层导入时不会加载 Qwen 的可选依赖。

## 4. Benchmark v1.3 的抗捷径设计

### 4.1 四种拓扑

每题从以下 DAG 结构之一生成：

- diamond_tail
- fanout_gate
- dual_lane
- dual_root_mesh

审批 barrier 和最终动作附着在拓扑主体之后。每题保存两条不同的合法 topological plan，终态 evaluator 接受任何满足约束的合法顺序。

### 4.2 Family×topology held-out

每个业务 family 固定一个 held-out topology：test 只使用该组合，train/dev 只使用同一 family 的另外三种拓扑。因此泛化单元是 family×topology 组合，不只是换 entity ID。

### 4.3 Opaque operation ID 与候选乱序

operation ID 由 generator version、task seed 与操作标签哈希成 op_<20 hex>，不携带 step 序号。FeatureEncoder 只保留状态 completed-operation 与候选 operation 的匹配统计，不恢复 ordinal。

Candidate Builder 使用公开 schema 与 visible state 派生局部 seed，对每个状态的候选做确定性乱序。重跑字节稳定，但 candidate index 不是固定标签。

### 4.4 合法但不推进

候选除 ready workflow action 外，还包括：

- inspect_status；
- 根据 priority/region/channel 选择的 contextual read-only query；
- 已完成操作的合法 idempotent replay。

这些调用可被环境和 Mask 接受，却消耗模拟 service time 且不推进目标。ActionMask 只回答“是否可执行”，Actor-Critic 仍须学习“是否有用”。

## 5. 一个动作级 Step

1. 环境输出 public Observation：visible state、user_goal、可用实体/工具和 step budget。
2. Candidate Builder 从公开 schema/state 构造并乱序候选。
3. ActionMask 为候选给出规则型可执行性预测；unmasked 变体不用于选动作，但预测仍可记录。
4. FeatureEncoder 去除 task/entity/approval/call ID 实例身份，保留去标识 user_goal 与 opaque-operation 匹配特征。
5. Actor-Critic 对数量可变的候选逐个打分，输出 action index 与 state value。
6. ToolRegistry 独立 dry-run；通过后原子提交 version/effects/audit/idempotency ledger，拒绝则业务状态不变。
7. 记录 task、progress、step、invalid、forbidden 五类奖励分量。
8. compact trace 保存可重放 ToolCall、candidate hash、环境结果和终态；full episode 另保存展开 observation/candidates。

## 6. Action Mask 的可观测边界

ActionMask 构造时即使接收 WorkflowTask，也只深拷贝公开 ToolSchema。它不读取 workflow nodes、oracle plans、隐藏 safety token、forbidden labels、processed call IDs 或 idempotency ledger。

action-validity-v2 的 20,000 个样本由环境 dry-run 冻结标签：

- 19,500 standard candidates：公开 schema/observation 足以判定；
- 500 hidden_ledger_collision：公开输入看起来合法，只有环境隐藏 ledger 能发现 key 已绑定到另一操作。

因此报告的是**规则型 ActionMask 在合成约束集上的 BAcc**。recall、BAcc、macro-F1 与混淆矩阵只对冻结 action-validity-v2 全局计算一次并写入 run bundle 的 benchmark/action_validity.metrics.json，不回写共享 benchmark，也不进入任一策略变体的 metrics.json；canonical report 仅引用这份全局证据。500 个隐藏挑战刻意展示部分可观测上限；该指标不是策略质量、规则发现能力或真实系统全状态安全率。

## 7. 学习阶段

### 7.1 共享 BC

每个 seed 只训练一个共享 BC checkpoint。A/B 直接复制它且参数 L2 delta 为 0；C/D/E/F 从同一 checkpoint 开始。B 与 E 的主比较因此共享初始化、test cases、候选生成和推理 mask。

### 7.2 Progress Estimator

~~~text
Phi(s) = P(task succeeds | visible state, user_goal)
F(s,s') = beta * (gamma * Phi(s') - Phi(s))
~~~

标签来自 train/dev noisy continuation。Estimator 看不到 test、oracle plan 和隐藏目标；训练后冻结，并记录 BCE、Brier、ECE、AUROC。只有 E/F 使用 Progress Reward。

### 7.3 动作级与序列级 PPO

动作级 PPO 在每个 ToolCall 保存 old log-prob、value、reward、done、trajectory ID；GAE 不跨 episode。Sequence PPO 将 episode 内 log-prob 求和，使用一个 sequence ratio、discounted return 和初始 value，不把终局 reward 复制到每步。

六变体支持以下成对解释：

- B−A：BC 下推理 mask 的系统影响；
- D−C：sparse PPO 下推理 mask 的系统影响；
- D−B：相同 mask、无 Progress 时的动作 PPO 增量；
- E−D：相同动作 PPO+mask 下的 Progress 增量；
- E−B：预注册主比较，测试 PPO+Progress 在相同 mask/BC 起点上的联合增量；
- E−F：动作级与序列级信用分配。

## 8. Service-time 口径

环境累积的是冻结的模拟工具 service execution time，不对应 Agent wall-clock 或线上端到端时延。发布指标为：

- successful_conditional_simulated_service_time_s：成功任务的未截断模拟 service time 均值，可为 null；
- timeout_penalized_simulated_cost_s：成功任务截断到 timeout，失败任务计 timeout 的全样本模拟成本。

写操作使用 Azure Functions 2019 invocation-weighted p85–p95，read-only、拒绝校验和 replay 分别使用冻结 p80、p75、p50。来源与推导见 [Benchmark 说明](benchmark.md#8-service-time-来源与指标)。

## 9. 统计报告

canonical report 的预注册 treatment 为 E-PPO-Progress-Mask，matched baseline 为 B-BC-Mask，补充 total-system baseline 为 A-BC-Unmasked。主统计量是每个完全配对 (seed, case_id) 的 E−B success 差。

bootstrap 每次分别有放回重采样五个 seed 和 1000 个完整 case clusters；treatment/baseline 配对关系始终保留。完整配置为 1000 replicates、95% percentile CI。CI 下界大于 0 表示预注册假设获支持。

报告把两个事实分开：

- canonical：六变体、五 seed、1000 cases、20k 动作集、配对覆盖和统计配置结构有效；
- hypothesis_passed/passed：E−B TSR difference 的 95% CI 下界是否大于 0。

假设失败不影响 canonical，也不让 ablate 退出失败；这是需要发布的反例结果。

## 10. 证据与恢复

source fingerprint 覆盖 src/**/*.py、configs/*.yaml、pyproject.toml 和 uv.lock。run ID 绑定 config、benchmark manifest、seed、变体和 source fingerprint。Resume Guard 再绑定 benchmark/config/source/checkpoint 四类 SHA256，禁止用新输入续写旧 trace。

运行开始时会把 train/dev/test、action-validity 数据、manifest 与全局指标复制到 `<run-id>/benchmark/`。run manifest 只保存以自身父目录为基准的规范相对路径，因此 `<run-id>/` 可作为单一自包含证据包搬迁或重命名；路径解析拒绝绝对路径、`..`、非规范分隔符及逃逸 bundle 的符号链接。

verify-run 会：

1. 确定性重生成 train/dev/test 与 action-validity-v2，核对 manifest、SHA256、held-out 和 challenge 分层；
2. 加载共享 BC 与变体 checkpoint，核对 training metadata、参数摘要和实际 L2 delta；
3. 证明 BC 无 PPO update/漂移，PPO 变体有 update、rollout 和参数变化；
4. 校验 Resume Guard 与精确 case manifest；
5. 逐步重放 trace 的候选、ToolCall、dry-run、事务结果、终态与模拟 service time；
6. 使用声明的 checkpoint、FeatureEncoder、Progress Estimator 和变体开关确定性重跑每个 case，递归核对动作、mask、log-prob、value、奖励与完整终态；
7. 从原始 trace 使用同一 compute_metrics 定义做一致性复算；
8. 验证 A–F 完整 outcomes，并重建配对 A/B/E canonical statistical report。

这里的 recompute 是“从原始 trace 重新计算并比对”，不是第二套独立指标实现。

## 11. 已知限制

- v1.3 是结构化合成环境，不覆盖视觉、自然语言歧义、真实故障和外部副作用。
- family×topology held-out 增强组合泛化测试，但仍来自同一生成器。
- 规则型 Mask 与环境共享公开规范；hidden-ledger 样本只覆盖一种不可观测状态。
- 模拟 service time 不包含模型推理、网络、队列和 cold start。
- 轻量核心结果不能外推到 Qwen。

v1.2 的 ceiling 与停止决定已归档在 [pilot-v1.2](results/pilot-v1.2.md)，不作为 v1.3 final 结果。
