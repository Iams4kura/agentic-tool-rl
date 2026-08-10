# Benchmark v1.4 development protocol

Status: development acceptance. This document does **not** unlock a canonical
run. A canonical final remains prohibited until the code commit, thresholds,
external timestamp, and future public randomness beacon are locked.

## Claim boundary

Benchmark v1.4 measures goal-conditioned long-horizon tool-action selection
under public executability constraints. It does not claim that a policy
independently discovers the workflow DAG; public schemas, candidate building,
and `ActionMask` continue to enforce executability.

The v1.3 release, tag, evidence, and reported numbers are immutable. The
[post-hoc audit](../results/posthoc-v1.3-public-ready.md) adds interpretation
without rewriting that history.

## Exact development dataset

| Split | Counterfactual groups | Cases | Goals per group |
|---|---:|---:|---:|
| train | 500 | 2,000 | 4 |
| dev | 100 | 400 | 4 |
| test | 250 | 1,000 | 4 |

Each group has one world, entity, seed, initial state, schema set, candidate
process, and action-mask process. Only opaque `case_id`, `user_goal`, hidden
goal predicates, oracle plans, and evaluator-only `goal_variant` may differ.
The four goals are the Cartesian product of two binary semantic choices. Each
world contains four isomorphic semantic lanes. Every lane repeats the same
family stages and topology, then exposes two checkpoints and four parallel
completion actions. The target operation sets are pairwise disjoint across the
four goals; choosing any other lane is safe and executable but exhausts the
exact step budget.

All mutating tools use per-world opaque role-neutral names and the same public
flags and safe side-effect shape. Goal, prefix, checkpoint, and completion
surfaces use separate paraphrase banks for the same two semantic axes. Within
a lane, all four completion candidates are simultaneously executable and have
matching public metadata. Consequently neither family/stage membership,
candidate-visible downstream centrality, a fixed candidate position, nor
`side_effect != null` identifies the target route or completion.

Test keeps one topology held out per family. Therefore the minimum-cell rule
applies to populated `family × difficulty × topology` cells; every populated
test cell must contain at least five groups. Empty family/topology combinations
are intentional holdouts, not silently missing strata.

## Information boundary

`PolicyInput` is the sole public decision DTO. It contains a de-identified
observation, candidate calls without `call_id`, action-mask bits, and the
public schema paired with each candidate. It excludes task/case identity,
workflow nodes, oracle plans or distance, positive labels, hidden predicates,
validity labels, evaluator outputs, and private ledgers.

Public baselines must accept `PolicyInput` rather than `WorkflowTask`. The
lightweight feature encoder and optional Qwen adapter must consume the same DTO
before this protocol can be frozen.

## Development benchmark gates

All conditions are required:

- Oracle TSR is 100%, forbidden side effects are zero, and every plan uses
  exactly `optimal_steps`; at least four deterministic shortest plans per case
  are generated and replayed.
- Counterfactual public-world, candidate-order, and mask invariants pass for
  every group.
- R1 public-ready greedy TSR is at most 25%.
- The strongest registered non-learning public rule has dev TSR below 80%.
- At least 20% of expert states have two or more optimal positive actions.
- At least 80% of oracle-reachable nonterminal states expose a safe,
  executable, goal-incompatible mutation.
- Splits share no group, entity, world seed, or case.
- Repeated generation is byte deterministic.

The registered rules are Oracle, R0 masked random, R1 public-ready greedy, R2
public-DAG greedy, R3 goal/schema token overlap, R4 frozen reviewer composite
shortcut, R5 frozen source-aware family/stage shortcut, and R6 public
candidate-frontier downstream centrality. R4 preserves the exact public-only
rule that previously
achieved 400/400 by composing an `optional advisory` filter, fixed goal-to-name
mappings, and a `finalize_` name marker. The development gate replays and
stores R4 traces on every run so this shortcut cannot silently return. R5
preserves the next discovered 400/400 rule: fixed family-stage membership,
finite goal/gate phrase tables, and final detection through a non-null public
side effect. R6 checks whether the prerequisite relations visible in the
current `PolicyInput` candidate frontier reveal a preferred route. R5 and R6
are likewise replayed and stored on every run.
G-shuffle is evaluated with learned models during the later dev pilot.

## Training correctness

For an expert state, the positive set is:

```text
A+(s) = {a | a is executable and d*(T(s,a)) = d*(s) - 1}
L_MP(s) = -log sum[a in A+(s)] pi(a|s)
```

`positive_mask` must be non-empty and a subset of `candidate_mask`. The frozen
v1.3 training path remains single-label; v1.4 tasks explicitly select the
multi-positive path.

Potential shaping uses:

```text
phi_next_used = 0 if done else Phi(next_state)
F = gamma * phi_next_used - Phi(state)
```

Success and timeout terminals must not call the Progress estimator. Progress
examples are selected deterministically across populated
`family × difficulty × topology` strata and include both `done=0` and `done=1`.

## Claim and publication separation

Passing these development gates permits benchmark iteration only. It does not
permit a research claim or canonical release. The later locked protocol uses
only B (multi-positive BC + mask), D (B + sparse Action PPO), and E (B + Action
PPO + corrected Progress), ten paired seeds each. H1 is D-B and H2 is E-D;
each claim requires both a positive 97.5% paired hierarchical-bootstrap lower
bound and at least a two-percentage-point effect.

Valid artifacts may be published when a hypothesis is null or negative. A
failed claim gate changes wording, not artifact validity. No threshold,
reward coefficient, or learner may be tuned after canonical test exposure.

## Development acceptance command

```bash
uv run --locked python scripts/run_benchmark_v14.py \
  --output artifacts/benchmark-v1.4-development
```

The command generates all three splits twice, checks byte determinism, validates
counterfactual invariants and oracles, runs all registered public baselines on
dev, writes per-case traces, and records `canonical=false` and
`canonical_final_executed=false` in the verification evidence.
