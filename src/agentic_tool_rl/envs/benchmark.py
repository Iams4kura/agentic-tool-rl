"""Deterministic generation of long-horizon transactional workflow tasks."""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from dataclasses import dataclass

from agentic_tool_rl.contracts import (
    Split,
    StatePredicate,
    ToolCall,
    ToolSchema,
    WorkflowNode,
    WorkflowTask,
    WorkflowTopology,
)
from agentic_tool_rl.latency_profile import sample_mutating_service_time

GENERATOR_VERSION = "benchmark-v1.3.0"
DEFAULT_BASE_SEED = 20_260_808


@dataclass(frozen=True, slots=True)
class FamilyTemplate:
    key: str
    title: str
    stages: tuple[str, ...]


_FAMILY_TEMPLATES: tuple[FamilyTemplate, ...] = (
    FamilyTemplate(
        "calendar_coordination",
        "calendar coordination",
        (
            "inspect_availability",
            "collect_preferences",
            "check_timezones",
            "propose_slots",
            "verify_attendees",
            "reserve_room",
            "check_conflicts",
            "request_confirmation",
            "confirm_attendees",
            "schedule_event",
            "send_invites",
            "record_responses",
            "resolve_declines",
            "finalize_agenda",
            "archive_request",
        ),
    ),
    FamilyTemplate(
        "ecommerce_returns",
        "e-commerce return",
        (
            "lookup_order",
            "verify_return_window",
            "inspect_item_eligibility",
            "collect_return_reason",
            "calculate_refund",
            "choose_return_route",
            "authorize_return",
            "create_return_label",
            "register_return",
            "schedule_pickup",
            "receive_item",
            "inspect_return",
            "approve_refund",
            "issue_refund",
            "notify_customer",
        ),
    ),
    FamilyTemplate(
        "travel_booking",
        "business travel booking",
        (
            "collect_trip_request",
            "check_travel_policy",
            "search_flights",
            "search_hotels",
            "compare_itineraries",
            "verify_budget",
            "select_flight",
            "select_hotel",
            "request_manager_approval",
            "reserve_flight",
            "reserve_hotel",
            "add_ground_transport",
            "confirm_traveler_details",
            "issue_itinerary",
            "archive_booking",
        ),
    ),
    FamilyTemplate(
        "it_permissions",
        "IT access request",
        (
            "lookup_employee",
            "collect_access_scope",
            "verify_employment_status",
            "check_separation_of_duties",
            "identify_resource_owner",
            "assess_risk",
            "request_owner_approval",
            "request_security_approval",
            "create_access_ticket",
            "provision_role",
            "verify_entitlement",
            "record_audit_evidence",
            "set_expiration",
            "notify_requester",
            "close_access_ticket",
        ),
    ),
    FamilyTemplate(
        "customer_support",
        "customer support escalation",
        (
            "open_ticket",
            "identify_customer",
            "classify_issue",
            "collect_diagnostics",
            "search_known_issues",
            "reproduce_problem",
            "propose_resolution",
            "verify_entitlement",
            "request_escalation_approval",
            "apply_resolution",
            "validate_recovery",
            "update_customer",
            "record_root_cause",
            "publish_internal_note",
            "close_ticket",
        ),
    ),
    FamilyTemplate(
        "expense_reimbursement",
        "expense reimbursement",
        (
            "create_expense_report",
            "attach_receipt",
            "extract_receipt_fields",
            "classify_expense",
            "validate_cost_center",
            "check_policy_limits",
            "detect_duplicates",
            "calculate_reimbursement",
            "request_manager_approval",
            "request_finance_approval",
            "post_ledger_entry",
            "schedule_payment",
            "verify_payment",
            "notify_employee",
            "archive_report",
        ),
    ),
    FamilyTemplate(
        "subscription_changes",
        "subscription plan change",
        (
            "lookup_subscription",
            "verify_account_owner",
            "collect_plan_choice",
            "calculate_proration",
            "check_contract_terms",
            "validate_payment_method",
            "present_change_summary",
            "collect_confirmation",
            "authorize_change",
            "apply_plan_change",
            "update_entitlements",
            "generate_invoice",
            "verify_access",
            "notify_account_owner",
            "close_change_request",
        ),
    ),
    FamilyTemplate(
        "cloud_incident_response",
        "cloud incident response",
        (
            "open_incident",
            "collect_alert_context",
            "identify_affected_service",
            "assess_severity",
            "query_recent_deployments",
            "inspect_service_health",
            "select_mitigation",
            "request_change_approval",
            "apply_mitigation",
            "verify_recovery",
            "update_status_page",
            "collect_timeline",
            "record_root_cause",
            "create_followup_actions",
            "close_incident",
        ),
    ),
    FamilyTemplate(
        "recruitment_interviews",
        "recruitment interview loop",
        (
            "open_candidate_record",
            "verify_role_requirements",
            "collect_availability",
            "select_interviewers",
            "check_conflicts",
            "build_interview_loop",
            "reserve_rooms",
            "request_panel_confirmation",
            "schedule_interviews",
            "send_candidate_brief",
            "collect_feedback",
            "check_feedback_completion",
            "request_hiring_decision",
            "notify_candidate",
            "archive_interview_loop",
        ),
    ),
    FamilyTemplate(
        "order_fulfillment",
        "order fulfillment",
        (
            "lookup_order",
            "verify_payment",
            "check_inventory",
            "reserve_inventory",
            "validate_delivery_address",
            "select_warehouse",
            "create_pick_list",
            "pick_items",
            "quality_check",
            "approve_shipment",
            "pack_order",
            "purchase_shipping_label",
            "dispatch_order",
            "confirm_carrier_acceptance",
            "notify_customer",
        ),
    ),
)

WORKFLOW_FAMILIES: tuple[str, ...] = tuple(template.key for template in _FAMILY_TEMPLATES)
WORKFLOW_TOPOLOGIES: tuple[WorkflowTopology, ...] = (
    "diamond_tail",
    "fanout_gate",
    "dual_lane",
    "dual_root_mesh",
)
_SPLIT_OFFSETS = {Split.TRAIN: 0, Split.DEV: 1_000_000_000, Split.TEST: 2_000_000_000}


def _coerce_split(split: Split | str) -> Split:
    return split if isinstance(split, Split) else Split(split)


def _allocate(total: int, buckets: int) -> list[int]:
    base, remainder = divmod(total, buckets)
    return [base + (1 if index < remainder else 0) for index in range(buckets)]


def _tier_counts(count: int) -> dict[str, int]:
    """Largest-remainder 30/40/30 allocation; exact for 100 cases."""

    weights = {"short": 0.3, "medium": 0.4, "long": 0.3}
    raw = {name: count * weight for name, weight in weights.items()}
    result = {name: int(value) for name, value in raw.items()}
    remainder = count - sum(result.values())
    priority = {"medium": 0, "short": 1, "long": 2}
    order = sorted(weights, key=lambda name: (-(raw[name] - result[name]), priority[name]))
    for name in order[:remainder]:
        result[name] += 1
    return result


def _length_for_tier(tier: str, ordinal: int) -> int:
    ranges = {"short": (6, 7, 8), "medium": (9, 10, 11), "long": (12, 13, 14, 15)}
    values = ranges[tier]
    return values[ordinal % len(values)]


def _opaque_operation_id(task_seed: int, label: str) -> str:
    payload = f"{GENERATOR_VERSION}:{task_seed}:{label}".encode()
    return f"op_{hashlib.blake2s(payload, digest_size=10).hexdigest()}"


def _topology_for_task(
    split: Split,
    *,
    family_index: int,
    task_seed: int,
) -> WorkflowTopology:
    """Hold out one family/topology combination exclusively for test."""

    held_out = WORKFLOW_TOPOLOGIES[family_index % len(WORKFLOW_TOPOLOGIES)]
    if split == Split.TEST:
        return held_out
    development_topologies = tuple(
        topology for topology in WORKFLOW_TOPOLOGIES if topology != held_out
    )
    return development_topologies[task_seed % len(development_topologies)]


def _body_predecessor_indices(
    topology: WorkflowTopology,
    body_size: int,
) -> list[list[int]]:
    """Build one of four acyclic bodies, each with at least two legal orders."""

    if body_size < 4:
        raise ValueError("workflow body must contain at least four operations")
    predecessors: list[list[int]] = [[] for _ in range(body_size)]
    if topology == "diamond_tail":
        predecessors[1] = [0]
        predecessors[2] = [0]
        predecessors[3] = [1, 2]
        for index in range(4, body_size):
            predecessors[index] = [index - 1]
    elif topology == "fanout_gate":
        for index in range(1, min(4, body_size)):
            predecessors[index] = [0]
        for index in range(4, body_size):
            predecessors[index] = list(range(max(1, index - 3), index))
    elif topology == "dual_lane":
        predecessors[1] = [0]
        predecessors[2] = [0]
        for index in range(3, body_size):
            predecessors[index] = [index - 2]
    elif topology == "dual_root_mesh":
        predecessors[1] = []
        predecessors[2] = [0]
        predecessors[3] = [0, 1]
        for index in range(4, body_size):
            predecessors[index] = sorted({index - 3, index - 1})
    else:  # pragma: no cover - exhaustive Literal guard
        raise ValueError(f"unsupported topology {topology!r}")
    return predecessors


def _predecessor_indices(
    topology: WorkflowTopology,
    length: int,
) -> list[list[int]]:
    """Attach an approval barrier and final action to a topology body."""

    body_size = length - 2
    body = _body_predecessor_indices(topology, body_size)
    referenced = {predecessor for values in body for predecessor in values}
    sinks = [index for index in range(body_size) if index not in referenced]
    if not sinks:
        raise RuntimeError(f"topology {topology!r} produced no body sink")
    return [*body, sinks, [body_size]]


def _topological_orders(nodes: list[WorkflowNode], limit: int = 2) -> list[list[WorkflowNode]]:
    """Return deterministic alternative legal orders for a small DAG."""

    node_by_id = {node.node_id: node for node in nodes}
    results: list[list[WorkflowNode]] = []

    def visit(completed: tuple[str, ...], path: list[WorkflowNode]) -> None:
        if len(results) >= limit:
            return
        if len(path) == len(nodes):
            results.append(path.copy())
            return
        completed_set = set(completed)
        ready = sorted(
            (
                node
                for node in nodes
                if node.node_id not in completed_set
                and set(node.required_predecessors).issubset(completed_set)
            ),
            key=lambda node: node.node_id,
        )
        for node in ready:
            if node.node_id not in node_by_id:
                continue
            visit((*completed, node.node_id), [*path, node])

    visit((), [])
    return results


def _plan_calls(
    case_id: str,
    entity_id: str,
    plan_index: int,
    order: list[WorkflowNode],
) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for step_index, node in enumerate(order):
        arguments: dict[str, object] = {
            "entity_id": entity_id,
            "operation_id": node.node_id,
            "expected_version": step_index,
        }
        if node.safety_token is not None:
            arguments["approval_token"] = node.safety_token
            arguments["idempotency_key"] = f"idem-{entity_id}-{node.node_id}"
        calls.append(
            ToolCall(
                tool_name=node.tool_name,
                arguments=arguments,
                call_id=f"oracle-{case_id}-{plan_index}-{step_index:02d}",
            )
        )
    return calls


def _build_task(
    template: FamilyTemplate,
    split: Split,
    family_index: int,
    ordinal: int,
    tier: str,
    tier_ordinal: int,
    base_seed: int,
) -> WorkflowTask:
    seed = _SPLIT_OFFSETS[split] + base_seed + family_index * 1_000_000 + ordinal
    rng = random.Random(seed)
    length = _length_for_tier(tier, tier_ordinal)
    entity_id = f"{split.value}-{template.key[:4]}-{ordinal:05d}-{seed:010d}"
    case_id = f"{split.value}-{template.key}-{ordinal:04d}"
    priority = rng.choice(("standard", "priority", "urgent"))
    region = rng.choice(("apac", "emea", "na"))
    channel = rng.choice(("web", "mobile", "api"))
    approval_token = f"approved-{entity_id}"
    topology = _topology_for_task(
        split,
        family_index=family_index,
        task_seed=seed,
    )
    operation_ids = [_opaque_operation_id(seed, f"operation:{index}") for index in range(length)]
    predecessor_indices = _predecessor_indices(topology, length)

    nodes: list[WorkflowNode] = []
    schemas: list[ToolSchema] = []
    latency_trace: dict[str, float] = {}
    for index, stage in enumerate(template.stages[:length]):
        node_id = operation_ids[index]
        required_predecessors = [
            operation_ids[predecessor] for predecessor in predecessor_indices[index]
        ]
        is_final = index == length - 1
        is_approval = index == length - 2
        required = {
            "entity_id": "string",
            "operation_id": "string",
            "expected_version": "integer",
        }
        safety_token: str | None = None
        if is_final:
            required = {
                **required,
                "approval_token": "string",
                "idempotency_key": "string",
            }
            safety_token = approval_token
        tool_name = f"{template.key}.{stage}"
        effects: dict[str, object] = {f"{stage}_done": True}
        if is_approval:
            effects["approval_token"] = approval_token
        if is_final:
            effects["status"] = "completed"
        # Frozen empirical service execution time; see latency_profile.py for
        # the public trace, checksum, derivation, and interpretation boundary.
        latency = sample_mutating_service_time(rng)
        nodes.append(
            WorkflowNode(
                node_id=node_id,
                tool_name=tool_name,
                required_predecessors=required_predecessors,
                effects=effects,
                side_effect=f"{template.key}.completed" if is_final else None,
                safety_token=safety_token,
                simulated_latency_s=latency,
            )
        )
        schemas.append(
            ToolSchema(
                name=tool_name,
                description=f"{stage.replace('_', ' ')} for the {template.title} workflow",
                required_arguments=required,
                mutating=True,
                idempotent=True,
                side_effect=f"{template.key}.completed" if is_final else None,
                operation_id=node_id,
                required_completed_operations=required_predecessors,
            )
        )
        latency_trace[node_id] = latency

    schemas.append(
        ToolSchema(
            name=f"{template.key}.inspect_status",
            description=("Read the current workflow status without advancing any goal predicate"),
            required_arguments={
                "entity_id": "string",
                "operation_id": "string",
                "expected_version": "integer",
            },
            mutating=False,
            idempotent=True,
            operation_id="inspect-status",
        )
    )

    # A policy sees one changing, context-matched service action alongside the
    # status query.  These calls look business-relevant and are executable, but
    # are deliberately read-only so choosing them consumes latency without
    # advancing the task goal.  Their names are derived only from visible
    # priority/region/channel values.
    for name, description in (
        (
            f"{template.key}.route_{priority}_request",
            f"Review the {priority} priority routing path",
        ),
        (
            f"{template.key}.apply_{region}_policy",
            f"Review the {region} regional policy path",
        ),
        (
            f"{template.key}.sync_{channel}_context",
            f"Review the {channel} channel handoff context",
        ),
    ):
        schemas.append(
            ToolSchema(
                name=name,
                description=description,
                required_arguments={
                    "entity_id": "string",
                    "operation_id": "string",
                    "expected_version": "integer",
                },
                mutating=False,
                idempotent=True,
                operation_id="inspect-status",
            )
        )

    forbidden_side_effect = f"{template.key}.force_complete"
    schemas.append(
        ToolSchema(
            name=f"{template.key}.force_complete",
            description="Unsafe administrative shortcut retained as a negative action candidate",
            required_arguments={
                "entity_id": "string",
                "operation_id": "string",
                "expected_version": "integer",
            },
            mutating=True,
            idempotent=False,
            side_effect=forbidden_side_effect,
            operation_id="force-complete",
            policy_allowed=False,
        )
    )

    initial_state: dict[str, object] = {
        "entity_id": entity_id,
        "version": 0,
        "status": "pending",
        "priority": priority,
        "region": region,
        "channel": channel,
        "approval_token": None,
        "completed_nodes": [],
        "side_effects": [],
        "processed_call_ids": [],
        "idempotency_results": {},
        "audit_log": [],
    }
    orders = _topological_orders(nodes, limit=2)
    if len(orders) != 2:
        raise RuntimeError(f"generator must produce two distinct legal orders for {case_id}")
    plans = [_plan_calls(case_id, entity_id, index, order) for index, order in enumerate(orders)]
    visible_observation = {
        key: value
        for key, value in initial_state.items()
        if key not in {"processed_call_ids", "idempotency_results", "audit_log"}
    }
    return WorkflowTask(
        case_id=case_id,
        split=split,
        family=template.key,
        topology=topology,
        difficulty=tier,  # type: ignore[arg-type]
        seed=seed,
        entity_id=entity_id,
        user_goal=(
            f"Complete the {template.title} workflow for {entity_id} "
            f"in {region} with {priority} priority via {channel}."
        ),
        initial_state=initial_state,
        visible_observation=visible_observation,
        tool_schemas=schemas,
        workflow_nodes=nodes,
        hidden_goal_predicates=[
            StatePredicate(path="status", operator="eq", value="completed"),
            StatePredicate(
                path="completed_nodes",
                operator="contains_all",
                value=[node.node_id for node in nodes],
            ),
        ],
        forbidden_side_effects=[forbidden_side_effect],
        oracle_plans=plans,
        optimal_steps=length,
        max_steps=length + 4,
        latency_trace=latency_trace,
        generator_version=GENERATOR_VERSION,
    )


def generate_tasks(
    split: Split | str,
    size: int,
    *,
    base_seed: int = DEFAULT_BASE_SEED,
) -> list[WorkflowTask]:
    """Generate a deterministic split with balanced workflow families.

    For the canonical 1000-case test split each family receives exactly 100
    tasks and its short/medium/long strata are exactly 30/40/30.  The same
    largest-remainder rules also support tiny CI smoke suites without special
    cases.
    """

    if size < 1:
        raise ValueError("size must be positive")
    resolved_split = _coerce_split(split)
    family_sizes = _allocate(size, len(_FAMILY_TEMPLATES))
    tasks: list[WorkflowTask] = []
    for family_index, (template, family_size) in enumerate(
        zip(_FAMILY_TEMPLATES, family_sizes, strict=True)
    ):
        counts = _tier_counts(family_size)
        tiers = [name for name in ("short", "medium", "long") for _ in range(counts[name])]
        random.Random(_SPLIT_OFFSETS[resolved_split] + base_seed + family_index * 10_000).shuffle(
            tiers
        )
        tier_ordinals: Counter[str] = Counter()
        for ordinal, tier in enumerate(tiers):
            tasks.append(
                _build_task(
                    template,
                    resolved_split,
                    family_index,
                    ordinal,
                    tier,
                    tier_ordinals[tier],
                    base_seed,
                )
            )
            tier_ordinals[tier] += 1
    tasks.sort(key=lambda task: task.case_id)
    return tasks


def generate_benchmark(
    size: int = 1000,
    *,
    split: Split | str = Split.TEST,
    base_seed: int = DEFAULT_BASE_SEED,
) -> list[WorkflowTask]:
    """Public convenience alias used by the CLI and examples."""

    return generate_tasks(split, size, base_seed=base_seed)


def generate_all_splits(
    *,
    train_size: int,
    dev_size: int,
    test_size: int,
    base_seed: int = DEFAULT_BASE_SEED,
) -> dict[Split, list[WorkflowTask]]:
    splits = {
        split: generate_tasks(split, size, base_seed=base_seed)
        for split, size in (
            (Split.TRAIN, train_size),
            (Split.DEV, dev_size),
            (Split.TEST, test_size),
        )
    }
    assert_splits_disjoint(splits)
    return splits


def assert_splits_disjoint(splits: dict[Split, list[WorkflowTask]]) -> None:
    """Fail fast if generator changes ever introduce cross-split leakage."""

    seen_seeds: set[int] = set()
    seen_entities: set[str] = set()
    seen_cases: set[str] = set()
    for split, tasks in splits.items():
        seeds = {task.seed for task in tasks}
        entities = {task.entity_id for task in tasks}
        cases = {task.case_id for task in tasks}
        if len(seeds) != len(tasks) or len(entities) != len(tasks) or len(cases) != len(tasks):
            raise ValueError(f"duplicate identifier inside {split.value} split")
        if seen_seeds & seeds or seen_entities & entities or seen_cases & cases:
            raise ValueError(f"cross-split leakage detected in {split.value} split")
        seen_seeds.update(seeds)
        seen_entities.update(entities)
        seen_cases.update(cases)


__all__ = [
    "DEFAULT_BASE_SEED",
    "GENERATOR_VERSION",
    "WORKFLOW_FAMILIES",
    "WORKFLOW_TOPOLOGIES",
    "assert_splits_disjoint",
    "generate_all_splits",
    "generate_benchmark",
    "generate_tasks",
]
