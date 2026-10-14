"""Metric comparisons preserve JSON boolean/number distinctions."""

import json
from pathlib import Path

import pytest

from agentic_tool_rl.evaluation.io import canonical_json, write_json_atomic
from agentic_tool_rl.evaluation.metrics import compute_metrics
from agentic_tool_rl.evaluation.recompute import diff_metrics, recompute_metrics


@pytest.mark.parametrize("boolean,number", [(True, 1), (True, 1.0), (False, 0), (False, 0.0)])
def test_nested_boolean_number_mismatches_are_symmetric(
    boolean: bool, number: int | float
) -> None:
    for expected, actual in ((boolean, number), (number, boolean)):
        differences = diff_metrics(
            {"nested": [{"value": expected}]},
            {"nested": [{"value": actual}]},
            tolerance=10.0,
        )
        assert len(differences) == 1
        difference = differences[0]
        assert difference.path == "nested[0].value"
        assert type(difference.expected) is type(expected)
        assert type(difference.actual) is type(actual)
        assert difference.absolute_difference is None


def test_boolean_equality_and_numeric_tolerance_are_preserved() -> None:
    assert diff_metrics({"a": [True, False, 1, 0.5]}, {"a": [True, False, 1.0, 0.5]}) == ()
    assert diff_metrics({"a": 1.0}, {"a": 1.0001}, tolerance=0.001) == ()
    assert len(diff_metrics({"a": 1.0}, {"a": 1.01}, tolerance=0.001)) == 1
    assert len(diff_metrics({"a": True}, {"a": False})) == 1


def test_recompute_reports_boolean_tampering_in_published_metrics(tmp_path: Path) -> None:
    trace = {
        "case_id": "one",
        "family": "test",
        "success": True,
        "steps": 1,
        "optimal_steps": 1,
        "simulated_latency_s": 1.0,
    }
    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_text(canonical_json(trace) + "\n", encoding="utf-8")
    published = compute_metrics([trace]).to_dict()
    published.update(tsr=True, successful_cases=True, invalid_action_rate=False)
    metrics_path = write_json_atomic(tmp_path / "metrics.json", published)
    output_path = tmp_path / "recompute.json"

    result = recompute_metrics(
        trace_path, published_metrics_path=metrics_path, output_path=output_path
    )

    assert not result.matches
    assert {item.path for item in result.differences} == {
        "tsr", "successful_cases", "invalid_action_rate"
    }
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["matches"] is False
    for difference in report["differences"]:
        assert isinstance(difference["expected"], bool)
        assert type(difference["actual"]) in (int, float)


@pytest.mark.parametrize(
    "payload",
    [
        '{"tsr":0.0,"tsr":1.0}',
        '{"tsr":1.0,"tsr":0.0}',
        '{"nested":{"value":1,"value":2}}',
        '{"nested":[{"value":1,"value":2}]}',
        r'{"tsr":0.0,"\u0074sr":1.0}',
    ],
)
def test_recompute_rejects_duplicate_published_keys_without_replacing_report(
    tmp_path: Path, payload: str
) -> None:
    trace = {
        "case_id": "one",
        "family": "test",
        "success": True,
        "steps": 1,
        "optimal_steps": 1,
        "simulated_latency_s": 1.0,
    }
    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_text(canonical_json(trace) + "\n", encoding="utf-8")
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(payload, encoding="utf-8")
    report_path = tmp_path / "report.json"
    previous_report = b"previous verified report\n"
    report_path.write_bytes(previous_report)

    with pytest.raises(ValueError, match=r"duplicate.*key") as error:
        recompute_metrics(
            trace_path, published_metrics_path=metrics_path, output_path=report_path
        )

    assert str(metrics_path) in str(error.value)
    assert report_path.read_bytes() == previous_report
    assert metrics_path.read_text(encoding="utf-8") == payload


def test_published_keys_can_recur_in_separate_objects(tmp_path: Path) -> None:
    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_text(
        canonical_json({
            "case_id": "one", "family": "test", "success": True,
            "steps": 1, "optimal_steps": 1, "simulated_latency_s": 1.0,
        }) + "\n",
        encoding="utf-8",
    )
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text('{"nested":[{"value":1},{"value":2}]}', encoding="utf-8")

    result = recompute_metrics(trace_path, published_metrics_path=metrics_path)

    assert not result.matches
    assert any(item.path == "nested" for item in result.differences)
