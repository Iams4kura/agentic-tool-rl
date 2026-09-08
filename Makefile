SHELL := /usr/bin/env bash

.DEFAULT_GOAL := help

UV ?= uv
SMOKE_CONFIG ?= configs/smoke.yaml
FULL_CONFIG ?= configs/cpu_full.yaml
ABLATION_CONFIG ?= configs/ablation.yaml
QWEN_CONFIG ?= configs/qwen3_lora_gpu.yaml
SMOKE_BENCHMARK_DIR ?= artifacts/benchmark-smoke
SMOKE_OUTPUT_DIR ?= artifacts/runs/smoke
FULL_BENCHMARK_DIR ?= artifacts/benchmark-v1
FULL_OUTPUT_DIR ?= artifacts/runs/full
BUILD_CONSTRAINTS ?= build-constraints.txt

.PHONY: help lock-check sync lint typecheck test check package-check verify-smoke verify-smoke-structure verify-semantic smoke qwen-test qwen-dry-run qwen-check ci benchmark full-benchmark

help:
	@printf '%s\n' \
		'lock-check      Verify pyproject.toml and uv.lock are synchronized' \
		'sync            Install locked runtime and development dependencies' \
		'lint            Run Ruff checks' \
		'typecheck       Run strict mypy checks' \
		'test            Run the pytest suite' \
		'check           Run lint, typecheck, and tests' \
		'package-check   Rebuild and inspect reproducible wheel/sdist artifacts' \
		'verify-smoke    Run structural smoke plus deterministic semantic canary' \
		'verify-semantic Run the deterministic learner semantic canary' \
		'qwen-dry-run    Validate Qwen/LoRA GPU configuration without model access' \
		'qwen-check      Run the optional Qwen adapter test surface' \
		'ci              Run the same lightweight acceptance surface as CI' \
		'benchmark       Run the full six-variant, five-seed CPU benchmark'

lock-check:
	$(UV) lock --check

sync:
	$(UV) sync --extra dev --locked

lint:
	$(UV) run --locked ruff check .

typecheck:
	$(UV) run --locked mypy

test:
	$(UV) run --locked pytest

check: lint typecheck test

package-check:
	$(UV) run --locked python scripts/check_distribution.py \
		--uv $(UV) \
		--build-constraints $(BUILD_CONSTRAINTS)

verify-smoke: verify-smoke-structure verify-semantic

verify-smoke-structure:
	$(UV) run --locked agentic-tool-rl smoke \
		--benchmark-dir $(SMOKE_BENCHMARK_DIR) \
		--output $(SMOKE_OUTPUT_DIR)

verify-semantic:
	$(UV) run --locked agentic-tool-rl semantic-canary

smoke: verify-smoke

qwen-dry-run:
	$(UV) run --locked agentic-tool-rl qwen-dry-run

qwen-test:
	$(UV) run --locked pytest -o addopts="-ra --strict-markers" -m qwen

qwen-check: qwen-test qwen-dry-run

ci:
	$(MAKE) lock-check
	$(MAKE) sync
	$(MAKE) check
	$(MAKE) verify-smoke
	$(MAKE) package-check

benchmark:
	$(UV) run --locked agentic-tool-rl ablate \
		--config $(FULL_CONFIG) \
		--ablation $(ABLATION_CONFIG) \
		--benchmark-dir $(FULL_BENCHMARK_DIR) \
		--output $(FULL_OUTPUT_DIR)

full-benchmark: benchmark
