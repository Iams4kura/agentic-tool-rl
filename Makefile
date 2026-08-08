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

.PHONY: help sync lint typecheck test check verify-smoke smoke qwen-test qwen-dry-run qwen-check ci benchmark full-benchmark

help:
	@printf '%s\n' \
		'sync            Install locked runtime and development dependencies' \
		'lint            Run Ruff checks' \
		'typecheck       Run strict mypy checks' \
		'test            Run the pytest suite' \
		'check           Run lint, typecheck, and tests' \
		'verify-smoke    Run the lightweight end-to-end acceptance loop' \
		'qwen-dry-run    Validate Qwen/LoRA GPU configuration without model access' \
		'qwen-check      Run the optional Qwen adapter test surface' \
		'ci              Run the same lightweight acceptance surface as CI' \
		'benchmark       Run the full six-variant, five-seed CPU benchmark'

sync:
	$(UV) sync --extra dev --frozen

lint:
	$(UV) run --frozen ruff check .

typecheck:
	$(UV) run --frozen mypy

test:
	$(UV) run --frozen pytest

check: lint typecheck test

verify-smoke:
	$(UV) run --frozen agentic-tool-rl smoke \
		--config $(SMOKE_CONFIG) \
		--ablation $(ABLATION_CONFIG) \
		--benchmark-dir $(SMOKE_BENCHMARK_DIR) \
		--output $(SMOKE_OUTPUT_DIR)

smoke: verify-smoke

qwen-dry-run:
	$(UV) run --frozen agentic-tool-rl qwen-dry-run --config $(QWEN_CONFIG)

qwen-test:
	$(UV) run --frozen pytest -o addopts="-ra --strict-markers" -m qwen

qwen-check: qwen-test qwen-dry-run

ci: sync
	$(MAKE) check
	$(MAKE) verify-smoke

benchmark:
	$(UV) run --frozen agentic-tool-rl ablate \
		--config $(FULL_CONFIG) \
		--ablation $(ABLATION_CONFIG) \
		--benchmark-dir $(FULL_BENCHMARK_DIR) \
		--output $(FULL_OUTPUT_DIR)

full-benchmark: benchmark
