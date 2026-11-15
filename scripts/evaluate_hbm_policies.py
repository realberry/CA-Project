#!/usr/bin/env python3
"""Evaluate predictor-driven expert prefetching with persistent HBM caches.

The evaluator uses held-out Mixtral routes and latency/energy values measured by
LLMSimulator. It compares on-demand LRU, prediction without prefetch, prefetch
with LRU, and prefetch with correlation-aware retention (CAR) for standalone
CCT and a causal adaptive HT-CCT ensemble.

CAR is training-free.  When space is required, it retains experts that are
predicted for the immediately following layer, appeared at the same layer in
the previous token, are frequent in the profile split, or are near in the
cyclic layer traversal.  LRU recency breaks equal-score ties.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from analyze_prefetch_sensitivity import measured_dynamic_energy_j
from simulate_prefetch import (
    build_profile,
    cct_scores,
    iter_sequences,
    top_indices,
    trace_prefix_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
OFFLOAD_BANDWIDTH_BYTES_S = 450e9


@dataclass
class CacheEntry:
    layer: int
    expert: int
    loaded_ms: float
    last_access: int
    inserted_at: int
    prefetch_target_event: int | None = None
    prefetch_bytes_ms: float = 0.0
    prefetch_used: bool = False

    @property
    def key(self) -> tuple[int, int]:
        return (self.layer, self.expert)


@dataclass
class PredictionRecord:
    candidates: set[tuple[int, int]]
    issue_state: dict[tuple[int, int], str]


@dataclass
class Metrics:
    tokens: int = 0
    demand_requests: int = 0
    eligible_requests: int = 0
    predicted_requests: int = 0
    correct_predictions: int = 0
    prediction_resident_hits: int = 0
    prefetches_issued: int = 0
    timely_prefetches: int = 0
    late_prefetches: int = 0
    hbm_hits: int = 0
    demand_misses: int = 0
    evictions: int = 0
    prefetch_evictions: int = 0
    prefetch_transfer_ms: float = 0.0
    demand_transfer_ms: float = 0.0
    unused_prefetch_ms: float = 0.0
    exposed_stall_ms: float = 0.0
    token_latency_sum_ms: float = 0.0
    token_latency_square_sum_ms: float = 0.0

    def add_token(self, latency_ms: float) -> None:
        self.tokens += 1
        self.token_latency_sum_ms += latency_ms
        self.token_latency_square_sum_ms += latency_ms * latency_ms


SCENARIOS = (
    ("baseline_lru", "none", False, "lru"),
    ("prediction_only", "cct", False, "lru"),
    ("prefetch_lru", "cct", True, "lru"),
    ("complete_car", "cct", True, "car"),
    ("adaptive_prediction_only", "adaptive_ht_cct", False, "lru"),
    ("adaptive_prefetch_lru", "adaptive_ht_cct", True, "lru"),
    ("adaptive_complete_car", "adaptive_ht_cct", True, "car"),
)


class AdaptiveEnsemble:
    """Per-layer causal ensemble using exponentially decayed predictor recall."""

    def __init__(self, layers: int, experts: int, beta: float) -> None:
        if not 0.0 <= beta < 1.0:
            raise ValueError("adaptive_decay_beta must be in [0, 1)")
        self.layers = layers
        self.experts = experts
        self.beta = beta
        self.ht_reliability = [0.5 for _ in range(layers)]
        self.cct_reliability = [0.5 for _ in range(layers)]

    @staticmethod
    def _normalize(scores: list[float]) -> list[float]:
        total = sum(scores)
        if total == 0.0:
            return [0.0 for _ in scores]
        return [score / total for score in scores]

    def rank_and_update(
        self,
        *,
        target_layer: int,
        cct_values: list[float],
        previous_group,
        actual_next: set[int],
        budget: int,
    ) -> list[int]:
        cct_values = self._normalize(cct_values)
        cct_ranking = top_indices(cct_values, budget)
        ht_values = [0.0 for _ in range(self.experts)]
        if previous_group is not None:
            for token in previous_group:
                ht_values[token[0][target_layer]] += 1.0
                ht_values[token[1][target_layer]] += 1.0
            ht_values = self._normalize(ht_values)
            denominator = (
                self.ht_reliability[target_layer]
                + self.cct_reliability[target_layer]
            )
            ht_weight = self.ht_reliability[target_layer] / denominator
            cct_weight = self.cct_reliability[target_layer] / denominator
            combined = [
                ht_weight * ht_values[expert]
                + cct_weight * cct_values[expert]
                for expert in range(self.experts)
            ]
            ranking = top_indices(combined, budget)
            ht_prediction = set(top_indices(ht_values, budget))
            ht_reward = len(ht_prediction & actual_next) / len(actual_next)
            self.ht_reliability[target_layer] = (
                self.beta * self.ht_reliability[target_layer]
                + (1.0 - self.beta) * ht_reward
            )
        else:
            # HT has no causal input for the first batch in an independent row.
            ranking = cct_ranking
        cct_reward = len(set(cct_ranking) & actual_next) / len(actual_next)
        self.cct_reliability[target_layer] = (
            self.beta * self.cct_reliability[target_layer]
            + (1.0 - self.beta) * cct_reward
        )
        return ranking

    def metadata(self) -> dict:
        weights = []
        for layer in range(self.layers):
            denominator = self.ht_reliability[layer] + self.cct_reliability[layer]
            weights.append(
                {
                    "layer": layer,
                    "ht_reliability": self.ht_reliability[layer],
                    "cct_reliability": self.cct_reliability[layer],
                    "ht_weight": self.ht_reliability[layer] / denominator,
                    "cct_weight": self.cct_reliability[layer] / denominator,
                }
            )
        return {"beta": self.beta, "final_layer_weights": weights}


def iter_evaluation_groups(
    trace: Path,
    profile_tokens: int,
    evaluation_tokens: int,
    batch_size: int,
):
    """Yield fixed-size routing batches without joining independent trace rows."""
    if evaluation_tokens % batch_size:
        raise ValueError("evaluation_tokens must be divisible by routing_batch_size")
    evaluated = 0
    # Extra input covers at most one dropped partial group per trace row.
    token_limit = profile_tokens + evaluation_tokens + 64 * (batch_size - 1)
    for offset, routes in iter_sequences(trace, token_limit):
        start = max(0, profile_tokens - offset)
        if start >= len(routes):
            continue
        position = start
        while position + batch_size <= len(routes) and evaluated < evaluation_tokens:
            group = routes[position : position + batch_size]
            previous_group = (
                routes[position - batch_size : position]
                if position - batch_size >= start
                else None
            )
            yield group, previous_group
            position += batch_size
            evaluated += batch_size
        if evaluated >= evaluation_tokens:
            return
    raise RuntimeError(
        f"Trace supplied {evaluated:,} grouped evaluation tokens; "
        f"{evaluation_tokens:,} are required"
    )


class CacheSimulator:
    def __init__(
        self,
        *,
        name: str,
        predictor: str,
        enable_prefetch: bool,
        eviction_policy: str,
        capacity: int,
        load_ms: float,
        overlap_ms: float,
        candidate_budget: int,
        layers: int,
        layer_frequency: list[list[int]],
    ) -> None:
        self.name = name
        self.predictor = predictor
        self.enable_prefetch = enable_prefetch
        self.eviction_policy = eviction_policy
        self.enable_prediction = predictor != "none"
        self.capacity = capacity
        self.load_ms = load_ms
        self.overlap_ms = overlap_ms
        self.candidate_budget = candidate_budget
        self.layers = layers
        self.layer_frequency = layer_frequency
        self.cache: dict[tuple[int, int], CacheEntry] = {}
        self.metrics = Metrics()
        self.clock = 0
        self.pending: PredictionRecord | None = None

    def _finish_prefetch_lifecycle(self, entry: CacheEntry) -> None:
        if entry.prefetch_bytes_ms > 0.0 and not entry.prefetch_used:
            self.metrics.unused_prefetch_ms += entry.prefetch_bytes_ms
            self.metrics.prefetch_evictions += 1

    def flush(self) -> None:
        for entry in self.cache.values():
            self._finish_prefetch_lifecycle(entry)
        self.cache.clear()
        self.pending = None

    def _car_score(
        self,
        entry: CacheEntry,
        *,
        current_layer: int,
        previous_group,
        next_event: int,
    ) -> float:
        immediate = 1.0 if entry.prefetch_target_event == next_event else 0.0
        history = 0.0
        if previous_group is not None:
            history = float(
                any(
                    entry.expert in (token[0][entry.layer], token[1][entry.layer])
                    for token in previous_group
                )
            )
        frequencies = self.layer_frequency[entry.layer]
        maximum = max(frequencies) or 1
        frequency = frequencies[entry.expert] / maximum
        distance = (entry.layer - current_layer) % self.layers
        if distance == 0:
            distance = self.layers
        proximity = 1.0 / distance
        age = max(0, self.clock - entry.last_access)
        recency = 1.0 / (1.0 + age)
        return 4.0 * immediate + 1.5 * history + frequency + 0.5 * proximity + 0.25 * recency

    def _evict_one(self, *, current_layer: int, previous_group, next_event: int) -> None:
        if not self.cache:
            raise RuntimeError("Cannot evict from an empty cache")
        if self.eviction_policy == "lru":
            victim = min(
                self.cache.values(),
                key=lambda entry: (entry.last_access, entry.inserted_at, entry.key),
            )
        elif self.eviction_policy == "car":
            victim = min(
                self.cache.values(),
                key=lambda entry: (
                    self._car_score(
                        entry,
                        current_layer=current_layer,
                        previous_group=previous_group,
                        next_event=next_event,
                    ),
                    entry.last_access,
                    entry.inserted_at,
                    entry.key,
                ),
            )
        else:
            raise ValueError(f"Unknown eviction policy: {self.eviction_policy}")
        self._finish_prefetch_lifecycle(victim)
        self.metrics.evictions += 1
        del self.cache[victim.key]

    def _ensure_entry(
        self,
        key: tuple[int, int],
        *,
        current_layer: int,
        previous_group,
        next_event: int,
    ) -> CacheEntry:
        entry = self.cache.get(key)
        if entry is not None:
            return entry
        while len(self.cache) >= self.capacity:
            self._evict_one(
                current_layer=current_layer,
                previous_group=previous_group,
                next_event=next_event,
            )
        entry = CacheEntry(
            layer=key[0],
            expert=key[1],
            loaded_ms=0.0,
            last_access=self.clock,
            inserted_at=self.clock,
        )
        self.cache[key] = entry
        return entry

    def demand(
        self,
        *,
        layer: int,
        actual_experts: set[int],
        previous_group,
        event_id: int,
    ) -> float:
        self.clock += 1
        actual_keys = {(layer, expert) for expert in actual_experts}
        self.metrics.demand_requests += len(actual_keys)

        if self.pending is not None and self.enable_prediction:
            correct = actual_keys & self.pending.candidates
            self.metrics.correct_predictions += len(correct)
            if self.enable_prefetch:
                for key in correct:
                    state = self.pending.issue_state.get(key, "not_issued")
                    if state == "resident":
                        self.metrics.prediction_resident_hits += 1
                    elif state == "issued":
                        entry = self.cache.get(key)
                        if entry is not None and math.isclose(
                            entry.loaded_ms, self.load_ms, rel_tol=0.0, abs_tol=1e-12
                        ):
                            self.metrics.timely_prefetches += 1
                        else:
                            self.metrics.late_prefetches += 1

        stall_ms = 0.0
        for key in sorted(actual_keys):
            entry = self.cache.get(key)
            if entry is not None and math.isclose(
                entry.loaded_ms, self.load_ms, rel_tol=0.0, abs_tol=1e-12
            ):
                self.metrics.hbm_hits += 1
            else:
                self.metrics.demand_misses += 1
                entry = self._ensure_entry(
                    key,
                    current_layer=layer,
                    previous_group=previous_group,
                    next_event=event_id + 1,
                )
                remaining = max(0.0, self.load_ms - entry.loaded_ms)
                stall_ms += remaining
                self.metrics.demand_transfer_ms += remaining
                entry.loaded_ms = self.load_ms
            if entry.prefetch_bytes_ms > 0.0:
                entry.prefetch_used = True
            entry.last_access = self.clock
            entry.prefetch_target_event = None

        self.metrics.exposed_stall_ms += stall_ms
        self.pending = None
        return stall_ms

    def predict_and_prefetch(
        self,
        *,
        target_layer: int,
        ranked_experts: list[int],
        actual_next: set[int],
        previous_group,
        target_event: int,
    ) -> None:
        if not self.enable_prediction:
            self.pending = None
            return
        candidates = {
            (target_layer, expert)
            for expert in ranked_experts[: self.candidate_budget]
        }
        self.metrics.eligible_requests += len(actual_next)
        self.metrics.predicted_requests += len(candidates)
        issue_state: dict[tuple[int, int], str] = {}

        if not self.enable_prefetch:
            self.pending = PredictionRecord(candidates, issue_state)
            return

        needing_transfer: list[tuple[int, int]] = []
        for key in sorted(candidates, key=lambda value: ranked_experts.index(value[1])):
            entry = self.cache.get(key)
            if entry is not None and math.isclose(
                entry.loaded_ms, self.load_ms, rel_tol=0.0, abs_tol=1e-12
            ):
                issue_state[key] = "resident"
            else:
                issue_state[key] = "issued"
                needing_transfer.append(key)

        if needing_transfer:
            progress_per_candidate = self.overlap_ms / len(needing_transfer)
            for key in needing_transfer:
                entry = self._ensure_entry(
                    key,
                    current_layer=target_layer - 1,
                    previous_group=previous_group,
                    next_event=target_event,
                )
                progress = min(
                    self.load_ms - entry.loaded_ms,
                    progress_per_candidate,
                )
                if progress > 0.0:
                    self.metrics.prefetches_issued += 1
                    self.metrics.prefetch_transfer_ms += progress
                    entry.loaded_ms += progress
                    entry.prefetch_bytes_ms += progress
                    entry.prefetch_target_event = target_event
                    entry.last_access = self.clock

        self.pending = PredictionRecord(candidates, issue_state)


def evaluate(config: dict) -> tuple[list[dict], dict]:
    trace = ROOT / config["trace"]
    profile_tokens = int(config["profile_tokens"])
    evaluation_tokens = int(config["evaluation_tokens"])
    layers = int(config["layers"])
    experts = int(config["experts_per_layer"])
    top_k = int(config["top_k"])
    candidate_budget = int(config.get("candidate_budget", top_k))
    routing_batch_size = int(config.get("routing_batch_size", 1))
    effective_candidate_budget = min(experts, candidate_budget * routing_batch_size)
    capacity = int(config.get("hbm_capacity_experts", 16))
    overlap_multiplier = float(config.get("overlap_multiplier", 1.0))
    adaptive_beta = float(config.get("adaptive_decay_beta", 0.99))

    total_load_ms = float(config["measured_offchip_expert_load_ms"])
    total_execution_ms = float(config["measured_offchip_expert_execution_ms"])
    decoder_ms = float(config["measured_offchip_decoder_ms"])
    non_expert_ms = decoder_ms - total_load_ms - total_execution_ms
    unique_requests = 0
    calibration_steps = 0
    for group, _ in iter_evaluation_groups(
        trace, profile_tokens, evaluation_tokens, routing_batch_size
    ):
        for layer in range(layers):
            unique_requests += len(
                {
                    token[route][layer]
                    for token in group
                    for route in range(top_k)
                }
            )
        calibration_steps += 1
    mean_unique_experts_per_layer = unique_requests / (calibration_steps * layers)
    load_per_expert_ms = total_load_ms / (layers * mean_unique_experts_per_layer)
    execution_per_layer_ms = total_execution_ms / layers
    overlap_ms = execution_per_layer_ms * overlap_multiplier

    cct, layer_frequency = build_profile(trace, profile_tokens, layers, experts)
    adaptive_ensemble = AdaptiveEnsemble(layers, experts, adaptive_beta)
    simulators = [
        CacheSimulator(
            name=name,
            predictor=predictor,
            enable_prefetch=enable_prefetch,
            eviction_policy=eviction,
            capacity=capacity,
            load_ms=load_per_expert_ms,
            overlap_ms=overlap_ms,
            candidate_budget=effective_candidate_budget,
            layers=layers,
            layer_frequency=layer_frequency,
        )
        for name, predictor, enable_prefetch, eviction in SCENARIOS
    ]

    evaluated_routes = 0
    event_id = 0
    for group, previous_group in iter_evaluation_groups(
        trace, profile_tokens, evaluation_tokens, routing_batch_size
    ):
        token_stalls = {simulator.name: 0.0 for simulator in simulators}
        for layer in range(layers):
            event_id += 1
            actual = {
                token[route][layer]
                for token in group
                for route in range(top_k)
            }
            for simulator in simulators:
                token_stalls[simulator.name] += simulator.demand(
                    layer=layer,
                    actual_experts=actual,
                    previous_group=previous_group,
                    event_id=event_id,
                )
            if layer + 1 < layers:
                aggregate_scores = [0.0 for _ in range(experts)]
                for token in group:
                    scores = cct_scores(cct, layer + 1, token, layer_frequency, experts)
                    for expert, score in enumerate(scores):
                        aggregate_scores[expert] += score
                cct_ranking = top_indices(
                    aggregate_scores, effective_candidate_budget
                )
                actual_next = {
                    token[route][layer + 1]
                    for token in group
                    for route in range(top_k)
                }
                adaptive_ranking = adaptive_ensemble.rank_and_update(
                    target_layer=layer + 1,
                    cct_values=aggregate_scores,
                    previous_group=previous_group,
                    actual_next=actual_next,
                    budget=effective_candidate_budget,
                )
                for simulator in simulators:
                    ranking = (
                        adaptive_ranking
                        if simulator.predictor == "adaptive_ht_cct"
                        else cct_ranking
                    )
                    simulator.predict_and_prefetch(
                        target_layer=layer + 1,
                        ranked_experts=ranking,
                        actual_next=actual_next,
                        previous_group=previous_group,
                        target_event=event_id + 1,
                    )
        for simulator in simulators:
            latency = non_expert_ms + total_execution_ms + token_stalls[simulator.name]
            simulator.metrics.add_token(latency)
        evaluated_routes += len(group)

    if evaluated_routes != evaluation_tokens:
        raise RuntimeError(
            f"Evaluated {evaluated_routes:,} routes, expected {evaluation_tokens:,}"
        )
    for simulator in simulators:
        simulator.flush()

    compute_dynamic_j, offload_dynamic_j = measured_dynamic_energy_j(config)
    expert_size_mib = (
        load_per_expert_ms / 1000.0 * OFFLOAD_BANDWIDTH_BYTES_S / 2**20
    )
    rows: list[dict] = []
    for simulator in simulators:
        metrics = simulator.metrics
        mean = metrics.token_latency_sum_ms / metrics.tokens
        variance = max(
            0.0,
            metrics.token_latency_square_sum_ms / metrics.tokens - mean * mean,
        )
        total_transfer_ms = metrics.demand_transfer_ms + metrics.prefetch_transfer_ms
        rows.append(
            {
                "scenario": simulator.name,
                "predictor": simulator.predictor,
                "prediction_enabled": simulator.enable_prediction,
                "prefetch_enabled": simulator.enable_prefetch,
                "eviction_policy": simulator.eviction_policy,
                "tokens": metrics.tokens,
                "routing_batch_size": routing_batch_size,
                "routing_tokens": metrics.tokens * routing_batch_size,
                "hbm_capacity_experts": capacity,
                "hbm_capacity_mib": capacity * expert_size_mib,
                "prediction_precision": (
                    metrics.correct_predictions / metrics.predicted_requests
                    if metrics.predicted_requests else 0.0
                ),
                "prediction_recall": (
                    metrics.correct_predictions / metrics.eligible_requests
                    if metrics.eligible_requests else 0.0
                ),
                "prefetch_coverage": (
                    (metrics.timely_prefetches + metrics.late_prefetches)
                    / metrics.eligible_requests
                    if metrics.eligible_requests else 0.0
                ),
                "timely_prefetch_rate": (
                    metrics.timely_prefetches
                    / (metrics.timely_prefetches + metrics.late_prefetches)
                    if metrics.timely_prefetches + metrics.late_prefetches else 0.0
                ),
                "late_prefetch_rate": (
                    metrics.late_prefetches
                    / (metrics.timely_prefetches + metrics.late_prefetches)
                    if metrics.timely_prefetches + metrics.late_prefetches else 0.0
                ),
                "unused_prefetch_traffic_rate": (
                    metrics.unused_prefetch_ms / metrics.prefetch_transfer_ms
                    if metrics.prefetch_transfer_ms else 0.0
                ),
                "unused_prefetch_rate": (
                    metrics.prefetch_evictions / metrics.prefetches_issued
                    if metrics.prefetches_issued else 0.0
                ),
                "hbm_hit_rate": metrics.hbm_hits / metrics.demand_requests,
                "exposed_stall_ms_per_token": metrics.exposed_stall_ms / metrics.tokens,
                "expert_load_latency_ms_per_token": metrics.exposed_stall_ms / metrics.tokens,
                "mean_token_latency_ms": mean,
                "normalized_latency_ms_per_sequence": mean / routing_batch_size,
                "stdev_token_latency_ms": math.sqrt(variance),
                "demand_transfer_experts_per_token": (
                    metrics.demand_transfer_ms / load_per_expert_ms / metrics.tokens
                ),
                "prefetch_transfer_experts_per_token": (
                    metrics.prefetch_transfer_ms / load_per_expert_ms / metrics.tokens
                ),
                "unused_prefetch_experts_per_token": (
                    metrics.unused_prefetch_ms / load_per_expert_ms / metrics.tokens
                ),
                "total_transfer_experts_per_token": (
                    total_transfer_ms / load_per_expert_ms / metrics.tokens
                ),
                "evictions_per_token": metrics.evictions / metrics.tokens,
                "prefetch_evictions_per_token": metrics.prefetch_evictions / metrics.tokens,
                "_transfer_ms": total_transfer_ms,
                "_raw": asdict(metrics),
            }
        )

    baseline = next(row for row in rows if row["scenario"] == "baseline_lru")
    baseline_transfer_ms = baseline["_transfer_ms"]
    baseline_latency = baseline["mean_token_latency_ms"]
    baseline_transfer_scale = baseline_transfer_ms / (calibration_steps * total_load_ms)
    baseline_dynamic_j = compute_dynamic_j + offload_dynamic_j * baseline_transfer_scale
    baseline_total_350_j = (
        baseline_dynamic_j
        + 350.0 * baseline["exposed_stall_ms_per_token"] / 1000.0
    )
    for row in rows:
        transfer_scale = row["_transfer_ms"] / (calibration_steps * total_load_ms)
        dynamic_j = compute_dynamic_j + offload_dynamic_j * transfer_scale
        total_350_j = dynamic_j + 350.0 * row["exposed_stall_ms_per_token"] / 1000.0
        row["speedup_vs_baseline"] = baseline_latency / row["mean_token_latency_ms"]
        row["latency_reduction_pct"] = (
            100.0 * (baseline_latency - row["mean_token_latency_ms"]) / baseline_latency
        )
        row["additional_traffic_pct_vs_baseline"] = (
            100.0 * (row["_transfer_ms"] - baseline_transfer_ms) / baseline_transfer_ms
        )
        row["dynamic_energy_j"] = dynamic_j
        row["dynamic_energy_change_pct_vs_baseline"] = (
            100.0 * (dynamic_j - baseline_dynamic_j) / baseline_dynamic_j
        )
        row["total_energy_350w_j"] = total_350_j
        row["total_energy_saving_350w_pct"] = (
            100.0 * (baseline_total_350_j - total_350_j) / baseline_total_350_j
        )
        del row["_transfer_ms"]

    metadata = {
        "trace": config["trace"],
        "trace_sha256": trace_prefix_sha256(trace),
        "profile_tokens": profile_tokens,
        "evaluation_tokens": evaluation_tokens,
        "layers": layers,
        "experts_per_layer": experts,
        "top_k": top_k,
        "candidate_budget": candidate_budget,
        "effective_batch_candidate_budget": effective_candidate_budget,
        "routing_batch_size": routing_batch_size,
        "decode_steps": calibration_steps,
        "mean_unique_experts_per_layer": mean_unique_experts_per_layer,
        "hbm_capacity_experts": capacity,
        "hbm_capacity_mib": capacity * expert_size_mib,
        "lookahead_layers": 1,
        "overlap_multiplier": overlap_multiplier,
        "overlap_ms_per_layer": overlap_ms,
        "expert_load_ms": load_per_expert_ms,
        "expert_size_mib": expert_size_mib,
        "adaptive_ensemble": adaptive_ensemble.metadata(),
        "metric_definitions": {
            "prefetch_coverage": "eligible demand requests receiving transferred prefetch bytes",
            "timely": "correct issued prefetch fully resident before demand",
            "late": "correct issued prefetch incomplete or evicted when demanded",
            "unused_traffic": "prefetched bytes discarded without any later demand use",
            "hbm_hit": "expert fully resident before its demand access",
            "CAR": "correlation-aware retention using immediate CCT target, previous-token history, profile frequency, layer proximity, and LRU tie-breaking",
        },
    }
    return rows, metadata


def csv_rows(rows: list[dict]) -> list[dict]:
    return [{key: value for key, value in row.items() if key != "_raw"} for row in rows]


def write_svg(rows: list[dict], destination: Path) -> None:
    width, height = 1100, 650
    left, plot_width = 235, 720
    maximum = max(row["mean_token_latency_ms"] for row in rows) * 1.08
    colors = {
        "baseline_lru": "#e45756",
        "prediction_only": "#9aa4b2",
        "prefetch_lru": "#2f6fed",
        "complete_car": "#13a89e",
        "adaptive_prediction_only": "#c0c6d0",
        "adaptive_prefetch_lru": "#7b61d1",
        "adaptive_complete_car": "#087f5b",
    }
    labels = {
        "baseline_lru": "On demand + LRU",
        "prediction_only": "Prediction only",
        "prefetch_lru": "CCT prefetch + LRU",
        "complete_car": "CCT prefetch + CAR",
        "adaptive_prediction_only": "Adaptive prediction only",
        "adaptive_prefetch_lru": "Adaptive prefetch + LRU",
        "adaptive_complete_car": "Adaptive prefetch + CAR",
    }
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#17233c}.title{font-size:25px;font-weight:700}.sub{font-size:14px;fill:#596579}.label{font-size:15px;font-weight:600}.value{font-size:14px;font-weight:700}.tick{font-size:12px;fill:#6b7280}</style>',
        '<text x="32" y="42" class="title">Component ablation: CCT and adaptive HT–CCT prefetch</text>',
        '<text x="32" y="68" class="sub">Held-out Mixtral routing · identical measured calibration, candidate budget, overlap, and HBM capacity</text>',
    ]
    for tick in range(0, 61, 10):
        x = left + plot_width * tick / maximum
        svg.append(f'<line x1="{x:.1f}" y1="90" x2="{x:.1f}" y2="515" stroke="#e2e7ef"/>')
        svg.append(f'<text x="{x:.1f}" y="540" text-anchor="middle" class="tick">{tick}</text>')
    for index, row in enumerate(rows):
        y = 102 + index * 58
        value = row["mean_token_latency_ms"]
        bar_width = plot_width * value / maximum
        scenario = row["scenario"]
        svg.append(f'<text x="{left-14}" y="{y+25}" text-anchor="end" class="label">{labels[scenario]}</text>')
        svg.append(f'<rect x="{left}" y="{y}" width="{bar_width:.1f}" height="34" rx="5" fill="{colors[scenario]}"/>')
        svg.append(f'<text x="{left+bar_width+9:.1f}" y="{y+23}" class="value">{value:.3f} ms · {row["speedup_vs_baseline"]:.3f}×</text>')
    svg.append(f'<text x="{left+plot_width/2}" y="570" text-anchor="middle" class="sub">Mean modeled token latency (ms)</text>')
    cct = next(row for row in rows if row["scenario"] == "complete_car")
    adaptive = next(row for row in rows if row["scenario"] == "adaptive_complete_car")
    svg.append(f'<text x="32" y="615" class="sub">Complete CCT: {cct["mean_token_latency_ms"]:.3f} ms · Complete adaptive ensemble: {adaptive["mean_token_latency_ms"]:.3f} ms · both use CAR and identical resources.</text>')
    svg.append('</svg>')
    destination.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/final_prefetch_single_gpu_batch1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/final_endsem/hbm_policies"),
    )
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    output = args.output if args.output.is_absolute() else ROOT / args.output
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)

    rows, metadata = evaluate(config)
    clean_rows = csv_rows(rows)
    with (output / "hbm_policy_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(clean_rows[0]))
        writer.writeheader()
        writer.writerows(clean_rows)
    (output / "hbm_policy_results.json").write_text(
        json.dumps({"metadata": metadata, "scenarios": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    write_svg(rows, output / "hbm_policy_comparison.svg")

    for row in rows:
        print(
            f'{row["scenario"]:16s} '
            f'latency={row["mean_token_latency_ms"]:.6f} ms '
            f'hbm_hit={100*row["hbm_hit_rate"]:.3f}% '
            f'timely={100*row["timely_prefetch_rate"]:.3f}% '
            f'late={100*row["late_prefetch_rate"]:.3f}% '
            f'traffic={row["additional_traffic_pct_vs_baseline"]:+.3f}%'
        )
    print(f"wrote={output}")


if __name__ == "__main__":
    main()
