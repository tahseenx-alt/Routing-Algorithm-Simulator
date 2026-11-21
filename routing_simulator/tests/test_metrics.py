"""Tests for Run All, the comparison table and the best-algorithm rule."""

from __future__ import annotations

import csv

import pytest

from algorithms.base import RoutingResult
from core.config import SimConfig
from core.graph import Network
from core.metrics import AlgorithmMetrics, eligibility, export_csv, run_all, select_best
from core.storage import load_sample

from conftest import all_test_networks

FAST = 2  # timing runs: keep the tests quick


def test_small_k100_flooding_never_best_and_dijkstra_equals_bf() -> None:
    network = load_sample("small")
    for p in (0, 1, 3):
        for split in (False, True):
            config = SimConfig(processing_delay_ms=p, k_packets=100, split_horizon=split)
            for s in network.router_ids():
                for d in network.router_ids():
                    if s == d:
                        continue
                    comp = run_all(network, s, d, config, timing_runs=FAST)
                    assert comp.best is not None and comp.best.name != "Flooding"
                    dij, bf = comp.row("Dijkstra"), comp.row("Bellman-Ford")
                    assert dij.result.delivery_time_ms == bf.result.delivery_time_ms


@pytest.mark.parametrize("name,network", all_test_networks())
def test_dijkstra_bf_flooding_same_delivery_everywhere(name, network) -> None:
    ids = network.router_ids()
    comp = run_all(network, ids[0], ids[-1], SimConfig(), timing_runs=1)
    times = {comp.row(n).result.delivery_time_ms for n in ("Dijkstra", "Bellman-Ford", "Flooding")}
    assert len(times) == 1


def test_comparison_contents_and_explanation(small) -> None:
    comp = run_all(small, "A", "E", SimConfig(k_packets=100), timing_runs=FAST)
    assert [r.name for r in comp.rows] == ["Dijkstra", "Bellman-Ford", "Hop Count", "Flooding", "Random"]
    for row in comp.rows:
        assert row.computation_us > 0
        assert row.total_messages == row.result.routing_overhead + 100 * row.result.per_packet_transmissions
    assert comp.row("Bellman-Ford").result.rounds is not None
    assert comp.row("Random").result.success_rate is not None
    text = comp.explanation
    assert f"{comp.best.name} is best for this network" in text
    assert "Dijkstra and Bellman-Ford find the same shortest-path cost" in text
    assert "Flooding matches the fastest delivery time but wastes bandwidth" in text
    assert "For 100 packets" in text


def _row(name, delivery, overhead, per_packet, micros, success=None, path=("A", "B")) -> AlgorithmMetrics:
    result = RoutingResult(name, "A", "B", path=list(path) if delivery is not None else None,
                           delivery_time_ms=delivery, routing_overhead=overhead,
                           per_packet_transmissions=per_packet, success_rate=success)
    ok, status = eligibility(result)
    return AlgorithmMetrics(name, result, micros, 100, ok, status)


def test_best_rule_order() -> None:
    # Rule 2: lowest delivery time wins even with many messages.
    rows = [_row("X", 9, 0, 50, 1), _row("Y", 8, 500, 5, 99)]
    assert select_best(rows).name == "Y"
    # Rule 3: tie on time -> fewest total messages.
    rows = [_row("X", 9, 40, 3, 50), _row("Y", 9, 72, 3, 1)]
    assert select_best(rows).name == "X"
    # Rule 4: still tied -> lowest computation time.
    rows = [_row("X", 9, 40, 3, 50), _row("Y", 9, 40, 3, 1)]
    assert select_best(rows).name == "Y"


def test_unreliable_random_and_undelivered_are_not_eligible() -> None:
    rows = [_row("Random", 2, 0, 3, 1, success=0.9), _row("Lost", None, 0, 0, 1),
            _row("Dijkstra", 9, 40, 3, 5)]
    assert rows[0].status == "not eligible (unreliable)"
    assert rows[1].status == "not delivered"
    assert select_best(rows).name == "Dijkstra"
    assert select_best([_row("Random", 2, 0, 3, 1, success=1.0)]).name == "Random"
    assert select_best(rows[:2]) is None


def test_no_algorithm_delivers() -> None:
    net = Network()
    net.add_router(0, 0, "A")
    net.add_router(0, 0, "B")
    comp = run_all(net, "A", "B", SimConfig(), timing_runs=1)
    assert comp.best is None and "No algorithm delivered" in comp.explanation


def test_same_source_and_destination(small) -> None:
    comp = run_all(small, "B", "B", SimConfig(), timing_runs=1)
    assert comp.best is not None
    assert "Source and destination are the same" in comp.explanation


def test_csv_export(tmp_path, small) -> None:
    comp = run_all(small, "A", "E", SimConfig(), timing_runs=1)
    path = tmp_path / "results.csv"
    export_csv(comp, path)
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert [r["Algorithm"] for r in rows] == [r.name for r in comp.rows]
    assert rows[0]["Delivery time (ms)"] == comp.rows[0].delivery_text
