"""Link failure recovery and the count-to-infinity demos."""

from __future__ import annotations

import math

import networkx as nx
import pytest

from algorithms.bellman_ford import BellmanFord
from algorithms.dijkstra import count_lsa_flood
from algorithms.distance_vector import (
    HOP_INFINITY,
    converge,
    count_to_infinity_demo,
    reconverge,
)
from core.config import SimConfig
from core.metrics import failure_test
from core.storage import load_sample

from conftest import to_networkx


# ------------------------------------------------------ count to infinity
def b_cost_to_c(demo) -> list[float]:
    """B's cost to C in every row after the failure."""
    return [row["B"] for label, row in demo.rows if label != "before"]


def test_linear_without_split_horizon_counts_to_16() -> None:
    net = load_sample("count_to_infinity")
    demo = count_to_infinity_demo(net, ("B", "C"), "C", ["A", "B"], False, SimConfig())
    costs = b_cost_to_c(demo)
    assert costs[-1] == HOP_INFINITY  # B's cost to C reaches 16
    assert demo.reached_infinity
    assert any(2 < c < HOP_INFINITY for c in costs)  # wrong finite costs on the way up
    assert costs == sorted(costs)  # it only ever rises: "counting"


def test_linear_with_split_horizon_converges_fast_and_correctly() -> None:
    net = load_sample("count_to_infinity")
    without = count_to_infinity_demo(net, ("B", "C"), "C", ["A", "B"], False, SimConfig())
    with_sh = count_to_infinity_demo(net, ("B", "C"), "C", ["A", "B"], True, SimConfig())
    assert with_sh.rounds < without.rounds
    assert not with_sh.reached_infinity
    # never a wrong finite cost: B is unreachable at once, A keeps its old 2 at most
    for label, row in with_sh.rows[1:]:
        assert row["B"] == HOP_INFINITY
        assert row["A"] in (2, HOP_INFINITY)
    assert with_sh.rows[-1][1] == {"A": HOP_INFINITY, "B": HOP_INFINITY}


def test_triangle_with_split_horizon_still_counts_to_16() -> None:
    net = load_sample("triangle_loop")
    demo = count_to_infinity_demo(net, ("C", "D"), "D", ["A", "B", "C"], True, SimConfig())
    assert demo.reached_infinity
    assert all(cost == HOP_INFINITY for cost in demo.rows[-1][1].values())
    assert demo.rounds > 5


def test_demo_does_not_modify_the_network() -> None:
    net = load_sample("count_to_infinity")
    count_to_infinity_demo(net, ("B", "C"), "C", ["A", "B"], False, SimConfig())
    assert net.get_link("B", "C").active


def test_delay_metric_count_to_infinity_ends_at_finite_infinity() -> None:
    net = load_sample("count_to_infinity")
    demo = count_to_infinity_demo(net, ("B", "C"), "C", ["A", "B"], False, SimConfig(),
                                  metric="delay")
    assert demo.converged and demo.infinity == 2 * (1 + 1) + 1
    assert demo.rows[-1][1]["B"] == demo.infinity


def test_max_rounds_cap_reports_not_converged() -> None:
    net = load_sample("count_to_infinity")
    demo = count_to_infinity_demo(net, ("B", "C"), "C", ["A", "B"], False,
                                  SimConfig(max_rounds=3))
    assert not demo.converged


# ------------------------------------------------- warm-start reconvergence
@pytest.mark.parametrize("split_horizon", [False, True])
@pytest.mark.parametrize("sample", ["small", "medium", "large"])
def test_reconverge_matches_networkx_after_failure(sample, split_horizon) -> None:
    net = load_sample(sample)
    config = SimConfig(split_horizon=split_horizon, processing_delay_ms=1)
    for link in list(net.sorted_links())[:4]:
        state = converge(net, "delay", split_horizon, config).state
        after = net.copy()
        after.fail_link(link.a, link.b)
        run = reconverge(after, state, config)
        assert run.converged
        graph = to_networkx(after, 1)
        for source in after.router_ids():
            expected = nx.single_source_dijkstra_path_length(graph, source, weight="cost")
            got = {y: c for y, c in run.state.dist[source].items() if c < run.state.infinity}
            assert got == pytest.approx(expected)


# -------------------------------------------------------------- failure test
def test_failure_test_reports_every_algorithm(small) -> None:
    config = SimConfig(update_period_ms=5)
    rows = {r.name: r for r in failure_test(small, ("C", "E"), "A", "E", config)}
    assert set(rows) == {"Dijkstra", "Bellman-Ford", "Hop Count", "Flooding", "Random"}
    dij = rows["Dijkstra"]
    after = small.copy()
    after.fail_link("C", "E")
    assert dij.extra_messages == count_lsa_flood(after, "C") + count_lsa_flood(after, "E")
    assert dij.rounds == 1 and dij.reconvergence_delay_ms == 5
    assert dij.old_path == ["A", "C", "E"] and "C" != dij.new_path[-2]
    assert dij.new_time_ms > dij.old_time_ms
    bf = rows["Bellman-Ford"]
    assert bf.new_time_ms == dij.new_time_ms  # same shortest path cost again
    assert bf.extra_messages > 0 and bf.rounds >= 1
    assert bf.reconvergence_delay_ms == max(1, bf.rounds) * 5
    assert rows["Flooding"].extra_messages == 0 and rows["Flooding"].new_time_ms == dij.new_time_ms
    assert small.get_link("C", "E").active  # original network untouched
    assert rows["Dijkstra"].as_row()["New path"].startswith("A→")


def test_failure_makes_destination_unreachable_cleanly() -> None:
    net = load_sample("count_to_infinity")
    rows = failure_test(net, ("B", "C"), "A", "C", SimConfig())
    for row in rows:
        assert row.new_path is None and row.new_time_ms is None
        assert row.as_row()["New time (ms)"] == "Not delivered"


def test_failure_test_rejects_missing_or_failed_link(small) -> None:
    with pytest.raises(ValueError):
        failure_test(small, ("A", "E"), "A", "E", SimConfig())
    small.fail_link("A", "B")
    with pytest.raises(ValueError):
        failure_test(small, ("A", "B"), "A", "E", SimConfig())


def test_bellman_ford_after_failure_on_failed_network(small) -> None:
    """Computing from scratch on a network with a failed link also works."""
    small.fail_link("C", "E")
    result = BellmanFord().compute(small, "A", "E", SimConfig())
    assert result.delivered and "C" not in result.path[-2:-1]
    assert all(v != math.inf for v in result.distances.values())
