"""Check our routing algorithms against NetworkX (used only as a reference)."""

from __future__ import annotations

import math

import networkx as nx
import pytest

from algorithms.base import path_delivery_time
from algorithms.bellman_ford import BellmanFord
from algorithms.dijkstra import Dijkstra, count_lsa_flood, lsa_overhead
from algorithms.distance_vector import converge, infinity_for
from algorithms.flooding import Flooding
from algorithms.hop_count import HopCount
from algorithms.random_routing import RandomRouting
from core.config import SimConfig
from core.graph import Network
from core.storage import load_sample

from conftest import all_test_networks, to_networkx

NETWORKS = all_test_networks()


def finite(distances: dict[str, float]) -> dict[str, float]:
    """Drop unreachable routers so we can compare with NetworkX dicts."""
    return {k: v for k, v in distances.items() if v != math.inf}


# ---------------------------------------------------------------- Dijkstra
@pytest.mark.parametrize("p", [0, 2])
@pytest.mark.parametrize("name,network", NETWORKS)
def test_dijkstra_matches_networkx(name, network, p) -> None:
    graph = to_networkx(network, p)
    config = SimConfig(processing_delay_ms=p)
    for source in network.router_ids():
        dest = network.router_ids()[-1]
        result = Dijkstra().compute(network, source, dest, config)
        expected = nx.single_source_dijkstra_path_length(graph, source, weight="cost")
        assert finite(result.distances) == pytest.approx(expected)
        # delivery time = sum of link delays + p x hops = Dijkstra distance
        assert result.delivery_time_ms == pytest.approx(expected[dest])
        assert result.delivery_time_ms == path_delivery_time(network, result.path, p)


def test_dijkstra_routing_table_and_steps(small) -> None:
    result = Dijkstra().compute(small, "A", "E", SimConfig(processing_delay_ms=1))
    # A-C (2+1) then C-E (8+1) = 12; A-B-D-E = 5+6+4 = 15
    assert result.path == ["A", "C", "E"] and result.delivery_time_ms == 12
    assert result.routing_table["E"] == ("C", 12)
    assert result.routing_table["B"] == ("B", 5)  # A-B = 4+1 ties A-C-B = 3+2; first found wins
    assert result.per_packet_transmissions == 2
    assert any(s.description.startswith("Visit") for s in result.steps)
    assert any(s.description.startswith("Relax") for s in result.steps)


def test_lsa_flood_count_formula(small) -> None:
    # Origin sends to all its neighbours, every other router to all but one:
    # transmissions = 2*links - (routers - 1) for each LSA.
    links, routers = len(small.links), len(small.routers)
    for r in small.router_ids():
        assert count_lsa_flood(small, r) == 2 * links - (routers - 1)
    assert lsa_overhead(small, "A") == routers * (2 * links - (routers - 1))


# ------------------------------------------------------------ Bellman-Ford
@pytest.mark.parametrize("p", [0, 2])
@pytest.mark.parametrize("split_horizon", [False, True])
@pytest.mark.parametrize("name,network", NETWORKS)
def test_bellman_ford_matches_networkx(name, network, p, split_horizon) -> None:
    graph = to_networkx(network, p)
    config = SimConfig(processing_delay_ms=p, split_horizon=split_horizon)
    source = network.router_ids()[0]
    dest = network.router_ids()[-1]
    result = BellmanFord().compute(network, source, dest, config)
    expected = nx.single_source_bellman_ford_path_length(graph, source, weight="cost")
    assert result.converged
    assert finite(result.distances) == pytest.approx(expected)
    dijkstra = Dijkstra().compute(network, source, dest, config)
    assert result.delivery_time_ms == pytest.approx(dijkstra.delivery_time_ms)


@pytest.mark.parametrize("p", [0, 2])
def test_infinity_bug_regression_large_sample(p) -> None:
    """Links over 16 ms must never make a real route 'unreachable'."""
    network = load_sample("large")
    assert max(l.delay_ms for l in network.links.values()) > 16
    assert infinity_for(network, "delay", p) > 16
    graph = to_networkx(network, p)
    for source in network.router_ids():
        result = BellmanFord().compute(network, source, "L", SimConfig(processing_delay_ms=p))
        expected = nx.single_source_bellman_ford_path_length(graph, source, weight="cost")
        assert finite(result.distances) == pytest.approx(expected)
        assert all(v != math.inf for v in result.distances.values())  # nothing unreachable


def test_bellman_ford_counts_messages_and_rounds(small) -> None:
    run = converge(small, "delay", False, SimConfig())
    # every round every router sends one vector per active link end
    assert run.messages == run.rounds_run * 2 * len(small.links)
    assert run.rounds_run == run.rounds + 1  # plus the final no-change round
    result = BellmanFord().compute(small, "A", "E", SimConfig())
    assert result.routing_overhead == run.messages and result.rounds == run.rounds
    assert any("updates cost to" in line for line in result.log)


# --------------------------------------------------------------- Hop count
@pytest.mark.parametrize("name,network", NETWORKS)
def test_hop_count_matches_networkx(name, network) -> None:
    graph = to_networkx(network, 0)
    source = network.router_ids()[0]
    dest = network.router_ids()[-1]
    result = HopCount().compute(network, source, dest, SimConfig())
    expected = nx.single_source_shortest_path_length(graph, source)
    assert finite(result.distances) == expected
    if dest in expected:
        assert result.hop_count == expected[dest]
    # RIP's own distance-vector tables agree on hop counts
    dv_hops = {y: c for y, c in result.dv_state.dist[source].items() if c < 16}
    assert dv_hops == expected


def test_hop_count_prefers_fewer_hops_then_lower_delay() -> None:
    net = Network()
    for rid in "ABCD":
        net.add_router(0, 0, rid)
    net.add_link("A", "D", 50)  # 1 hop but slow
    net.add_link("A", "B", 1)
    net.add_link("B", "D", 1)   # 2 hops, fast
    config = SimConfig(processing_delay_ms=1)
    hop = HopCount().compute(net, "A", "D", config)
    dij = Dijkstra().compute(net, "A", "D", config)
    assert hop.path == ["A", "D"] and hop.delivery_time_ms == 51
    assert dij.path == ["A", "B", "D"] and dij.delivery_time_ms == 4
    net.add_link("A", "C", 1)
    net.add_link("C", "D", 9)
    net.remove_link("A", "D")
    assert HopCount().compute(net, "A", "D", config).path == ["A", "B", "D"]  # tie -> lower delay


# --------------------------------------------------------------- edge cases
@pytest.mark.parametrize("algorithm", [Dijkstra(), BellmanFord(), HopCount()])
def test_same_source_and_destination(small, algorithm) -> None:
    result = algorithm.compute(small, "B", "B", SimConfig())
    assert result.path == ["B"] and result.delivery_time_ms == 0
    assert result.per_packet_transmissions == 0
    assert result.note == "Source and destination are the same"


@pytest.mark.parametrize("algorithm", [Dijkstra(), BellmanFord(), HopCount()])
def test_disconnected_network(algorithm) -> None:
    net = Network()
    for rid in "ABCD":
        net.add_router(0, 0, rid)
    net.add_link("A", "B", 3)
    net.add_link("C", "D", 3)
    result = algorithm.compute(net, "A", "D", SimConfig())
    assert result.path is None and result.delivery_time_ms is None
    assert not result.delivered and result.distances["D"] == math.inf
    assert "No path" in result.note


# ---------------------------------------------------------------- Flooding
@pytest.mark.parametrize("p", [0, 1, 3])
@pytest.mark.parametrize("name,network", NETWORKS)
def test_flooding_delivery_equals_dijkstra(name, network, p) -> None:
    config = SimConfig(processing_delay_ms=p)
    ids = network.router_ids()
    for source, dest in [(ids[0], ids[-1]), (ids[-1], ids[len(ids) // 2])]:
        flood = Flooding().compute(network, source, dest, config)
        best = Dijkstra().compute(network, source, dest, config)
        assert flood.delivery_time_ms == pytest.approx(best.delivery_time_ms)
        # a first copy never beats the shortest path (it can be later only when the
        # shortest path runs through the destination, which does not forward)
        for router, t in finite(flood.distances).items():
            assert t >= best.distances[router] - 1e-9
        assert flood.routing_overhead == 0


def test_flooding_counts_every_copy(small) -> None:
    result = Flooding().compute(small, "A", "E", SimConfig())
    # Every router except E forwards on every link except the incoming one;
    # A (the source) sends on all its links.
    degrees = {r: len(small.neighbor_ids(r)) for r in small.router_ids()}
    expected = degrees["A"] + sum(degrees[r] - 1 for r in "BCD")
    assert result.per_packet_transmissions == expected
    assert any("duplicate" in s.description for s in result.steps)


def test_flooding_uses_event_time_not_hop_order() -> None:
    """A slow 1-hop link loses to a fast 3-hop route; BFS order would get this wrong."""
    net = Network()
    for rid in "ABCD":
        net.add_router(0, 0, rid)
    net.add_link("A", "D", 30)
    net.add_link("A", "B", 1)
    net.add_link("B", "C", 1)
    net.add_link("C", "D", 1)
    result = Flooding().compute(net, "A", "D", SimConfig(processing_delay_ms=1))
    assert result.path == ["A", "B", "C", "D"] and result.delivery_time_ms == 6


# ------------------------------------------------------------ Random routing
def test_random_routing_same_seed_same_result(small) -> None:
    a = RandomRouting().compute(small, "A", "E", SimConfig(seed=5))
    b = RandomRouting().compute(small, "A", "E", SimConfig(seed=5))
    assert (a.path, a.delivery_time_ms, a.success_rate, a.per_packet_transmissions, a.log) == \
           (b.path, b.delivery_time_ms, b.success_rate, b.per_packet_transmissions, b.log)


def test_random_routing_different_seed_different_result() -> None:
    network = load_sample("medium")  # many alternative paths
    logs = {tuple(RandomRouting().compute(network, "A", "H", SimConfig(seed=s)).log)
            for s in range(5)}
    assert len(logs) > 1


def test_random_routing_report(small) -> None:
    result = RandomRouting().compute(small, "A", "E", SimConfig(seed=1))
    assert 0 <= result.success_rate <= 1 and result.routing_overhead == 0
    assert len([l for l in result.log if l.startswith("Trial")]) == 20
    if result.path:
        assert result.path[0] == "A" and result.path[-1] == "E"
        assert result.steps[-1].links == list(zip(result.path, result.path[1:]))


def test_random_routing_ttl_drops_packet() -> None:
    net = Network()  # A - B - C - D in a line, plus a loop B-E-F-B to wander in
    for rid in "ABCDEF":
        net.add_router(0, 0, rid)
    for a, b in ["AB", "BC", "CD", "BE", "EF", "FB"]:
        net.add_link(a, b, 1)
    result = RandomRouting().compute(net, "A", "D", SimConfig(seed=3, random_trials=50))
    assert any("TTL expired" in line for line in result.log)
    assert result.success_rate < 1


# --------------------------------------------------- edge cases, all five
ALL_FIVE = [Dijkstra(), BellmanFord(), HopCount(), Flooding(), RandomRouting()]


@pytest.mark.parametrize("algorithm", ALL_FIVE)
def test_all_algorithms_same_source_destination(small, algorithm) -> None:
    result = algorithm.compute(small, "C", "C", SimConfig())
    assert result.path == ["C"] and result.delivery_time_ms == 0
    assert result.note == "Source and destination are the same"


@pytest.mark.parametrize("algorithm", ALL_FIVE)
def test_all_algorithms_disconnected(algorithm) -> None:
    net = Network()
    for rid in "ABC":
        net.add_router(0, 0, rid)
    net.add_link("A", "B", 2)
    result = algorithm.compute(net, "A", "C", SimConfig())
    assert not result.delivered and result.note


@pytest.mark.parametrize("algorithm", ALL_FIVE)
def test_unknown_router_raises_clear_error(small, algorithm) -> None:
    from algorithms.base import AlgorithmError
    with pytest.raises(AlgorithmError, match="does not exist"):
        algorithm.compute(small, "A", "Z", SimConfig())
