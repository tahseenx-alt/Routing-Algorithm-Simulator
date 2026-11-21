"""Tests for the hop-by-hop packet simulation (core/packet_sim.py)."""

from __future__ import annotations

import pytest

from algorithms import ALGORITHMS
from algorithms.base import path_delivery_time
from algorithms.bellman_ford import BellmanFord
from algorithms.dijkstra import Dijkstra
from algorithms.flooding import Flooding
from algorithms.hop_count import HopCount
from algorithms.random_routing import RandomRouting
from core.config import SimConfig
from core.graph import Network
from core.packet_sim import (
    NOT_DELIVERED,
    LinkFailure,
    PacketSimError,
    simulate_packet,
    simulate_path,
)
from core.storage import load_sample

from conftest import all_test_networks


@pytest.mark.parametrize("p", [0, 1, 3])
@pytest.mark.parametrize("name,network", all_test_networks())
def test_delivery_time_is_sum_of_delays_plus_p_times_hops(name, network, p) -> None:
    config = SimConfig(processing_delay_ms=p)
    ids = network.router_ids()
    result = Dijkstra().compute(network, ids[0], ids[-1], config)
    if not result.delivered:
        return
    packet = simulate_path(network, result.path, config)
    delays = sum(network.get_link(a, b).delay_ms for a, b in zip(result.path, result.path[1:]))
    assert packet.delivered
    assert packet.delivery_time_ms == delays + p * (len(result.path) - 1)
    assert packet.delivery_time_ms == pytest.approx(result.delivery_time_ms)


def test_timeline_text(small) -> None:
    packet = simulate_packet(small, Dijkstra(), "A", "E", SimConfig(processing_delay_ms=1))
    assert packet.timeline == [
        "t=0 ms: A sends packet to C",
        "t=2 ms: packet arrives at C",
        "t=3 ms: C forwards to E",
        "t=11 ms: packet arrives at E",
        "t=12 ms: packet delivered at E",
    ]
    assert [(h.src, h.dst, h.depart_ms, h.arrive_ms) for h in packet.hops] == \
           [("A", "C", 0, 2), ("C", "E", 3, 11)]


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_every_algorithm_delivers_on_sample(algorithm) -> None:
    network = load_sample("medium")
    config = SimConfig(processing_delay_ms=2)
    result = algorithm.compute(network, "A", "H", config)
    packet = simulate_packet(network, algorithm, "A", "H", config, initial=result)
    assert packet.delivered == result.delivered
    if algorithm.name != "Random":
        assert packet.delivery_time_ms == pytest.approx(result.delivery_time_ms)
    else:  # the animated packet follows the first successful trial
        assert packet.delivery_time_ms == path_delivery_time(network, result.path, 2)


# ----------------------------------------------------- failure during transfer
@pytest.mark.parametrize("algorithm", [Dijkstra(), BellmanFord(), HopCount()])
@pytest.mark.parametrize("fail_time", [0, 1, 5])  # before / during / ahead of the packet
def test_failure_rerouting_off_is_not_delivered(small, algorithm, fail_time) -> None:
    config = SimConfig(rerouting=False)
    path = algorithm.compute(small, "A", "E", config).path
    failure = LinkFailure(path[1], path[2], fail_time)  # the 2nd link of the path
    packet = simulate_packet(small, algorithm, "A", "E", config, failure)
    assert not packet.delivered and packet.status == NOT_DELIVERED
    assert any("rerouting is OFF" in line for line in packet.timeline)


@pytest.mark.parametrize("algorithm", [Dijkstra(), BellmanFord(), HopCount()])
@pytest.mark.parametrize("fail_time", [0, 1, 5])
def test_failure_rerouting_on_delivers_later(small, algorithm, fail_time) -> None:
    config = SimConfig(rerouting=True, update_period_ms=5)
    normal = simulate_packet(small, algorithm, "A", "E", config)
    path = algorithm.compute(small, "A", "E", config).path
    failure = LinkFailure(path[1], path[2], fail_time)
    packet = simulate_packet(small, algorithm, "A", "E", config, failure)
    assert packet.delivered and packet.rerouted
    assert packet.reconvergence_delay_ms >= config.update_period_ms
    assert packet.delivery_time_ms >= normal.delivery_time_ms + packet.reconvergence_delay_ms
    assert not any(h.src in path[1:3] and h.dst in path[1:3] and not h.lost
                   for h in packet.hops)  # never crossed the failed link


def test_copy_lost_on_link_versus_stopped_before_link(small) -> None:
    config = SimConfig(rerouting=False, processing_delay_ms=1)
    # Dijkstra path A-C (2 ms) then C-E (8 ms), leaving C at t=3.
    lost = simulate_packet(small, Dijkstra(), "A", "E", config, LinkFailure("C", "E", 6))
    assert lost.hops[-1].lost and lost.hops[-1].fraction == pytest.approx(3 / 8)
    assert any("copy lost" in line for line in lost.timeline)
    stopped = simulate_packet(small, Dijkstra(), "A", "E", config, LinkFailure("C", "E", 1))
    assert any("packet stops at C" in line for line in stopped.timeline)


def test_dijkstra_reconvergence_delay_is_one_update_period(small) -> None:
    config = SimConfig(update_period_ms=7)
    packet = simulate_packet(small, Dijkstra(), "A", "E", config, LinkFailure("C", "E", 0))
    assert packet.reconvergence_delay_ms == 7


def test_distance_vector_delay_is_rounds_times_period(small) -> None:
    config = SimConfig(update_period_ms=5)
    packet = simulate_packet(small, BellmanFord(), "A", "E", config, LinkFailure("C", "E", 0))
    assert packet.reconvergence_delay_ms % 5 == 0 and packet.reconvergence_delay_ms >= 5


def test_failure_not_on_path_changes_nothing(small) -> None:
    config = SimConfig()
    normal = simulate_packet(small, Dijkstra(), "A", "E", config)
    packet = simulate_packet(small, Dijkstra(), "A", "E", config, LinkFailure("B", "D", 0))
    assert packet.delivery_time_ms == normal.delivery_time_ms and not packet.rerouted


def test_flooding_survives_a_failure(small) -> None:
    config = SimConfig(rerouting=False)
    normal = simulate_packet(small, Flooding(), "A", "E", config)
    packet = simulate_packet(small, Flooding(), "A", "E", config, LinkFailure("C", "E", 0))
    assert packet.delivered  # other copies continue, no rerouting needed
    assert packet.delivery_time_ms >= normal.delivery_time_ms
    assert packet.path_taken[-2] != "C"


def test_random_routing_with_failure_never_crashes(small) -> None:
    for t in range(0, 30, 3):
        packet = simulate_packet(small, RandomRouting(), "A", "E", SimConfig(seed=4),
                                 LinkFailure("C", "E", t))
        assert packet.status


def test_destination_unreachable_after_failure() -> None:
    net = load_sample("count_to_infinity")  # A-B-C in a line
    for algorithm in ALGORITHMS:
        packet = simulate_packet(net, algorithm, "A", "C", SimConfig(rerouting=True),
                                 LinkFailure("B", "C", 0))
        assert not packet.delivered and packet.status == NOT_DELIVERED


def test_same_source_destination_and_disconnected() -> None:
    net = Network()
    for rid in "ABC":
        net.add_router(0, 0, rid)
    net.add_link("A", "B", 3)
    for algorithm in ALGORITHMS:
        same = simulate_packet(net, algorithm, "A", "A", SimConfig())
        assert same.delivered and same.delivery_time_ms == 0
        cut = simulate_packet(net, algorithm, "A", "C", SimConfig())
        assert not cut.delivered


def test_invalid_failure_is_a_clear_error(small) -> None:
    with pytest.raises(PacketSimError, match="no link"):
        simulate_packet(small, Dijkstra(), "A", "E", SimConfig(), LinkFailure("A", "E", 1))
    small.fail_link("A", "B")
    with pytest.raises(PacketSimError, match="already failed"):
        simulate_packet(small, Dijkstra(), "A", "E", SimConfig(), LinkFailure("A", "B", 1))
