"""Tests for the network model (core/graph.py) and JSON save/load (core/storage.py)."""

from __future__ import annotations

import json

import pytest

from core.config import ConfigError, SimConfig
from core.graph import MAX_ROUTERS, Network, NetworkError, generate_random_network
from core.storage import (
    StorageError,
    list_samples,
    load_network,
    load_sample,
    network_from_dict,
    save_network,
)

from conftest import all_test_networks


# ----------------------------------------------------------------- samples
def test_all_five_samples_exist_and_load() -> None:
    assert set(list_samples()) >= {
        "small", "medium", "large", "count_to_infinity", "triangle_loop"
    }
    sizes = {"small": 5, "medium": 8, "large": 12, "count_to_infinity": 3, "triangle_loop": 4}
    for name, size in sizes.items():
        net = load_sample(name)
        assert len(net.routers) == size
        assert net.is_connected()


def test_large_sample_has_links_over_16ms() -> None:
    net = load_sample("large")
    assert any(link.delay_ms > 16 for link in net.links.values())


def test_count_to_infinity_and_triangle_samples_have_1ms_links() -> None:
    lin = load_sample("count_to_infinity")
    assert sorted(lin.links) == [("A", "B"), ("B", "C")]
    tri = load_sample("triangle_loop")
    assert sorted(tri.links) == [("A", "B"), ("A", "C"), ("B", "C"), ("C", "D")]
    for net in (lin, tri):
        assert all(link.delay_ms == 1 for link in net.links.values())


# ------------------------------------------------------------ save / load
@pytest.mark.parametrize("name,network", all_test_networks())
def test_save_then_load_gives_identical_network(tmp_path, name, network) -> None:
    network.fail_link(*network.sorted_links()[0].key)  # failed state is saved too
    path = tmp_path / f"{name}.json"
    save_network(network, path)
    assert load_network(path) == network


def test_saved_file_matches_documented_format(tmp_path, small) -> None:
    path = tmp_path / "net.json"
    save_network(small, path)
    data = json.loads(path.read_text())
    assert data["routers"][0] == {"id": "A", "x": 120, "y": 270}
    assert data["links"][0] == {"a": "A", "b": "B", "delay_ms": 4, "active": True}


@pytest.mark.parametrize(
    "data,message",
    [
        ([], "JSON object"),
        ({"routers": "A"}, "must both be lists"),
        ({"routers": [{"id": "A"}, {"id": "A"}], "links": []}, "more than once"),
        ({"routers": [{"id": "Z"}], "links": []}, "invalid id"),
        ({"routers": [{"id": "A", "x": "left", "y": 0}], "links": []}, "numeric"),
        ({"routers": [{"id": "A"}], "links": [{"a": "A", "b": "B", "delay_ms": 3}]}, "does not exist"),
        ({"routers": [{"id": "A"}, {"id": "B"}], "links": [{"a": "A", "b": "B", "delay_ms": 0}]}, "between 1 and 100"),
        ({"routers": [{"id": "A"}, {"id": "B"}], "links": [{"a": "A", "b": "B", "delay_ms": 101}]}, "between 1 and 100"),
        ({"routers": [{"id": "A"}, {"id": "B"}], "links": [{"a": "A", "b": "B", "delay_ms": 2.5}]}, "whole number"),
        ({"routers": [{"id": "A"}, {"id": "B"}], "links": [{"a": "A", "b": "B", "delay_ms": 2}, {"a": "B", "b": "A", "delay_ms": 2}]}, "already exists"),
        ({"routers": [{"id": c} for c in "ABCDEFGHIJKLMNOP"], "links": []}, "Too many routers"),
    ],
)
def test_invalid_data_raises_clear_error(data, message) -> None:
    with pytest.raises(StorageError, match=message):
        network_from_dict(data)


def test_invalid_json_file_raises_clear_error(tmp_path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{ not json")
    with pytest.raises(StorageError, match="Not a valid JSON"):
        load_network(path)
    with pytest.raises(StorageError, match="File not found"):
        load_network(tmp_path / "missing.json")


# --------------------------------------------------------------- the model
def test_router_and_link_editing() -> None:
    net = Network()
    a = net.add_router(10, 20)
    b = net.add_router(30, 40)
    assert (a.id, b.id) == ("A", "B")
    net.add_link("B", "A", 5)
    assert net.get_link("A", "B").delay_ms == 5  # undirected: B-A == A-B
    net.set_delay("A", "B", 7)
    assert net.get_link("B", "A").delay_ms == 7
    net.move_router("A", 50, 60)
    assert (net.routers["A"].x, net.routers["A"].y) == (50, 60)
    net.fail_link("A", "B")
    assert net.neighbors("A") == [] and not net.is_connected()
    net.restore_link("A", "B")
    assert net.neighbor_ids("A") == ["B"] and net.is_connected()
    net.remove_router("B")
    assert not net.links and net.next_free_id() == "B"


def test_model_rejects_bad_edits() -> None:
    net = Network()
    net.add_router(0, 0)
    with pytest.raises(NetworkError):
        net.add_link("A", "A", 3)
    with pytest.raises(NetworkError):
        net.add_link("A", "Q", 3)
    for _ in range(MAX_ROUTERS - 1):
        net.add_router(0, 0)
    with pytest.raises(NetworkError, match="at most 15"):
        net.add_router(0, 0)


@pytest.mark.parametrize("n", range(2, 16))
def test_random_network_is_connected_and_reproducible(n) -> None:
    net = generate_random_network(n, seed=7)
    assert len(net.routers) == n and net.is_connected()
    assert all(1 <= link.delay_ms <= 20 for link in net.links.values())
    assert generate_random_network(n, seed=7) == net


def test_random_network_rejects_bad_sizes() -> None:
    for n in (1, 16):
        with pytest.raises(NetworkError):
            generate_random_network(n, seed=1)


def test_config_validation() -> None:
    assert SimConfig().validate().processing_delay_ms == 1
    with pytest.raises(ConfigError):
        SimConfig(processing_delay_ms=51).validate()
    with pytest.raises(ConfigError):
        SimConfig(k_packets=0).validate()
