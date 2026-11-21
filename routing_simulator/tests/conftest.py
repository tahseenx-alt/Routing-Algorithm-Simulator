"""Shared pytest helpers: sample networks, random networks and NetworkX conversion.

NetworkX is used ONLY here in the tests, to check our own algorithm results.
"""

from __future__ import annotations

import networkx as nx
import pytest

from core.graph import Network, generate_random_network
from core.storage import list_samples, load_sample

SAMPLE_NAMES = list_samples()
RANDOM_SEEDS = list(range(20))


def random_networks() -> list[tuple[str, Network]]:
    """20 reproducible random networks of different sizes (2..15 routers)."""
    nets = []
    for seed in RANDOM_SEEDS:
        n = 2 + seed % 14  # sizes 2..15
        nets.append((f"random-{seed}-n{n}", generate_random_network(n, seed)))
    return nets


def all_test_networks() -> list[tuple[str, Network]]:
    """All bundled samples plus the 20 random networks."""
    return [(name, load_sample(name)) for name in SAMPLE_NAMES] + random_networks()


def to_networkx(network: Network, p: float) -> nx.Graph:
    """Build a NetworkX graph of the ACTIVE links with weight 'cost' = delay + p."""
    graph = nx.Graph()
    graph.add_nodes_from(network.router_ids())
    for link in network.active_links():
        graph.add_edge(link.a, link.b, cost=link.delay_ms + p, delay=link.delay_ms)
    return graph


@pytest.fixture
def small() -> Network:
    """The 5-router sample network."""
    return load_sample("small")
