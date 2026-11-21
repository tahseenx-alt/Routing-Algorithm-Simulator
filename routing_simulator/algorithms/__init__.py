"""Routing algorithms. None of them import Tkinter.

ALGORITHMS lists one instance of each, in the order used by "Run All".
"""

from algorithms.base import RoutingAlgorithm, RoutingResult, Step
from algorithms.bellman_ford import BellmanFord
from algorithms.dijkstra import Dijkstra
from algorithms.flooding import Flooding
from algorithms.hop_count import HopCount
from algorithms.random_routing import RandomRouting

ALGORITHMS: list[RoutingAlgorithm] = [
    Dijkstra(), BellmanFord(), HopCount(), Flooding(), RandomRouting()
]


def get_algorithm(name: str) -> RoutingAlgorithm:
    """Return the algorithm instance with this display name."""
    for algorithm in ALGORITHMS:
        if algorithm.name == name:
            return algorithm
    raise KeyError(f"Unknown algorithm: {name}")


__all__ = ["ALGORITHMS", "get_algorithm", "RoutingAlgorithm", "RoutingResult", "Step"]
