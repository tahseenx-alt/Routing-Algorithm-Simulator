"""Common interface and result types shared by every routing algorithm.

Each algorithm takes the same inputs (network, source, destination, config)
and returns a RoutingResult, so the GUI and the comparison code can treat all
of them the same way. Algorithms never import Tkinter: they only *record*
Steps, and the GUI replays those steps as an animation.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from core.config import SimConfig
from core.graph import Network

INF = math.inf
SAME_SOURCE_DEST = "Source and destination are the same"


class AlgorithmError(ValueError):
    """Raised for bad inputs such as a router id that does not exist."""


@dataclass
class Step:
    """One frame of an algorithm animation.

    Attributes:
        description: What happens in this step, in plain English.
        current: Router(s) being worked on now (drawn yellow).
        finalized: Routers whose answer is final (drawn green).
        links: Link(s) to highlight, as (a, b) pairs.
        distances: Snapshot of the distance label of every router.
    """

    description: str
    current: list[str] = field(default_factory=list)
    finalized: list[str] = field(default_factory=list)
    links: list[tuple[str, str]] = field(default_factory=list)
    distances: dict[str, float] = field(default_factory=dict)


@dataclass
class RoutingResult:
    """Everything an algorithm reports about one run.

    Attributes:
        name: Algorithm name shown in the GUI.
        source / destination: The routers the run was for.
        distances: Cost from the source to every router (math.inf = unreachable).
        routing_table: Source's table: destination -> (next hop, cost).
        path: Router ids from source to destination, or None if not delivered.
        delivery_time_ms: Time for one packet to arrive, or None.
        routing_overhead: One-time routing messages to build the tables.
        per_packet_transmissions: Link transmissions caused by one data packet.
        rounds: Distance-vector rounds to converge (None for others).
        converged: False if distance vector hit the max_rounds safety cap.
        success_rate: Fraction of successful trials (random routing only).
        steps: Animation frames.
        log: Text log lines for the Log tab.
        note: Extra message, e.g. "Source and destination are the same".
        dv_state: Converged distance-vector state, kept for warm-start
            reconvergence after a link failure.
    """

    name: str
    source: str
    destination: str
    distances: dict[str, float] = field(default_factory=dict)
    routing_table: dict[str, tuple[str | None, float]] = field(default_factory=dict)
    path: list[str] | None = None
    delivery_time_ms: float | None = None
    routing_overhead: int = 0
    per_packet_transmissions: float = 0.0
    rounds: int | None = None
    converged: bool = True
    success_rate: float | None = None
    steps: list[Step] = field(default_factory=list)
    log: list[str] = field(default_factory=list)
    note: str = ""
    dv_state: Any = None

    @property
    def delivered(self) -> bool:
        """True when the packet reached the destination."""
        return self.path is not None and self.delivery_time_ms is not None

    @property
    def hop_count(self) -> int | None:
        """Number of links on the path, or None if there is no path."""
        return None if self.path is None else len(self.path) - 1

    def total_messages(self, k_packets: int) -> float:
        """Routing overhead + K x per-packet transmissions."""
        return self.routing_overhead + k_packets * self.per_packet_transmissions


class RoutingAlgorithm(ABC):
    """Abstract base class: every routing algorithm implements compute()."""

    name: str = "Routing algorithm"

    @abstractmethod
    def compute(
        self, network: Network, source: str, destination: str, config: SimConfig
    ) -> RoutingResult:
        """Run the algorithm from source to destination and return the result."""

    @staticmethod
    def check_endpoints(network: Network, source: str, destination: str) -> None:
        """Raise AlgorithmError if source or destination is not in the network."""
        for label, rid in (("Source", source), ("Destination", destination)):
            if rid not in network.routers:
                raise AlgorithmError(f"{label} router {rid!r} does not exist.")


# ---------------------------------------------------------------- helpers
def fmt(value: float) -> str:
    """Format a cost for display: whole numbers without decimals, inf as '∞'."""
    if value is None or value == INF:
        return "∞"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.2f}"


def path_links(path: list[str] | None) -> list[tuple[str, str]]:
    """Return the consecutive (a, b) link pairs along a path."""
    if not path:
        return []
    return list(zip(path, path[1:]))


def path_delivery_time(network: Network, path: list[str], processing_delay_ms: float) -> float:
    """Delivery time of a path = sum of link delays + p x number of hops.

    The source sends at t = 0 without processing; every router that receives
    the packet (including the destination) spends p ms processing it.
    """
    total = 0.0
    for a, b in path_links(path):
        link = network.get_link(a, b)
        if link is None:
            raise AlgorithmError(f"Path uses a link {a}-{b} that does not exist.")
        total += link.delay_ms
    return total + processing_delay_ms * (len(path) - 1)


def path_from_predecessors(
    predecessors: dict[str, str | None], source: str, destination: str
) -> list[str] | None:
    """Rebuild source -> destination by walking predecessor links backwards."""
    if destination == source:
        return [source]
    if predecessors.get(destination) is None:
        return None
    path = [destination]
    while path[-1] != source:
        prev = predecessors.get(path[-1])
        if prev is None or len(path) > len(predecessors):
            return None
        path.append(prev)
    return path[::-1]


def table_from_predecessors(
    predecessors: dict[str, str | None], distances: dict[str, float], source: str
) -> dict[str, tuple[str | None, float]]:
    """Build the source's routing table (destination -> first hop, cost).

    The first hop is found by walking back from each destination until the
    router just after the source. This is exactly what a router stores after
    running SPF: only the next hop matters, not the whole path.
    """
    table: dict[str, tuple[str | None, float]] = {}
    for dest in sorted(distances):
        path = path_from_predecessors(predecessors, source, dest)
        if dest == source:
            table[dest] = (None, 0)
        elif path is None:
            table[dest] = (None, INF)
        else:
            table[dest] = (path[1], distances[dest])
    return table


def finish_path_result(result: RoutingResult, network: Network, p: float) -> None:
    """Fill in delivery time, per-packet transmissions (= hops) and a final
    step that highlights the chosen path. Used by the path-based algorithms."""
    if result.path is None:
        result.note = f"No path from {result.source} to {result.destination}."
        result.log.append(result.note)
        return
    if result.source == result.destination:
        result.note = SAME_SOURCE_DEST
    result.delivery_time_ms = path_delivery_time(network, result.path, p)
    result.per_packet_transmissions = len(result.path) - 1
    text = (f"Path {'→'.join(result.path)}: delivery time "
            f"{fmt(result.delivery_time_ms)} ms")
    result.log.append(text)
    last = result.steps[-1] if result.steps else Step("")
    result.steps.append(Step(text, current=[result.destination],
                             finalized=list(last.finalized) or list(result.path),
                             links=path_links(result.path),
                             distances=dict(result.distances)))
