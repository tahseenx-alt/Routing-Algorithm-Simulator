"""Minimum-hop routing: the metric used by RIP.

RIP ignores link speed and simply counts routers on the way (hops), with
16 meaning "unreachable". So a 2-hop path over slow links beats a 3-hop path
over fast links. Delivery time is still measured with the REAL delays, which
is why hop count can be slower than Dijkstra.

Path: a priority queue on the pair (hops, total delay), so among paths with
the same number of hops the one with lower delay wins (tie break).
Overhead: RIP builds its tables with the distance-vector engine (metric =
hops, infinity = 16), so we count the vectors that engine sends.
"""

from __future__ import annotations

import heapq

from algorithms.base import (
    INF,
    RoutingAlgorithm,
    RoutingResult,
    Step,
    finish_path_result,
    path_from_predecessors,
    table_from_predecessors,
)
from algorithms.distance_vector import converge
from core.config import SimConfig
from core.graph import Network


def min_hop_paths(network: Network, source: str) -> tuple[dict[str, float], dict[str, float],
                                                          dict[str, str | None], list[Step]]:
    """Search ordered by (hops, delay). Returns (hops, delay, predecessors, steps)."""
    ids = network.router_ids()
    hops: dict[str, float] = {r: INF for r in ids}
    delay: dict[str, float] = {r: INF for r in ids}
    prev: dict[str, str | None] = {r: None for r in ids}
    hops[source], delay[source] = 0, 0
    heap: list[tuple[float, float, str]] = [(0, 0, source)]
    done: set[str] = set()
    steps = [Step(f"Start at {source}: 0 hops", current=[source], distances=dict(hops))]
    while heap:
        h, d, u = heapq.heappop(heap)
        if u in done:
            continue
        done.add(u)
        tree = [(prev[u], u)] if prev[u] else []
        steps.append(Step(f"Visit {u} ({int(h)} hops, {int(d)} ms), finalise {u}", current=[u],
                          finalized=sorted(done), links=tree, distances=dict(hops)))
        for v, link in network.neighbors(u):
            if v in done:
                continue
            candidate = (h + 1, d + link.delay_ms)
            if candidate < (hops[v], delay[v]):  # fewer hops, or same hops but faster
                hops[v], delay[v], prev[v] = candidate[0], candidate[1], u
                heapq.heappush(heap, (candidate[0], candidate[1], v))
                steps.append(Step(f"{v} = {int(candidate[0])} hops via {u}", current=[u, v],
                                  finalized=sorted(done), links=[(u, v)], distances=dict(hops)))
    return hops, delay, prev, steps


class HopCount(RoutingAlgorithm):
    """RIP-style minimum-hop routing."""

    name = "Hop Count"

    def compute(
        self, network: Network, source: str, destination: str, config: SimConfig
    ) -> RoutingResult:
        """Find the fewest-hop path and count RIP's distance-vector messages."""
        self.check_endpoints(network, source, destination)
        hops, _delay, prev, steps = min_hop_paths(network, source)
        run = converge(network, "hops", config.split_horizon, config)
        result = RoutingResult(
            name=self.name, source=source, destination=destination, distances=hops,
            routing_table=table_from_predecessors(prev, hops, source),
            path=path_from_predecessors(prev, source, destination),
            routing_overhead=run.messages, rounds=run.rounds, converged=run.converged,
            steps=steps, dv_state=run.state,
        )
        result.log.append(f"RIP tables (hop metric, infinity 16) built in {run.rounds} "
                          f"rounds with {run.messages} vectors.")
        result.log.extend(s.description for s in steps)
        finish_path_result(result, network, config.processing_delay_ms)
        return result
