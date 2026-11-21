"""Dijkstra's shortest path algorithm: the heart of LINK STATE routing (OSPF).

How OSPF works, in short:
  1. Every router floods a Link State Advertisement (LSA) describing its own
     links. After flooding, every router has the same map of the network.
  2. Every router runs Dijkstra (Shortest Path First, SPF) on that map to
     find the cheapest path to every destination and stores the next hop.

Here the link cost is the *effective* cost delay_ms + p, so the cheapest path
is also the path that delivers a packet fastest.
"""

from __future__ import annotations

import heapq
from collections import deque

from algorithms.base import (
    INF,
    RoutingAlgorithm,
    RoutingResult,
    Step,
    finish_path_result,
    fmt,
    path_from_predecessors,
    table_from_predecessors,
)
from core.config import SimConfig
from core.graph import Network


def count_lsa_flood(network: Network, origin: str) -> int:
    """Count transmissions needed to flood ONE LSA originated by `origin`.

    Flooding rule (as in OSPF): a router that receives an LSA for the first
    time forwards it to every active neighbour except the one it came from.
    Duplicates are dropped. Every forwarded copy is one transmission.
    """
    seen = {origin}
    transmissions = 0
    queue: deque[tuple[str, str | None]] = deque([(origin, None)])
    while queue:
        router, came_from = queue.popleft()
        for nbr in network.neighbor_ids(router):
            if nbr == came_from:
                continue
            transmissions += 1  # one copy sent over this link
            if nbr not in seen:  # first copy: accept and forward later
                seen.add(nbr)
                queue.append((nbr, router))
            # else: duplicate LSA, the neighbour drops it
    return transmissions


def lsa_overhead(network: Network, source: str) -> int:
    """Routing overhead of link state: every router in the source's part of
    the network originates one LSA, and each LSA is flooded."""
    return sum(count_lsa_flood(network, r) for r in sorted(network.reachable_from(source)))


def shortest_paths(
    network: Network, source: str, processing_delay_ms: float
) -> tuple[dict[str, float], dict[str, str | None], list[Step]]:
    """Run Dijkstra from source using effective cost (delay + p).

    Returns (distances, predecessors, animation steps).
    """
    dist: dict[str, float] = {r: INF for r in network.router_ids()}
    prev: dict[str, str | None] = {r: None for r in network.router_ids()}
    dist[source] = 0
    finalized: set[str] = set()
    heap: list[tuple[float, str]] = [(0, source)]  # priority queue ordered by cost
    steps = [Step(f"Start at {source}: cost 0, every other router ∞",
                  current=[source], distances=dict(dist))]

    while heap:
        cost, u = heapq.heappop(heap)
        if u in finalized:
            continue  # stale queue entry, a cheaper one was already used
        finalized.add(u)  # the cheapest unvisited router is now final
        tree = [(prev[u], u)] if prev[u] else []
        steps.append(Step(f"Visit {u} (cost {fmt(cost)}), finalise {u}",
                          current=[u], finalized=sorted(finalized),
                          links=tree, distances=dict(dist)))
        for v, link in network.neighbors(u):
            if v in finalized:
                continue
            new_cost = cost + link.effective_cost(processing_delay_ms)
            if new_cost < dist[v]:  # "relaxation": found a cheaper way to v
                dist[v], prev[v] = new_cost, u
                heapq.heappush(heap, (new_cost, v))
                text = f"Relax {u}–{v}: {v} = {fmt(new_cost)} via {u}"
            else:
                text = f"Check {u}–{v}: {fmt(new_cost)} is not better, {v} stays {fmt(dist[v])}"
            steps.append(Step(text, current=[u, v], finalized=sorted(finalized),
                              links=[(u, v)], distances=dict(dist)))
    return dist, prev, steps


class Dijkstra(RoutingAlgorithm):
    """Link state routing (OSPF-like) using Dijkstra's algorithm."""

    name = "Dijkstra"

    def compute(
        self, network: Network, source: str, destination: str, config: SimConfig
    ) -> RoutingResult:
        """Build the source's routing table with SPF and find the path."""
        self.check_endpoints(network, source, destination)
        p = config.processing_delay_ms
        dist, prev, steps = shortest_paths(network, source, p)
        path = path_from_predecessors(prev, source, destination)
        overhead = lsa_overhead(network, source)

        result = RoutingResult(
            name=self.name, source=source, destination=destination,
            distances=dist, routing_table=table_from_predecessors(prev, dist, source),
            path=path, routing_overhead=overhead, steps=steps,
        )
        result.log.append(f"Link state: {len(network.reachable_from(source))} routers each "
                          f"flood one LSA = {overhead} LSA transmissions.")
        result.log.extend(s.description for s in steps)
        finish_path_result(result, network, p)
        return result

