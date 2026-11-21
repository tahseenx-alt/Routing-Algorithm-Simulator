"""Bellman-Ford as DISTANCE VECTOR routing by delay (a RIP-style process).

Unlike Dijkstra, no router ever sees the whole map. Each router only trades
distance vectors with its neighbours, round after round, until nothing
changes. The final costs are the same as Dijkstra's (both find the true
shortest path), but building them costs more messages and several rounds.
"""

from __future__ import annotations

from algorithms.base import (
    RoutingAlgorithm,
    RoutingResult,
    Step,
    finish_path_result,
    fmt,
)
from algorithms.distance_vector import (
    DVRun,
    as_distances,
    converge,
    follow_next_hops,
)
from core.config import SimConfig
from core.graph import Network


def dv_steps(run: DVRun, source: str, title: str) -> list[Step]:
    """Turn distance-vector rounds into animation steps.

    Distance labels show the SOURCE's view of the network after each round;
    routers whose table changed in that round are highlighted.
    """
    inf = run.state.infinity
    steps = [Step(f"{title}: every router knows only itself (cost 0), all else ∞",
                  current=[source],
                  distances={y: (0 if y == source else float("inf")) for y in run.state.dist})]
    for record in run.history:
        if record.number == 0:
            continue
        changed = sorted({c.router for c in record.changes})
        src_changes = [c for c in record.changes if c.router == source]
        detail = ", ".join(f"{c.destination}={fmt(c.new_cost if c.new_cost < inf else float('inf'))}"
                           for c in src_changes)
        text = (f"Round {record.number}: {record.messages} vectors sent, "
                f"{len(changed)} router(s) changed their table")
        if detail:
            text += f"; {source} now has {detail}"
        if not record.changes:
            text = f"Round {record.number}: {record.messages} vectors sent, no change: converged"
        links = [(source, record.next_hop[source][c.destination]) for c in src_changes
                 if record.next_hop[source][c.destination]]
        dist = {y: (float("inf") if c >= inf else c) for y, c in record.dist[source].items()}
        steps.append(Step(text, current=changed, links=links, distances=dist))
    return steps


def dv_result(name: str, source: str, destination: str, run: DVRun) -> RoutingResult:
    """Package a distance-vector run as a RoutingResult for the source."""
    state = run.state
    dist = as_distances(state, source)
    table = {y: (state.next_hop[source][y], dist[y]) for y in sorted(dist)}
    path = follow_next_hops(state, source, destination) if dist[destination] != float("inf") else None
    result = RoutingResult(
        name=name, source=source, destination=destination, distances=dist,
        routing_table=table, path=path, routing_overhead=run.messages,
        rounds=run.rounds, converged=run.converged, dv_state=state,
        log=list(run.log),
    )
    return result


class BellmanFord(RoutingAlgorithm):
    """Distance vector routing using delay + p as the metric."""

    name = "Bellman-Ford"

    def compute(
        self, network: Network, source: str, destination: str, config: SimConfig
    ) -> RoutingResult:
        """Converge every router's table, then read the source's route."""
        self.check_endpoints(network, source, destination)
        run = converge(network, "delay", config.split_horizon, config)
        result = dv_result(self.name, source, destination, run)
        result.steps = dv_steps(run, source, "Distance vector (delay)")
        if not run.converged:
            result.path = None
            result.note = f"Did not converge within {config.max_rounds} rounds."
        finish_path_result(result, network, config.processing_delay_ms)
        return result
