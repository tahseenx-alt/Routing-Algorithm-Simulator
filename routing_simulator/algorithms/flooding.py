"""Flooding: send the packet everywhere, no routing tables at all.

Each router that receives a packet for the first time forwards a copy on
every link except the one it came in on; later duplicates are dropped. The
first copy to arrive always travels along the fastest path, so flooding is
as fast as Dijkstra, but it sends many copies for EVERY packet. It needs no
setup messages (routing overhead = 0) and is very robust to failures.

The simulation is event driven: a heap of events ordered by time, so the
copy that really arrives first is processed first (not BFS / hop order).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

from algorithms.base import (
    INF,
    SAME_SOURCE_DEST,
    RoutingAlgorithm,
    RoutingResult,
    Step,
    fmt,
    path_from_predecessors,
    path_links,
)
from core.config import SimConfig
from core.graph import Network, link_key


@dataclass
class FloodEvent:
    """Something that happened to one copy during the flood."""

    time_ms: float
    router: str
    came_from: str | None
    text: str


@dataclass
class FloodRun:
    """Outcome of flooding one packet.

    Attributes:
        copies: Copies transmitted until no copies remained.
        first_arrival: Time the first copy arrived at each router.
        parent: Router each first copy came from (gives the first-copy path).
        delivery_time_ms: Finish time of the first copy at the destination.
        events: Time-ordered list of everything that happened.
        seen: Routers that received the packet.
    """

    copies: int = 0
    first_arrival: dict[str, float] = field(default_factory=dict)
    parent: dict[str, str | None] = field(default_factory=dict)
    delivery_time_ms: float | None = None
    events: list[FloodEvent] = field(default_factory=list)
    seen: set[str] = field(default_factory=set)


def flood_packet(network: Network, source: str, destination: str, p: float,
                 failed_link: tuple[str, str] | None = None,
                 fail_time_ms: float = INF) -> FloodRun:
    """Flood one packet with an event queue ordered by time.

    Optional link failure: from fail_time_ms on, `failed_link` is down. A copy
    already travelling on it at that moment is lost; no new copies are sent
    on it afterwards. All other copies simply continue.
    """
    run = FloodRun(seen={source}, first_arrival={source: 0.0}, parent={source: None})
    events: list[tuple[float, int, str, str]] = []  # (arrival time, order, router, came from)
    order = 0
    broken = link_key(*failed_link) if failed_link else None

    def send_copies(router: str, came_from: str | None, at: float) -> list[str]:
        """Send a copy to every active neighbour except came_from."""
        nonlocal order
        targets = []
        for nbr, link in network.neighbors(router):
            if nbr == came_from:
                continue
            on_broken = link.key == broken
            if on_broken and at >= fail_time_ms:
                continue  # link already down: the router does not use it
            run.copies += 1
            arrive = at + link.delay_ms
            if on_broken and fail_time_ms < arrive:
                run.events.append(FloodEvent(fail_time_ms, router, None,
                                             f"link {router}–{nbr} fails: copy to {nbr} lost"))
                continue
            order += 1
            heapq.heappush(events, (arrive, order, nbr, router))
            targets.append(nbr)
        return targets

    if source == destination:
        run.delivery_time_ms = 0.0
        return run
    sent = send_copies(source, None, 0.0)
    run.events.append(FloodEvent(0.0, source, None,
                                 f"{source} sends copies to {', '.join(sent) or 'nobody'}"))
    while events:
        arrival, _, router, came_from = heapq.heappop(events)
        if router in run.seen:
            text = f"duplicate at {router} from {came_from}, dropped"
        else:
            run.seen.add(router)
            run.parent[router] = came_from
            run.first_arrival[router] = arrival
            if router == destination:  # the destination keeps it, never forwards
                run.delivery_time_ms = arrival + p
                text = (f"first copy reaches {router} from {came_from}; "
                        f"delivered at t={fmt(arrival + p)} ms")
            else:
                targets = send_copies(router, came_from, arrival + p)
                text = (f"{router} gets first copy from {came_from}, forwards "
                        f"{len(targets)} cop{'y' if len(targets) == 1 else 'ies'} "
                        f"at t={fmt(arrival + p)} ms")
        run.events.append(FloodEvent(arrival, router, came_from, text))
    run.events.sort(key=lambda e: e.time_ms)
    return run


class Flooding(RoutingAlgorithm):
    """Event-driven flooding with duplicate suppression."""

    name = "Flooding"

    def compute(
        self, network: Network, source: str, destination: str, config: SimConfig
    ) -> RoutingResult:
        """Flood one packet from source and record every copy."""
        self.check_endpoints(network, source, destination)
        p = config.processing_delay_ms
        run = flood_packet(network, source, destination, p)
        result = RoutingResult(name=self.name, source=source, destination=destination,
                               per_packet_transmissions=run.copies)
        # Distance label = time the router finished processing its first copy.
        dist = {r: INF for r in network.router_ids()}
        for router, arrival in run.first_arrival.items():
            dist[router] = arrival + (p if router != source else 0)
        result.distances = dist

        seen: set[str] = set()
        labels: dict[str, float] = {r: INF for r in dist}
        for event in run.events:
            seen.add(event.router)
            labels[event.router] = dist[event.router]
            links = [(event.came_from, event.router)] if event.came_from else []
            result.steps.append(Step(f"t={fmt(event.time_ms)} ms: {event.text}",
                                     current=[event.router], finalized=sorted(seen),
                                     links=links, distances=dict(labels)))
        result.log = [s.description for s in result.steps]

        result.path = path_from_predecessors(run.parent, source, destination)
        if source == destination:
            result.path, result.delivery_time_ms, result.note = [source], 0, SAME_SOURCE_DEST
            return result
        summary = f"Flooding finished: {run.copies} copies transmitted for one packet."
        if result.path is None or run.delivery_time_ms is None:
            result.path = None
            result.note = f"No path from {source} to {destination}."
            summary += " " + result.note
        else:
            result.delivery_time_ms = run.delivery_time_ms
            summary += (f" First copy path {'→'.join(result.path)}, "
                        f"delivered in {fmt(result.delivery_time_ms)} ms.")
            result.steps.append(Step(summary, current=[destination], finalized=sorted(seen),
                                     links=path_links(result.path), distances=dict(dist)))
        result.log.append(summary)
        return result
