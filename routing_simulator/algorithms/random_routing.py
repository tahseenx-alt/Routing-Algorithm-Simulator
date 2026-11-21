"""Random routing: every router forwards the packet to a random neighbour.

No routing tables and no setup messages, but the path is a random walk, so
the packet may wander, take a long time or be dropped when its TTL (time to
live) runs out. To avoid bouncing straight back, a router does not send the
packet back to the router it came from unless that is the only way out.

We run several trials with a local random.Random(seed), so the same seed
always gives the same results (reproducible experiments).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from algorithms.base import (
    INF,
    SAME_SOURCE_DEST,
    RoutingAlgorithm,
    RoutingResult,
    Step,
    fmt,
    path_links,
)
from core.config import SimConfig
from core.graph import Network


@dataclass
class Trial:
    """One random walk of a single packet."""

    delivered: bool
    path: list[str]
    time_ms: float
    transmissions: int
    steps: list[Step] = field(default_factory=list)
    reason: str = ""


def random_walk(network: Network, source: str, destination: str, rng: random.Random,
                p: float, ttl: int | None = None, previous: str | None = None,
                start_ms: float = 0.0) -> Trial:
    """Send one packet by random forwarding until delivered or dropped.

    ttl, previous and start_ms let a walk continue from the middle (used by
    the packet simulation after a link failure). TTL defaults to 2 x routers.
    """
    ttl = 2 * len(network.routers) if ttl is None else ttl
    path, time, sent = [source], start_ms, 0
    times = {source: start_ms}
    steps = [Step(f"Packet starts at {source} with TTL {ttl}", current=[source],
                  distances={source: start_ms})]
    while path[-1] != destination:
        here = path[-1]
        options = network.neighbor_ids(here)
        if not options:
            return Trial(False, path, time, sent, steps, f"{here} has no active links")
        choices = [n for n in options if n != previous] or options  # dead end: go back
        nxt = rng.choice(choices)
        time += network.get_link(here, nxt).delay_ms + p  # travel + processing at nxt
        sent += 1
        ttl -= 1
        previous = here
        path.append(nxt)
        times[nxt] = time
        steps.append(Step(f"{here} picks {nxt} at random from {', '.join(choices)} "
                          f"(t={fmt(time)} ms, TTL {ttl})", current=[nxt],
                          finalized=sorted(set(path)), links=[(here, nxt)], distances=dict(times)))
        if nxt != destination and ttl <= 0:  # TTL stops packets from looping forever
            steps.append(Step(f"TTL reached 0 at {nxt}: packet dropped", current=[nxt],
                              finalized=sorted(set(path)), distances=dict(times)))
            return Trial(False, path, time, sent, steps, "TTL expired")
    return Trial(True, path, time, sent, steps)


class RandomRouting(RoutingAlgorithm):
    """Random-walk forwarding, averaged over several seeded trials."""

    name = "Random"

    def compute(
        self, network: Network, source: str, destination: str, config: SimConfig
    ) -> RoutingResult:
        """Run config.random_trials random walks and summarise them."""
        self.check_endpoints(network, source, destination)
        rng = random.Random(config.seed)  # local generator: reproducible
        trials = [random_walk(network, source, destination, rng, config.processing_delay_ms)
                  for _ in range(config.random_trials)]
        good = [t for t in trials if t.delivered]
        shown = good[0] if good else trials[0]

        result = RoutingResult(name=self.name, source=source, destination=destination)
        result.success_rate = len(good) / len(trials)
        result.per_packet_transmissions = sum(t.transmissions for t in trials) / len(trials)
        result.steps = list(shown.steps)
        result.distances = {r: INF for r in network.router_ids()}
        for step in shown.steps:  # time each router was (last) reached in the shown walk
            result.distances.update(step.distances)
        if good:
            result.path = shown.path
            result.delivery_time_ms = sum(t.time_ms for t in good) / len(good)
            result.steps.append(Step(f"Shown trial delivered via {'→'.join(shown.path)}",
                                     current=[destination], finalized=sorted(set(shown.path)),
                                     links=path_links(shown.path),
                                     distances=dict(shown.steps[-1].distances)))
        if source == destination:
            result.note = SAME_SOURCE_DEST
        elif not good:
            result.note = f"All {len(trials)} trials failed ({trials[0].reason})."
        for i, t in enumerate(trials, 1):
            status = (f"delivered in {fmt(t.time_ms)} ms" if t.delivered
                      else f"dropped ({t.reason})")
            result.log.append(f"Trial {i}: {'→'.join(t.path)} – {status}, "
                              f"{t.transmissions} transmissions")
        result.log.append(f"Success rate {len(good)}/{len(trials)}, average "
                          f"{result.per_packet_transmissions:.1f} transmissions per trial.")
        return result
