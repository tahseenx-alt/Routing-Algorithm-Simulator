"""Hop-by-hop simulation of one data packet, with an optional link failure.

Timing model (same everywhere in the project):
  * The source sends at t = 0 with no processing.
  * Crossing a link takes delay_ms.
  * Every router that receives the packet, including the destination, spends
    p ms processing it before forwarding or delivering it.
  => delivery time = sum of link delays + p x hops.

Link failure during transfer ("fail link X-Y at t ms"):
  * Packet travelling on that link at that moment: this copy is lost.
  * Packet that has not reached the link yet: it stops at the router before it.
  * Rerouting OFF: the packet is "Not delivered".
  * Rerouting ON: the packet waits at the last router it reached for the
    reconvergence delay, the algorithm recomputes from there and the packet
    continues on the new path (or is "Not delivered" if there is none).

Reconvergence delay:
  * Distance vector (Bellman-Ford, Hop Count): rounds to reconverge x
    update_period_ms (at least one period).
  * Link state (Dijkstra): 1 x update_period_ms (LSA flood + SPF).
  * Flooding and random routing: no reconvergence; other copies / random
    choices simply continue.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from algorithms.base import (
    RoutingAlgorithm,
    RoutingResult,
    fmt,
    path_delivery_time,
    path_from_predecessors,
    path_links,
)
from algorithms.dijkstra import Dijkstra, count_lsa_flood
from algorithms.distance_vector import follow_next_hops, reconverge
from algorithms.flooding import flood_packet
from algorithms.hop_count import min_hop_paths
from algorithms.random_routing import random_walk
from core.config import SimConfig
from core.graph import Network, link_key

NOT_DELIVERED = "Not delivered"


class PacketSimError(ValueError):
    """Bad simulation input, e.g. a scheduled failure on a missing link."""


@dataclass
class LinkFailure:
    """A failure scheduled before sending: link a-b goes down at time_ms."""

    a: str
    b: str
    time_ms: float

    def matches(self, x: str, y: str) -> bool:
        """True if x-y is the failing link (either direction)."""
        return link_key(x, y) == link_key(self.a, self.b)


@dataclass
class TimelineEvent:
    """One line of the packet timeline."""

    time_ms: float
    router: str
    text: str

    def __str__(self) -> str:
        """E.g. 't=4 ms: packet arrives at B'."""
        return f"t={fmt(self.time_ms)} ms: {self.text}"


@dataclass
class Hop:
    """The packet crossing one link (used to animate the moving dot).

    If the copy is lost mid-link, `fraction` says how far it got (0..1).
    """

    src: str
    dst: str
    depart_ms: float
    arrive_ms: float
    lost: bool = False
    fraction: float = 1.0


@dataclass
class Recovery:
    """How an algorithm recovers from a link failure.

    Attributes:
        path: New path from the router where the packet waits (or None).
        delay_ms: Reconvergence delay before the packet can continue.
        extra_messages: Routing messages caused by the failure.
        rounds: Rounds needed to reconverge (1 for link state).
        converged: False if distance vector hit max_rounds.
        explain: Short text for the timeline and the GUI.
        result: The recomputed RoutingResult (for tables / logs), if any.
        log: Detailed log lines of the reconvergence.
    """

    path: list[str] | None
    delay_ms: float = 0.0
    extra_messages: int = 0
    rounds: int = 0
    converged: bool = True
    explain: str = ""
    result: RoutingResult | None = None
    log: list[str] = field(default_factory=list)


@dataclass
class PacketResult:
    """Everything that happened to the packet."""

    algorithm: str
    source: str
    destination: str
    delivered: bool = False
    delivery_time_ms: float | None = None
    path_taken: list[str] = field(default_factory=list)
    hops: list[Hop] = field(default_factory=list)
    events: list[TimelineEvent] = field(default_factory=list)
    rerouted: bool = False
    reconvergence_delay_ms: float = 0.0
    status: str = ""

    @property
    def timeline(self) -> list[str]:
        """Timeline as text lines, sorted by time."""
        return [str(e) for e in sorted(self.events, key=lambda e: e.time_ms)]

    def log(self, time_ms: float, router: str, text: str) -> None:
        """Add one timeline event."""
        self.events.append(TimelineEvent(time_ms, router, text))


# ----------------------------------------------------------------- recovery
def recover(algorithm: RoutingAlgorithm, network_after: Network, previous: RoutingResult,
            router: str, destination: str, config: SimConfig,
            failed: tuple[str, str]) -> Recovery:
    """Recompute `algorithm` after link `failed` went down (network_after has it failed).

    Returns the new path from `router`, the reconvergence delay and the extra
    routing messages caused by the failure.
    """
    period = config.update_period_ms
    name = algorithm.name
    if name == "Dijkstra":
        # Both routers at the ends of the link flood a new LSA, then every
        # router reruns SPF: treated as one update period.
        new = Dijkstra().compute(network_after, router, destination, config)
        extra = sum(count_lsa_flood(network_after, end) for end in failed)
        return Recovery(new.path, period, extra, 1, True,
                        f"{failed[0]} and {failed[1]} flood new LSAs ({extra} messages), "
                        f"SPF recomputed: 1 update period", new)
    if name in ("Bellman-Ford", "Hop Count"):
        if previous.dv_state is None:
            raise PacketSimError(f"{name} has no converged state to warm start from.")
        run = reconverge(network_after, previous.dv_state, config)
        if name == "Bellman-Ford":
            path = follow_next_hops(run.state, router, destination) if run.converged else None
        else:
            _, _, prev, _ = min_hop_paths(network_after, router)
            path = path_from_predecessors(prev, router, destination) if run.converged else None
        if path is not None and run.state.dist[router][destination] >= run.state.infinity:
            path = None
        rounds = max(1, run.rounds)
        text = (f"distance vector reconverged in {run.rounds} round(s), "
                f"{run.messages} vectors" if run.converged
                else f"did not reconverge within {config.max_rounds} rounds")
        return Recovery(path, rounds * period, run.messages, run.rounds, run.converged,
                        text, None, list(run.log))
    # Flooding and random routing keep no tables: nothing to reconverge.
    new = algorithm.compute(network_after, router, destination, config)
    return Recovery(new.path, 0.0, 0, 0, True, "no routing tables, nothing to reconverge", new)


# --------------------------------------------------------------- simulation
def simulate_path(network: Network, path: list[str], config: SimConfig,
                  algorithm_name: str = "") -> PacketResult:
    """Simulate a packet along a fixed path with no failures."""
    out = PacketResult(algorithm_name, path[0], path[-1], path_taken=[path[0]])
    outcome, _, t = _travel(network, path, 0.0, config.processing_delay_ms, None, out, True)
    _finish(out, outcome, t)
    return out


def simulate_packet(network: Network, algorithm: RoutingAlgorithm, source: str,
                    destination: str, config: SimConfig,
                    failure: LinkFailure | None = None,
                    initial: RoutingResult | None = None) -> PacketResult:
    """Send one packet with `algorithm`, optionally failing a link mid-transfer.

    `initial` can pass an already computed RoutingResult to avoid recomputing.
    The given network is never modified.
    """
    config.validate()
    _check_failure(network, failure)
    initial = initial or algorithm.compute(network, source, destination, config)
    p = config.processing_delay_ms
    out = PacketResult(algorithm.name, source, destination, path_taken=[source])

    if algorithm.name == "Flooding":
        return _simulate_flooding(network, source, destination, config, failure, out)
    if not initial.delivered:
        reason = initial.note or f"no path from {source} to {destination}"
        out.log(0, source, f"{algorithm.name}: {reason}")
        _finish(out, "none", 0)
        return out

    outcome, router, t = _travel(network, initial.path, 0.0, p, failure, out, True)
    if outcome in ("stopped", "lost"):
        assert failure is not None
        after = network.copy()
        after.fail_link(failure.a, failure.b)
        if algorithm.name == "Random":
            outcome, t = _continue_random(after, out, outcome, router, t, config)
        elif not config.rerouting:
            out.log(t, router, "rerouting is OFF: packet lost")
            outcome = "dropped"
        else:
            outcome, t = _reroute(algorithm, after, initial, out, router, t, config,
                                  link_key(failure.a, failure.b))
    _finish(out, outcome, t)
    return out


def _reroute(algorithm: RoutingAlgorithm, after: Network, initial: RoutingResult,
             out: PacketResult, router: str, t: float, config: SimConfig,
             failed: tuple[str, str]) -> tuple[str, float]:
    """Wait for reconvergence at `router`, then continue on the new path."""
    recovery = recover(algorithm, after, initial, router, out.destination, config, failed)
    out.reconvergence_delay_ms = recovery.delay_ms
    out.log(t, router, f"packet waits at {router} {fmt(recovery.delay_ms)} ms: {recovery.explain}")
    t += recovery.delay_ms
    if recovery.path is None:
        out.log(t, router, f"no new path from {router} to {out.destination}")
        return "none", t
    out.rerouted = True
    out.log(t, router, f"new path {'→'.join(recovery.path)}")
    outcome, _, t = _travel(after, recovery.path, t, config.processing_delay_ms, None, out, False)
    return outcome, t


def _continue_random(after: Network, out: PacketResult, outcome: str, router: str,
                     t: float, config: SimConfig) -> tuple[str, float]:
    """Random routing after a failure: the single copy is lost, or it simply
    picks another random neighbour (no reconvergence needed)."""
    if outcome == "lost":
        out.log(t, router, "random routing has only one copy: packet lost")
        return "dropped", t
    used = len(out.path_taken) - 1
    ttl = 2 * len(after.routers) - used
    previous = out.path_taken[-2] if len(out.path_taken) > 1 else None
    trial = random_walk(after, router, out.destination, random.Random(config.seed),
                        config.processing_delay_ms, ttl, previous, t)
    out.log(t, router, f"{router} picks another random neighbour")
    p = config.processing_delay_ms
    for a, b in path_links(trial.path):
        link = after.get_link(a, b)
        out.log(t, a, f"{a} forwards to {b}")
        out.hops.append(Hop(a, b, t, t + link.delay_ms))
        out.path_taken.append(b)
        t += link.delay_ms
        out.log(t, b, f"packet arrives at {b}")
        t += p
    if trial.delivered:
        out.log(t, out.destination, f"packet delivered at {out.destination}")
        return "delivered", t
    out.log(t, out.path_taken[-1], f"packet dropped ({trial.reason})")
    return "dropped", t


def _travel(network: Network, path: list[str], start_ms: float, p: float,
            failure: LinkFailure | None, out: PacketResult,
            from_source: bool) -> tuple[str, str, float]:
    """Move the packet along `path` from path[0], ready to send at start_ms.

    Returns (outcome, router, time): outcome is "delivered", "stopped"
    (link down before the packet reached it) or "lost" (link failed while the
    packet was on it). `router` is where the packet now is (or last was).
    """
    t = start_ms
    if len(path) == 1:
        out.log(t, path[0], "Source and destination are the same: delivered at once")
        return "delivered", path[0], t
    for i, (u, v) in enumerate(path_links(path)):
        link = network.get_link(u, v)
        on_failing_link = failure is not None and failure.matches(u, v)
        if on_failing_link and t >= failure.time_ms:
            out.log(t, u, f"link {u}–{v} is down: packet stops at {u}")
            return "stopped", u, t
        verb = "sends packet to" if (from_source and i == 0) else "forwards to"
        out.log(t, u, f"{u} {verb} {v}")
        arrive = t + link.delay_ms
        if on_failing_link and failure.time_ms < arrive:
            fraction = (failure.time_ms - t) / link.delay_ms
            out.hops.append(Hop(u, v, t, failure.time_ms, lost=True, fraction=fraction))
            out.log(failure.time_ms, u, f"link {u}–{v} fails while the packet is on it: copy lost")
            return "lost", u, failure.time_ms
        out.hops.append(Hop(u, v, t, arrive))
        out.path_taken.append(v)
        out.log(arrive, v, f"packet arrives at {v}")
        t = arrive + p  # processing at the receiving router
        if v == path[-1]:
            out.log(t, v, f"packet delivered at {v}")
            return "delivered", v, t
    return "delivered", path[-1], t


def _simulate_flooding(network: Network, source: str, destination: str, config: SimConfig,
                       failure: LinkFailure | None, out: PacketResult) -> PacketResult:
    """Flooding: copies go everywhere; a failure only removes some copies."""
    p = config.processing_delay_ms
    failed = (failure.a, failure.b) if failure else None
    run = flood_packet(network, source, destination, p, failed,
                       failure.time_ms if failure else float("inf"))
    for event in run.events:
        out.log(event.time_ms, event.router, event.text)
    path = path_from_predecessors(run.parent, source, destination)
    if path is None or run.delivery_time_ms is None:
        _finish(out, "none" if failure is None else "dropped", 0)
        return out
    out.path_taken = path
    for a, b in path_links(path):  # animate the first copy that reached the destination
        depart = 0.0 if a == source else run.first_arrival[a] + p
        out.hops.append(Hop(a, b, depart, run.first_arrival[b]))
    _finish(out, "delivered", run.delivery_time_ms)
    return out


def _finish(out: PacketResult, outcome: str, t: float) -> None:
    """Set the final status of the packet."""
    if outcome == "delivered":
        out.delivered, out.delivery_time_ms = True, t
        out.status = f"Delivered in {fmt(t)} ms"
        if out.rerouted:
            out.status += f" (rerouted, waited {fmt(out.reconvergence_delay_ms)} ms)"
    else:
        out.delivered, out.delivery_time_ms = False, None
        out.status = NOT_DELIVERED


def _check_failure(network: Network, failure: LinkFailure | None) -> None:
    """The scheduled failure must be on an existing, active link at t >= 0."""
    if failure is None:
        return
    link = network.get_link(failure.a, failure.b)
    if link is None:
        raise PacketSimError(f"There is no link {failure.a}–{failure.b} to fail.")
    if not link.active:
        raise PacketSimError(f"Link {failure.a}–{failure.b} has already failed.")
    if failure.time_ms < 0:
        raise PacketSimError("Failure time must be 0 ms or later.")


def expected_delivery_time(network: Network, path: list[str], p: float) -> float:
    """Convenience re-export of the timing formula (sum of delays + p x hops)."""
    return path_delivery_time(network, path, p)
