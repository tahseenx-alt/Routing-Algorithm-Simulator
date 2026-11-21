"""Synchronous DISTANCE VECTOR engine (the idea behind RIP).

How distance vector routing works:
  * Every router only knows its own neighbours and a table ("distance
    vector") of its best known cost to every destination.
  * Periodically (one *round*), every router sends its vector to each
    neighbour. Each vector sent is one routing message.
  * After receiving the vectors, each router recomputes its table with the
    Bellman-Ford equation:
        D_x(y) = min over neighbours v of [ cost(x, v) + D_v(y) ]
  * Repeat until a round changes nothing: the network has *converged*.

This one engine serves two algorithms:
  * metric = "delay": cost(x, v) = delay + p  (Bellman-Ford, our delay-based RIP)
  * metric = "hops":  cost(x, v) = 1, infinity = 16  (real RIP metric)

Split horizon: a router does not advertise a route back to the neighbour it
learned that route from (that destination is left out, which the receiver
treats as infinity). This stops simple two-router loops.

Warm start: reconverge() continues from an already converged state after a
link failure, which is exactly how the famous *count-to-infinity* problem
appears in real networks.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from algorithms.base import fmt
from core.config import SimConfig
from core.graph import Link, Network

HOP_INFINITY = 16  # RIP's "unreachable" value
METRICS = ("delay", "hops")

Vector = dict[str, float]


def link_cost(link: Link, metric: str, processing_delay_ms: float) -> float:
    """Cost of crossing one link: 1 hop, or delay + processing delay."""
    return 1 if metric == "hops" else link.effective_cost(processing_delay_ms)


def infinity_for(network: Network, metric: str, processing_delay_ms: float) -> float:
    """Value that means "unreachable".

    Hops: 16, as in RIP. Delay: (sum of effective costs of ALL links) + 1,
    which is larger than any real path, so a long route (e.g. one 18 ms link)
    is never mistaken for unreachable, yet count-to-infinity still ends.
    """
    if metric == "hops":
        return HOP_INFINITY
    return network.total_effective_cost(processing_delay_ms) + 1


@dataclass
class DVState:
    """The routing tables of every router at one moment.

    Attributes:
        metric: "delay" or "hops".
        split_horizon: Whether split horizon is used when advertising.
        infinity: The "unreachable" value for this metric.
        processing_delay_ms: p used for delay-metric link costs.
        dist: dist[x][y] = router x's current cost to y.
        next_hop: next_hop[x][y] = neighbour x uses to reach y (None if none).
        received: received[x][v] = the vector neighbour v last sent to x.
    """

    metric: str
    split_horizon: bool
    infinity: float
    processing_delay_ms: float
    dist: dict[str, Vector] = field(default_factory=dict)
    next_hop: dict[str, dict[str, str | None]] = field(default_factory=dict)
    received: dict[str, dict[str, Vector]] = field(default_factory=dict)

    def copy(self) -> "DVState":
        """Deep copy, so a warm start never changes the original state."""
        return copy.deepcopy(self)


@dataclass
class Change:
    """One table entry that changed in a round."""

    router: str
    destination: str
    old_cost: float
    new_cost: float
    via: str | None


@dataclass
class RoundRecord:
    """What happened in one round (used for logs, animation and the demo table)."""

    number: int
    messages: int
    changes: list[Change]
    dist: dict[str, Vector]
    next_hop: dict[str, dict[str, str | None]]


@dataclass
class DVRun:
    """Outcome of running rounds until convergence (or the max_rounds cap).

    Attributes:
        state: Final routing tables.
        rounds: Rounds up to and including the last round that changed a table.
        rounds_run: Rounds actually executed (includes the final no-change round).
        messages: Vectors sent in every round that was run.
        converged: False if max_rounds was reached while tables still changed.
        history: One RoundRecord per round.
        log: Human-readable lines ("Round 2: B updates cost to D = 5 via A").
    """

    state: DVState
    rounds: int = 0
    rounds_run: int = 0
    messages: int = 0
    converged: bool = True
    history: list[RoundRecord] = field(default_factory=list)
    log: list[str] = field(default_factory=list)


# --------------------------------------------------------------- engine
def initial_state(network: Network, metric: str, split_horizon: bool,
                  processing_delay_ms: float) -> DVState:
    """Cold start: cost to itself 0, to everything else infinity."""
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}")
    inf = infinity_for(network, metric, processing_delay_ms)
    ids = network.router_ids()
    state = DVState(metric, split_horizon, inf, processing_delay_ms)
    for x in ids:
        state.dist[x] = {y: (0 if y == x else inf) for y in ids}
        state.next_hop[x] = {y: None for y in ids}
        state.received[x] = {}
    return state


def advertise(state: DVState, sender: str, receiver: str) -> Vector:
    """The vector `sender` sends to `receiver` this round.

    With split horizon, routes learned from `receiver` are left out; the
    receiver then treats them as infinity (we store infinity directly).
    """
    vector: Vector = {}
    for dest, cost in state.dist[sender].items():
        if state.split_horizon and state.next_hop[sender][dest] == receiver:
            vector[dest] = state.infinity  # omitted by split horizon
        else:
            vector[dest] = cost
    return vector


def recompute(network: Network, state: DVState, x: str) -> tuple[Vector, dict[str, str | None]]:
    """Bellman-Ford equation for router x, from scratch:

        D_x(y) = min over active neighbours v of [cost(x, v) + D_v(y)]

    using the vectors x last received. Costs are capped at infinity.
    Ties go to the neighbour with the smallest id (deterministic).
    """
    inf = state.infinity
    dist: Vector = {}
    nh: dict[str, str | None] = {}
    neighbours = [(v, link) for v, link in network.neighbors(x) if v in state.received[x]]
    for y in state.dist[x]:
        if y == x:
            dist[y], nh[y] = 0, None
            continue
        best, best_via = inf, None
        for v, link in neighbours:
            candidate = link_cost(link, state.metric, state.processing_delay_ms) \
                + state.received[x][v].get(y, inf)
            if candidate < best:
                best, best_via = candidate, v
        dist[y], nh[y] = (best, best_via) if best < inf else (inf, None)
    return dist, nh


def _apply(state: DVState, x: str, dist: Vector, nh: dict[str, str | None],
           round_no: int, log: list[str]) -> list[Change]:
    """Store router x's new table and log each entry that changed."""
    changes = []
    for y in dist:
        old = state.dist[x][y]
        if old != dist[y] or state.next_hop[x][y] != nh[y]:
            changes.append(Change(x, y, old, dist[y], nh[y]))
            if dist[y] >= state.infinity:
                log.append(f"Round {round_no}: {x} marks {y} unreachable (cost {fmt(dist[y])})")
            else:
                log.append(f"Round {round_no}: {x} updates cost to {y} = {fmt(dist[y])} via {nh[y]}")
    state.dist[x], state.next_hop[x] = dist, nh
    return changes


def run_rounds(network: Network, state: DVState, max_rounds: int,
               run: DVRun | None = None) -> DVRun:
    """Run synchronous rounds on `state` (modified in place) until no change.

    Each round:
      1. Every router sends its vector to each active neighbour (1 message each).
      2. All routers recompute their tables at the same time.
    The final round that changes nothing is still counted in messages.
    """
    run = run or DVRun(state=state)
    ids = network.router_ids()
    for round_no in range(1, max_rounds + 1):
        # 1. Exchange: build every advertisement from the OLD tables first.
        outbox = []
        for x in ids:
            for v in network.neighbor_ids(x):
                outbox.append((x, v, advertise(state, x, v)))
        for sender, receiver, vector in outbox:
            state.received[receiver][sender] = vector
        # 2. Everyone recomputes simultaneously.
        new_tables = {x: recompute(network, state, x) for x in ids}
        changes: list[Change] = []
        for x in ids:
            changes += _apply(state, x, *new_tables[x], round_no, run.log)

        run.rounds_run = round_no
        run.messages += len(outbox)
        run.history.append(RoundRecord(round_no, len(outbox), changes,
                                       copy.deepcopy(state.dist), copy.deepcopy(state.next_hop)))
        if not changes:
            run.log.append(f"Round {round_no}: no table changed, converged "
                           f"after {run.rounds} round(s), {run.messages} vectors sent.")
            run.converged = True
            return run
        run.rounds = round_no
    run.converged = False
    run.log.append(f"Stopped after {max_rounds} rounds: did not converge.")
    return run


def converge(network: Network, metric: str, split_horizon: bool, config: SimConfig) -> DVRun:
    """Cold start: build all routing tables from nothing."""
    state = initial_state(network, metric, split_horizon, config.processing_delay_ms)
    return run_rounds(network, state, config.max_rounds)


def reconverge(network: Network, previous_state: DVState, config: SimConfig) -> DVRun:
    """Warm start after a topology change (e.g. a link failure).

    The routers at the ends of a failed link notice immediately that the
    neighbour is gone: they forget that neighbour's vector and recompute from
    the vectors they still have (round 0). Then normal rounds continue from
    the OLD tables. Because the other routers still hold stale routes, this is
    where count-to-infinity can happen.
    """
    state = previous_state.copy()
    run = DVRun(state=state)
    for x in network.router_ids():
        active = set(network.neighbor_ids(x))
        lost = sorted(v for v in state.received[x] if v not in active)
        if not lost:
            continue
        for v in lost:
            del state.received[x][v]
            run.log.append(f"Round 0: {x} detects that link {x}–{v} is down")
        _apply(state, x, *recompute(network, state, x), 0, run.log)
    run.history.append(RoundRecord(0, 0, [], copy.deepcopy(state.dist),
                                   copy.deepcopy(state.next_hop)))
    return run_rounds(network, state, config.max_rounds, run)


def follow_next_hops(state: DVState, source: str, destination: str) -> list[str] | None:
    """Build a path by following next hops, like a packet would.

    Returns None if a hop is missing or a routing loop is detected.
    """
    path = [source]
    while path[-1] != destination:
        hop = state.next_hop[path[-1]].get(destination)
        if hop is None or hop in path:
            return None
        path.append(hop)
    return path


def as_distances(state: DVState, router: str) -> dict[str, float]:
    """Router's vector with 'infinity' turned into math.inf for display."""
    return {y: (float("inf") if c >= state.infinity else c) for y, c in state.dist[router].items()}


# ------------------------------------------------- count-to-infinity demo
@dataclass
class CountToInfinityDemo:
    """Round-by-round costs after a failure, for the demo table.

    Attributes:
        title: e.g. "Without split horizon".
        destination: The router that became unreachable.
        watch: Routers whose cost to `destination` is shown.
        rows: (round label, {router: cost}) - "before", then round 0, 1, 2...
        rounds: Rounds to reconverge.
        messages: Vectors sent during reconvergence.
        converged: False if max_rounds was hit.
        reached_infinity: True if any watched cost climbed all the way to infinity
            through finite values bigger than the original (= counting to infinity).
        infinity: The infinity value (16 for hops).
        log: Detailed reconvergence log.
    """

    title: str
    destination: str
    watch: list[str]
    rows: list[tuple[str, dict[str, float]]]
    rounds: int
    messages: int
    converged: bool
    reached_infinity: bool
    infinity: float
    log: list[str]


def count_to_infinity_demo(network: Network, failed_link: tuple[str, str], destination: str,
                           watch: list[str], split_horizon: bool, config: SimConfig,
                           metric: str = "hops") -> CountToInfinityDemo:
    """Converge, fail `failed_link`, then record how costs to `destination` change.

    The given network is not modified.
    """
    net = network.copy()
    before = converge(net, metric, split_horizon, config)
    net.fail_link(*failed_link)
    run = reconverge(net, before.state, config)
    inf = run.state.infinity
    rows = [("before", {w: before.state.dist[w][destination] for w in watch})]
    for record in run.history:
        rows.append((str(record.number), {w: record.dist[w][destination] for w in watch}))
    original = rows[0][1]
    climbed = any(
        any(original[w] < row[w] < inf for _, row in rows[1:]) and rows[-1][1][w] >= inf
        for w in watch
    )
    title = "With split horizon" if split_horizon else "Without split horizon"
    return CountToInfinityDemo(title, destination, list(watch), rows, run.rounds, run.messages,
                               run.converged, climbed, inf, list(run.log))
