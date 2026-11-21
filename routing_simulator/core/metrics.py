"""Run all algorithms on the same input, compare them fairly and pick the best.

Fair comparison rules:
  1. Only algorithms that fully delivered the packet are eligible. Random
     routing counts only with a 100% success rate ("not eligible (unreliable)").
  2. Lowest delivery time wins.
  3. Tie: fewest total messages for K packets
     (routing overhead + K x per-packet transmissions).
  4. Still tied: lowest computation time.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path

from algorithms import ALGORITHMS
from algorithms.base import RoutingAlgorithm, RoutingResult, fmt, path_delivery_time
from core.config import SimConfig
from core.graph import Network
from core.packet_sim import recover

TIE_TOLERANCE = 1e-9


@dataclass
class AlgorithmMetrics:
    """One row of the comparison table."""

    name: str
    result: RoutingResult
    computation_us: float
    k_packets: int
    eligible: bool
    status: str  # "eligible", "not delivered", "not eligible (unreliable)", ...

    @property
    def path_text(self) -> str:
        """Path as 'A→C→E' or '-'."""
        return "→".join(self.result.path) if self.result.path else "-"

    @property
    def delivery_text(self) -> str:
        """Delivery time in ms or 'Not delivered'."""
        if not self.result.delivered:
            return "Not delivered"
        return fmt(round(self.result.delivery_time_ms, 2))

    @property
    def total_messages(self) -> float:
        """Routing overhead + K x per-packet transmissions."""
        return self.result.total_messages(self.k_packets)

    def as_row(self) -> dict[str, str]:
        """Values for the table / CSV, in display form."""
        r = self.result
        return {
            "Algorithm": self.name,
            "Path": self.path_text,
            "Hops": "-" if r.hop_count is None else str(r.hop_count),
            "Delivery time (ms)": self.delivery_text,
            "Routing overhead": str(r.routing_overhead),
            "Per-packet transmissions": _num(r.per_packet_transmissions),
            f"Total messages (K={self.k_packets})": _num(self.total_messages),
            "Computation time (µs)": f"{self.computation_us:.1f}",
            "Rounds": "-" if r.rounds is None else str(r.rounds) + ("" if r.converged else " (did not converge)"),
            "Success rate": "-" if r.success_rate is None else f"{r.success_rate:.0%}",
            "Status": self.status,
        }


@dataclass
class Comparison:
    """Result of Run All & Compare."""

    rows: list[AlgorithmMetrics]
    best: AlgorithmMetrics | None
    explanation: str
    k_packets: int

    def row(self, name: str) -> AlgorithmMetrics:
        """Return the row of one algorithm by name."""
        return next(r for r in self.rows if r.name == name)


def measure_time_us(algorithm: RoutingAlgorithm, network: Network, source: str,
                    destination: str, config: SimConfig, runs: int) -> float:
    """Average computation time in microseconds over `runs` runs.

    Uses time.perf_counter. For random routing one run = all its trials.
    """
    start = time.perf_counter()
    for _ in range(runs):
        algorithm.compute(network, source, destination, config)
    return (time.perf_counter() - start) / runs * 1e6


def eligibility(result: RoutingResult) -> tuple[bool, str]:
    """Rule 1: may this algorithm be chosen as best?"""
    if not result.converged:
        return False, "not eligible (did not converge)"
    if not result.delivered:
        return False, "not delivered"
    if result.success_rate is not None and result.success_rate < 1:
        return False, "not eligible (unreliable)"
    return True, "eligible"


def select_best(rows: list[AlgorithmMetrics]) -> AlgorithmMetrics | None:
    """Rules 2-4: lowest delivery time, then fewest total messages, then
    lowest computation time. Returns None if nothing is eligible."""
    candidates = [r for r in rows if r.eligible]
    if not candidates:
        return None
    return min(candidates, key=lambda r: (round(r.result.delivery_time_ms / TIE_TOLERANCE),
                                          r.total_messages, r.computation_us))


def run_all(network: Network, source: str, destination: str, config: SimConfig,
            algorithms: list[RoutingAlgorithm] | None = None,
            timing_runs: int | None = None) -> Comparison:
    """Run every algorithm on the same network, endpoints and config."""
    config.validate()
    runs = config.timing_runs if timing_runs is None else timing_runs
    rows = []
    for algorithm in algorithms or ALGORITHMS:
        result = algorithm.compute(network, source, destination, config)
        micros = measure_time_us(algorithm, network, source, destination, config, runs)
        ok, status = eligibility(result)
        rows.append(AlgorithmMetrics(algorithm.name, result, micros, config.k_packets, ok, status))
    best = select_best(rows)
    if best is not None:
        best.status = "BEST"
    return Comparison(rows, best, explain(rows, best, config.k_packets), config.k_packets)


def explain(rows: list[AlgorithmMetrics], best: AlgorithmMetrics | None, k: int) -> str:
    """Plain-English explanation of why the best algorithm won."""
    by_name = {r.name: r for r in rows}
    if rows and rows[0].result.source == rows[0].result.destination:
        prefix = "Source and destination are the same, so every algorithm delivers at once. "
    else:
        prefix = ""
    if best is None:
        return prefix + "No algorithm delivered the packet, so there is no best algorithm."

    fastest = best.result.delivery_time_ms
    tied = [r for r in rows if r.eligible
            and abs(r.result.delivery_time_ms - fastest) <= TIE_TOLERANCE]
    names = _join([r.name for r in tied])
    parts = [prefix]
    if len(tied) > 1:
        parts.append(f"{names} all delivered in {fmt(fastest)} ms. ")
        costs = [f"{r.name} {'needs ' if i == 0 else ''}{_num(r.result.routing_overhead)} setup + "
                 f"{_num(k * r.result.per_packet_transmissions)} packet = "
                 f"{_num(r.total_messages)} messages" for i, r in enumerate(tied)]
        parts.append(f"For {k:,} packets, {_join(costs)}. ")
        same_msgs = [r for r in tied if abs(r.total_messages - best.total_messages) < 1e-9]
        if len(same_msgs) > 1:
            parts.append(f"{_join([r.name for r in same_msgs])} also tie on messages, so the "
                         f"lowest computation time ({best.computation_us:.1f} µs) decides. ")
    else:
        parts.append(f"{best.name} delivered fastest, in {fmt(fastest)} ms. ")
    parts.append(f"{best.name} is best for this network.")

    notes = []
    dij, bf = by_name.get("Dijkstra"), by_name.get("Bellman-Ford")
    if dij and bf:
        notes.append("Dijkstra and Bellman-Ford find the same shortest-path cost, so they are "
                     "separated by routing overhead and convergence, not by delivery time.")
    flood = by_name.get("Flooding")
    if flood and flood.result.delivered:
        notes.append(f"Flooding matches the fastest delivery time but wastes bandwidth on every "
                     f"packet ({_num(flood.result.per_packet_transmissions)} copies per packet).")
    hop = by_name.get("Hop Count")
    if hop and hop.result.delivered and hop.result.delivery_time_ms > fastest + TIE_TOLERANCE:
        notes.append(f"Hop Count takes {fmt(hop.result.delivery_time_ms)} ms because it picks "
                     f"the fewest hops, not the lowest delay.")
    rnd = by_name.get("Random")
    if rnd and rnd.result.success_rate is not None and rnd.result.success_rate < 1:
        notes.append(f"Random routing delivered only {rnd.result.success_rate:.0%} of its "
                     f"trials, so it is not eligible (unreliable).")
    return "".join(parts) + ("\n\n" + " ".join(notes) if notes else "")


def export_csv(comparison: Comparison, path: str | Path) -> None:
    """Save the comparison table as a CSV file (for the project report)."""
    rows = [r.as_row() for r in comparison.rows]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@dataclass
class FailureRow:
    """How one algorithm recovers from a link failure (Failure Test)."""

    name: str
    old_path: list[str] | None
    old_time_ms: float | None
    new_path: list[str] | None
    new_time_ms: float | None
    extra_messages: int
    rounds: int | None
    reconvergence_delay_ms: float
    converged: bool
    log: list[str]

    def as_row(self) -> dict[str, str]:
        """Display values for the Failure Test table."""
        def path(p: list[str] | None) -> str:
            return "→".join(p) if p else "-"

        def ms(t: float | None) -> str:
            return "Not delivered" if t is None else fmt(round(t, 2))

        return {
            "Algorithm": self.name,
            "Old path": path(self.old_path),
            "Old time (ms)": ms(self.old_time_ms),
            "New path": path(self.new_path),
            "New time (ms)": ms(self.new_time_ms),
            "Extra routing messages": str(self.extra_messages),
            "Rounds to reconverge": "-" if self.rounds is None else
            str(self.rounds) + ("" if self.converged else " (did not converge)"),
            "Reconvergence delay (ms)": fmt(self.reconvergence_delay_ms),
        }


def failure_test(network: Network, link: tuple[str, str], source: str, destination: str,
                 config: SimConfig,
                 algorithms: list[RoutingAlgorithm] | None = None) -> list[FailureRow]:
    """Fail `link`, reconverge every algorithm and report how each recovers.

    * Dijkstra: the two routers at the ends of the link each flood a new LSA
      (both floods counted); recovery takes one update period.
    * Bellman-Ford / Hop Count: warm-start reconvergence from the old tables;
      vectors sent during reconvergence are the extra messages.
    * Flooding / Random: no tables, so no extra messages; just send again.
    The given network is not modified.
    """
    config.validate()
    existing = network.get_link(*link)
    if existing is None or not existing.active:
        raise ValueError(f"Link {link[0]}–{link[1]} must exist and be active to fail it.")
    after = network.copy()
    after.fail_link(*link)
    rows = []
    for algorithm in algorithms or ALGORITHMS:
        before = algorithm.compute(network, source, destination, config)
        rec = recover(algorithm, after, before, source, destination, config, existing.key)
        if rec.result is not None:
            new_time = rec.result.delivery_time_ms
            new_path = rec.result.path
        else:
            new_path = rec.path
            new_time = (path_delivery_time(after, new_path, config.processing_delay_ms)
                        if new_path else None)
        rounds = rec.rounds if algorithm.name in ("Dijkstra", "Bellman-Ford", "Hop Count") else None
        rows.append(FailureRow(algorithm.name, before.path, before.delivery_time_ms, new_path,
                               new_time, rec.extra_messages, rounds, rec.delay_ms,
                               rec.converged, [rec.explain] + rec.log))
    return rows


def _join(items: list[str]) -> str:
    """'A', 'A and B', 'A, B and C'."""
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _num(value: float) -> str:
    """Number with thousands separators; one decimal only if needed."""
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.1f}"
