"""Simulation settings shared by every algorithm and the GUI."""

from __future__ import annotations

from dataclasses import dataclass, replace


class ConfigError(ValueError):
    """Raised when a setting is out of range (friendly message for the GUI)."""


@dataclass
class SimConfig:
    """All tunable simulation parameters.

    Attributes:
        processing_delay_ms: p, time every receiving router spends processing
            a packet (0-50 ms). Effective link cost = delay_ms + p.
        k_packets: K, number of data packets used for "total messages"
            (routing overhead + K x per-packet transmissions).
        update_period_ms: Time of one routing update period. Used to turn
            reconvergence rounds into a delay (rounds x period).
        seed: Seed for every random choice (random networks, random routing).
        split_horizon: Distance vector does not advertise a route back to the
            neighbour it was learned from.
        rerouting: During a packet transfer, recompute a new path after a link
            failure instead of dropping the packet.
        max_rounds: Safety cap for distance-vector rounds.
        random_trials: Number of trials for random routing.
        timing_runs: Runs averaged when measuring computation time.
    """

    processing_delay_ms: int = 1
    k_packets: int = 100
    update_period_ms: int = 5
    seed: int = 42
    split_horizon: bool = False
    rerouting: bool = True
    max_rounds: int = 500
    random_trials: int = 20
    timing_runs: int = 100

    def validate(self) -> "SimConfig":
        """Check every value is in range. Returns self so calls can be chained."""
        _check_int("Processing delay", self.processing_delay_ms, 0, 50)
        _check_int("K packets", self.k_packets, 1, 1_000_000)
        _check_int("Update period", self.update_period_ms, 1, 10_000)
        _check_int("Random seed", self.seed, -(2**31), 2**31 - 1)
        _check_int("Max rounds", self.max_rounds, 1, 100_000)
        _check_int("Random trials", self.random_trials, 1, 10_000)
        _check_int("Timing runs", self.timing_runs, 1, 100_000)
        return self

    def with_changes(self, **changes: object) -> "SimConfig":
        """Return a copy of this config with some fields changed."""
        return replace(self, **changes)


def _check_int(label: str, value: object, low: int, high: int) -> None:
    """Raise ConfigError unless value is an int in [low, high]."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{label} must be a whole number.")
    if not low <= value <= high:
        raise ConfigError(f"{label} must be between {low} and {high} (got {value}).")
