"""Save and load networks as JSON files.

File format:
    {
      "routers": [{"id": "A", "x": 100, "y": 120}],
      "links":   [{"a": "A", "b": "B", "delay_ms": 4, "active": true}]
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.graph import MAX_ROUTERS, ROUTER_IDS, Network, NetworkError, validate_delay

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


class StorageError(ValueError):
    """Raised when a network file cannot be read; message is user friendly."""


def network_to_dict(network: Network) -> dict[str, Any]:
    """Convert a network into a plain JSON-ready dict."""
    return {
        "routers": [
            {"id": r.id, "x": _num(r.x), "y": _num(r.y)} for r in network.iter_routers()
        ],
        "links": [
            {"a": l.a, "b": l.b, "delay_ms": l.delay_ms, "active": l.active}
            for l in network.sorted_links()
        ],
    }


def network_from_dict(data: Any) -> Network:
    """Build a network from a dict, validating everything.

    Checks: correct structure, unique router ids from A to O, at most 15
    routers, links refer to existing routers, delays are 1-100 ms, no
    duplicate links. Raises StorageError with a friendly message.
    """
    if not isinstance(data, dict):
        raise StorageError("The file must contain a JSON object with 'routers' and 'links'.")
    routers = data.get("routers")
    links = data.get("links", [])
    if not isinstance(routers, list) or not isinstance(links, list):
        raise StorageError("'routers' and 'links' must both be lists.")
    if len(routers) > MAX_ROUTERS:
        raise StorageError(f"Too many routers: {len(routers)} (maximum is {MAX_ROUTERS}).")

    network = Network()
    for i, item in enumerate(routers, start=1):
        if not isinstance(item, dict):
            raise StorageError(f"Router #{i} must be an object.")
        rid, x, y = item.get("id"), item.get("x", 0), item.get("y", 0)
        if not isinstance(rid, str) or rid not in ROUTER_IDS:
            raise StorageError(f"Router #{i} has an invalid id {rid!r} (use one letter A-O).")
        if rid in network.routers:
            raise StorageError(f"Router id {rid} is used more than once.")
        if not _is_number(x) or not _is_number(y):
            raise StorageError(f"Router {rid} needs numeric x and y positions.")
        network.add_router(x, y, rid)

    for i, item in enumerate(links, start=1):
        if not isinstance(item, dict):
            raise StorageError(f"Link #{i} must be an object.")
        a, b = item.get("a"), item.get("b")
        if a not in network.routers or b not in network.routers:
            raise StorageError(f"Link #{i} ({a}-{b}) refers to a router that does not exist.")
        active = item.get("active", True)
        if not isinstance(active, bool):
            raise StorageError(f"Link {a}-{b}: 'active' must be true or false.")
        try:
            delay = validate_delay(item.get("delay_ms"))
            network.add_link(a, b, delay, active)
        except NetworkError as exc:
            raise StorageError(f"Link {a}-{b}: {exc}") from exc
    return network


def save_network(network: Network, path: str | Path) -> None:
    """Write the network to a JSON file."""
    data = network_to_dict(network)
    # One router / link per line keeps the file short and easy to read.
    parts = []
    for section in ("routers", "links"):
        rows = ",\n".join(f"    {json.dumps(item)}" for item in data[section])
        parts.append(f'  "{section}": [\n{rows}\n  ]' if rows else f'  "{section}": []')
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("{\n" + ",\n".join(parts) + "\n}\n")


def load_network(path: str | Path) -> Network:
    """Read and validate a network JSON file. Raises StorageError on problems."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError as exc:
        raise StorageError(f"File not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise StorageError(f"Not a valid JSON file (line {exc.lineno}): {exc.msg}") from exc
    except OSError as exc:
        raise StorageError(f"Could not read file: {exc}") from exc
    return network_from_dict(data)


def list_samples() -> list[str]:
    """Return the names (without .json) of the bundled sample networks."""
    return sorted(p.stem for p in SAMPLES_DIR.glob("*.json"))


def load_sample(name: str) -> Network:
    """Load one of the bundled sample networks by name, e.g. "small"."""
    return load_network(SAMPLES_DIR / f"{name}.json")


def _is_number(value: Any) -> bool:
    """True for int/float values (but not bool)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _num(value: float) -> float | int:
    """Store whole-number coordinates as ints to keep files tidy."""
    return int(value) if float(value).is_integer() else value
