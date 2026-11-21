"""Network model: routers, links and the network that holds them.

In computer networks a *router* is a node that forwards packets and a *link*
is the physical (or logical) connection between two routers. Every link has a
propagation/transmission delay in milliseconds. A link can also *fail*
(cable cut, interface down); a failed link stays in the network so it can be
restored later, but routing algorithms must not use it.

This module has no GUI code, so the algorithms and the tests can use it
directly.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Iterator

# Router ids are single letters. 15 routers (A..O) keeps every simple path
# under 15 hops, so RIP's hop-count infinity of 16 is always valid.
ROUTER_IDS: str = "ABCDEFGHIJKLMNO"
MIN_ROUTERS: int = 2
MAX_ROUTERS: int = len(ROUTER_IDS)  # 15
MIN_DELAY_MS: int = 1
MAX_DELAY_MS: int = 100


class NetworkError(ValueError):
    """Raised for any invalid change to the network (friendly message)."""


@dataclass
class Router:
    """A router drawn on the canvas.

    Attributes:
        id: Single-letter name, "A" to "O".
        x: Horizontal canvas position in pixels.
        y: Vertical canvas position in pixels.
    """

    id: str
    x: float
    y: float


@dataclass
class Link:
    """An undirected link between two routers.

    Attributes:
        a: Id of the first router (always the alphabetically smaller one).
        b: Id of the second router.
        delay_ms: Link delay in milliseconds (integer, 1-100).
        active: False when the link has failed.
    """

    a: str
    b: str
    delay_ms: int
    active: bool = True

    @property
    def key(self) -> tuple[str, str]:
        """Return the sorted (a, b) pair that identifies this link."""
        return link_key(self.a, self.b)

    def other(self, router_id: str) -> str:
        """Return the router at the other end of the link."""
        return self.b if router_id == self.a else self.a

    def effective_cost(self, processing_delay_ms: float) -> float:
        """Cost used by shortest-path algorithms: link delay + processing delay.

        Every router that receives the packet spends p ms processing it, so
        crossing a link really costs delay_ms + p (see README, timing model).
        """
        return self.delay_ms + processing_delay_ms


def link_key(a: str, b: str) -> tuple[str, str]:
    """Return a canonical (sorted) key for the undirected link a-b."""
    return (a, b) if a <= b else (b, a)


def validate_delay(delay_ms: object) -> int:
    """Check that a delay is an integer from 1 to 100 and return it as int."""
    if isinstance(delay_ms, bool):
        raise NetworkError("Delay must be a whole number of milliseconds.")
    if isinstance(delay_ms, float) and delay_ms.is_integer():
        delay_ms = int(delay_ms)
    if not isinstance(delay_ms, int):
        raise NetworkError("Delay must be a whole number of milliseconds.")
    if not MIN_DELAY_MS <= delay_ms <= MAX_DELAY_MS:
        raise NetworkError(
            f"Delay must be between {MIN_DELAY_MS} and {MAX_DELAY_MS} ms "
            f"(got {delay_ms})."
        )
    return delay_ms


class Network:
    """A collection of routers and undirected links.

    Routers are kept in a dict keyed by id; links in a dict keyed by the
    sorted pair of router ids, so A-B and B-A are the same link.
    """

    def __init__(self) -> None:
        """Create an empty network."""
        self.routers: dict[str, Router] = {}
        self.links: dict[tuple[str, str], Link] = {}

    # ------------------------------------------------------------------ routers
    def router_ids(self) -> list[str]:
        """Return all router ids in alphabetical order."""
        return sorted(self.routers)

    def next_free_id(self) -> str | None:
        """Return the first unused id A..O, or None if the network is full."""
        for rid in ROUTER_IDS:
            if rid not in self.routers:
                return rid
        return None

    def add_router(self, x: float, y: float, router_id: str | None = None) -> Router:
        """Add a router at (x, y). Uses the next free letter if no id is given."""
        if len(self.routers) >= MAX_ROUTERS:
            raise NetworkError(f"A network can have at most {MAX_ROUTERS} routers.")
        if router_id is None:
            router_id = self.next_free_id()
        assert router_id is not None
        if router_id not in ROUTER_IDS:
            raise NetworkError(f"Router id must be one letter from A to O (got {router_id!r}).")
        if router_id in self.routers:
            raise NetworkError(f"Router {router_id} already exists.")
        router = Router(router_id, float(x), float(y))
        self.routers[router_id] = router
        return router

    def remove_router(self, router_id: str) -> None:
        """Remove a router together with every link attached to it."""
        self._require_router(router_id)
        for key in [k for k in self.links if router_id in k]:
            del self.links[key]
        del self.routers[router_id]

    def move_router(self, router_id: str, x: float, y: float) -> None:
        """Move a router to a new canvas position."""
        router = self._require_router(router_id)
        router.x, router.y = float(x), float(y)

    # -------------------------------------------------------------------- links
    def add_link(self, a: str, b: str, delay_ms: int, active: bool = True) -> Link:
        """Connect routers a and b with a link of the given delay."""
        self._require_router(a)
        self._require_router(b)
        if a == b:
            raise NetworkError("A link must connect two different routers.")
        key = link_key(a, b)
        if key in self.links:
            raise NetworkError(f"Link {key[0]}-{key[1]} already exists.")
        link = Link(key[0], key[1], validate_delay(delay_ms), bool(active))
        self.links[key] = link
        return link

    def remove_link(self, a: str, b: str) -> None:
        """Delete the link between a and b."""
        self.links.pop(self._require_link(a, b).key)

    def get_link(self, a: str, b: str) -> Link | None:
        """Return the link between a and b, or None if there is none."""
        return self.links.get(link_key(a, b))

    def set_delay(self, a: str, b: str, delay_ms: int) -> None:
        """Change the delay of the link a-b."""
        self._require_link(a, b).delay_ms = validate_delay(delay_ms)

    def fail_link(self, a: str, b: str) -> None:
        """Mark link a-b as failed (it stays in the network but is unusable)."""
        self._require_link(a, b).active = False

    def restore_link(self, a: str, b: str) -> None:
        """Bring a failed link a-b back up."""
        self._require_link(a, b).active = True

    def toggle_link(self, a: str, b: str) -> bool:
        """Fail an active link or restore a failed one. Returns the new state."""
        link = self._require_link(a, b)
        link.active = not link.active
        return link.active

    def active_links(self) -> list[Link]:
        """Return all links that are currently up."""
        return [link for link in self.sorted_links() if link.active]

    def sorted_links(self) -> list[Link]:
        """Return all links (active and failed) sorted by their key."""
        return [self.links[k] for k in sorted(self.links)]

    # ---------------------------------------------------------------- topology
    def neighbors(self, router_id: str) -> list[tuple[str, Link]]:
        """Return (neighbour id, link) for every *active* link of a router.

        The list is sorted by neighbour id so every algorithm is deterministic.
        """
        result = []
        for link in self.links.values():
            if link.active and router_id in (link.a, link.b):
                result.append((link.other(router_id), link))
        result.sort(key=lambda item: item[0])
        return result

    def neighbor_ids(self, router_id: str) -> list[str]:
        """Return the ids of the active neighbours of a router."""
        return [n for n, _ in self.neighbors(router_id)]

    def reachable_from(self, source: str) -> set[str]:
        """Return all routers reachable from source over active links (BFS)."""
        if source not in self.routers:
            return set()
        seen = {source}
        queue = deque([source])
        while queue:
            current = queue.popleft()
            for nbr in self.neighbor_ids(current):
                if nbr not in seen:
                    seen.add(nbr)
                    queue.append(nbr)
        return seen

    def is_connected(self) -> bool:
        """True when every router can reach every other over active links."""
        if not self.routers:
            return True
        first = self.router_ids()[0]
        return len(self.reachable_from(first)) == len(self.routers)

    def total_effective_cost(self, processing_delay_ms: float) -> float:
        """Sum of (delay + p) over ALL links, active or failed.

        Used to build the delay-metric "infinity" for distance vector.
        """
        return sum(link.effective_cost(processing_delay_ms) for link in self.links.values())

    def iter_routers(self) -> Iterator[Router]:
        """Iterate over routers in id order."""
        for rid in self.router_ids():
            yield self.routers[rid]

    # ------------------------------------------------------------------- misc
    def copy(self) -> "Network":
        """Return an independent deep copy of the network."""
        clone = Network()
        for r in self.iter_routers():
            clone.add_router(r.x, r.y, r.id)
        for link in self.sorted_links():
            clone.add_link(link.a, link.b, link.delay_ms, link.active)
        return clone

    def __eq__(self, other: object) -> bool:
        """Two networks are equal when they have the same routers and links."""
        if not isinstance(other, Network):
            return NotImplemented
        return self.routers == other.routers and self.links == other.links

    def __repr__(self) -> str:
        """Short text form, e.g. Network(routers=5, links=7)."""
        return f"Network(routers={len(self.routers)}, links={len(self.links)})"

    def _require_router(self, router_id: str) -> Router:
        """Return a router or raise a friendly NetworkError."""
        if router_id not in self.routers:
            raise NetworkError(f"Router {router_id} does not exist.")
        return self.routers[router_id]

    def _require_link(self, a: str, b: str) -> Link:
        """Return a link or raise a friendly NetworkError."""
        link = self.get_link(a, b)
        if link is None:
            raise NetworkError(f"There is no link between {a} and {b}.")
        return link


def generate_random_network(
    n: int,
    seed: int,
    width: int = 760,
    height: int = 540,
    min_delay: int = 1,
    max_delay: int = 20,
) -> Network:
    """Create a random *connected* network with n routers.

    Steps (all driven by random.Random(seed), so the same seed gives the same
    network):
      1. Place routers roughly on a circle so the drawing is readable.
      2. Build a random spanning tree: each new router links to a random
         router already in the tree. This guarantees connectivity.
      3. Add a few extra random links to create alternative paths/loops.
      4. Every link gets a random delay from min_delay to max_delay ms.
    """
    if not MIN_ROUTERS <= n <= MAX_ROUTERS:
        raise NetworkError(f"Number of routers must be between {MIN_ROUTERS} and {MAX_ROUTERS}.")
    rng = random.Random(seed)
    net = Network()
    cx, cy = width / 2, height / 2
    radius = min(width, height) / 2 - 60
    for i in range(n):
        angle = 2 * math.pi * i / n - math.pi / 2
        jitter = rng.uniform(-0.12, 0.12) * radius
        x = cx + (radius + jitter) * math.cos(angle)
        y = cy + (radius + jitter) * math.sin(angle)
        net.add_router(round(x), round(y), ROUTER_IDS[i])

    ids = net.router_ids()
    order = ids[:]
    rng.shuffle(order)
    for i in range(1, n):  # random spanning tree
        net.add_link(order[i], rng.choice(order[:i]), rng.randint(min_delay, max_delay))

    max_links = n * (n - 1) // 2
    extra = min(max_links - (n - 1), max(1, n // 2))
    attempts = 0
    while extra > 0 and attempts < 1000:  # a few extra links for alternative paths
        attempts += 1
        a, b = rng.sample(ids, 2)
        if net.get_link(a, b) is None:
            net.add_link(a, b, rng.randint(min_delay, max_delay))
            extra -= 1
    return net
