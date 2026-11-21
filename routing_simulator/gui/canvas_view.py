"""The network canvas: draws routers and links, handles editing with the
mouse, and shows algorithm steps and moving packets.

The canvas never runs algorithms itself. It only draws the Network and the
"overlay" (highlighted routers/links and distance labels) the app gives it.
"""

from __future__ import annotations

import math
import tkinter as tk
from dataclasses import dataclass, field
from typing import Callable

from algorithms.base import Step, fmt
from core.graph import Network, NetworkError, link_key

ROUTER_RADIUS = 19

# Colours used for the visualisation (kept in one place so they are easy to change).
COLORS = {
    "bg": "#f8fafc",
    "router": "#dbeafe",
    "router_outline": "#1e3a8a",
    "current": "#facc15",      # yellow: router being worked on
    "finalized": "#86efac",    # green: router whose answer is final
    "selected": "#f97316",     # orange outline: selected router
    "link": "#64748b",
    "failed": "#dc2626",       # red dashed: failed link
    "path": "#2563eb",         # thick blue: chosen path / highlighted link
    "label_bg": "#ffffff",
    "dist": "#7c3aed",
    "pending": "#f97316",
}

EDIT_MODES = {
    "add_router": "Add Router",
    "add_link": "Add Link",
    "move": "Move",
    "delete": "Delete",
    "set_delay": "Set Delay",
    "toggle_link": "Fail/Restore Link",
}


@dataclass
class Overlay:
    """What to highlight on top of the plain network drawing."""

    current: set[str] = field(default_factory=set)
    finalized: set[str] = field(default_factory=set)
    links: set[tuple[str, str]] = field(default_factory=set)
    distances: dict[str, float] = field(default_factory=dict)
    caption: str = ""

    @classmethod
    def from_step(cls, step: Step, caption: str = "") -> "Overlay":
        """Build an overlay from one algorithm Step."""
        return cls(set(step.current), set(step.finalized),
                   {link_key(a, b) for a, b in step.links if a and b},
                   dict(step.distances), caption)


@dataclass
class PacketTrack:
    """One packet to animate: its hops (src, dst, depart, arrive, lost, fraction)."""

    hops: list
    color: str
    label: str
    end_time_ms: float
    delivered: bool


class NetworkCanvas(tk.Canvas):
    """Canvas that draws and edits a Network.

    Callbacks (set by the app):
        on_change(): the network was edited.
        on_select(router_id | None): a router was clicked.
        on_message(text): show a friendly message in the status bar.
        ask_delay(title, default) -> int | None: ask the user for a delay.
    """

    def __init__(self, master: tk.Misc, network: Network,
                 on_change: Callable[[], None],
                 on_select: Callable[[str | None], None],
                 on_message: Callable[[str], None],
                 ask_delay: Callable[[str, int], int | None]) -> None:
        """Create the canvas and bind the mouse."""
        super().__init__(master, bg=COLORS["bg"], highlightthickness=0, width=780, height=540)
        self.network = network
        self.on_change, self.on_select = on_change, on_select
        self.on_message, self.ask_delay = on_message, ask_delay
        self.mode = "move"
        self.selected: str | None = None
        self.link_start: str | None = None  # first router clicked in Add Link mode
        self.overlay = Overlay()
        self.legend: list[tuple[str, str]] = []
        self._drag: str | None = None
        self._anim_job: str | None = None
        self.bind("<Button-1>", self._on_click)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Configure>", lambda _e: self.redraw())

    # ------------------------------------------------------------- public
    def set_network(self, network: Network) -> None:
        """Show a different network (after New / Open / Random)."""
        self.network = network
        self.selected = None
        self.link_start = None
        self.clear_overlay()

    def set_mode(self, mode: str) -> None:
        """Switch edit mode (see EDIT_MODES)."""
        self.mode = mode
        self.link_start = None
        self.redraw()
        hints = {
            "add_router": "Click on empty space to add a router (max 15).",
            "add_link": "Click two routers to connect them.",
            "move": "Drag a router to move it. Click a router to see its routing table.",
            "delete": "Click a router or a link to delete it.",
            "set_delay": "Click a link to change its delay.",
            "toggle_link": "Click a link to fail it (red dashed) or restore it.",
        }
        self.on_message(hints.get(mode, ""))

    def show_step(self, step: Step, caption: str = "") -> None:
        """Highlight one algorithm step."""
        self.overlay = Overlay.from_step(step, caption)
        self.redraw()

    def show_path(self, path: list[str] | None, caption: str = "") -> None:
        """Highlight a whole path in thick blue."""
        links = {link_key(a, b) for a, b in zip(path or [], (path or [])[1:])}
        self.overlay = Overlay(finalized=set(path or []), links=links, caption=caption)
        self.redraw()

    def clear_overlay(self) -> None:
        """Remove every highlight, distance label, legend and packet."""
        self.stop_animation()
        self.overlay = Overlay()
        self.legend = []
        self.redraw()

    # ------------------------------------------------------------ drawing
    def redraw(self) -> None:
        """Draw everything from scratch (networks are small, so this is fast)."""
        self.delete("all")
        for link in self.network.sorted_links():
            self._draw_link(link)
        for router in self.network.iter_routers():
            self._draw_router(router.id, router.x, router.y)
        if self.overlay.caption:
            self.create_text(10, 10, anchor="nw", text=self.overlay.caption, width=560,
                             font=("TkDefaultFont", 10, "bold"), fill="#0f172a", tags="caption")
        self._draw_legend()
        if not self.network.routers:
            self.create_text(self.winfo_width() / 2 or 390, 270, fill="#94a3b8",
                             text="Empty network: choose 'Add Router' and click here,\n"
                                  "or use Load Sample / Random Network.", justify="center")

    def _draw_link(self, link) -> None:
        """A link is a line with its delay written in the middle."""
        a, b = self.network.routers[link.a], self.network.routers[link.b]
        highlighted = link.key in self.overlay.links
        if not link.active:
            color, width, dash = COLORS["failed"], 2, (6, 4)
        elif highlighted:
            color, width, dash = COLORS["path"], 5, ()
        else:
            color, width, dash = COLORS["link"], 2, ()
        self.create_line(a.x, a.y, b.x, b.y, fill=color, width=width, dash=dash,
                         tags=("link", f"link:{link.a}{link.b}"))
        mx, my = (a.x + b.x) / 2, (a.y + b.y) / 2
        text = f"{link.delay_ms} ms" + ("" if link.active else " ✕")
        label = self.create_text(mx, my, text=text, font=("TkDefaultFont", 9),
                                 fill=COLORS["failed"] if not link.active else "#0f172a",
                                 tags=("link", f"link:{link.a}{link.b}"))
        x1, y1, x2, y2 = self.bbox(label)
        box = self.create_rectangle(x1 - 2, y1 - 1, x2 + 2, y2 + 1, fill=COLORS["label_bg"],
                                    outline=color, tags=("link", f"link:{link.a}{link.b}"))
        self.tag_lower(box, label)

    def _draw_router(self, rid: str, x: float, y: float) -> None:
        """A router is a labelled circle; colour shows its animation state."""
        fill = COLORS["router"]
        if rid in self.overlay.finalized:
            fill = COLORS["finalized"]
        if rid in self.overlay.current:
            fill = COLORS["current"]
        outline, width = COLORS["router_outline"], 2
        if rid == self.selected:
            outline, width = COLORS["selected"], 4
        if rid == self.link_start:
            outline, width = COLORS["pending"], 4
        r = ROUTER_RADIUS
        self.create_oval(x - r, y - r, x + r, y + r, fill=fill, outline=outline, width=width,
                         tags=("router", f"router:{rid}"))
        self.create_text(x, y, text=rid, font=("TkDefaultFont", 12, "bold"),
                         tags=("router", f"router:{rid}"))
        if rid in self.overlay.distances:
            value = fmt(self.overlay.distances[rid])
            label = self.create_text(x + r + 4, y - r - 2, text=value, anchor="w",
                                     font=("TkDefaultFont", 10, "bold"), fill="white", tags="dist")
            x1, y1, x2, y2 = self.bbox(label)
            box = self.create_rectangle(x1 - 3, y1 - 1, x2 + 3, y2 + 1, fill=COLORS["dist"],
                                        outline="", tags="dist")
            self.tag_lower(box, label)

    def _draw_legend(self) -> None:
        """Legend for the Run All packet colours (top-right corner)."""
        if not self.legend:
            return
        x = max(self.winfo_width(), 780) - 170
        y = 10
        height = 20 * len(self.legend) + 10
        self.create_rectangle(x - 8, y - 4, x + 160, y + height, fill="white", outline="#cbd5e1",
                              tags="legend")
        for i, (color, label) in enumerate(self.legend):
            cy = y + 10 + 20 * i
            self.create_oval(x, cy - 6, x + 12, cy + 6, fill=color, outline="", tags="legend")
            self.create_text(x + 20, cy, text=label, anchor="w", font=("TkDefaultFont", 9),
                             tags="legend")

    # -------------------------------------------------------- hit testing
    def router_at(self, x: float, y: float) -> str | None:
        """Router whose circle contains (x, y), if any."""
        for router in self.network.iter_routers():
            if math.hypot(router.x - x, router.y - y) <= ROUTER_RADIUS + 2:
                return router.id
        return None

    def link_at(self, x: float, y: float) -> tuple[str, str] | None:
        """Link whose line passes within 7 px of (x, y), if any."""
        best, best_d = None, 7.0
        for link in self.network.sorted_links():
            a, b = self.network.routers[link.a], self.network.routers[link.b]
            d = _point_segment_distance(x, y, a.x, a.y, b.x, b.y)
            if d <= best_d:
                best, best_d = link.key, d
        return best

    # -------------------------------------------------------------- mouse
    def _on_click(self, event: tk.Event) -> None:
        """Dispatch a click according to the edit mode."""
        try:
            self._handle_click(event.x, event.y)
        except NetworkError as exc:
            self.on_message(str(exc))
            self.bell()

    def _handle_click(self, x: float, y: float) -> None:
        """Edit-mode logic for one click."""
        rid = self.router_at(x, y)
        link = None if rid else self.link_at(x, y)
        mode = self.mode
        if mode == "add_router" and rid is None and link is None:
            new = self.network.add_router(x, y)
            self._changed(f"Added router {new.id}.")
        elif mode == "add_link" and rid:
            self._add_link_click(rid)
        elif mode == "move" and rid:
            self._drag = rid
        elif mode == "delete" and rid:
            self.network.remove_router(rid)
            self.selected = None if self.selected == rid else self.selected
            self._changed(f"Deleted router {rid} and its links.")
            return
        elif mode == "delete" and link:
            self.network.remove_link(*link)
            self._changed(f"Deleted link {link[0]}–{link[1]}.")
        elif mode == "set_delay" and link:
            current = self.network.get_link(*link).delay_ms
            value = self.ask_delay(f"Delay of link {link[0]}–{link[1]} (1-100 ms)", current)
            if value is not None:
                self.network.set_delay(*link, value)
                self._changed(f"Link {link[0]}–{link[1]} delay set to {value} ms.")
        elif mode == "toggle_link" and link:
            up = self.network.toggle_link(*link)
            self._changed(f"Link {link[0]}–{link[1]} {'restored' if up else 'FAILED'}.")
        if rid and mode != "delete":
            self.selected = rid
            self.on_select(rid)
            self.redraw()

    def _add_link_click(self, rid: str) -> None:
        """First click remembers a router, second click creates the link."""
        if self.link_start is None or self.link_start == rid:
            self.link_start = rid
            self.on_message(f"Link from {rid}: now click the second router.")
            return
        a, b = self.link_start, rid
        self.link_start = None
        if self.network.get_link(a, b):
            raise NetworkError(f"Link {a}–{b} already exists.")
        value = self.ask_delay(f"Delay of new link {a}–{b} (1-100 ms)", 5)
        if value is not None:
            self.network.add_link(a, b, value)
            self._changed(f"Added link {a}–{b} ({value} ms).")

    def _on_drag(self, event: tk.Event) -> None:
        """Move the dragged router with the mouse (kept inside the canvas)."""
        if self.mode == "move" and self._drag:
            x = min(max(event.x, ROUTER_RADIUS), max(self.winfo_width(), 100) - ROUTER_RADIUS)
            y = min(max(event.y, ROUTER_RADIUS), max(self.winfo_height(), 100) - ROUTER_RADIUS)
            self.network.move_router(self._drag, x, y)
            self.redraw()

    def _on_release(self, _event: tk.Event) -> None:
        """Finish a drag (moving does not change any routing result)."""
        self._drag = None

    def _changed(self, message: str) -> None:
        """Tell the app the network changed, then redraw."""
        self.on_message(message)
        self.on_change()
        self.redraw()

    # ---------------------------------------------------------- animation
    def stop_animation(self) -> None:
        """Cancel a running packet animation."""
        if self._anim_job is not None:
            self.after_cancel(self._anim_job)
            self._anim_job = None
        self.delete("packet")

    def animate_packets(self, tracks: list[PacketTrack], speed: Callable[[], float],
                        on_done: Callable[[], None] | None = None,
                        sequential: bool = True,
                        failure: tuple[str, str, float] | None = None) -> None:
        """Move a dot along each track's hops with a running ms timer.

        speed() returns simulated milliseconds per real second, read every
        frame so the slider works while playing. Tracks play one after another
        (sequential=True) or all at once. `failure` = (a, b, time) marks the
        link with a red cross once the simulated clock passes that time.
        """
        self.stop_animation()
        self.legend = [(t.color, t.label) for t in tracks] if len(tracks) > 1 else []
        self.redraw()
        queue = list(tracks) if sequential else [tracks]
        frame_ms = 30

        def start_next() -> None:
            if not queue:
                self._anim_job = None
                if on_done:
                    on_done()
                return
            group = queue.pop(0)
            group = group if isinstance(group, list) else [group]
            end = max((t.end_time_ms for t in group), default=0)
            run_frame(group, 0.0, end)

        def run_frame(group: list[PacketTrack], t: float, end: float) -> None:
            self.delete("packet")
            if failure and t >= failure[2]:
                self._draw_failure_mark(failure[0], failure[1])
            for i, track in enumerate(group):
                self._draw_packet(track, t, offset=i * 3)
            names = ", ".join(tr.label for tr in group)
            self.create_text(10, max(self.winfo_height(), 540) - 12, anchor="sw",
                             text=f"{names}   t = {t:.1f} ms", font=("TkFixedFont", 12, "bold"),
                             fill="#0f172a", tags="packet")
            if t >= end:
                self._anim_job = self.after(700, start_next)
                return
            step = speed() * frame_ms / 1000.0
            self._anim_job = self.after(frame_ms, run_frame, group, min(end, t + step), end)

        start_next()

    def _draw_failure_mark(self, a: str, b: str) -> None:
        """Red cross + dashed line on a link that just failed mid-transfer."""
        ra, rb = self.network.routers.get(a), self.network.routers.get(b)
        if ra is None or rb is None:
            return
        self.create_line(ra.x, ra.y, rb.x, rb.y, fill=COLORS["failed"], width=3, dash=(6, 4),
                         tags="packet")
        mx, my = (ra.x + rb.x) / 2, (ra.y + rb.y) / 2
        self.create_text(mx, my - 16, text="✕ link down", fill=COLORS["failed"],
                         font=("TkDefaultFont", 10, "bold"), tags="packet")

    def _draw_packet(self, track: PacketTrack, t: float, offset: int = 0) -> None:
        """Draw one packet dot at simulated time t.

        Between hops the dot waits at a router (processing / reconvergence).
        A copy lost on a failing link leaves a red cross where it was lost.
        """
        pos: tuple[float, float] | None = None
        lost_marks = []
        gone = False  # the only copy was lost and nothing follows
        for i, hop in enumerate(track.hops):
            a, b = self.network.routers.get(hop.src), self.network.routers.get(hop.dst)
            if a is None or b is None:
                continue
            if t < hop.depart_ms:
                pos = pos or (a.x, a.y)
                break
            span = max(hop.arrive_ms - hop.depart_ms, 1e-9)
            frac = min(1.0, (t - hop.depart_ms) / span)
            if hop.lost:
                frac *= hop.fraction
            here = (a.x + (b.x - a.x) * frac, a.y + (b.y - a.y) * frac)
            if t <= hop.arrive_ms:
                pos = here
                break
            if hop.lost:
                lost_marks.append(here)
                pos = (a.x, a.y)  # the router still has the packet and may reroute it
                gone = i == len(track.hops) - 1
                continue
            pos = (b.x, b.y)
        for x, y in lost_marks:
            self.create_text(x, y, text="✕", fill=COLORS["failed"],
                             font=("TkDefaultFont", 16, "bold"), tags="packet")
        if pos is None or gone:
            return
        x, y = pos[0] + offset, pos[1] - offset
        self.create_oval(x - 7, y - 7, x + 7, y + 7, fill=track.color, outline="black",
                         tags="packet")


def _point_segment_distance(px: float, py: float, ax: float, ay: float,
                            bx: float, by: float) -> float:
    """Shortest distance from point P to segment AB (used to click on links)."""
    dx, dy = bx - ax, by - ay
    if dx == dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))
