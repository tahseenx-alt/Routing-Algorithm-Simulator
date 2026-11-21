"""Main window: puts the canvas, control panel and bottom tabs together and
connects every button to the algorithm / simulation code.

Layout:
    +-----------------------------+--------------+
    |  network canvas (large)     | control panel|
    +-----------------------------+--------------+
    |  Routing Table | Log | Timeline  (tabs)    |
    +--------------------------------------------+
    |  status bar                                |
"""

from __future__ import annotations

import functools
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Any, Callable

from algorithms import get_algorithm
from algorithms.base import AlgorithmError, RoutingResult, Step
from core.config import ConfigError, SimConfig
from core.graph import (MAX_ROUTERS, MIN_ROUTERS, Network, NetworkError,
                        generate_random_network, link_key)
from core.storage import StorageError, load_network, load_sample, save_network
from algorithms.distance_vector import count_to_infinity_demo as build_count_demo
from core.metrics import failure_test as run_failure_test
from core.metrics import run_all as compare_all
from core.packet_sim import LinkFailure, PacketResult, PacketSimError, simulate_packet
from gui.canvas_view import NetworkCanvas, PacketTrack
from gui.control_panel import ControlPanel
from gui.log_panel import LogPanel
from gui.results_window import CountToInfinityWindow, FailureWindow, ResultsWindow
from gui.table_panel import RoutingTablePanel

FRIENDLY_ERRORS = (ConfigError, NetworkError, StorageError, AlgorithmError, PacketSimError,
                   ValueError)

# One colour per algorithm for packet dots (Run All draws them one after another).
PACKET_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


def _end_time(packet: PacketResult) -> float:
    """When the packet animation should stop: delivery or the last hop."""
    if packet.delivery_time_ms is not None:
        return packet.delivery_time_ms
    return max((h.arrive_ms for h in packet.hops), default=0.0) + 1


def safe(method: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator for GUI callbacks: show a friendly message instead of crashing."""

    @functools.wraps(method)
    def wrapper(self: "RoutingSimulatorApp", *args: Any, **kwargs: Any) -> Any:
        try:
            return method(self, *args, **kwargs)
        except FRIENDLY_ERRORS as exc:
            self.status(str(exc))
            messagebox.showwarning("Please check", str(exc), parent=self.root)
        except Exception as exc:  # last safety net: never crash the app
            self.status(f"Unexpected error: {exc}")
            messagebox.showerror("Unexpected error", f"{type(exc).__name__}: {exc}",
                                 parent=self.root)
        return None

    return wrapper


class RoutingSimulatorApp:
    """The application object: owns the network and all widgets."""

    def __init__(self, root: tk.Tk) -> None:
        """Build the window and load the small sample network."""
        self.root = root
        root.title("Routing Algorithm Simulator and Performance Comparator")
        root.geometry("1280x820")
        root.minsize(1000, 680)
        ttk.Style(root).theme_use("clam")

        self.network = Network()
        self.result: RoutingResult | None = None   # last visualised result
        self.steps: list[Step] = []
        self.step_index = 0
        self.play_job: str | None = None
        self.selected_router: str | None = None

        self._build_layout()
        self.load_sample("small")
        self.status("Welcome! Pick an algorithm, a source and a destination, then "
                    "press 'Visualize Selected'.")

    # ------------------------------------------------------------ layout
    def _build_layout(self) -> None:
        """Create canvas, control panel, notebook and status bar."""
        self.status_var = tk.StringVar()
        ttk.Label(self.root, textvariable=self.status_var, anchor="w", relief="sunken",
                  padding=(6, 2)).pack(side="bottom", fill="x")

        vertical = ttk.PanedWindow(self.root, orient="vertical")
        vertical.pack(fill="both", expand=True)
        top = ttk.Frame(vertical)
        vertical.add(top, weight=4)

        self.panel = ControlPanel(top, self)
        self.panel.pack(side="right", fill="y")
        self.canvas = NetworkCanvas(top, self.network, on_change=self.on_network_changed,
                                    on_select=self.on_router_selected,
                                    on_message=self.status, ask_delay=self.ask_delay)
        self.canvas.pack(side="left", fill="both", expand=True)

        self.tabs = ttk.Notebook(vertical)
        vertical.add(self.tabs, weight=1)
        self.table = RoutingTablePanel(self.tabs)
        self.log = LogPanel(self.tabs)
        self.timeline = LogPanel(self.tabs)
        self.tabs.add(self.table, text="Routing Table")
        self.tabs.add(self.log, text="Log")
        self.tabs.add(self.timeline, text="Timeline")
        # Give the canvas most of the height at start-up (the user can drag the sash).
        self.root.after(100, lambda: vertical.sashpos(0, max(420, self.root.winfo_height() - 250)))

    # ----------------------------------------------------------- helpers
    def status(self, text: str) -> None:
        """Show a message in the status bar."""
        self.status_var.set(text)

    def ask_delay(self, title: str, default: int) -> int | None:
        """Ask for a link delay (1-100 ms). Returns None if cancelled."""
        return simpledialog.askinteger("Link delay", title, initialvalue=default,
                                       minvalue=1, maxvalue=100, parent=self.root)

    def config(self) -> SimConfig:
        """Current settings from the control panel (validated)."""
        return self.panel.get_config()

    def endpoints(self) -> tuple[str, str]:
        """Selected source and destination, checked to exist."""
        source, dest = self.panel.source.get(), self.panel.destination.get()
        if len(self.network.routers) < MIN_ROUTERS:
            raise NetworkError("Add at least two routers first.")
        if source not in self.network.routers or dest not in self.network.routers:
            raise NetworkError("Please choose a source and a destination router.")
        return source, dest

    def set_network(self, network: Network, message: str) -> None:
        """Replace the whole network (New / Open / Sample / Random)."""
        self.network = network
        self.canvas.set_network(network)
        self.selected_router = None
        self.on_network_changed()
        self.panel.source.set(network.router_ids()[0] if network.routers else "")
        self.panel.destination.set(network.router_ids()[-1] if network.routers else "")
        self.status(message)

    # ------------------------------------------------------ network events
    def on_network_changed(self) -> None:
        """Any edit invalidates previous results (they must be rerun)."""
        self.pause()
        self.result, self.steps, self.step_index = None, [], 0
        self.canvas.clear_overlay()
        self.panel.set_step_label(0, 0)
        links = [f"{l.a}-{l.b}" for l in self.network.sorted_links() if l.active]
        self.panel.refresh_routers(self.network.router_ids(), links)
        if self.selected_router not in self.network.routers:
            self.selected_router = None
        self.refresh_table()

    def on_router_selected(self, router_id: str | None) -> None:
        """Clicking a router shows its routing table."""
        self.selected_router = router_id
        self.refresh_table()
        self.tabs.select(self.table)

    def on_selection_changed(self) -> None:
        """Algorithm / source / destination dropdown changed."""
        self.refresh_table()

    def set_mode(self, mode: str) -> None:
        """Edit mode radio button changed."""
        self.canvas.set_mode(mode)

    def refresh_table(self) -> None:
        """Compute the selected algorithm's table for the selected router."""
        router = self.selected_router
        if router is None:
            self.table.show_message("Click a router to see its routing table.")
            return
        try:
            config = self.config()
        except ConfigError as exc:
            self.table.show_message(str(exc))
            return
        algorithm = get_algorithm(self.panel.algorithm.get())
        if algorithm.name in ("Flooding", "Random"):
            self.table.show_message(f"{algorithm.name} keeps no routing table: router {router} "
                                    f"forwards {'to every neighbour' if algorithm.name == 'Flooding' else 'to a random neighbour'}.")
            return
        result = algorithm.compute(self.network, router, router, config)
        unit = "hops" if algorithm.name == "Hop Count" else "ms (delay + p)"
        self.table.show(f"Routing table of {router} ({algorithm.name}, cost in {unit})",
                        result.routing_table, own_id=router)

    # ------------------------------------------------------- file buttons
    @safe
    def new_network(self) -> None:
        """Start with an empty canvas."""
        self.set_network(Network(), "New empty network. Choose 'Add Router' and click the canvas.")
        self.panel.mode.set("add_router")
        self.canvas.set_mode("add_router")

    @safe
    def load_sample(self, name: str) -> None:
        """Load one of the bundled sample networks."""
        if not name:
            raise StorageError("Choose a sample first.")
        self.set_network(load_sample(name), f"Loaded sample '{name}'.")

    @safe
    def random_network(self, n_text: str) -> None:
        """Generate a random connected network with N routers."""
        try:
            n = int(n_text)
        except ValueError:
            raise NetworkError("Number of routers must be a whole number.") from None
        if not MIN_ROUTERS <= n <= MAX_ROUTERS:
            raise NetworkError(f"Number of routers must be between {MIN_ROUTERS} and {MAX_ROUTERS}.")
        seed = self.config().seed
        self.set_network(generate_random_network(n, seed),
                         f"Random connected network with {n} routers (seed {seed}).")

    @safe
    def save_file(self) -> None:
        """Save the network as JSON."""
        path = filedialog.asksaveasfilename(parent=self.root, defaultextension=".json",
                                            filetypes=[("Network JSON", "*.json")])
        if path:
            save_network(self.network, path)
            self.status(f"Saved to {path}")

    @safe
    def open_file(self) -> None:
        """Open a network JSON file (validated, friendly error if invalid)."""
        path = filedialog.askopenfilename(parent=self.root,
                                          filetypes=[("Network JSON", "*.json"), ("All", "*.*")])
        if path:
            self.set_network(load_network(path), f"Opened {path}")

    # ------------------------------------------------------------ actions
    @safe
    def visualize_selected(self) -> None:
        """Run the selected algorithm and show its first step."""
        source, dest = self.endpoints()
        config = self.config()
        algorithm = get_algorithm(self.panel.algorithm.get())
        result = algorithm.compute(self.network, source, dest, config)
        self.pause()
        self.canvas.stop_animation()
        self.canvas.legend = []
        self.result = result
        self.steps = result.steps
        self.step_index = 0
        lines = [s.description for s in self.steps]
        details = [line for line in result.log if line not in lines]
        if details:  # e.g. "Round 2: B updates cost to D = 5 via A", random trials
            lines += ["", "Details:"] + details
        self.log.set_lines(lines, heading=f"{algorithm.name}: {source} → {dest}")
        self._show_current_step()
        self.selected_router = source
        self.refresh_table()
        self.tabs.select(self.log)
        summary = self.summary(result)
        self.status(f"{summary}  Use Step / Play to walk through {len(self.steps)} steps.")

    def summary(self, result: RoutingResult) -> str:
        """One-line description of a result for the status bar."""
        if result.note and not result.delivered:
            return f"{result.name}: {result.note}"
        text = f"{result.name}: path {'→'.join(result.path)}, {result.delivery_time_ms:g} ms"
        if result.note:
            text = f"{result.name}: {result.note}."
        return text

    @safe
    def run_all(self) -> None:
        """Run all five algorithms, open the results window and animate each
        algorithm's packet one after another in its own colour."""
        source, dest = self.endpoints()
        config = self.config()
        self.status("Running all algorithms (timing 100 runs each)…")
        self.root.config(cursor="watch")
        self.root.update_idletasks()
        try:
            comparison = compare_all(self.network, source, dest, config)
        finally:
            self.root.config(cursor="")
        ResultsWindow(self.root, comparison, f"{source} → {dest}, p = "
                      f"{config.processing_delay_ms} ms, K = {config.k_packets}")

        tracks, lines = [], []
        for i, row in enumerate(comparison.rows):
            packet = simulate_packet(self.network, get_algorithm(row.name), source, dest,
                                     config, initial=row.result)
            tracks.append(PacketTrack(packet.hops, PACKET_COLORS[i % len(PACKET_COLORS)],
                                      row.name, _end_time(packet), packet.delivered))
            lines += [f"--- {row.name}: {packet.status}"] + packet.timeline
        self.timeline.set_lines(lines, heading=f"Run All: {source} → {dest}")
        self.log.set_lines(comparison.explanation.split("\n"), heading="Comparison")
        self.tabs.select(self.timeline)
        self.pause()
        self.steps = []
        best = comparison.best
        self.canvas.show_path(best.result.path if best else None,
                              caption=f"Run All: best = {best.name if best else 'none'}")
        self.canvas.animate_packets(tracks, self.packet_speed, sequential=True,
                                    on_done=lambda: self.status("Run All animation finished."))
        self.status(comparison.explanation.split("\n")[0])

    @safe
    def send_packet(self) -> None:
        """Simulate one packet hop by hop and animate it as a moving dot."""
        source, dest = self.endpoints()
        config = self.config()
        algorithm = get_algorithm(self.panel.algorithm.get())
        failure = self.scheduled_failure()
        initial = algorithm.compute(self.network, source, dest, config)
        packet = simulate_packet(self.network, algorithm, source, dest, config, failure, initial)

        heading = f"{algorithm.name}: packet {source} → {dest}"
        if failure:
            heading += (f"  (link {failure.a}–{failure.b} fails at t={failure.time_ms:g} ms, "
                        f"rerouting {'ON' if config.rerouting else 'OFF'})")
        self.timeline.set_lines(packet.timeline + ["", f"Result: {packet.status}"], heading)
        self.tabs.select(self.timeline)
        self.pause()
        self.steps = []
        self.canvas.show_path(initial.path, caption=heading)
        for hop in packet.hops:  # also highlight any detour taken after a failure
            if not hop.lost:
                self.canvas.overlay.links.add(link_key(hop.src, hop.dst))
        track = PacketTrack(packet.hops, PACKET_COLORS[0], algorithm.name,
                            _end_time(packet), packet.delivered)
        fail_mark = (failure.a, failure.b, failure.time_ms) if failure else None
        self.canvas.animate_packets([track], self.packet_speed, failure=fail_mark,
                                    on_done=lambda: self.status(f"{algorithm.name}: {packet.status}"))
        self.status(f"Sending packet with {algorithm.name}…")

    def scheduled_failure(self) -> LinkFailure | None:
        """Read the optional 'fail link X-Y at t ms' inputs."""
        if not self.panel.fail_enabled.get():
            return None
        text = self.panel.fail_link.get()
        if "-" not in text:
            raise NetworkError("Choose which link should fail.")
        a, b = text.split("-")
        try:
            time_ms = float(self.panel.fail_time.get())
        except ValueError:
            raise ConfigError("Failure time must be a number of milliseconds.") from None
        if time_ms < 0:
            raise ConfigError("Failure time must be 0 ms or later.")
        return LinkFailure(a, b, time_ms)

    def packet_speed(self) -> float:
        """Simulated milliseconds per real second for the speed slider."""
        return 2.0 * max(1.0, self.panel.speed.get()) ** 1.5

    @safe
    def failure_test(self) -> None:
        """Fail a link, reconverge every algorithm and show how each recovers."""
        source, dest = self.endpoints()
        config = self.config()
        active = [f"{l.a}-{l.b}" for l in self.network.active_links()]
        if not active:
            raise NetworkError("There is no active link to fail.")
        # Suggest the last link of the current shortest path: that forces a reroute.
        path = get_algorithm("Dijkstra").compute(self.network, source, dest, config).path
        default = active[0]
        if path and len(path) > 1:
            default = "-".join(sorted(path[-2:]))
        choice = self.choose_link("Failure Test", "Which link should fail?", active, default)
        if choice is None:
            return
        link = tuple(choice.split("-"))
        rows = run_failure_test(self.network, link, source, dest, config)
        self.network.fail_link(*link)  # show the failure on the canvas
        self.on_network_changed()
        new_path = rows[0].new_path
        self.canvas.show_path(new_path, caption=f"Link {link[0]}–{link[1]} failed. "
                              f"Dijkstra's new path: {'→'.join(new_path) if new_path else 'none'}")
        lines = []
        for row in rows:
            lines += [f"--- {row.name}"] + row.log
        self.log.set_lines(lines, heading=f"Failure Test: link {link[0]}–{link[1]} failed")
        self.tabs.select(self.log)
        FailureWindow(self.root, rows, link, f"{source} → {dest}")
        self.status(f"Link {link[0]}–{link[1]} failed. Click it in 'Fail/Restore Link' mode "
                    f"to restore it.")

    def choose_link(self, title: str, prompt: str, options: list[str],
                    default: str) -> str | None:
        """Small modal dialog with a dropdown of links. Returns None if cancelled."""
        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.transient(self.root)
        dialog.resizable(False, False)
        var = tk.StringVar(value=default)
        chosen: list[str] = []
        ttk.Label(dialog, text=prompt, padding=8).pack()
        ttk.Combobox(dialog, textvariable=var, values=options, state="readonly",
                     width=10).pack(padx=10)
        buttons = ttk.Frame(dialog, padding=8)
        buttons.pack()

        def ok() -> None:
            chosen.append(var.get())
            dialog.destroy()

        ttk.Button(buttons, text="Fail link", command=ok).pack(side="left", padx=4)
        ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side="left", padx=4)
        dialog.bind("<Return>", lambda _e: ok())
        dialog.grab_set()
        self.root.wait_window(dialog)
        return chosen[0] if chosen else None

    @safe
    def count_to_infinity_demo(self) -> None:
        """Linear A–B–C with/without split horizon, then the triangle loop."""
        config = self.config()
        linear = load_sample("count_to_infinity")
        triangle = load_sample("triangle_loop")
        demos = (
            build_count_demo(linear, ("B", "C"), "C", ["A", "B"], False, config),
            build_count_demo(linear, ("B", "C"), "C", ["A", "B"], True, config),
        )
        tri = build_count_demo(triangle, ("C", "D"), "D", ["A", "B", "C"], True, config)

        def show(tab: int) -> None:
            """Draw the network of the selected demo tab, with its failed link."""
            name, link = (("count_to_infinity", ("B", "C")) if tab == 0
                          else ("triangle_loop", ("C", "D")))
            net = load_sample(name)
            net.fail_link(*link)
            self.set_network(net, f"Count-to-infinity demo: {name}.json with link "
                                  f"{link[0]}–{link[1]} failed.")
            demo_logs = demos[0].log if tab == 0 else tri.log
            self.log.set_lines(demo_logs, heading="Reconvergence log (hop metric)")
            self.tabs.select(self.log)

        show(0)
        CountToInfinityWindow(self.root, demos, tri, show)

    # ---------------------------------------------------------- animation
    def _show_current_step(self) -> None:
        """Draw step number self.step_index on the canvas."""
        if not self.steps:
            return
        step = self.steps[self.step_index]
        caption = f"{self.result.name if self.result else ''} – step {self.step_index + 1}: " \
                  f"{step.description}"
        self.canvas.show_step(step, caption)
        self.log.highlight(self.step_index)
        self.panel.set_step_label(self.step_index + 1, len(self.steps))

    def step_forward(self) -> None:
        """Show the next precomputed step."""
        if self.steps and self.step_index < len(self.steps) - 1:
            self.step_index += 1
            self._show_current_step()
        elif not self.steps:
            self.status("Press 'Visualize Selected' first.")

    def step_back(self) -> None:
        """Show the previous step (steps are precomputed, so this is easy)."""
        if self.steps and self.step_index > 0:
            self.step_index -= 1
            self._show_current_step()

    def play(self) -> None:
        """Advance automatically; the speed slider sets the interval."""
        if not self.steps:
            self.status("Press 'Visualize Selected' first.")
            return
        self.pause()
        if self.step_index >= len(self.steps) - 1:
            self.step_index = -1

        def tick() -> None:
            if self.step_index < len(self.steps) - 1:
                self.step_index += 1
                self._show_current_step()
                self.play_job = self.root.after(self._interval(), tick)
            else:
                self.play_job = None

        tick()

    def pause(self) -> None:
        """Stop automatic stepping."""
        if self.play_job is not None:
            self.root.after_cancel(self.play_job)
            self.play_job = None

    def reset(self) -> None:
        """Back to the first step (or a clean canvas)."""
        self.pause()
        self.canvas.stop_animation()
        if self.steps:
            self.step_index = 0
            self._show_current_step()
        else:
            self.canvas.clear_overlay()

    def _interval(self) -> int:
        """Milliseconds between steps for the current slider position."""
        return int(2200 / max(1.0, self.panel.speed.get()))


def run() -> None:
    """Create the Tk root window and start the event loop."""
    root = tk.Tk()
    RoutingSimulatorApp(root)
    root.mainloop()
