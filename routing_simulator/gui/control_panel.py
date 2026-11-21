"""Right-hand control panel: edit modes, file buttons, algorithm choice,
actions, animation controls and simulation settings.

The panel only collects input. Every button calls a method on the app,
which does the actual work.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

from algorithms import ALGORITHMS
from core.config import ConfigError, SimConfig
from core.storage import list_samples
from gui.canvas_view import EDIT_MODES

if TYPE_CHECKING:  # only for type hints, avoids a circular import
    from gui.app import RoutingSimulatorApp


class ScrollableFrame(ttk.Frame):
    """A frame with a vertical scrollbar, so the panel fits small screens."""

    def __init__(self, master: tk.Misc, width: int = 300) -> None:
        """Create the canvas + inner frame + scrollbar."""
        super().__init__(master)
        self.canvas = tk.Canvas(self, width=width, highlightthickness=0)
        bar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.inner.bind("<Configure>",
                        lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=bar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.inner.bind_all(seq, self._wheel, add="+")

    def _wheel(self, event: tk.Event) -> None:
        """Scroll with the mouse wheel when the pointer is over the panel."""
        widget = self.winfo_containing(event.x_root, event.y_root)
        while widget is not None and widget is not self:
            widget = widget.master
        if widget is None:
            return
        if getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0:
            self.canvas.yview_scroll(-2, "units")
        else:
            self.canvas.yview_scroll(2, "units")


class ControlPanel(ScrollableFrame):
    """All buttons, dropdowns, sliders and inputs on the right."""

    def __init__(self, master: tk.Misc, app: "RoutingSimulatorApp") -> None:
        """Build every section of the panel."""
        super().__init__(master)
        self.app = app
        self.mode = tk.StringVar(value="move")
        self.sample = tk.StringVar()
        self.random_n = tk.StringVar(value="8")
        self.algorithm = tk.StringVar(value=ALGORITHMS[0].name)
        self.source = tk.StringVar()
        self.destination = tk.StringVar()
        self.speed = tk.DoubleVar(value=4)
        self.p = tk.StringVar(value="1")
        self.k = tk.StringVar(value="100")
        self.period = tk.StringVar(value="5")
        self.seed = tk.StringVar(value="42")
        self.split_horizon = tk.BooleanVar(value=False)
        self.rerouting = tk.BooleanVar(value=True)
        self.fail_enabled = tk.BooleanVar(value=False)
        self.fail_link = tk.StringVar()
        self.fail_time = tk.StringVar(value="3")

        body = self.inner
        self._build_edit(body)
        self._build_file(body)
        self._build_algorithm(body)
        self._build_actions(body)
        self._build_animation(body)
        self._build_settings(body)
        self._build_packet(body)
        # Make the scroll area exactly as wide as its contents (no clipping).
        self.inner.update_idletasks()
        self.canvas.configure(width=self.inner.winfo_reqwidth())

    # ---------------------------------------------------------- sections
    def _section(self, parent: tk.Misc, title: str) -> ttk.LabelFrame:
        """A titled box that stretches to the panel width."""
        frame = ttk.LabelFrame(parent, text=title, padding=6)
        frame.pack(fill="x", padx=6, pady=4)
        return frame

    def _build_edit(self, parent: tk.Misc) -> None:
        """Edit mode radio buttons (two columns)."""
        box = self._section(parent, "Edit mode")
        for i, (key, label) in enumerate(EDIT_MODES.items()):
            ttk.Radiobutton(box, text=label, value=key, variable=self.mode,
                            command=lambda: self.app.set_mode(self.mode.get())
                            ).grid(row=i // 2, column=i % 2, sticky="w", padx=2)

    def _build_file(self, parent: tk.Misc) -> None:
        """New / Load Sample / Random / Save / Open."""
        box = self._section(parent, "Network")
        ttk.Button(box, text="New", command=self.app.new_network).grid(row=0, column=0, sticky="ew")
        ttk.Button(box, text="Open…", command=self.app.open_file).grid(row=0, column=1, sticky="ew")
        ttk.Button(box, text="Save…", command=self.app.save_file).grid(row=0, column=2, sticky="ew")
        samples = list_samples()
        self.sample.set("small" if "small" in samples else (samples[0] if samples else ""))
        ttk.Combobox(box, textvariable=self.sample, values=samples, state="readonly",
                     width=16).grid(row=1, column=0, columnspan=2, sticky="ew", pady=3)
        ttk.Button(box, text="Load Sample",
                   command=lambda: self.app.load_sample(self.sample.get())
                   ).grid(row=1, column=2, sticky="ew")
        ttk.Label(box, text="Routers (2-15):").grid(row=2, column=0, sticky="w")
        ttk.Spinbox(box, from_=2, to=15, textvariable=self.random_n, width=5
                    ).grid(row=2, column=1, sticky="w")
        ttk.Button(box, text="Random Network",
                   command=lambda: self.app.random_network(self.random_n.get())
                   ).grid(row=2, column=2, sticky="ew")
        for c in range(3):
            box.columnconfigure(c, weight=1)

    def _build_algorithm(self, parent: tk.Misc) -> None:
        """Algorithm, source and destination dropdowns."""
        box = self._section(parent, "Routing")
        names = [a.name for a in ALGORITHMS]
        rows = [("Algorithm", self.algorithm, names), ("Source", self.source, []),
                ("Destination", self.destination, [])]
        self.router_boxes: list[ttk.Combobox] = []
        for i, (label, var, values) in enumerate(rows):
            ttk.Label(box, text=label).grid(row=i, column=0, sticky="w")
            combo = ttk.Combobox(box, textvariable=var, values=values, state="readonly", width=16)
            combo.grid(row=i, column=1, sticky="ew", pady=1)
            combo.bind("<<ComboboxSelected>>", lambda _e: self.app.on_selection_changed())
            if label != "Algorithm":
                self.router_boxes.append(combo)
        box.columnconfigure(1, weight=1)

    def _build_actions(self, parent: tk.Misc) -> None:
        """The five main action buttons."""
        box = self._section(parent, "Actions")
        actions = [
            ("Visualize Selected", self.app.visualize_selected),
            ("Run All & Compare", self.app.run_all),
            ("Send Packet", self.app.send_packet),
            ("Failure Test", self.app.failure_test),
            ("Count-to-Infinity Demo", self.app.count_to_infinity_demo),
        ]
        for text, command in actions:
            ttk.Button(box, text=text, command=command).pack(fill="x", pady=1)

    def _build_animation(self, parent: tk.Misc) -> None:
        """Step / Back / Play / Pause / Reset and the speed slider."""
        box = self._section(parent, "Animation")
        buttons = [("◀ Back", self.app.step_back), ("Step ▶", self.app.step_forward),
                   ("Play", self.app.play), ("Pause", self.app.pause), ("Reset", self.app.reset)]
        for i, (text, command) in enumerate(buttons):
            ttk.Button(box, text=text, command=command, width=7
                       ).grid(row=i // 3, column=i % 3, sticky="ew", padx=1, pady=1)
        ttk.Label(box, text="Speed").grid(row=2, column=0, sticky="w")
        ttk.Scale(box, from_=1, to=10, variable=self.speed, orient="horizontal"
                  ).grid(row=2, column=1, columnspan=2, sticky="ew")
        self.step_label = ttk.Label(box, text="Step 0 / 0", foreground="#475569")
        self.step_label.grid(row=3, column=0, columnspan=3, sticky="w")
        for c in range(3):
            box.columnconfigure(c, weight=1)

    def _build_settings(self, parent: tk.Misc) -> None:
        """Numeric inputs and the Split Horizon / Rerouting checkboxes."""
        box = self._section(parent, "Settings")
        fields = [("Processing delay p (ms)", self.p), ("K packets", self.k),
                  ("Update period (ms)", self.period), ("Random seed", self.seed)]
        for i, (label, var) in enumerate(fields):
            ttk.Label(box, text=label).grid(row=i, column=0, sticky="w")
            ttk.Entry(box, textvariable=var, width=8).grid(row=i, column=1, sticky="e", pady=1)
        ttk.Checkbutton(box, text="Split Horizon", variable=self.split_horizon
                        ).grid(row=4, column=0, sticky="w")
        ttk.Checkbutton(box, text="Rerouting", variable=self.rerouting
                        ).grid(row=4, column=1, sticky="w")
        box.columnconfigure(0, weight=1)

    def _build_packet(self, parent: tk.Misc) -> None:
        """Optional link failure scheduled for Send Packet."""
        box = self._section(parent, "Send Packet: scheduled failure")
        ttk.Checkbutton(box, text="Fail link", variable=self.fail_enabled
                        ).grid(row=0, column=0, sticky="w")
        self.fail_box = ttk.Combobox(box, textvariable=self.fail_link, state="readonly", width=7)
        self.fail_box.grid(row=0, column=1, sticky="w")
        ttk.Label(box, text="at t =").grid(row=1, column=0, sticky="e")
        ttk.Entry(box, textvariable=self.fail_time, width=6).grid(row=1, column=1, sticky="w")
        ttk.Label(box, text="ms").grid(row=1, column=2, sticky="w")

    # -------------------------------------------------------------- data
    def refresh_routers(self, router_ids: list[str], links: list[str]) -> None:
        """Update the router and link dropdowns after the network changed."""
        for combo, var, default in ((self.router_boxes[0], self.source, 0),
                                    (self.router_boxes[1], self.destination, -1)):
            combo["values"] = router_ids
            if var.get() not in router_ids:
                var.set(router_ids[default] if router_ids else "")
        self.fail_box["values"] = links
        if self.fail_link.get() not in links:
            self.fail_link.set(links[0] if links else "")

    def get_config(self) -> SimConfig:
        """Read all settings into a validated SimConfig (raises ConfigError)."""
        config = SimConfig(
            processing_delay_ms=_parse_int(self.p.get(), "Processing delay"),
            k_packets=_parse_int(self.k.get(), "K packets"),
            update_period_ms=_parse_int(self.period.get(), "Update period"),
            seed=_parse_int(self.seed.get(), "Random seed"),
            split_horizon=self.split_horizon.get(),
            rerouting=self.rerouting.get(),
        )
        return config.validate()

    def set_step_label(self, index: int, total: int) -> None:
        """Show 'Step i / n' under the animation buttons."""
        self.step_label.configure(text=f"Step {index} / {total}")


def _parse_int(text: str, label: str) -> int:
    """Turn user text into an int, with a friendly error message."""
    try:
        return int(text.strip())
    except ValueError:
        raise ConfigError(f"{label} must be a whole number (you typed {text!r}).") from None
