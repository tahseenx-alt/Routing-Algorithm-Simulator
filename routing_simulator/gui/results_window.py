"""Results window for "Run All & Compare": table, best-algorithm banner,
three Matplotlib bar charts and CSV / PNG export.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from typing import Callable

from algorithms.distance_vector import CountToInfinityDemo
from core.metrics import Comparison, FailureRow, export_csv

# Chart colours (one hue per job; text stays in neutral ink).
BAR = "#2a78d6"          # single-series bars
SETUP = "#2a78d6"        # stacked: routing overhead (one-time)
PACKETS = "#eb6834"      # stacked: K x per-packet transmissions
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e5e4e0"


class ResultsWindow(tk.Toplevel):
    """Shows a Comparison produced by core.metrics.run_all."""

    def __init__(self, master: tk.Misc, comparison: Comparison, title: str) -> None:
        """Build banner, table, charts and export buttons."""
        super().__init__(master)
        self.comparison = comparison
        self.title(f"Results – {title}")
        self.geometry("1180x760")
        self._build_buttons()  # packed first at the bottom so it is never cut off
        self._build_banner()
        self._build_table()
        self._build_charts()

    def _build_banner(self) -> None:
        """Highlighted banner: which algorithm is best and why."""
        best = self.comparison.best
        bg = "#dcfce7" if best else "#fee2e2"
        frame = tk.Frame(self, bg=bg, padx=10, pady=8)
        frame.pack(fill="x", padx=8, pady=(8, 4))
        headline = f"★ Best algorithm: {best.name}" if best else "No algorithm delivered the packet"
        tk.Label(frame, text=headline, bg=bg, fg="#14532d" if best else "#7f1d1d",
                 font=("TkDefaultFont", 14, "bold")).pack(anchor="w")
        tk.Label(frame, text=self.comparison.explanation, bg=bg, justify="left",
                 wraplength=1120, font=("TkDefaultFont", 10)).pack(anchor="w")

    def _build_table(self) -> None:
        """Treeview with every metric from section 8."""
        rows = [r.as_row() for r in self.comparison.rows]
        columns = list(rows[0])
        frame = ttk.Frame(self)
        frame.pack(fill="x", padx=8)
        tree = ttk.Treeview(frame, columns=columns, show="headings", height=len(rows))
        widths = {"Algorithm": 95, "Path": 150, "Status": 160}
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=widths.get(col, max(80, 7 * len(col))), anchor="center",
                        stretch=True)
        tree.tag_configure("best", background="#bbf7d0")
        tree.tag_configure("out", foreground="#94a3b8")
        for metrics, values in zip(self.comparison.rows, rows):
            tag = "best" if metrics is self.comparison.best else ("" if metrics.eligible else "out")
            tree.insert("", "end", values=[values[c] for c in columns], tags=(tag,))
        bar = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(xscrollcommand=bar.set)
        tree.pack(fill="x")
        bar.pack(fill="x")

    def _build_charts(self) -> None:
        """Delivery time, stacked total messages, computation time."""
        rows = self.comparison.rows
        names = [r.name for r in rows]
        k = self.comparison.k_packets
        best = self.comparison.best.name if self.comparison.best else None
        self.figure = Figure(figsize=(11.5, 3.4), dpi=100, facecolor="white")
        axes = self.figure.subplots(1, 3)
        x = range(len(rows))

        # 1. Delivery time (not delivered -> no bar, "N/A" label)
        times = [r.result.delivery_time_ms if r.result.delivered else 0 for r in rows]
        bars = axes[0].bar(x, times, color=BAR, width=0.6, edgecolor="white", linewidth=2)
        for bar, r in zip(bars, rows):
            text = f"{r.result.delivery_time_ms:.1f}".removesuffix(".0") if r.result.delivered else "N/A"
            axes[0].annotate(text, (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                             ha="center", va="bottom", fontsize=8, color=INK,
                             xytext=(0, 2), textcoords="offset points")
        axes[0].set_title("Delivery time (ms)")

        # 2. Total messages for K packets, stacked: setup + K x per packet
        setup = [r.result.routing_overhead for r in rows]
        packets = [k * r.result.per_packet_transmissions for r in rows]
        axes[1].bar(x, setup, color=SETUP, width=0.6, label="Routing overhead (setup)",
                    edgecolor="white", linewidth=2)
        axes[1].bar(x, packets, bottom=setup, color=PACKETS, width=0.6,
                    label=f"{k} × per-packet", edgecolor="white", linewidth=2)
        for i, (s, p) in enumerate(zip(setup, packets)):
            axes[1].annotate(f"{s + p:,.0f}", (i, s + p), ha="center", va="bottom", fontsize=8,
                             color=INK, xytext=(0, 2), textcoords="offset points")
        axes[1].set_title(f"Total messages for K = {k} packets")
        axes[1].legend(fontsize=8, frameon=False, loc="upper left")

        # 3. Computation time (µs)
        micros = [r.computation_us for r in rows]
        bars = axes[2].bar(x, micros, color=BAR, width=0.6, edgecolor="white", linewidth=2)
        for bar, m in zip(bars, micros):
            axes[2].annotate(f"{m:.0f}", (bar.get_x() + bar.get_width() / 2, m), ha="center",
                             va="bottom", fontsize=8, color=INK, xytext=(0, 2),
                             textcoords="offset points")
        axes[2].set_title("Computation time (µs)")

        for ax in axes:
            ax.set_xticks(list(x))
            ax.set_xticklabels([f"★ {n}" if n == best else n for n in names],
                               rotation=20, fontsize=8, color=INK)
            ax.title.set_fontsize(10)
            ax.tick_params(axis="y", labelsize=8, colors=MUTED)
            ax.grid(axis="y", color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            ax.margins(y=0.15)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(GRID)
        self.figure.tight_layout()
        canvas = FigureCanvasTkAgg(self.figure, master=self)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=4)

    def _build_buttons(self) -> None:
        """Export CSV / Export PNG / Close."""
        frame = ttk.Frame(self, padding=6)
        frame.pack(side="bottom", fill="x")
        ttk.Button(frame, text="Export table as CSV…", command=self.export_csv).pack(side="left")
        ttk.Button(frame, text="Export charts as PNG…", command=self.export_png
                   ).pack(side="left", padx=6)
        ttk.Button(frame, text="Close", command=self.destroy).pack(side="right")

    def export_csv(self, path: str | None = None) -> None:
        """Save the comparison table as CSV (for the project report)."""
        path = path or filedialog.asksaveasfilename(parent=self, defaultextension=".csv",
                                                    filetypes=[("CSV", "*.csv")])
        if path:
            try:
                export_csv(self.comparison, path)
                messagebox.showinfo("Saved", f"Table saved to {path}", parent=self)
            except OSError as exc:
                messagebox.showerror("Could not save", str(exc), parent=self)

    def export_png(self, path: str | None = None) -> None:
        """Save the three charts as one PNG image."""
        path = path or filedialog.asksaveasfilename(parent=self, defaultextension=".png",
                                                    filetypes=[("PNG image", "*.png")])
        if path:
            try:
                self.figure.savefig(path, dpi=150, bbox_inches="tight")
                messagebox.showinfo("Saved", f"Charts saved to {path}", parent=self)
            except OSError as exc:
                messagebox.showerror("Could not save", str(exc), parent=self)


# ------------------------------------------------------------ Failure Test
class FailureWindow(tk.Toplevel):
    """Table showing how every algorithm recovers from one link failure."""

    def __init__(self, master: tk.Misc, rows: list[FailureRow], link: tuple[str, str],
                 title: str) -> None:
        """Build the table and the explanation."""
        super().__init__(master)
        self.title(f"Failure Test – link {link[0]}–{link[1]} failed ({title})")
        self.geometry("1100x520")
        intro = (f"Link {link[0]}–{link[1]} failed. Each algorithm recovers in its own way: "
                 "Dijkstra (link state) floods two new LSAs from the routers at the ends of "
                 "the link and reruns SPF (one update period). Bellman-Ford and Hop Count "
                 "(distance vector) continue from their old tables round by round "
                 "(rounds × update period). Flooding and Random keep no tables, so they need "
                 "no extra routing messages.")
        tk.Label(self, text=intro, justify="left", wraplength=1060, bg="#fef9c3", padx=8,
                 pady=6).pack(fill="x", padx=8, pady=8)
        values = [r.as_row() for r in rows]
        columns = list(values[0])
        tree = ttk.Treeview(self, columns=columns, show="headings", height=len(rows))
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=max(90, 8 * len(col)), anchor="center")
        for v in values:
            tree.insert("", "end", values=[v[c] for c in columns])
        tree.pack(fill="x", padx=8)
        text = tk.Text(self, height=12, wrap="word", font=("TkFixedFont", 9))
        for r in rows:
            text.insert("end", f"{r.name}: {r.log[0]}\n")
        for r in rows:
            if len(r.log) > 1:
                text.insert("end", f"\n{r.name} reconvergence log:\n" + "\n".join(r.log[1:]) + "\n")
        text.configure(state="disabled")
        text.pack(fill="both", expand=True, padx=8, pady=8)
        ttk.Button(self, text="Close", command=self.destroy).pack(pady=(0, 8))


# ---------------------------------------------------- Count-to-infinity
class CountToInfinityWindow(tk.Toplevel):
    """Two tabs: linear A–B–C (with vs without split horizon) and the triangle."""

    def __init__(self, master: tk.Misc, linear: tuple[CountToInfinityDemo, CountToInfinityDemo],
                 triangle: CountToInfinityDemo, on_tab: Callable[[int], None]) -> None:
        """Build both tabs; on_tab(i) lets the app draw the matching network."""
        super().__init__(master)
        self.title("Count-to-Infinity Demo (hop metric, infinity = 16)")
        self.geometry("1100x690")
        ttk.Button(self, text="Close", command=self.destroy).pack(side="bottom", pady=(0, 6))
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=6, pady=6)
        notebook.add(self._linear_tab(notebook, *linear), text="1. Linear A–B–C, fail B–C")
        notebook.add(self._triangle_tab(notebook, triangle), text="2. Triangle + D, fail C–D")
        notebook.bind("<<NotebookTabChanged>>", lambda _e: on_tab(notebook.index("current")))

    def _linear_tab(self, parent: tk.Misc, without: CountToInfinityDemo,
                    with_sh: CountToInfinityDemo) -> ttk.Frame:
        """Side-by-side tables + chart of B's cost to C."""
        tab = ttk.Frame(parent, padding=6)
        explain = (
            "A–B–C, every link 1 hop. After B–C fails, B has no route to C. Without split "
            f"horizon, A still advertises 'C is 2 hops away' to B, so B believes C is 3 hops "
            f"away via A, A then believes 4 via B, and so on: the costs count up until they "
            f"reach infinity = 16 ({without.rounds} rounds, {without.messages} vectors). "
            "With split horizon, A does not advertise C back to B (it learned C from B), so B "
            f"immediately sees C as unreachable: converged in {with_sh.rounds} round(s) with "
            f"{with_sh.messages} vectors.")
        _explanation(tab, explain)
        tables = ttk.Frame(tab)
        tables.pack(fill="x")
        for demo in (without, with_sh):
            box = ttk.LabelFrame(tables, text=f"{demo.title}: rounds {demo.rounds}, "
                                              f"vectors {demo.messages}", padding=4)
            box.pack(side="left", fill="both", expand=True, padx=4)
            _demo_table(box, demo, height=10)
        _demo_chart(tab, [(f"B's cost to C – {d.title.lower()}", d, "B") for d in (without, with_sh)],
                    "B's cost to C after B–C fails")
        return tab

    def _triangle_tab(self, parent: tk.Misc, demo: CountToInfinityDemo) -> ttk.Frame:
        """Table + chart for the triangle loop with split horizon ON."""
        tab = ttk.Frame(parent, padding=6)
        explain = (
            "Triangle A–B–C with D attached to C, split horizon ON. After C–D fails, A and B "
            "each still have a route to D through C, and they tell each other about it "
            "(split horizon only blocks advertising a route back to the router you learned it "
            "from). The stale route keeps travelling around the loop (C learns it from A, "
            "B from C, A from B, ...) and the costs still count up to 16 "
            f"({demo.rounds} rounds, {demo.messages} vectors).\n\nSplit horizon only stops loops "
            "between two neighbours. Loops through three or more routers can still count to "
            "infinity.")
        _explanation(tab, explain)
        box = ttk.LabelFrame(tab, text=f"{demo.title}: cost to D", padding=4)
        box.pack(fill="x")
        _demo_table(box, demo, height=8)
        _demo_chart(tab, [(f"{w}'s cost to D", demo, w) for w in demo.watch],
                    "Costs to D after C–D fails (split horizon ON)")
        return tab


def _explanation(parent: tk.Misc, text: str) -> None:
    """Yellow explanation box at the top of a demo tab."""
    tk.Label(parent, text=text, justify="left", wraplength=1040, bg="#fef9c3", padx=8,
             pady=6).pack(fill="x", pady=(0, 6))


def _demo_table(parent: tk.Misc, demo: CountToInfinityDemo, height: int) -> None:
    """Round-by-round table: round, then each watched router's cost."""
    columns = ["Round"] + [f"{w}'s cost to {demo.destination}" for w in demo.watch]
    tree = ttk.Treeview(parent, columns=columns, show="headings", height=height)
    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, width=110, anchor="center")
    tree.tag_configure("inf", foreground="#dc2626")
    for label, costs in demo.rows:
        cells = [label] + [("16 (∞)" if costs[w] >= demo.infinity else f"{costs[w]:g}")
                           for w in demo.watch]
        tags = ("inf",) if all(costs[w] >= demo.infinity for w in demo.watch) else ()
        tree.insert("", "end", values=cells, tags=tags)
    bar = ttk.Scrollbar(parent, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=bar.set)
    tree.pack(side="left", fill="both", expand=True)
    bar.pack(side="right", fill="y")


def _demo_chart(parent: tk.Misc, series: list[tuple[str, CountToInfinityDemo, str]],
                title: str) -> None:
    """Line chart of cost per round, with the infinity line dashed."""
    figure = Figure(figsize=(10.5, 2.3), dpi=100, facecolor="white")
    ax = figure.add_subplot(1, 1, 1)
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    for (label, demo, router), color in zip(series, colors):
        rows = demo.rows[1:]  # rounds after the failure
        xs = [int(r[0]) for r in rows]
        ys = [min(r[1][router], demo.infinity) for r in rows]
        ax.plot(xs, ys, marker="o", markersize=4, linewidth=2, color=color, label=label)
    ax.axhline(series[0][1].infinity, color="#dc2626", linestyle="--", linewidth=1,
               label="infinity (16)")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Round", fontsize=8, color=MUTED)
    ax.set_ylabel("Hops", fontsize=8, color=MUTED)
    ax.tick_params(labelsize=8, colors=MUTED)
    ax.grid(color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    figure.tight_layout()
    canvas = FigureCanvasTkAgg(figure, master=parent)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True, pady=4)
