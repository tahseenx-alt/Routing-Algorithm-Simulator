"""Routing Table tab: destination, next hop and cost for the selected router."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from algorithms.base import fmt


class RoutingTablePanel(ttk.Frame):
    """A title plus a three-column Treeview."""

    def __init__(self, master: tk.Misc) -> None:
        """Create the title label and the table."""
        super().__init__(master, padding=4)
        self.title = ttk.Label(self, text="Click a router to see its routing table.",
                               font=("TkDefaultFont", 10, "bold"))
        self.title.pack(anchor="w")
        frame = ttk.Frame(self)
        frame.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(frame, columns=("dest", "hop", "cost"), show="headings", height=6)
        for col, text, width in (("dest", "Destination", 120), ("hop", "Next hop", 120),
                                 ("cost", "Cost", 120)):
            self.tree.heading(col, text=text)
            self.tree.column(col, width=width, anchor="center")
        bar = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=bar.set)
        self.tree.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")

    def show(self, title: str, table: dict[str, tuple[str | None, float]],
             own_id: str | None = None) -> None:
        """Fill the table. A router's own entry shows next hop '(local)'."""
        self.title.configure(text=title)
        self.tree.delete(*self.tree.get_children())
        for dest, (hop, cost) in sorted(table.items()):
            if dest == own_id:
                hop_text = "(local)"
            elif hop is None:
                hop_text = "-"
            else:
                hop_text = hop
            self.tree.insert("", "end", values=(dest, hop_text, fmt(cost)))

    def show_message(self, title: str) -> None:
        """Show only a message (e.g. flooding has no routing table)."""
        self.title.configure(text=title)
        self.tree.delete(*self.tree.get_children())
