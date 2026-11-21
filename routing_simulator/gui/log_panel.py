"""A read-only scrolling text box, used for the Log and Timeline tabs."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class LogPanel(ttk.Frame):
    """Text log with a scrollbar and an optional highlighted line."""

    def __init__(self, master: tk.Misc) -> None:
        """Create the text widget."""
        super().__init__(master, padding=2)
        self.text = tk.Text(self, height=8, wrap="word", font=("TkFixedFont", 10),
                            state="disabled", background="#ffffff")
        bar = ttk.Scrollbar(self, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=bar.set)
        self.text.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")
        self.text.tag_configure("current", background="#fef9c3")
        self.text.tag_configure("heading", font=("TkFixedFont", 10, "bold"))

    def set_lines(self, lines: list[str], heading: str = "") -> None:
        """Replace the whole log."""
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        if heading:
            self.text.insert("end", heading + "\n", "heading")
        for line in lines:
            self.text.insert("end", line + "\n")
        self.text.configure(state="disabled")
        self._offset = 1 if heading else 0

    def append(self, line: str) -> None:
        """Add one line at the end and scroll to it."""
        self.text.configure(state="normal")
        self.text.insert("end", line + "\n")
        self.text.configure(state="disabled")
        self.text.see("end")

    def highlight(self, index: int | None) -> None:
        """Highlight line `index` (0-based, not counting the heading)."""
        self.text.tag_remove("current", "1.0", "end")
        if index is None:
            return
        line = index + 1 + getattr(self, "_offset", 0)
        self.text.tag_add("current", f"{line}.0", f"{line}.end")
        self.text.see(f"{line}.0")

    def clear(self) -> None:
        """Empty the log."""
        self.set_lines([])
