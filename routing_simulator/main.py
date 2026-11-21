"""Entry point: python main.py

Starts the Routing Algorithm Simulator and Performance Comparator GUI.
"""

from __future__ import annotations

import os
import sys

# Make "core", "algorithms" and "gui" importable no matter where we start from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    """Launch the application, with a clear message if Tkinter is missing."""
    try:
        import tkinter  # noqa: F401  (only checking it is installed)
    except ImportError:
        print("Tkinter is not installed. On Ubuntu/Debian run: sudo apt install python3-tk")
        sys.exit(1)
    from gui.app import run

    run()


if __name__ == "__main__":
    main()
