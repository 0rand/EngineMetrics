#!/usr/bin/env python3
"""vLLM / TensorFold / mlx-serve / oMLX Performance Dashboard TUI — real-time engine monitor.

Auto-detects the backend (see engines/__init__.py), or force it with --engine.
Each backend lives in engines/<name>.py; shared plumbing in engines/common.py.

Usage:
    python3 engine_metrics.py [--host HOST] [--port PORT] [--engine ENGINE] [--interval SECONDS]

Keys: q quit | r reset uptime | c clear rate history
"""

import argparse
import curses
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engines import ENGINES, detect_backend
from engines.common import COLOR_WARN, init_colors


def main(stdscr, args):
    init_colors()
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(args.interval * 1000)

    state = {
        "start_time": time.time(),
        "prev": None,
        "history": deque(maxlen=max(2, args.window)),
        "rate_hist": deque(),
        "model_name": None,
    }
    backend = args.engine if args.engine != "auto" else None

    while True:
        if backend is None:
            backend = detect_backend(args.host, args.port)
        if backend is None:
            stdscr.erase()
            try:
                stdscr.addstr(0, 0, f"Waiting for engine at {args.host}:{args.port} ... (q to quit)",
                              curses.color_pair(COLOR_WARN))
            except curses.error:
                pass
            stdscr.refresh()
            key = stdscr.getch()
            if key in (ord("q"), ord("Q")):
                break
            time.sleep(args.interval)
            continue

        ENGINES[backend].tick(stdscr, args, state)

        key = stdscr.getch()
        if key in (ord("q"), ord("Q")):
            break
        if key in (ord("r"), ord("R")):
            state["start_time"] = time.time()
        if key in (ord("c"), ord("C")):
            state["history"].clear()
            state["prev"] = None
            state["rate_hist"].clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vLLM / TensorFold / mlx-serve / oMLX Performance Dashboard TUI")
    parser.add_argument("--host", default="localhost", help="engine host (default: localhost)")
    parser.add_argument("--port", type=int, default=8100, help="engine port (default: 8100)")
    parser.add_argument("--engine", choices=["auto"] + list(ENGINES), default="auto",
                        help="force backend type instead of auto-detection (default: auto)")
    parser.add_argument("--interval", type=int, default=2, help="refresh interval seconds (default: 2)")
    parser.add_argument("--window", type=int, default=15, help="rate-averaging window in samples (default: 15)")
    args = parser.parse_args()
    curses.wrapper(main, args)
