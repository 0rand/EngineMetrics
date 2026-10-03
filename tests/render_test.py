"""Headless render test: one clean frame per engine backend (live where reachable).

Usage: python3 tests/render_test.py [omlx|mlx-serve]
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import curses  # noqa: E402

# Headless: curses.color_pair raises before start_color(); those raises are
# swallowed by the draw helpers' `except curses.error`, so stub the accessors.
curses.color_pair = lambda n: n
curses.A_BOLD = 0

import engines.mlx_serve as mlx_serve  # noqa: E402
import engines.omlx as omlx  # noqa: E402


class FakeScr:
    def __init__(self, w=100):
        self.w = w
        self.buf = []

    def erase(self):
        self.buf = []

    def getmaxyx(self):
        return (60, self.w)

    def addstr(self, y, x, s, attr=0):
        while len(self.buf) <= y:
            self.buf.append("")
        row = self.buf[y]
        self.buf[y] = row + " " * (x - len(row)) + s if x > len(row) else row[:x] + s

    def refresh(self):
        pass

    def getch(self):
        return -1


def render(label, fn):
    scr = FakeScr()
    fn(scr)
    print(f"===== {label} =====")
    print("\n".join(scr.buf))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "omlx"
    if which == "omlx":
        cur = omlx.sample("localhost", 8000)
        render("oMLX :8000", lambda scr: omlx.draw(scr, cur, [cur], "localhost", 8000, 2))
    elif which == "mlx-serve":
        cur = mlx_serve.sample("localhost", 8000)
        render("mlx-serve :8000", lambda scr: mlx_serve.draw(
            scr, cur, None, [cur], time.time() - 60, "localhost", 8000, 2, rate_hist=None))
