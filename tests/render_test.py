"""Headless render test: two live samples 3s apart during an in-flight stream, one clean frame."""
import os
import sys
import time
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import engine_metrics as em

# Headless: curses.color_pair raises before start_color(); those raises are
# swallowed by the draw helpers' `except curses.error`, so stub the module.
em.curses = types.SimpleNamespace(color_pair=lambda n: n, A_BOLD=0, error=Exception)


class FakeScr:
    def __init__(self, w=100):
        self.w = w
        self.buf = []

    def erase(self):
        self.buf = []

    def getmaxyx(self):
        return (50, self.w)

    def addstr(self, y, x, s, attr=0):
        while len(self.buf) <= y:
            self.buf.append("")
        row = self.buf[y]
        self.buf[y] = row + " " * (x - len(row)) + s if x > len(row) else row[:x] + s

    def refresh(self):
        pass

    def getch(self):
        return -1

    def color_pair(self, n):
        return n


scr = FakeScr()
a = em.mlx_sample("localhost", 8000)
time.sleep(3)
b = em.mlx_sample("localhost", 8000)
hist = [a, b]
em.draw_mlx(scr, b, a, hist, time.time() - 3600, "localhost", 8000, 2, rate_hist=None)
print("\n".join(scr.buf))
