"""Shared plumbing: HTTP, parsing, formatting, rate math, curses draw helpers."""

import curses
import json
import re
import time
import urllib.request
import urllib.error

# Color pair IDs
COLOR_TITLE = 1
COLOR_LABEL = 2
COLOR_VALUE = 3
COLOR_WARN = 4
COLOR_GOOD = 5
COLOR_HEADER = 6
COLOR_BORDER = 7
COLOR_DIM = 8
COLOR_RED = 9
COLOR_GREEN = 10


def init_colors():
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(COLOR_TITLE, curses.COLOR_CYAN, -1)
    curses.init_pair(COLOR_LABEL, curses.COLOR_YELLOW, -1)
    curses.init_pair(COLOR_VALUE, curses.COLOR_WHITE, -1)
    curses.init_pair(COLOR_WARN, curses.COLOR_YELLOW, -1)
    curses.init_pair(COLOR_GOOD, curses.COLOR_GREEN, -1)
    curses.init_pair(COLOR_HEADER, curses.COLOR_CYAN, -1)
    curses.init_pair(COLOR_BORDER, curses.COLOR_CYAN, -1)
    curses.init_pair(COLOR_DIM, curses.COLOR_WHITE, -1)
    curses.init_pair(COLOR_RED, curses.COLOR_RED, -1)
    curses.init_pair(COLOR_GREEN, curses.COLOR_GREEN, -1)


def http_get(url, timeout=5):
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def fetch_json(host, port, path):
    try:
        return json.loads(http_get(f"http://{host}:{port}{path}"))
    except Exception:
        return None


def fetch_metrics(host, port):
    """Fetch /metrics, parse ALL known prefixes (vllm:, tensorfold:).
    Returns (metrics_dict, model_name). Labeled lines keep last label set."""
    try:
        text = http_get(f"http://{host}:{port}/metrics")
    except Exception as e:
        return {"_error": str(e)}, None
    metrics, model_name = {}, None
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "_created" in line or "_bucket" in line:
            continue
        m = re.match(r"^([\w:]+)\{([^}]*)\}\s+([\d.e+-]+)$", line)
        if m:
            name, labels, value = m.group(1).strip(), m.group(2), float(m.group(3))
            metrics[name] = value
            metrics[name + "@labels"] = labels
            mm = re.search(r'model_name="([^"]+)"', labels)
            if mm and model_name is None:
                model_name = mm.group(1)
        else:
            m2 = re.match(r"^([\w:]+)\s+([\d.e+-]+)$", line)
            if m2:
                metrics[m2.group(1).strip()] = float(m2.group(2))
    return metrics, model_name


def get_value(metrics, key, default=0.0):
    return metrics.get(key, default)


_LABEL_PAIR_RE = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')
_METRIC_LINE_RE = re.compile(r"^([\w:]+)(?:\{([^}]*)\})?\s+([\d.eE+-]+)$")


def parse_labels(label_str):
    """Parse a Prometheus label string 'k="v",k2="v2"' into a dict."""
    if not label_str:
        return {}
    return {k: v.replace('\\"', '"') for k, v in _LABEL_PAIR_RE.findall(label_str)}


def fetch_metrics_labeled(host, port, keep_buckets=False):
    """Label-aware /metrics parse.

    Returns (flat, series, model_name):
      flat   — {name: last_value} (backward-compatible with fetch_metrics)
      series — {name: [(labels_dict, value), ...]} every labeled series kept
      model_name — first model_name label seen, or None

    `_created` lines are always dropped; `_bucket` lines only kept when
    keep_buckets=True (they are the raw material for percentile estimates).
    """
    try:
        text = http_get(f"http://{host}:{port}/metrics")
    except Exception as e:
        return {"_error": str(e)}, {}, None
    flat, series, model_name = {}, {}, None
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _METRIC_LINE_RE.match(line)
        if not m:
            continue
        name, labels_s, value = m.group(1), m.group(2) or "", float(m.group(3))
        if "_created" in name:
            continue
        if name.endswith("_bucket") and not keep_buckets:
            continue
        labels = parse_labels(labels_s)
        flat[name] = value
        series.setdefault(name, []).append((labels, value))
        if labels.get("model_name") and model_name is None:
            model_name = labels["model_name"]
    return flat, series, model_name


def percentile_from_buckets(buckets, total, q):
    """Estimate the q-quantile (0..1) from cumulative histogram buckets.

    buckets: {upper_bound: cumulative_count} (le="+Inf" entries are ignored —
    they cannot be interpolated). total: total observation count. Linear
    interpolation inside the bucket, Prometheus-text style. None when empty.
    """
    if total <= 0 or not buckets:
        return None
    pts = sorted((u, c) for u, c in buckets.items() if u != float("inf"))
    if not pts:
        return None
    target = q * total
    prev_u, prev_c = 0.0, 0.0
    for u, c in pts:
        if c >= target:
            if c <= prev_c:
                return u
            return prev_u + (target - prev_c) / (c - prev_c) * (u - prev_u)
        prev_u, prev_c = u, c
    return pts[-1][0]


def bucket_delta(cur_buckets, prev_buckets):
    """Per-bucket deltas between two cumulative bucket dicts (engine restart
   -safe: negative totals are caught by the caller via total <= 0)."""
    if prev_buckets is None:
        return dict(cur_buckets)
    return {u: c - prev_buckets.get(u, 0.0) for u, c in cur_buckets.items()}


def format_num(n):
    if n >= 1e9:
        return f"{n/1e9:.1f}B"
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    if n >= 1e3:
        return f"{n/1e3:.1f}K"
    return f"{n:,.0f}"


def format_speed(v):
    if v is None:
        return "—"
    if v >= 1e6:
        return f"{v/1e6:.2f}M"
    if v >= 1e3:
        return f"{v/1e3:.1f}K"
    return f"{v:.1f}"


def format_uptime(seconds):
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    s = int(seconds) % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_bytes(n):
    if n >= 1e9:
        return f"{n/1e9:.1f}G"
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    if n >= 1e3:
        return f"{n/1e3:.1f}K"
    return f"{n:.0f}"


# ---------------- box/row drawing ----------------

def draw_box_start(stdscr, y, x, width, title, color=COLOR_BORDER):
    if y < 0:
        return
    if title:
        header = f"┌─ {title} " + "─" * max(0, width - len(title) - 5) + "┐"
    else:
        header = "┌" + "─" * max(0, width - 2) + "┐"
    try:
        stdscr.addstr(y, x, header[:width], curses.color_pair(color) | curses.A_BOLD)
    except curses.error:
        pass


def draw_box_end(stdscr, y, x, width):
    if y < 0:
        return
    footer = "└" + "─" * max(0, width - 2) + "┘"
    try:
        stdscr.addstr(y, x, footer[:width], curses.color_pair(COLOR_BORDER))
    except curses.error:
        pass


def draw_row(stdscr, y, x, label, value, width, label_color=COLOR_LABEL, value_color=COLOR_VALUE):
    if y < 0:
        return
    label_str = f"│ {label}"
    value_str = f" {value}"
    try:
        stdscr.addstr(y, x, label_str[:width], curses.color_pair(label_color))
        stdscr.addstr(y, x + len(label_str), value_str[: max(0, width - len(label_str))], curses.color_pair(value_color))
    except curses.error:
        pass


def draw_bar_row(stdscr, y, x, label, pct, width, color):
    """Row like '│ Usage: 12.3% [████░░░░]'."""
    if y < 0:
        return
    bar_width = max(10, min(30, width - len(label) - 14))
    filled = int(max(0.0, min(100.0, pct)) / 100 * bar_width)
    bar = "█" * filled + "░" * (bar_width - filled)
    text = f"│ {label} {pct:5.1f}%  ["
    try:
        stdscr.addstr(y, x, text[:width], curses.color_pair(COLOR_LABEL))
        stdscr.addstr(y, x + len(text), bar[: max(0, width - len(text) - 1)], curses.color_pair(color))
        stdscr.addstr(y, x + len(text) + len(bar), "]", curses.color_pair(COLOR_LABEL))
    except curses.error:
        pass


# ---------------- rate math ----------------

def rate(cur, prev, num_key, den_key):
    """Rate from counter deltas; None when no progress or bad dt."""
    if prev is None:
        return None
    dn = cur.get(num_key, 0.0) - prev.get(num_key, 0.0)
    dd = cur.get(den_key, 0.0) - prev.get(den_key, 0.0)
    if dd <= 0:
        return None
    return dn / dd


def wall_rate(cur, prev, num_key):
    """Tokens per wall second from a cumulative token counter."""
    if prev is None:
        return None
    dt = cur["t"] - prev["t"]
    if dt <= 0:
        return None
    return (cur[num_key] - prev[num_key]) / dt


def avg_rate(history, num_key, den_key):
    """Rate across the whole rolling window (first vs last sample)."""
    if len(history) < 2:
        return None
    first, last = history[0], history[-1]
    dn = last[num_key] - first[num_key]
    dd = last[den_key] - first[den_key]
    if dd <= 0:
        return None
    return dn / dd


def avg_wall_rate(history, num_key):
    if len(history) < 2:
        return None
    first, last = history[0], history[-1]
    dt = last["t"] - first["t"]
    if dt <= 0:
        return None
    return (last[num_key] - first[num_key]) / dt


SPARK = "▁▂▃▄▅▆▇█"

def sparkline(points):
    """points: list of floats-or-None (gaps). Normalized to window max."""
    if not points:
        return ""
    vmax = max((v for v in points if v is not None), default=0.0)
    if vmax <= 0:
        return "·" * len(points)
    out = []
    for v in points:
        if v is None:
            out.append("·")
        else:
            out.append(SPARK[min(7, max(0, int(v / vmax * 7.999)))])
    return "".join(out)


def prune_rate_hist(rate_hist, seconds=120.0):
    now = time.time()
    while rate_hist and now - rate_hist[0][0] > seconds:
        rate_hist.popleft()
