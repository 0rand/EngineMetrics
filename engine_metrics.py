#!/usr/bin/env python3
"""vLLM / TensorFold / mlx-serve Performance Dashboard TUI — real-time engine monitor.

Auto-detects the backend from /health + /metrics:
  - vLLM      : parses vllm:* Prometheus metrics (gauges give rates directly)
  - TensorFold: parses tensorfold:* metrics + /health JSON; cumulative counters are
                delta-sampled between polls for live PP/TG, pool occupancy for
                live cache, histograms for averages.
  - mlx-serve : parses vllm:* + mlx_serve:* metrics; mlx_serve:generation_tokens_live
                is a real-time gauge (includes in-flight slots), cumulative counters
                are delta-sampled, vllm:* histograms give cumulative averages.
                Force with --engine mlx-serve if auto-detection fails.

Usage:
    python3 engine_metrics.py [--host HOST] [--port PORT] [--engine ENGINE] [--interval SECONDS]

Keys: q quit | r reset uptime | c clear rate history
"""

import argparse
import curses
import json
import re
import time
import urllib.request
import urllib.error
from collections import deque
from datetime import datetime

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


# ---------------- TensorFold sampling ----------------

TF_COUNTERS = [
    "tensorfold:prompt_tokens_total",
    "tensorfold:generation_tokens_total",
    "tensorfold:mtp_drafted_total",
    "tensorfold:mtp_accepted_total",
    "tensorfold:request_latency_seconds_sum",
    "tensorfold:request_latency_seconds_count",
    "tensorfold:time_to_first_token_seconds_sum",
    "tensorfold:time_to_first_token_seconds_count",
]

HEALTH_COUNTERS = [
    "requests_total", "prompt_tokens_total", "completion_tokens_total",
    "cached_tokens_total", "rounds_total", "drafted_total", "accepted_total",
    "prefill_seconds_total", "decode_seconds_total",
    "pool_tokens", "pool_free_tokens", "kept_prompts", "context_length",
]


def tf_sample(host, port):
    """One TensorFold sample: wall time + counters from /metrics + /health."""
    metrics, _ = fetch_metrics(host, port)
    health = fetch_json(host, port, "/health") or {}
    s = {"t": time.time(), "_error": metrics.get("_error")}
    for k in TF_COUNTERS:
        s[k] = get_value(metrics, k)
    for k in HEALTH_COUNTERS:
        s[k] = float(health.get(k, 0) or 0)
    # generation tokens only exist as a /metrics counter (health has completion_tokens_total)
    s["generation_tokens_total"] = get_value(metrics, "tensorfold:generation_tokens_total")
    streams = health.get("streams", {}) or {}
    s["streams"] = streams
    s["multi_prefill"] = health.get("multi_prefill", {}) or {}
    s["busy"] = bool(health.get("busy", False))
    s["ok"] = bool(health.get("ok", False))
    return s


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


# ---------------- TensorFold dashboard ----------------

def draw_tensorfold(stdscr, cur, prev, history, start_time, host, port, interval, model_name, rate_hist=None):
    stdscr.erase()
    height, width = stdscr.getmaxyx()
    x = 0
    y = 0
    title = "TensorFold Performance Dashboard"
    try:
        stdscr.addstr(y, x, "╔" + "═" * (width - 2) + "╗", curses.color_pair(COLOR_BORDER) | curses.A_BOLD)
        y += 1
        tpad = max(0, (width - 2 - len(title)) // 2)
        line = "║" + " " * tpad + title + " " * max(0, width - 2 - tpad - len(title)) + "║"
        stdscr.addstr(y, x, line[:width], curses.color_pair(COLOR_TITLE) | curses.A_BOLD)
        y += 1
        info = f"  Backend: tensorfold  Model: {model_name or 'unknown'}  Host: {host}:{port}  Uptime: {format_uptime(time.time() - start_time)}"
        stdscr.addstr(y, x, "║" + info[: width - 2].ljust(width - 2) + "║", curses.color_pair(COLOR_DIM))
        y += 1
        stdscr.addstr(y, x, "╚" + "═" * (width - 2) + "╝", curses.color_pair(COLOR_BORDER) | curses.A_BOLD)
        y += 1
    except curses.error:
        pass

    if cur.get("_error"):
        try:
            stdscr.addstr(y, x, f"  Status: DOWN  {cur['_error']}"[:width], curses.color_pair(COLOR_RED) | curses.A_BOLD)
        except curses.error:
            pass
        stdscr.refresh()
        return
    y += 1
    status = "BUSY" if cur["busy"] else "IDLE"
    scolor = COLOR_WARN if cur["busy"] else COLOR_GREEN
    try:
        stdscr.addstr(y, x, f"  Status: OK ({status})", curses.color_pair(scolor) | curses.A_BOLD)
    except curses.error:
        pass
    y += 1

    # ---- Live throughput (delta-sampled) ----
    # engine-time rates (finished-request based, health counters mirror /metrics)
    pp_engine_rate = rate(cur, prev, "prompt_tokens_total", "prefill_seconds_total")
    tg_engine_rate = rate(cur, prev, "generation_tokens_total", "decode_seconds_total")
    # live wall rates (completion_tokens_total includes in-flight replies)
    tg_live = wall_rate(cur, prev, "completion_tokens_total")
    pp_live = wall_rate(cur, prev, "prompt_tokens_total")
    # window averages
    pp_eng_avg = avg_rate(history, "prompt_tokens_total", "prefill_seconds_total")
    tg_eng_avg = avg_rate(history, "generation_tokens_total", "decode_seconds_total")
    tg_live_avg = avg_wall_rate(history, "completion_tokens_total")

    # 120s rate history for sparklines
    if rate_hist is not None:
        rate_hist.append((cur["t"], pp_engine_rate, tg_live if tg_live is not None else tg_engine_rate))
        prune_rate_hist(rate_hist, 120.0)

    draw_box_start(stdscr, y, x, width, "Live Throughput (delta-sampled)")
    y += 1
    draw_row(stdscr, y, x, "Prefill PP (engine):", f"{format_speed(pp_engine_rate)} tok/s   avg {format_speed(pp_eng_avg)}", width)
    y += 1
    draw_row(stdscr, y, x, "Decode TG (engine):", f"{format_speed(tg_engine_rate)} tok/s   avg {format_speed(tg_eng_avg)}", width)
    y += 1
    tg_color = COLOR_GOOD if (tg_live or 0) >= 30 else COLOR_WARN if (tg_live or 0) > 0 else COLOR_DIM
    draw_row(stdscr, y, x, "TG live (wall):", f"{format_speed(tg_live)} tok/s   avg {format_speed(tg_live_avg)}", width, value_color=tg_color)
    y += 1
    st = cur["streams"]
    draw_row(stdscr, y, x, "Streams:",
             f"decoding {int(st.get('decoding', 0))}  prefilling {int(st.get('prefilling', 0))}  "
             f"filling {int(st.get('filling', 0))}  paused {int(st.get('paused', 0))}  / max {int(st.get('max', 0))}",
             width)
    y += 1
    # 120s sparklines (PP engine rate, TG live rate)
    if rate_hist is not None and len(rate_hist) >= 2:
        pp_pts = [r[1] for r in rate_hist]
        tg_pts = [r[2] for r in rate_hist]
        pp_max = max((v for v in pp_pts if v is not None), default=0.0)
        tg_max = max((v for v in tg_pts if v is not None), default=0.0)
        draw_row(stdscr, y, x, f"PP 120s [{format_speed(pp_max)}]:",
                 f"[{sparkline(pp_pts)}]", width)
        y += 1
        draw_row(stdscr, y, x, f"TG 120s [{format_speed(tg_max)}]:",
                 f"[{sparkline(tg_pts)}]", width)
        y += 1
    draw_box_end(stdscr, y, x, width)
    y += 1

    # ---- Live cache ----
    pool = cur["pool_tokens"]
    free = cur["pool_free_tokens"]
    used_pct = (pool - free) / pool * 100 if pool > 0 else 0.0
    kv_ratio = get_value(cur, "tensorfold:kv_cache_usage_ratio") * 100
    kept = int(cur["kept_prompts"])
    ctx = int(cur["context_length"])
    cache_color = COLOR_GREEN if used_pct < 70 else COLOR_WARN if used_pct < 90 else COLOR_RED

    draw_box_start(stdscr, y, x, width, "Live Cache")
    y += 1
    draw_bar_row(stdscr, y, x, "Pool occupancy:", used_pct, width, cache_color)
    y += 1
    draw_row(stdscr, y, x, "Pool:", f"{format_num(pool - free)} / {format_num(pool)} rows free {format_num(free)}", width)
    y += 1
    draw_row(stdscr, y, x, "Stream KV ratio:", f"{kv_ratio:.1f}% (per-stream / ctx window)", width)
    y += 1
    draw_row(stdscr, y, x, "Kept prompts:", f"{kept}   context window {format_num(ctx)}", width)
    y += 1
    draw_box_end(stdscr, y, x, width)
    y += 1

    # ---- Averages (cumulative histograms) ----
    lat_sum = cur["tensorfold:request_latency_seconds_sum"]
    lat_n = cur["tensorfold:request_latency_seconds_count"]
    ttft_sum = cur["tensorfold:time_to_first_token_seconds_sum"]
    ttft_n = cur["tensorfold:time_to_first_token_seconds_count"]
    avg_lat = lat_sum / lat_n if lat_n else 0.0
    avg_ttft = ttft_sum / ttft_n if ttft_n else 0.0
    drafted = cur["drafted_total"]
    accepted = cur["accepted_total"]
    acc_rate = accepted / drafted * 100 if drafted else 0.0
    ptok = cur["prompt_tokens_total"]
    ctok = cur["cached_tokens_total"]
    hit = ctok / ptok * 100 if ptok else 0.0

    draw_box_start(stdscr, y, x, width, "Averages (cumulative)")
    y += 1
    draw_row(stdscr, y, x, "Avg Latency:", f"{avg_lat:.2f}s  (n={int(lat_n)})", width)
    y += 1
    draw_row(stdscr, y, x, "Avg TTFT:", f"{avg_ttft:.2f}s  (n={int(ttft_n)})", width)
    y += 1
    acc_color = COLOR_GREEN if acc_rate >= 70 else COLOR_WARN if acc_rate >= 50 else COLOR_RED
    if not drafted:
        acc_color = COLOR_DIM
    draw_row(stdscr, y, x, "MTP Acceptance:", f"{acc_rate:.1f}%  ({format_num(accepted)}/{format_num(drafted)})", width, value_color=acc_color)
    y += 1
    hit_color = COLOR_GREEN if hit >= 70 else COLOR_WARN
    draw_row(stdscr, y, x, "Cache Hit (tokens):", f"{hit:.1f}%  ({format_num(ctok)}/{format_num(ptok)})", width, value_color=hit_color)
    y += 1
    draw_box_end(stdscr, y, x, width)
    y += 1

    # ---- Totals ----
    draw_box_start(stdscr, y, x, width, "Totals")
    y += 1
    draw_row(stdscr, y, x, "Requests:", f"{int(cur['requests_total'])}  (running {int(get_value(cur, 'tensorfold:requests_running'))}, waiting {int(get_value(cur, 'tensorfold:requests_waiting'))})", width)
    y += 1
    draw_row(stdscr, y, x, "Prompt / Completion:", f"{format_num(cur['prompt_tokens_total'])} / {format_num(cur['completion_tokens_total'])}", width)
    y += 1
    draw_row(stdscr, y, x, "Engine time:", f"prefill {cur['prefill_seconds_total']:.0f}s  decode {cur['decode_seconds_total']:.0f}s  rounds {int(cur['rounds_total'])}", width)
    y += 1
    mp = cur["multi_prefill"]
    draw_row(stdscr, y, x, "Multi-prefill:", f"chunks {int(mp.get('chunks', 0))}  pieces {int(mp.get('pieces', 0))}  rows {format_num(mp.get('rows', 0))}", width)
    y += 1
    draw_box_end(stdscr, y, x, width)
    y += 1

    try:
        stdscr.addstr(y, x, "─" * width, curses.color_pair(COLOR_BORDER))
        y += 1
        footer = (f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  q=quit  r=reset  c=clear rates  "
                  f"|  {interval}s refresh  |  window {len(history)} samples")
        stdscr.addstr(y, x, footer[:width], curses.color_pair(COLOR_DIM))
    except curses.error:
        pass
    stdscr.refresh()


# ---------------- vLLM dashboard (legacy path, preserved) ----------------

def draw_vllm(stdscr, metrics, start_time, host, port, interval, model_name):
    stdscr.erase()
    height, width = stdscr.getmaxyx()
    x = 0
    y = 0
    title = "vLLM Performance Dashboard"
    try:
        stdscr.addstr(y, x, "╔" + "═" * (width - 2) + "╗", curses.color_pair(COLOR_BORDER) | curses.A_BOLD)
        y += 1
        tpad = max(0, (width - 2 - len(title)) // 2)
        line = "║" + " " * tpad + title + " " * max(0, width - 2 - tpad - len(title)) + "║"
        stdscr.addstr(y, x, line[:width], curses.color_pair(COLOR_TITLE) | curses.A_BOLD)
        y += 1
        info = f"  Model: {model_name or 'unknown'}     Host: {host}:{port}     Uptime: {format_uptime(time.time() - start_time)}"
        stdscr.addstr(y, x, "║" + info[: width - 2].ljust(width - 2) + "║", curses.color_pair(COLOR_DIM))
        y += 1
        stdscr.addstr(y, x, "╚" + "═" * (width - 2) + "╝", curses.color_pair(COLOR_BORDER) | curses.A_BOLD)
        y += 1
    except curses.error:
        pass

    if "_error" in metrics:
        try:
            stdscr.addstr(y, x, f"  Status: DOWN  {metrics['_error']}"[:width], curses.color_pair(COLOR_RED) | curses.A_BOLD)
        except curses.error:
            pass
        stdscr.refresh()
        return
    try:
        stdscr.addstr(y, x, "  Status: OK", curses.color_pair(COLOR_GREEN) | curses.A_BOLD)
    except curses.error:
        pass
    y += 1

    e2e_sum = get_value(metrics, "vllm:e2e_request_latency_seconds_sum")
    e2e_n = get_value(metrics, "vllm:e2e_request_latency_seconds_count")
    pre_sum = get_value(metrics, "vllm:request_prefill_time_seconds_sum")
    pre_n = get_value(metrics, "vllm:request_prefill_time_seconds_count")
    dec_sum = get_value(metrics, "vllm:request_decode_time_seconds_sum")
    dec_n = get_value(metrics, "vllm:request_decode_time_seconds_count")
    ttft_sum = get_value(metrics, "vllm:time_to_first_token_seconds_sum")
    ttft_n = get_value(metrics, "vllm:time_to_first_token_seconds_count")
    tpt_sum = get_value(metrics, "vllm:request_time_per_output_token_seconds_sum")
    tpt_n = get_value(metrics, "vllm:request_time_per_output_token_seconds_count")

    draw_box_start(stdscr, y, x, width, "Performance")
    y += 1
    draw_row(stdscr, y, x, "Avg E2E Latency:", f"{(e2e_sum/e2e_n if e2e_n else 0):.2f}s", width); y += 1
    draw_row(stdscr, y, x, "Avg Prefill Time:", f"{(pre_sum/pre_n if pre_n else 0):.2f}s", width); y += 1
    draw_row(stdscr, y, x, "Avg Decode Time:", f"{(dec_sum/dec_n if dec_n else 0):.2f}s", width); y += 1
    draw_row(stdscr, y, x, "Avg TTFT:", f"{(ttft_sum/ttft_n if ttft_n else 0):.2f}s", width); y += 1
    draw_row(stdscr, y, x, "Avg Time/Token:", f"{(tpt_sum/tpt_n if tpt_n else 0):.2f}s", width); y += 1
    draw_box_end(stdscr, y, x, width); y += 1

    kv = 100 * get_value(metrics, "vllm:kv_cache_usage_perc")
    draw_box_start(stdscr, y, x, width, "KV Cache & Queue")
    y += 1
    draw_bar_row(stdscr, y, x, "Usage:", kv, width,
                 COLOR_GREEN if kv < 70 else COLOR_WARN if kv < 90 else COLOR_RED)
    y += 1
    draw_row(stdscr, y, x, "Running / Waiting:",
             f"{int(get_value(metrics, 'vllm:num_requests_running'))} / {int(get_value(metrics, 'vllm:num_requests_waiting'))}", width)
    y += 1
    draw_row(stdscr, y, x, "Prompt / Gen / Cached:",
             f"{format_num(get_value(metrics, 'vllm:prompt_tokens_total'))} / "
             f"{format_num(get_value(metrics, 'vllm:generation_tokens_total'))} / "
             f"{format_num(get_value(metrics, 'vllm:prompt_tokens_cached_total'))}", width)
    y += 1
    hits = get_value(metrics, "vllm:prefix_cache_hits_total")
    queries = get_value(metrics, "vllm:prefix_cache_queries_total")
    draw_row(stdscr, y, x, "Prefix Cache Hit:", f"{(hits/queries*100 if queries else 0):.1f}%", width)
    y += 1
    draw_box_end(stdscr, y, x, width); y += 1

    try:
        stdscr.addstr(y, x, "─" * width, curses.color_pair(COLOR_BORDER))
        y += 1
        footer = f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  q=quit  r=reset  |  {interval}s refresh"
        stdscr.addstr(y, x, footer[:width], curses.color_pair(COLOR_DIM))
    except curses.error:
        pass
    stdscr.refresh()


# ---------------- mlx-serve sampling ----------------

MLX_KEYS = [
    # vllm:* compat surface
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:request_success_total",
    "vllm:request_cancelled_total",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:prefix_cache_hits_total",
    "vllm:prefix_cache_queries_total",
    "vllm:e2e_request_latency_seconds_sum",
    "vllm:e2e_request_latency_seconds_count",
    "vllm:request_prefill_time_seconds_sum",
    "vllm:request_prefill_time_seconds_count",
    "vllm:request_decode_time_seconds_sum",
    "vllm:request_decode_time_seconds_count",
    "vllm:time_to_first_token_seconds_sum",
    "vllm:time_to_first_token_seconds_count",
    # mlx_serve:* native surface
    "mlx_serve:prefill_tokens_total",
    "mlx_serve:prefix_cache_tokens_total",
    "mlx_serve:generation_tokens_live",
    "mlx_serve:prefill_tokens_live",
    "mlx_serve:prefill_tokens_expected",
    "mlx_serve:gpu_utilization_pct",
    "mlx_serve:memory_mb",
    "mlx_serve:mlx_active_bytes",
    "mlx_serve:mlx_cache_bytes",
    "mlx_serve:ngram_warm_bytes",
    "mlx_serve:requests_prefilling",
    "mlx_serve:batched_group_size",
]


def mlx_model_info(host, port):
    """Loaded model card from /v1/models (mlx-serve lists all with loaded/state)."""
    mm = fetch_json(host, port, "/v1/models")
    if not isinstance(mm, dict) or not mm.get("data"):
        return None
    data = mm["data"]
    loaded = [m for m in data if m.get("loaded")] or data
    m = loaded[0]
    meta = m.get("meta") or {}
    return {
        "id": m.get("id"),
        "arch": meta.get("architecture"),
        "quant": meta.get("quantization"),
        "ctx": m.get("context_length") or meta.get("context_length") or 0,
        "mtp": bool(meta.get("mtp_loaded")),
        "resident": m.get("bytes_resident") or 0,
        "loaded": bool(m.get("loaded")),
    }


def mlx_sample(host, port):
    """One mlx-serve sample: wall time + counters from /metrics + model card."""
    metrics, _ = fetch_metrics(host, port)
    s = {"t": time.time(), "_error": metrics.get("_error")}
    for k in MLX_KEYS:
        s[k] = get_value(metrics, k)
    # fetch_metrics keeps only the last label set per name; decode_serial_total is
    # emitted per-reason, so re-scan the raw text for the full breakdown
    s["decode_serial"] = {}
    if not s["_error"]:
        try:
            text = http_get(f"http://{host}:{port}/metrics")
            for mm in re.finditer(r'^mlx_serve:decode_serial_total\{reason="([^"]+)"\}\s+([\d.eE+-]+)$', text, re.M):
                s["decode_serial"][mm.group(1)] = float(mm.group(2))
        except Exception:
            pass
    s["model"] = mlx_model_info(host, port)
    return s


# ---------------- mlx-serve dashboard ----------------

def draw_mlx(stdscr, cur, prev, history, start_time, host, port, interval, rate_hist=None):
    stdscr.erase()
    height, width = stdscr.getmaxyx()
    x = 0
    y = 0
    title = "mlx-serve Performance Dashboard"
    try:
        stdscr.addstr(y, x, "╔" + "═" * (width - 2) + "╗", curses.color_pair(COLOR_BORDER) | curses.A_BOLD)
        y += 1
        tpad = max(0, (width - 2 - len(title)) // 2)
        line = "║" + " " * tpad + title + " " * max(0, width - 2 - tpad - len(title)) + "║"
        stdscr.addstr(y, x, line[:width], curses.color_pair(COLOR_TITLE) | curses.A_BOLD)
        y += 1
        info = f"  Backend: mlx-serve  Host: {host}:{port}  Uptime: {format_uptime(time.time() - start_time)}"
        stdscr.addstr(y, x, "║" + info[: width - 2].ljust(width - 2) + "║", curses.color_pair(COLOR_DIM))
        y += 1
        stdscr.addstr(y, x, "╚" + "═" * (width - 2) + "╝", curses.color_pair(COLOR_BORDER) | curses.A_BOLD)
        y += 1
    except curses.error:
        pass

    if cur.get("_error"):
        try:
            stdscr.addstr(y, x, f"  Status: DOWN  {cur['_error']}"[:width], curses.color_pair(COLOR_RED) | curses.A_BOLD)
        except curses.error:
            pass
        stdscr.refresh()
        return
    y += 1

    busy = cur["vllm:num_requests_running"] > 0 or cur["mlx_serve:requests_prefilling"] > 0
    status = "BUSY" if busy else "IDLE"
    scolor = COLOR_WARN if busy else COLOR_GREEN
    try:
        stdscr.addstr(y, x, f"  Status: OK ({status})", curses.color_pair(scolor) | curses.A_BOLD)
    except curses.error:
        pass
    y += 1

    # ---- Model ----
    model = cur.get("model") or {}
    draw_box_start(stdscr, y, x, width, "Model")
    y += 1
    mstate = "loaded" if model.get("loaded", True) else "NOT LOADED"
    mcolor = COLOR_VALUE if model.get("loaded", True) else COLOR_RED
    draw_row(stdscr, y, x, "", f"{model.get('id') or 'unknown'}  ({mstate})", width, value_color=mcolor)
    y += 1
    ctx = model.get("ctx") or 0
    detail = " · ".join(v for v in [
        model.get("arch"),
        model.get("quant"),
        f"ctx {format_num(ctx)}" if ctx else None,
        "MTP loaded" if model.get("mtp") else None,
        f"resident {format_bytes(model.get('resident') or 0)}B" if model.get("resident") else None,
    ] if v)
    draw_row(stdscr, y, x, "", detail or "—", width)
    y += 1
    draw_box_end(stdscr, y, x, width)
    y += 1

    # ---- Live throughput (delta-sampled) ----
    # generation_tokens_live is a gauge that includes in-flight slots → true wall rate
    tg_live = wall_rate(cur, prev, "mlx_serve:generation_tokens_live")
    # prefill_tokens_total counts only actually-computed prefill tokens (excludes
    # prefix-cache restores) → the correct prefill tok/s numerator
    pp_wall = wall_rate(cur, prev, "mlx_serve:prefill_tokens_total")
    tg_avg = avg_wall_rate(history, "mlx_serve:generation_tokens_live")
    pp_avg = avg_wall_rate(history, "mlx_serve:prefill_tokens_total")

    if rate_hist is not None:
        rate_hist.append((cur["t"], pp_wall, tg_live))
        prune_rate_hist(rate_hist, 120.0)

    draw_box_start(stdscr, y, x, width, "Live Throughput (delta-sampled)")
    y += 1
    tg_color = COLOR_GOOD if (tg_live or 0) >= 30 else COLOR_WARN if (tg_live or 0) > 0 else COLOR_DIM
    draw_row(stdscr, y, x, "Decode TG (wall):", f"{format_speed(tg_live)} tok/s   avg {format_speed(tg_avg)}", width, value_color=tg_color)
    y += 1
    draw_row(stdscr, y, x, "Prefill PP (wall):", f"{format_speed(pp_wall)} tok/s   avg {format_speed(pp_avg)}", width)
    y += 1
    exp = cur["mlx_serve:prefill_tokens_expected"]
    live = cur["mlx_serve:prefill_tokens_live"]
    if exp > 0:
        draw_bar_row(stdscr, y, x, "Prefill progress:", min(100.0, live / exp * 100), width, COLOR_WARN)
        y += 1
        draw_row(stdscr, y, x, "", f"{live:,.0f} / {exp:,.0f} tokens computed this prefill", width)
        y += 1
    else:
        draw_row(stdscr, y, x, "Prefill progress:", "idle", width)
        y += 1
    draw_row(stdscr, y, x, "GPU / footprint:",
             f"{cur['mlx_serve:gpu_utilization_pct']:.0f}%  /  {cur['mlx_serve:memory_mb']/1024:.1f} GB", width)
    y += 1
    draw_row(stdscr, y, x, "MLX bytes:",
             f"active {format_bytes(cur['mlx_serve:mlx_active_bytes'])}B  cache {format_bytes(cur['mlx_serve:mlx_cache_bytes'])}B  ngram-warm {format_bytes(cur['mlx_serve:ngram_warm_bytes'])}B",
             width)
    y += 1
    draw_row(stdscr, y, x, "Slots:",
             f"running {int(cur['vllm:num_requests_running'])}  waiting {int(cur['vllm:num_requests_waiting'])}  "
             f"prefilling {int(cur['mlx_serve:requests_prefilling'])}  batch-group {int(cur['mlx_serve:batched_group_size'])}",
             width)
    y += 1
    if rate_hist is not None and len(rate_hist) >= 2:
        pp_pts = [r[1] for r in rate_hist]
        tg_pts = [r[2] for r in rate_hist]
        pp_max = max((v for v in pp_pts if v is not None), default=0.0)
        tg_max = max((v for v in tg_pts if v is not None), default=0.0)
        draw_row(stdscr, y, x, f"PP 120s [{format_speed(pp_max)}]:", f"[{sparkline(pp_pts)}]", width)
        y += 1
        draw_row(stdscr, y, x, f"TG 120s [{format_speed(tg_max)}]:", f"[{sparkline(tg_pts)}]", width)
        y += 1
    draw_box_end(stdscr, y, x, width)
    y += 1

    # ---- Prefix cache ----
    ptok = cur["vllm:prompt_tokens_total"]
    restored = cur["mlx_serve:prefix_cache_tokens_total"]
    tok_hit = restored / ptok * 100 if ptok else 0.0
    hits = cur["vllm:prefix_cache_hits_total"]
    queries = cur["vllm:prefix_cache_queries_total"]
    q_hit = hits / queries * 100 if queries else 0.0

    draw_box_start(stdscr, y, x, width, "Prefix Cache")
    y += 1
    tok_color = COLOR_GREEN if tok_hit >= 70 else COLOR_WARN
    draw_row(stdscr, y, x, "Token hit rate:", f"{tok_hit:.1f}%  ({format_num(restored)}/{format_num(ptok)} restored)", width, value_color=tok_color)
    y += 1
    draw_row(stdscr, y, x, "Query hit rate:", f"{q_hit:.1f}%  ({int(hits)}/{int(queries)})", width)
    y += 1
    draw_box_end(stdscr, y, x, width)
    y += 1

    # ---- Averages (cumulative histograms) ----
    e2e_s = cur["vllm:e2e_request_latency_seconds_sum"]
    e2e_n = cur["vllm:e2e_request_latency_seconds_count"]
    ttft_s = cur["vllm:time_to_first_token_seconds_sum"]
    ttft_n = cur["vllm:time_to_first_token_seconds_count"]
    pre_s = cur["vllm:request_prefill_time_seconds_sum"]
    pre_n = cur["vllm:request_prefill_time_seconds_count"]
    dec_s = cur["vllm:request_decode_time_seconds_sum"]
    dec_n = cur["vllm:request_decode_time_seconds_count"]

    draw_box_start(stdscr, y, x, width, "Averages (cumulative)")
    y += 1
    draw_row(stdscr, y, x, "Avg E2E Latency:", f"{(e2e_s/e2e_n if e2e_n else 0):.2f}s  (n={int(e2e_n)})", width)
    y += 1
    draw_row(stdscr, y, x, "Avg TTFT:", f"{(ttft_s/ttft_n if ttft_n else 0):.2f}s  (n={int(ttft_n)})", width)
    y += 1
    draw_row(stdscr, y, x, "Avg Prefill / Decode Time:",
             f"{(pre_s/pre_n if pre_n else 0):.2f}s / {(dec_s/dec_n if dec_n else 0):.2f}s", width)
    y += 1
    draw_box_end(stdscr, y, x, width)
    y += 1

    # ---- Totals ----
    draw_box_start(stdscr, y, x, width, "Totals")
    y += 1
    draw_row(stdscr, y, x, "Requests:",
             f"{int(cur['vllm:request_success_total'])} ok, {int(cur['vllm:request_cancelled_total'])} cancelled  "
             f"(running {int(cur['vllm:num_requests_running'])}, waiting {int(cur['vllm:num_requests_waiting'])})", width)
    y += 1
    draw_row(stdscr, y, x, "Prompt / Completion:",
             f"{format_num(cur['vllm:prompt_tokens_total'])} / {format_num(cur['vllm:generation_tokens_total'])}", width)
    y += 1
    draw_row(stdscr, y, x, "Prefill computed / cache-restored:",
             f"{format_num(cur['mlx_serve:prefill_tokens_total'])} / {format_num(restored)}", width)
    y += 1
    serial = cur.get("decode_serial") or {}
    tops = sorted(((k, v) for k, v in serial.items() if v > 0), key=lambda kv: -kv[1])[:3]
    draw_row(stdscr, y, x, "Serial decode:",
             "  ".join(f"{k} {int(v)}" for k, v in tops) or "none", width)
    y += 1
    draw_box_end(stdscr, y, x, width)
    y += 1

    try:
        stdscr.addstr(y, x, "─" * width, curses.color_pair(COLOR_BORDER))
        y += 1
        footer = (f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  q=quit  r=reset  c=clear rates  "
                  f"|  {interval}s refresh  |  window {len(history)} samples")
        stdscr.addstr(y, x, footer[:width], curses.color_pair(COLOR_DIM))
    except curses.error:
        pass
    stdscr.refresh()


# ---------------- main ----------------

def detect_backend(host, port):
    """Return 'mlx-serve' | 'tensorfold' | 'vllm' | None."""
    h = fetch_json(host, port, "/health")
    if isinstance(h, dict) and h.get("backend"):
        return h["backend"]
    m, _ = fetch_metrics(host, port)
    # mlx-serve also emits vllm:* compat counters, so check its native prefix FIRST
    if any(k.startswith("mlx_serve:") for k in m):
        return "mlx-serve"
    if any(k.startswith("vllm:") for k in m):
        return "vllm"
    if any(k.startswith("tensorfold") for k in m):
        return "tensorfold"
    return None


def main(stdscr, args):
    init_colors()
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(args.interval * 1000)

    start_time = time.time()
    backend = args.engine if args.engine != "auto" else None
    model_name = None
    prev = None
    history = deque(maxlen=max(2, args.window))
    rate_hist = deque()  # (t, pp_rate, tg_rate) for 120s sparklines

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

        if backend == "tensorfold":
            cur = tf_sample(args.host, args.port)
            if not model_name:
                mm = fetch_json(args.host, args.port, "/v1/models")
                if mm and mm.get("data"):
                    model_name = mm["data"][0].get("id")
            if not cur.get("_error"):
                history.append(cur)
            draw_tensorfold(stdscr, cur, prev, history, start_time, args.host, args.port, args.interval, model_name, rate_hist)
            prev = cur if not cur.get("_error") else prev
        elif backend == "mlx-serve":
            cur = mlx_sample(args.host, args.port)
            if not cur.get("_error"):
                history.append(cur)
            draw_mlx(stdscr, cur, prev, history, start_time, args.host, args.port, args.interval, rate_hist)
            prev = cur if not cur.get("_error") else prev
        else:
            metrics, fetched = fetch_metrics(args.host, args.port)
            if fetched:
                model_name = fetched
            draw_vllm(stdscr, metrics, start_time, args.host, args.port, args.interval, model_name)

        key = stdscr.getch()
        if key in (ord("q"), ord("Q")):
            break
        if key in (ord("r"), ord("R")):
            start_time = time.time()
        if key in (ord("c"), ord("C")):
            history.clear()
            prev = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vLLM / TensorFold / mlx-serve Performance Dashboard TUI")
    parser.add_argument("--host", default="localhost", help="engine host (default: localhost)")
    parser.add_argument("--port", type=int, default=8100, help="engine port (default: 8100)")
    parser.add_argument("--engine", choices=["auto", "vllm", "tensorfold", "mlx-serve"], default="auto",
                        help="force backend type instead of auto-detection (default: auto)")
    parser.add_argument("--interval", type=int, default=2, help="refresh interval seconds (default: 2)")
    parser.add_argument("--window", type=int, default=15, help="rate-averaging window in samples (default: 15)")
    args = parser.parse_args()
    curses.wrapper(main, args)
