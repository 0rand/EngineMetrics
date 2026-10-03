"""TensorFold backend: tensorfold:* metrics + /health JSON; cumulative counters
are delta-sampled between polls for live PP/TG."""

import curses
import time
from collections import deque
from datetime import datetime

from .common import (
    COLOR_BORDER, COLOR_DIM, COLOR_GOOD, COLOR_GREEN, COLOR_RED, COLOR_TITLE,
    COLOR_WARN, avg_rate, avg_wall_rate, draw_bar_row, draw_box_end,
    draw_box_start, draw_row, fetch_json, fetch_metrics, format_num,
    format_speed, format_uptime, get_value, prune_rate_hist, rate, sparkline,
    wall_rate,
)

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


def sample(host, port):
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


def tick(stdscr, args, state):
    cur = sample(args.host, args.port)
    if not state["model_name"]:
        mm = fetch_json(args.host, args.port, "/v1/models")
        if mm and mm.get("data"):
            state["model_name"] = mm["data"][0].get("id")
    if not cur.get("_error"):
        state["history"].append(cur)
    draw(stdscr, cur, state["prev"], state["history"], state["start_time"],
         args.host, args.port, args.interval, state["model_name"], state["rate_hist"])
    state["prev"] = cur if not cur.get("_error") else state["prev"]


def draw(stdscr, cur, prev, history, start_time, host, port, interval, model_name, rate_hist=None):
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
