"""mlx-serve backend: vllm:* compat counters + mlx_serve:* natives.

mlx_serve:generation_tokens_live is a real-time gauge (includes in-flight slots),
cumulative counters are delta-sampled, vllm:* histograms give cumulative averages.
"""

import curses
import re
import time
from datetime import datetime

from .common import (
    COLOR_BORDER, COLOR_DIM, COLOR_GOOD, COLOR_GREEN, COLOR_RED, COLOR_TITLE,
    COLOR_VALUE, COLOR_WARN, avg_wall_rate, draw_bar_row, draw_box_end,
    draw_box_start, draw_row, fetch_json, fetch_metrics, format_bytes,
    format_num, format_speed, format_uptime, get_value, http_get,
    prune_rate_hist, sparkline, wall_rate,
)

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


def sample(host, port):
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


def tick(stdscr, args, state):
    cur = sample(args.host, args.port)
    if not cur.get("_error"):
        state["history"].append(cur)
    draw(stdscr, cur, state["prev"], state["history"], state["start_time"],
         args.host, args.port, args.interval, state["rate_hist"])
    state["prev"] = cur if not cur.get("_error") else state["prev"]


def draw(stdscr, cur, prev, history, start_time, host, port, interval, rate_hist=None):
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
