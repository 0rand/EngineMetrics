"""vLLM backend: vllm:* Prometheus metrics (gauges give rates directly)."""

import curses
import time
from datetime import datetime

from .common import (
    COLOR_BORDER, COLOR_DIM, COLOR_GREEN, COLOR_RED, COLOR_TITLE, COLOR_WARN,
    draw_bar_row, draw_box_end, draw_box_start, draw_row, fetch_metrics,
    format_num, format_uptime, get_value,
)


def tick(stdscr, args, state):
    metrics, fetched = fetch_metrics(args.host, args.port)
    if fetched:
        state["model_name"] = fetched
    draw(stdscr, metrics, state["start_time"], args.host, args.port, args.interval, state["model_name"])


def draw(stdscr, metrics, start_time, host, port, interval, model_name):
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
