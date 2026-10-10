"""Splash backend (incoai/splash): Apple-silicon native engine, Metal + ANE.

Data source is the JSON /status endpoint (schema_version 6) — it mirrors every
counter /metrics renders (server/metrics.py PROMETHEUS_SERIES), so one fetch
per poll covers the whole dashboard. Cumulative counters (prefill input rows,
decode output tokens, decode wall ms) are delta-sampled between polls for live
PP/TG; batch-width mix, KV pages, memory governor, admission queue, draft
acceptance and TTFT/ITL percentiles are gauges read straight off the status.
"""

import curses
import time
from datetime import datetime

from .common import (
    COLOR_BORDER, COLOR_DIM, COLOR_GOOD, COLOR_GREEN, COLOR_RED, COLOR_TITLE,
    COLOR_VALUE, COLOR_WARN, avg_rate, avg_wall_rate, draw_bar_row,
    draw_box_end, draw_box_start, draw_row, fetch_json, format_bytes,
    format_num, format_speed, format_uptime, prune_rate_hist, rate,
    sparkline, wall_rate,
)


def _num(d, *path, default=0.0):
    """Walk a dict path; float value or default (bools rejected)."""
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    if isinstance(cur, bool) or not isinstance(cur, (int, float)):
        return default
    return float(cur)


def _bool(d, *path, default=False):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return bool(cur)


def _str(d, *path, default=""):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur if isinstance(cur, str) else default


def _hist_avg(st, *path):
    """sum/count of a cumulative histogram; None when no samples."""
    n = _num(st, *path, "count")
    if n <= 0:
        return None
    return _num(st, *path, "sum") / n


def _fmt_ms(v):
    if v is None:
        return "—"
    if v >= 1000:
        return f"{v/1000:.2f}s"
    return f"{v:.1f}ms"


def sample(host, port):
    """One Splash sample: /status JSON (single fetch, mirrors /metrics)."""
    st = fetch_json(host, port, "/status")
    s = {"t": time.time()}
    if not isinstance(st, dict):
        s["_error"] = "/status unavailable"
        return s
    s["_error"] = None
    s["ready"] = _bool(st, "ready")
    s["pressure"] = _str(st, "memory_pressure", default="normal")
    s["thermal"] = _str(st, "thermal_state", default="nominal")
    s["max_ctx"] = int(_num(st, "maximum_context_tokens"))
    s["model_name"] = (_str(st, "instance", "model")
                       or _str(st, "memory_plan", "model", "model_name"))

    started = _num(st, "instance", "started_at")
    s["engine_uptime"] = (time.time() - started) if started > 0 else 0.0

    # ---- scheduler ----
    for key, dst in (
        ("queued", "sched_queued"),
        ("waiting_resources", "sched_wait_res"),
        ("waiting_prefix", "sched_wait_prefix"),
        ("prefilling", "sched_prefilling"),
        ("decoding", "sched_decoding"),
        ("waiting_mask", "sched_wait_mask"),
        ("terminal", "sched_terminal"),
    ):
        s[dst] = int(_num(st, "scheduler", key))
    s["prefill_batches"] = int(_num(st, "scheduler", "prefill_batches"))
    s["decode_batches"] = int(_num(st, "scheduler", "decode_batches"))
    s["max_batch_width"] = int(
        _num(st, "memory_plan", "budget", "maximum_batch_width", default=4))
    for w in ("b1", "b2", "b3", "b4"):
        s["decode_" + w] = int(_num(st, "scheduler", "decode_batches_by_width", w))

    # ---- admission / frontend / loop ----
    s["adm_waiting"] = int(_num(st, "admission", "waiting"))
    s["adm_mem"] = int(_num(st, "admission", "waiting_memory"))
    s["adm_conc"] = int(_num(st, "admission", "waiting_concurrency"))
    s["adm_oldest_ms"] = _num(st, "admission", "oldest_wait_ms")
    s["fe_active"] = int(_num(st, "frontend", "active"))
    s["fe_waiting"] = int(_num(st, "frontend", "waiting"))
    s["fe_cap"] = int(_num(st, "frontend", "preparation_capacity", default=4))
    s["loop_tick_ms"] = _num(st, "loop", "max_tick_ms")

    # ---- KV cache ----
    s["kv_pages_allocated"] = int(_num(st, "kv", "pages_allocated"))
    s["kv_pages_active"] = int(_num(st, "kv", "pages_active"))
    s["kv_pages_cache"] = int(_num(st, "kv", "pages_cache"))
    s["kv_pages_free"] = int(_num(st, "kv", "pages_free"))
    s["kv_capacity_pages"] = int(_num(st, "memory_plan", "budget", "kv_capacity_pages"))
    s["kv_alloc_bytes"] = _num(st, "kv", "allocated_bytes")
    s["kv_reclaim_bytes"] = _num(st, "kv", "reclaimable_bytes")
    s["kv_block_tokens"] = int(_num(st, "kv", "block_tokens", default=32))
    s["kv_extent_alloc"] = int(_num(st, "kv", "extent_allocations"))
    s["kv_extent_rel"] = int(_num(st, "kv", "extent_releases"))
    s["kv_extent_compact"] = int(_num(st, "kv", "extent_compactions"))
    s["kv_extent_alloc_ms"] = _num(st, "kv", "extent_allocate_max_ms")
    s["kv_format"] = _str(st, "memory_plan", "model", "kv_format", default="?")

    # ---- prompt cache / state / disk ----
    s["cache_hits"] = int(_num(st, "cache", "hits"))
    s["cache_cold"] = int(_num(st, "cache", "cold_misses"))
    s["cache_hit_rate"] = _num(st, "cache", "hit_rate")
    s["cache_reused"] = int(_num(st, "cache", "reused_tokens"))
    s["cache_kv_hit_tokens"] = int(_num(st, "cache", "kv_hit_tokens"))
    s["state_entries"] = int(_num(st, "state", "entries"))
    s["state_in_use"] = int(_num(st, "state", "in_use"))
    s["state_bytes"] = _num(st, "state", "bytes")
    s["state_lanes"] = int(_num(st, "state", "active_lanes"))
    s["state_evictions"] = int(_num(st, "state", "evictions"))
    s["disk_capacity"] = _num(st, "disk", "capacity_bytes")
    s["disk_kv_blocks"] = int(_num(st, "disk", "kv_blocks"))

    # ---- memory ----
    s["mem_limit"] = _num(st, "memory_governor", "limit_bytes")
    s["mem_charged"] = _num(st, "memory_governor", "charged_bytes")
    s["mem_headroom"] = _num(st, "memory_governor", "headroom_bytes")
    s["mem_current"] = _num(st, "memory_actual", "current_bytes")
    s["mem_peak"] = _num(st, "memory_actual", "peak_bytes")
    s["host_available"] = _num(st, "memory_governor", "host_available_bytes")
    s["denied_res"] = int(_num(st, "memory_governor", "denied_reservations"))
    s["weights_released"] = _bool(st, "weights", "released")
    s["idle_release_s"] = _num(st, "weights", "idle_release_seconds")
    s["weight_restores"] = int(_num(st, "weights", "restores"))
    s["weight_failures"] = int(_num(st, "weights", "restore_failures"))

    # ---- device ----
    s["device_name"] = _str(st, "memory_plan", "device", "device_name", default="?")
    s["gpu_cores"] = int(_num(st, "memory_plan", "device", "gpu_core_count"))
    s["physical_mem"] = _num(st, "memory_plan", "device", "physical_memory_bytes")
    s["macos"] = _str(st, "memory_plan", "device", "macos_version")

    # ---- draft (DFlash2 speculative) ----
    s["drafted"] = int(_num(st, "metrics", "drafted_tokens"))
    s["accepted"] = int(_num(st, "metrics", "accepted_draft_tokens"))
    s["draft_rate"] = _num(st, "metrics", "draft_acceptance_rate")
    s["draft_ctx_avoided"] = int(_num(st, "draft_context", "avoided_rows"))
    s["draft_ctx_resets"] = int(_num(st, "draft_context", "resets"))

    # ---- latency (rolling p50/p95 gauges) ----
    s["ttft_p50_ms"] = _num(st, "metrics", "ttft_ms", "p50")
    s["ttft_p95_ms"] = _num(st, "metrics", "ttft_ms", "p95")
    s["ttft_n"] = int(_num(st, "metrics", "ttft_ms", "samples"))
    s["itl_p50_ms"] = _num(st, "metrics", "itl_ms", "p50")
    s["itl_p95_ms"] = _num(st, "metrics", "itl_ms", "p95")
    s["itl_n"] = int(_num(st, "metrics", "itl_ms", "samples"))
    s["http_avg_s"] = _hist_avg(st, "latency", "http_request")
    s["http_n"] = int(_num(st, "latency", "http_request", "count"))
    s["prep_avg_ms"] = None
    prep = _hist_avg(st, "latency", "preparation")
    if prep is not None:
        s["prep_avg_ms"] = prep * 1000.0
    s["queue_avg_ms"] = None
    q = _hist_avg(st, "latency", "native_queue")
    if q is not None:
        s["queue_avg_ms"] = q * 1000.0

    # ---- cumulative counters (delta-sampled between polls) ----
    s["prefill_rows"] = _num(st, "metrics", "prefill_input_tokens")
    s["prefill_wall_s"] = _num(st, "metrics", "prefill_wall_ms") / 1000.0
    s["decode_rows"] = _num(st, "metrics", "decode_output_tokens")
    s["decode_wall_s"] = _num(st, "metrics", "decode_wall_ms") / 1000.0
    s["decode_cycle_s"] = _num(st, "metrics", "decode_cycle_ms") / 1000.0
    s["pp_lifetime_tps"] = _num(st, "metrics", "prefill_tokens_per_second")
    s["tg_lifetime_tps"] = _num(st, "metrics", "decode_tokens_per_second")
    s["req_submitted"] = int(_num(st, "requests", "submitted"))
    s["req_completed"] = int(_num(st, "requests", "completed"))
    s["req_cancelled"] = int(_num(st, "requests", "cancelled"))
    s["req_failed"] = int(_num(st, "requests", "failed"))
    s["capacity_failures"] = int(_num(st, "metrics", "capacity_failures"))
    s["metal_failures"] = int(_num(st, "metrics", "metal_failures"))

    # ---- last batch gauges ----
    s["last_ppb_valid"] = _bool(st, "metrics", "current_prefill_batch", "valid")
    s["last_ppb_width"] = int(_num(st, "metrics", "current_prefill_batch", "width"))
    s["last_ppb_tok"] = int(_num(st, "metrics", "current_prefill_batch", "input_tokens"))
    s["last_ppb_tps"] = _num(st, "metrics", "current_prefill_batch", "tokens_per_second")
    s["last_tgb_valid"] = _bool(st, "metrics", "current_decode_batch", "valid")
    s["last_tgb_width"] = int(_num(st, "metrics", "current_decode_batch", "width"))
    s["last_tgb_out"] = int(_num(st, "metrics", "current_decode_batch", "output_tokens"))
    s["last_tgb_draft"] = int(_num(st, "metrics", "current_decode_batch", "drafted_tokens"))
    s["last_tgb_acc"] = int(_num(st, "metrics", "current_decode_batch", "accepted_draft_tokens"))
    s["last_tgb_tps"] = _num(st, "metrics", "current_decode_batch", "tokens_per_second")

    # ---- engine health ----
    s["metal_healthy"] = _bool(st, "metal", "healthy", default=True)
    s["transport_ready"] = _bool(st, "transport", "ready", default=True)
    s["transport_pending"] = int(_num(st, "transport", "pending"))
    s["transport_pending_limit"] = int(_num(st, "transport", "pending_limit", default=32))
    s["transport_restarts"] = int(_num(st, "transport", "restarts"))
    s["ane_state"] = _str(st, "ane_ffn", "state", default="off")
    s["ane_share"] = _num(st, "ane_ffn", "share")
    s["warmup_prefill"] = _bool(st, "warmup", "prefill_2048")
    s["warmup_b"] = all(_bool(st, "warmup", f"decode_b{i}") for i in (1, 2, 3, 4))
    s["warmup_restore"] = _bool(st, "warmup", "composite_state_restore")

    s["busy"] = (s["sched_decoding"] + s["sched_prefilling"] + s["sched_queued"]
                 + s["sched_wait_res"] + s["sched_wait_prefix"] + s["sched_wait_mask"]
                 + s["fe_active"] + s["adm_waiting"]) > 0
    return s


def tick(stdscr, args, state):
    cur = sample(args.host, args.port)
    if not state["model_name"] and cur.get("model_name"):
        state["model_name"] = cur["model_name"]
    if not cur.get("_error"):
        state["history"].append(cur)
    draw(stdscr, cur, state["prev"], state["history"], state["start_time"],
         args.host, args.port, args.interval, state["model_name"], state["rate_hist"])
    state["prev"] = cur if not cur.get("_error") else state["prev"]


def draw(stdscr, cur, prev, history, start_time, host, port, interval,
         model_name, rate_hist=None):
    stdscr.erase()
    height, width = stdscr.getmaxyx()
    x = 0
    y = 0
    title = "Splash Performance Dashboard"
    try:
        stdscr.addstr(y, x, "╔" + "═" * (width - 2) + "╗", curses.color_pair(COLOR_BORDER) | curses.A_BOLD)
        y += 1
        tpad = max(0, (width - 2 - len(title)) // 2)
        line = "║" + " " * tpad + title + " " * max(0, width - 2 - tpad - len(title)) + "║"
        stdscr.addstr(y, x, line[:width], curses.color_pair(COLOR_TITLE) | curses.A_BOLD)
        y += 1
        uptime = cur.get("engine_uptime") or (time.time() - start_time)
        info = (f"  Backend: splash  Model: {model_name or cur.get('model_name') or 'unknown'}  "
                f"Host: {host}:{port}  Engine uptime: {format_uptime(uptime)}")
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

    status = "BUSY" if cur["busy"] else ("READY" if cur["ready"] else "STARTING")
    scolor = COLOR_WARN if cur["busy"] else COLOR_GREEN
    pcol = {
        "normal": COLOR_GREEN, "warning": COLOR_WARN, "critical": COLOR_RED,
    }.get(cur["pressure"], COLOR_VALUE)
    tcol = {
        "nominal": COLOR_GREEN, "fair": COLOR_GOOD,
        "serious": COLOR_WARN, "critical": COLOR_RED,
    }.get(cur["thermal"], COLOR_VALUE)
    try:
        line = f"  Status: OK ({status})   mem pressure: "
        stdscr.addstr(y, x, line, curses.color_pair(scolor) | curses.A_BOLD)
        cx = x + len(line)
        seg = f"{cur['pressure']}   thermal: "
        stdscr.addstr(y, cx, seg, curses.color_pair(pcol))
        cx += len(seg)
        stdscr.addstr(y, cx, cur["thermal"] + "   metal: " +
                      ("OK" if cur["metal_healthy"] else "FAIL"),
                      curses.color_pair(tcol if cur["metal_healthy"] else COLOR_RED))
    except curses.error:
        pass
    y += 1

    # ---- Live throughput (delta-sampled) ----
    pp_rate = rate(cur, prev, "prefill_rows", "prefill_wall_s")
    tg_rate = rate(cur, prev, "decode_rows", "decode_wall_s")
    tg_live = wall_rate(cur, prev, "decode_rows")
    pp_avg = avg_rate(history, "prefill_rows", "prefill_wall_s")
    tg_avg = avg_rate(history, "decode_rows", "decode_wall_s")
    tg_live_avg = avg_wall_rate(history, "decode_rows")

    if rate_hist is not None:
        rate_hist.append((cur["t"], pp_rate, tg_live if tg_live is not None else tg_rate))
        prune_rate_hist(rate_hist, 120.0)

    draw_box_start(stdscr, y, x, width, "Live Throughput (delta-sampled)")
    y += 1
    draw_row(stdscr, y, x, "Prefill PP (engine):",
             f"{format_speed(pp_rate)} tok/s   avg {format_speed(pp_avg)}   lifetime {format_speed(cur['pp_lifetime_tps'])}",
             width)
    y += 1
    draw_row(stdscr, y, x, "Decode TG (engine):",
             f"{format_speed(tg_rate)} tok/s   avg {format_speed(tg_avg)}   lifetime {format_speed(cur['tg_lifetime_tps'])}",
             width)
    y += 1
    tg_color = COLOR_GOOD if (tg_live or 0) >= 30 else COLOR_WARN if (tg_live or 0) > 0 else COLOR_DIM
    draw_row(stdscr, y, x, "TG live (wall):",
             f"{format_speed(tg_live)} tok/s   avg {format_speed(tg_live_avg)}", width, value_color=tg_color)
    y += 1
    parts = []
    if cur["last_ppb_valid"]:
        parts.append(f"PP w{cur['last_ppb_width']} {format_speed(cur['last_ppb_tps'])} tok/s ({cur['last_ppb_tok']} tok)")
    if cur["last_tgb_valid"]:
        spec = f" +{cur['last_tgb_draft']} draft/{cur['last_tgb_acc']} acc" if cur["last_tgb_draft"] else ""
        parts.append(f"TG w{cur['last_tgb_width']} {format_speed(cur['last_tgb_tps'])} tok/s ({cur['last_tgb_out']} tok{spec})")
    draw_row(stdscr, y, x, "Last batch:", "   ".join(parts) if parts else "—", width)
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

    # ---- Scheduler & admission ----
    draw_box_start(stdscr, y, x, width, "Scheduler & Admission")
    y += 1
    draw_row(stdscr, y, x, "Streams:",
             f"decoding {cur['sched_decoding']}  prefilling {cur['sched_prefilling']}  "
             f"queued {cur['sched_queued']}  terminal {cur['sched_terminal']}", width)
    y += 1
    draw_row(stdscr, y, x, "Waiting:",
             f"resources {cur['sched_wait_res']}  prefix {cur['sched_wait_prefix']}  mask {cur['sched_wait_mask']}",
             width)
    y += 1
    db = max(1, cur["decode_batches"])
    wm = "  ".join(f"{w} {cur['decode_' + w] / db * 100:.0f}%" for w in ("b1", "b2", "b3", "b4"))
    draw_row(stdscr, y, x, "Batch width:", f"{wm}  / max {cur['max_batch_width']}", width)
    y += 1
    draw_row(stdscr, y, x, "Batches:",
             f"prefill {format_num(cur['prefill_batches'])}   decode {format_num(cur['decode_batches'])}", width)
    y += 1
    draw_row(stdscr, y, x, "Admission:",
             f"waiting {cur['adm_waiting']} (mem {cur['adm_mem']}, conc {cur['adm_conc']})  "
             f"oldest {_fmt_ms(cur['adm_oldest_ms'])}   frontend {cur['fe_active']}/{cur['fe_cap']} active",
             width, value_color=COLOR_WARN if cur["adm_waiting"] else COLOR_VALUE)
    y += 1
    draw_box_end(stdscr, y, x, width)
    y += 1

    # ---- KV cache ----
    cap = cur["kv_capacity_pages"]
    kv_pct = cur["kv_pages_allocated"] / cap * 100 if cap else 0.0
    kv_color = COLOR_GREEN if kv_pct < 70 else COLOR_WARN if kv_pct < 90 else COLOR_RED
    draw_box_start(stdscr, y, x, width, "KV Cache")
    y += 1
    draw_bar_row(stdscr, y, x, "Pool occupancy:", kv_pct, width, kv_color)
    y += 1
    draw_row(stdscr, y, x, "Pages:",
             f"{format_num(cur['kv_pages_allocated'])} / {format_num(cap)}   "
             f"active {format_num(cur['kv_pages_active'])}   cached {format_num(cur['kv_pages_cache'])}   free {format_num(cur['kv_pages_free'])}",
             width)
    y += 1
    draw_row(stdscr, y, x, "Bytes:",
             f"allocated {format_bytes(cur['kv_alloc_bytes'])}B   reclaimable {format_bytes(cur['kv_reclaim_bytes'])}B   block {cur['kv_block_tokens']} tok   format {cur['kv_format']}",
             width)
    y += 1
    draw_row(stdscr, y, x, "Extents:",
             f"alloc {cur['kv_extent_alloc']}  rel {cur['kv_extent_rel']}  compact {cur['kv_extent_compact']}  "
             f"max alloc {_fmt_ms(cur['kv_extent_alloc_ms'])}", width)
    y += 1
    draw_box_end(stdscr, y, x, width)
    y += 1

    # ---- Prompt cache ----
    draw_box_start(stdscr, y, x, width, "Prompt Cache")
    y += 1
    hit = cur["cache_hit_rate"] * 100
    draw_row(stdscr, y, x, "Hit rate:",
             f"{hit:.1f}%  ({format_num(cur['cache_hits'])} hits / {format_num(cur['cache_cold'])} cold misses)",
             width, value_color=COLOR_GREEN if hit >= 70 else COLOR_WARN)
    y += 1
    draw_row(stdscr, y, x, "Tokens:",
             f"reused {format_num(cur['cache_reused'])}   KV-hit {format_num(cur['cache_kv_hit_tokens'])}", width)
    y += 1
    draw_row(stdscr, y, x, "State:",
             f"{cur['state_entries']} entries   {format_bytes(cur['state_bytes'])}B   "
             f"in-use {cur['state_in_use']}   lanes {cur['state_lanes']}   evictions {cur['state_evictions']}",
             width)
    y += 1
    disk = (f"SSD cache: {format_num(cur['disk_kv_blocks'])} blocks"
            if cur["disk_capacity"] > 0 else "SSD cache: off")
    draw_row(stdscr, y, x, "Draft ctx:",
             f"avoided {format_num(cur['draft_ctx_avoided'])} rows   resets {cur['draft_ctx_resets']}   {disk}", width)
    y += 1
    draw_box_end(stdscr, y, x, width)
    y += 1

    # ---- Memory & thermal ----
    mcap = cur["mem_limit"]
    mem_pct = cur["mem_charged"] / mcap * 100 if mcap else 0.0
    mem_color = COLOR_GREEN if mem_pct < 70 else COLOR_WARN if mem_pct < 90 else COLOR_RED
    draw_box_start(stdscr, y, x, width, "Memory")
    y += 1
    draw_bar_row(stdscr, y, x, "Governor:", mem_pct, width, mem_color)
    y += 1
    draw_row(stdscr, y, x, "Charged:",
             f"{format_bytes(cur['mem_charged'])}B / {format_bytes(mcap)}B   headroom {format_bytes(cur['mem_headroom'])}B",
             width)
    y += 1
    draw_row(stdscr, y, x, "Actual:",
             f"current {format_bytes(cur['mem_current'])}B   peak {format_bytes(cur['mem_peak'])}B   "
             f"host available {format_bytes(cur['host_available'])}B   denied {cur['denied_res']}", width)
    y += 1
    if cur["weights_released"]:
        wtxt = f"RELEASED (idle >{cur['idle_release_s'] / 60:.0f}m, restores {cur['weight_restores']})"
        wcol = COLOR_WARN
    else:
        wtxt = (f"resident (idle-release {cur['idle_release_s'] / 60:.0f}m, "
                f"restores {cur['weight_restores']}, failures {cur['weight_failures']})")
        wcol = COLOR_GOOD
    draw_row(stdscr, y, x, "Weights:", wtxt, width, value_color=wcol)
    y += 1
    draw_box_end(stdscr, y, x, width)
    y += 1

    # ---- Latency & speculative ----
    draw_box_start(stdscr, y, x, width, "Latency & Speculative")
    y += 1
    draw_row(stdscr, y, x, "TTFT:",
             f"p50 {_fmt_ms(cur['ttft_p50_ms'])}  p95 {_fmt_ms(cur['ttft_p95_ms'])}  (n={cur['ttft_n']})", width)
    y += 1
    draw_row(stdscr, y, x, "ITL:",
             f"p50 {_fmt_ms(cur['itl_p50_ms'])}  p95 {_fmt_ms(cur['itl_p95_ms'])}  (n={format_num(cur['itl_n'])})", width)
    y += 1
    draw_row(stdscr, y, x, "HTTP req:",
             f"avg {cur['http_avg_s']:.2f}s  prep {_fmt_ms(cur['prep_avg_ms'])}  queue {_fmt_ms(cur['queue_avg_ms'])}  (n={cur['http_n']})",
             width)
    y += 1
    acc = cur["draft_rate"] * 100
    acc_color = COLOR_GREEN if acc >= 70 else COLOR_WARN if acc >= 50 else COLOR_RED
    if not cur["drafted"]:
        acc_color = COLOR_DIM
    draw_row(stdscr, y, x, "Draft accept:",
             f"{acc:.1f}%  ({format_num(cur['accepted'])}/{format_num(cur['drafted'])} DFlash2 tokens)", width,
             value_color=acc_color)
    y += 1
    fcol = COLOR_GOOD if not cur["capacity_failures"] and not cur["metal_failures"] else COLOR_RED
    draw_row(stdscr, y, x, "Failures:",
             f"capacity {cur['capacity_failures']}   metal {cur['metal_failures']}", width, value_color=fcol)
    y += 1
    draw_box_end(stdscr, y, x, width)
    y += 1

    # ---- Engine & device ----
    draw_box_start(stdscr, y, x, width, "Engine & Device")
    y += 1
    draw_row(stdscr, y, x, "Chip:",
             f"{cur['device_name']}   {cur['gpu_cores']} GPU cores   "
             f"{format_bytes(cur['physical_mem'])}B RAM   macOS {cur['macos'] or '?'}", width)
    y += 1
    ctx = cur["max_ctx"]
    draw_row(stdscr, y, x, "Model:",
             f"{model_name or cur.get('model_name') or '?'}   ctx {format_num(ctx)}   "
             f"max batch {cur['max_batch_width']}   ANE FFN: {cur['ane_state']} ({cur['ane_share'] * 100:.1f}%)",
             width)
    y += 1
    tcol = COLOR_GOOD if cur["transport_ready"] else COLOR_RED
    draw_row(stdscr, y, x, "Runtime:",
             f"Metal {'OK' if cur['metal_healthy'] else 'FAIL'}   "
             f"transport {'ready' if cur['transport_ready'] else 'DOWN'} "
             f"(pending {cur['transport_pending']}/{cur['transport_pending_limit']}, restarts {cur['transport_restarts']})   "
             f"loop max tick {_fmt_ms(cur['loop_tick_ms'])}",
             width, value_color=tcol)
    y += 1
    draw_row(stdscr, y, x, "Warmup:",
             f"prefill 2048 {'✓' if cur['warmup_prefill'] else '—'}   "
             f"decode b1–b4 {'✓' if cur['warmup_b'] else '—'}   "
             f"state restore {'✓' if cur['warmup_restore'] else '—'}", width)
    y += 1
    draw_row(stdscr, y, x, "Requests:",
             f"submitted {format_num(cur['req_submitted'])}   completed {format_num(cur['req_completed'])}   "
             f"cancelled {cur['req_cancelled']}   failed {cur['req_failed']}", width)
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
