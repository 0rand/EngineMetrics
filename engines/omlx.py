"""oMLX backend (oMLX.app / omlx-server): JSON admin API, no Prometheus.

Live rates come from /admin/api/activity per-model slot lists:
  generating[]  → per-request tokens_per_second (engine-computed, in-flight)
  prefilling[]  → per-request processed/total/speed/eta (native progress bar)
/api/status totals are finished-request counters (delta-sampling shows nothing
mid-flight) — used for cumulative averages and totals only.
"""

import curses
import time
from datetime import datetime

from .common import (
    COLOR_BORDER, COLOR_DIM, COLOR_GOOD, COLOR_GREEN, COLOR_RED, COLOR_TITLE,
    COLOR_VALUE, COLOR_WARN, draw_bar_row, draw_box_end, draw_box_start,
    draw_row, fetch_json, format_bytes, format_num, format_speed,
    format_uptime,
)


def sample(host, port):
    """One oMLX sample: /api/status + activity + model states + device info."""
    s = {"t": time.time(), "_error": None}
    status = fetch_json(host, port, "/api/status")
    if not isinstance(status, dict):
        s["_error"] = "/api/status unavailable"
        return s
    s["status"] = status
    s["activity"] = fetch_json(host, port, "/admin/api/activity") or {}
    s["models_status"] = fetch_json(host, port, "/v1/models/status") or {}
    s["device"] = fetch_json(host, port, "/admin/api/device-info") or {}
    return s


def tick(stdscr, args, state):
    cur = sample(args.host, args.port)
    if not cur.get("_error"):
        state["history"].append(cur)
    draw(stdscr, cur, state["history"], args.host, args.port, args.interval)
    # omlx live rates come from in-slot gauges, no prev-delta needed


def _slots(cur):
    """(generating[], prefilling[]) across all loaded models."""
    models = ((cur.get("activity") or {}).get("active_models") or {}).get("models") or []
    gen, pre = [], []
    for m in models:
        gen.extend(m.get("generating") or [])
        pre.extend(m.get("prefilling") or [])
    return gen, pre


def draw(stdscr, cur, history, host, port, interval):
    stdscr.erase()
    height, width = stdscr.getmaxyx()
    x = 0
    y = 0
    title = "oMLX Performance Dashboard"
    try:
        stdscr.addstr(y, x, "╔" + "═" * (width - 2) + "╗", curses.color_pair(COLOR_BORDER) | curses.A_BOLD)
        y += 1
        tpad = max(0, (width - 2 - len(title)) // 2)
        line = "║" + " " * tpad + title + " " * max(0, width - 2 - tpad - len(title)) + "║"
        stdscr.addstr(y, x, line[:width], curses.color_pair(COLOR_TITLE) | curses.A_BOLD)
        y += 1
        st = cur.get("status") or {}
        info = (f"  Backend: omlx v{st.get('version', '?')}  Host: {host}:{port}  "
                f"Engine uptime: {format_uptime(st.get('uptime_seconds', 0))}")
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

    st = cur["status"]
    gen, pre = _slots(cur)
    busy = st.get("active_requests", 0) > 0 or st.get("waiting_requests", 0) > 0 or gen or pre
    status = "BUSY" if busy else "IDLE"
    scolor = COLOR_WARN if busy else COLOR_GREEN
    try:
        stdscr.addstr(y, x, f"  Status: OK ({status})", curses.color_pair(scolor) | curses.A_BOLD)
    except curses.error:
        pass
    y += 1

    # ---- Models ----
    ms = cur.get("models_status") or {}
    all_models = ms.get("models") or []
    loaded = [m for m in all_models if m.get("loaded")]
    loading = [m for m in all_models if m.get("is_loading")]

    draw_box_start(stdscr, y, x, width, "Models")
    y += 1
    draw_row(stdscr, y, x, "Default:", f"{st.get('default_model', 'unknown')}   "
             f"({st.get('models_discovered', 0)} discovered, {len(loaded)} loaded, {st.get('models_loading', 0)} loading)", width)
    y += 1
    for m in loaded[:3]:
        size = m.get("actual_size") or m.get("estimated_size") or 0
        detail = " · ".join(v for v in [
            m.get("engine_type"),
            f"ctx {format_num(m.get('model_context_length') or 0)}" if m.get("model_context_length") else None,
            f"{format_bytes(size)}B" if size else None,
            "pinned" if m.get("pinned") else None,
        ] if v)
        draw_row(stdscr, y, x, "", f"▸ {m.get('id')}  ({detail or '—'})", width)
        y += 1
    for m in loading[:2]:
        draw_row(stdscr, y, x, "", f"▸ {m.get('id')}  (LOADING…)", width, value_color=COLOR_WARN)
        y += 1
    draw_box_end(stdscr, y, x, width)
    y += 1

    # ---- Live throughput (in-slot gauges) ----
    tg_live = sum((g.get("tokens_per_second") or 0.0) for g in gen)
    tg_avg = st.get("avg_generation_tps") or 0.0
    pp_avg = st.get("avg_prefill_tps") or 0.0

    draw_box_start(stdscr, y, x, width, "Live Throughput (in-flight slots)")
    y += 1
    tg_color = COLOR_GOOD if tg_live >= 30 else COLOR_WARN if tg_live > 0 else COLOR_DIM
    draw_row(stdscr, y, x, "Decode TG (live):",
             f"{format_speed(tg_live)} tok/s   engine avg {format_speed(tg_avg)}   slots {len(gen)}", width,
             value_color=tg_color)
    y += 1
    if pre:
        p = pre[0]
        total = p.get("total") or 0
        processed = p.get("processed") or 0
        pct = processed / total * 100 if total else 0.0
        draw_bar_row(stdscr, y, x, "Prefill progress:", min(100.0, pct), width, COLOR_WARN)
        y += 1
        eta = p.get("eta")
        eta_s = f"  eta {eta:.0f}s" if isinstance(eta, (int, float)) else ""
        extra = f"{processed:,.0f} / {total:,.0f} tokens  ({format_speed(p.get('speed') or 0.0)} tok/s{eta_s})"
        if len(pre) > 1:
            extra += f"   +{len(pre) - 1} more prefilling"
        draw_row(stdscr, y, x, "", extra, width)
        y += 1
    else:
        draw_row(stdscr, y, x, "Prefill progress:", "idle", width)
        y += 1
    draw_row(stdscr, y, x, "Prefill engine avg:", f"{format_speed(pp_avg)} tok/s", width)
    y += 1
    draw_row(stdscr, y, x, "Requests:",
             f"active {st.get('active_requests', 0)}  waiting {st.get('waiting_requests', 0)}  total {st.get('total_requests', 0)}", width)
    y += 1
    draw_box_end(stdscr, y, x, width)
    y += 1

    # ---- Memory ----
    act_models = (cur.get("activity") or {}).get("active_models") or {}
    mem_used = act_models.get("model_memory_used") or 0
    mem_max = act_models.get("model_memory_max") or 1
    mem_pct = mem_used / mem_max * 100 if mem_max else 0.0
    mp = act_models.get("memory_pressure") or {}
    level = mp.get("pressure_level", "ok")
    level_color = COLOR_GREEN if level == "ok" else COLOR_RED if level == "critical" else COLOR_WARN

    draw_box_start(stdscr, y, x, width, "Memory")
    y += 1
    draw_bar_row(stdscr, y, x, "Model pool:", mem_pct, width,
                 COLOR_GREEN if mem_pct < 70 else COLOR_WARN if mem_pct < 90 else COLOR_RED)
    y += 1
    draw_row(stdscr, y, x, "Pool:", f"{format_bytes(mem_used)}B / {format_bytes(mem_max)}B"
             f"  (ceiling {st.get('model_memory_max_formatted', '?')})", width)
    y += 1
    draw_row(stdscr, y, x, "Pressure:",
             f"{mp.get('current_formatted', '?')}  soft {mp.get('soft_formatted', '?')}  hard {mp.get('hard_formatted', '?')}",
             width, value_color=level_color)
    y += 1
    draw_box_end(stdscr, y, x, width)
    y += 1

    # ---- Cache ----
    ptok = st.get("total_prompt_tokens") or 0
    ctok = st.get("total_cached_tokens") or 0
    eff = st.get("cache_efficiency") or (ctok / ptok * 100 if ptok else 0.0)

    draw_box_start(stdscr, y, x, width, "Cache")
    y += 1
    hit_color = COLOR_GREEN if eff >= 70 else COLOR_WARN
    draw_row(stdscr, y, x, "Cache efficiency:", f"{eff:.1f}%  ({format_num(ctok)}/{format_num(ptok)} tokens)", width,
             value_color=hit_color)
    y += 1
    draw_box_end(stdscr, y, x, width)
    y += 1

    # ---- Device & engines ----
    dev = cur.get("device") or {}
    kernels = st.get("custom_kernels") or {}
    k_ok = [k for k, v in kernels.items() if isinstance(v, dict) and v.get("available")]
    k_bad = [k for k, v in kernels.items() if isinstance(v, dict) and not v.get("available")]

    draw_box_start(stdscr, y, x, width, "Device & Engines")
    y += 1
    draw_row(stdscr, y, x, "Chip:",
             f"{dev.get('chip_name', '?')} {dev.get('chip_variant', '')}  ·  {dev.get('gpu_cores', '?')} GPU cores  ·  {dev.get('memory_gb', '?')} GB", width)
    y += 1
    draw_row(stdscr, y, x, "Custom kernels:",
             (", ".join(k_ok) or "none") + (f"  (unavailable: {', '.join(k_bad)})" if k_bad else ""), width,
             value_color=COLOR_GOOD if k_ok and not k_bad else COLOR_VALUE)
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
