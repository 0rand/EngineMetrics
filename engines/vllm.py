"""vLLM backend: vllm:* Prometheus metrics, delta-sampled between polls.

Replicates the TensorFold dashboard's live panels (delta-sampled PP/TG,
sparklines, cache occupancy, spec-decode acceptance) and adds vLLM-only
surface: ITL/TTFT/E2E percentiles from histogram buckets, per-position
acceptance decay, finish-reason accounting, scheduler fairness gauges,
tool-call traffic profile and a model card from cache_config_info.

All rates are counter deltas between polls; `—` means no progress in the
window (honest gap), never a fake 0. Negative deltas (engine restart) are
treated as gaps too.
"""

import curses
import time
from datetime import datetime

from .common import (
    COLOR_BORDER, COLOR_DIM, COLOR_GOOD, COLOR_GREEN, COLOR_RED, COLOR_TITLE,
    COLOR_WARN, avg_rate, avg_wall_rate, bucket_delta, draw_bar_row,
    draw_box_end, draw_box_start, draw_row, fetch_metrics_labeled,
    format_num, format_speed, format_uptime, percentile_from_buckets,
    prune_rate_hist, sparkline, wall_rate,
)

COUNTERS = [
    "vllm:prompt_tokens_total",
    "vllm:prompt_tokens_cached_total",
    "vllm:generation_tokens_total",
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
    "vllm:num_preemptions_total",
    "vllm:prefix_cache_hits_total",
    "vllm:prefix_cache_queries_total",
]

HISTOGRAMS = [
    "vllm:time_to_first_token_seconds",
    "vllm:inter_token_latency_seconds",
    "vllm:request_time_per_output_token_seconds",
    "vllm:e2e_request_latency_seconds",
    "vllm:request_queue_time_seconds",
    "vllm:request_prefill_time_seconds",
    "vllm:request_decode_time_seconds",
    "vllm:iteration_tokens_total",
    "vllm:request_prefill_kv_computed_tokens",
]

POS_ACCEPTED = "vllm:spec_decode_num_accepted_tokens_per_pos_total"
POS_DRAFTED = "vllm:spec_decode_num_draft_tokens_per_pos_total"

FINISH_REASONS = ("stop", "length", "abort", "error", "repetition")


def _series_map(series, name, label_key):
    """{label_value: value} for one labeled metric family."""
    out = {}
    for labels, value in series.get(name, []):
        key = labels.get(label_key)
        if key is not None:
            try:
                ik = int(key)
            except ValueError:
                ik = key
            out[ik] = out.get(ik, 0.0) + value
    return out


def sample(host, port):
    """One vLLM sample: wall time + counters + histogram buckets + gauges."""
    flat, series, model_name = fetch_metrics_labeled(host, port, keep_buckets=True)
    s = {"t": time.time(), "_error": flat.get("_error"), "_model": model_name}
    if s["_error"]:
        return s
    for k in COUNTERS:
        s[k] = flat.get(k, 0.0)
    for h in HISTOGRAMS:
        s[h + "_sum"] = flat.get(h + "_sum", 0.0)
        s[h + "_count"] = flat.get(h + "_count", 0.0)
        s[h + "_buckets"] = {
            labels.get("le", ""): value
            for labels, value in series.get(h + "_bucket", [])
        }
    s["pos_accepted"] = _series_map(series, POS_ACCEPTED, "position")
    s["pos_drafted"] = _series_map(series, POS_DRAFTED, "position")
    s["finish"] = _series_map(series, "vllm:request_success_total", "finished_reason")
    s["tool"] = {}
    for labels, value in series.get("vllm:tool_call_parser_invocations_total", []):
        if labels.get("mode") == "streaming":
            outcome = labels.get("outcome", "?")
            s["tool"][outcome] = s["tool"].get(outcome, 0.0) + value
    s["waiting_reason"] = _series_map(series, "vllm:num_requests_waiting_by_reason", "reason")
    s["compute_class"] = _series_map(series, "vllm:scheduler_compute_seconds_total", "class")
    s["prompt_source"] = _series_map(series, "vllm:prompt_tokens_by_source_total", "source")
    s["http_total"] = sum(v for _, v in series.get("http_requests_total", []))
    # gauges
    s["running"] = flat.get("vllm:num_requests_running", 0.0)
    s["waiting"] = flat.get("vllm:num_requests_waiting", 0.0)
    s["kv_usage"] = flat.get("vllm:kv_cache_usage_perc", 0.0)
    s["fair_share"] = flat.get("vllm:scheduler_prefill_compute_share", 0.0)
    s["fair_pressure"] = flat.get("vllm:scheduler_compute_pressure", 0.0)
    s["fair_backlog"] = flat.get("vllm:scheduler_local_prefill_backlog_tokens", 0.0)
    s["proc_rss"] = flat.get("process_resident_memory_bytes", 0.0)
    s["proc_cpu"] = flat.get("process_cpu_seconds_total", 0.0)
    # model card from cache_config_info labels
    info = series.get("vllm:cache_config_info", [])
    s["cache_info"] = info[0][0] if info else {}
    return s


def _delta(cur, prev, key):
    """Counter delta; None on first sample or engine restart (value went down)."""
    if prev is None:
        return None
    c, p = cur.get(key, 0.0), prev.get(key, 0.0)
    if c < p:
        return None
    return c - p


def _engine_rate(cur, prev, num_key, den_key):
    """Δnum / Δden over counter+histogram pair; None when no progress."""
    if prev is None:
        return None
    dn = _delta(cur, prev, num_key)
    dd = _delta(cur, prev, den_key)
    if dn is None or dd is None or dd <= 0:
        return None
    return dn / dd


def _pct_delta(cur, prev, hist, q):
    """Window percentile from cumulative bucket deltas; None when empty."""
    if prev is None:
        return None
    d = bucket_delta(cur.get(hist + "_buckets") or {}, prev.get(hist + "_buckets") or {})
    total = d.get("+Inf", 0.0)
    if total <= 0:
        return None
    pts = {}
    for k, v in d.items():
        if k in ("+Inf", ""):
            continue
        try:
            pts[float(k)] = v
        except ValueError:
            continue
    return percentile_from_buckets(pts, total, q)


def _fmt_ms(seconds):
    if seconds is None:
        return "—"
    if seconds >= 1.0:
        return f"{seconds:.2f}s"
    return f"{seconds * 1000:.0f}ms"


def tick(stdscr, args, state):
    cur = sample(args.host, args.port)
    if not state["model_name"]:
        state["model_name"] = cur.get("_model")
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
    title = "vLLM Performance Dashboard"
    try:
        stdscr.addstr(y, x, "╔" + "═" * (width - 2) + "╗", curses.color_pair(COLOR_BORDER) | curses.A_BOLD)
        y += 1
        tpad = max(0, (width - 2 - len(title)) // 2)
        line = "║" + " " * tpad + title + " " * max(0, width - 2 - tpad - len(title)) + "║"
        stdscr.addstr(y, x, line[:width], curses.color_pair(COLOR_TITLE) | curses.A_BOLD)
        y += 1
        info = (f"  Backend: vllm  Model: {model_name or 'unknown'}  Host: {host}:{port}"
                f"  Uptime: {format_uptime(time.time() - start_time)}")
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
    busy = cur.get("running", 0.0) > 0
    status = "BUSY" if busy else "IDLE"
    scolor = COLOR_WARN if busy else COLOR_GREEN
    try:
        stdscr.addstr(y, x, f"  Status: OK ({status})", curses.color_pair(scolor) | curses.A_BOLD)
    except curses.error:
        pass
    y += 1

    # ---- Live throughput (delta-sampled) ----
    tg_live = wall_rate(cur, prev, "vllm:generation_tokens_total")
    pp_live = wall_rate(cur, prev, "vllm:prompt_tokens_total")
    tg_engine = _engine_rate(cur, prev, "vllm:generation_tokens_total", "vllm:request_decode_time_seconds_sum")
    pp_engine = _engine_rate(cur, prev, "vllm:prompt_tokens_total", "vllm:request_prefill_time_seconds_sum")
    pp_computed = wall_rate(cur, prev, "vllm:request_prefill_kv_computed_tokens_sum")
    tg_live_avg = avg_wall_rate(history, "vllm:generation_tokens_total")
    pp_live_avg = avg_wall_rate(history, "vllm:prompt_tokens_total")
    tg_engine_avg = avg_rate(history, "vllm:generation_tokens_total", "vllm:request_decode_time_seconds_sum")
    pp_engine_avg = avg_rate(history, "vllm:prompt_tokens_total", "vllm:request_prefill_time_seconds_sum")
    # avg tokens per engine step (batch shape)
    iter_dsum = _delta(cur, prev, "vllm:iteration_tokens_total_sum")
    iter_dcount = _delta(cur, prev, "vllm:iteration_tokens_total_count")
    tok_per_step = iter_dsum / iter_dcount if iter_dsum is not None and iter_dcount and iter_dcount > 0 else None

    if rate_hist is not None:
        rate_hist.append((cur["t"], pp_live, tg_live))
        prune_rate_hist(rate_hist, 120.0)

    draw_box_start(stdscr, y, x, width, "Live Throughput (delta-sampled)")
    y += 1
    tg_color = COLOR_GOOD if (tg_live or 0) >= 30 else COLOR_WARN if (tg_live or 0) > 0 else COLOR_DIM
    draw_row(stdscr, y, x, "Decode TG (engine):",
             f"{format_speed(tg_engine)} tok/s   avg {format_speed(tg_engine_avg)}", width)
    y += 1
    draw_row(stdscr, y, x, "TG live (wall):",
             f"{format_speed(tg_live)} tok/s   avg {format_speed(tg_live_avg)}", width, value_color=tg_color)
    y += 1
    draw_row(stdscr, y, x, "Prefill PP (engine):",
             f"{format_speed(pp_engine)} tok/s   avg {format_speed(pp_engine_avg)}", width)
    y += 1
    draw_row(stdscr, y, x, "PP live (wall):",
             f"{format_speed(pp_live)} tok/s   avg {format_speed(pp_live_avg)}"
             f"   computed {format_speed(pp_computed)}", width)
    y += 1
    draw_row(stdscr, y, x, "Batch shape:",
             f"{'—' if tok_per_step is None else f'{tok_per_step:,.0f}'} tokens/step"
             f"   running {int(cur.get('running', 0))}  waiting {int(cur.get('waiting', 0))}", width)
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

    # ---- Spec decode / MTP panel ----
    d_acc = _delta(cur, prev, "vllm:spec_decode_num_accepted_tokens_total")
    d_dtok = _delta(cur, prev, "vllm:spec_decode_num_draft_tokens_total")
    d_drafts = _delta(cur, prev, "vllm:spec_decode_num_drafts_total")
    acc_live = d_acc / d_dtok * 100 if d_acc is not None and d_dtok and d_dtok > 0 else None
    tau_live = 1 + d_acc / d_drafts if d_acc is not None and d_drafts and d_drafts > 0 else None
    k_live = d_dtok / d_drafts if d_dtok is not None and d_drafts and d_drafts > 0 else None
    # cumulative fallbacks (always meaningful over engine lifetime)
    c_acc = cur.get("vllm:spec_decode_num_accepted_tokens_total", 0.0)
    c_dtok = cur.get("vllm:spec_decode_num_draft_tokens_total", 0.0)
    c_drafts = cur.get("vllm:spec_decode_num_drafts_total", 0.0)
    acc_cum = c_acc / c_dtok * 100 if c_dtok > 0 else 0.0
    tau_cum = 1 + c_acc / c_drafts if c_drafts > 0 else 0.0
    # per-position (window deltas, fall back to cumulative when no window data)
    pos_parts = []
    positions = sorted(set(cur.get("pos_accepted", {})) | set(cur.get("pos_drafted", {})))
    for p in positions:
        a = _series_delta(cur, prev, "pos_accepted", p)
        d = _series_delta(cur, prev, "pos_drafted", p)
        if a is None or d is None or d <= 0:
            ca = cur.get("pos_accepted", {}).get(p, 0.0)
            cd = cur.get("pos_drafted", {}).get(p, 0.0)
            pos_parts.append(f"p{p} {ca / cd * 100:.0f}%" if cd > 0 else f"p{p} —")
        else:
            pos_parts.append(f"p{p} {a / d * 100:.0f}%")

    draw_box_start(stdscr, y, x, width, "Spec Decode / MTP")
    y += 1
    acc_color = COLOR_DIM if not c_dtok else COLOR_GREEN if acc_cum >= 70 else COLOR_WARN if acc_cum >= 50 else COLOR_RED
    draw_row(stdscr, y, x, "Acceptance:",
             f"live {fmt_pct(acc_live)}   cum {acc_cum:.1f}%  ({format_num(c_acc)}/{format_num(c_dtok)})",
             width, value_color=acc_color)
    y += 1
    draw_row(stdscr, y, x, "τ accept length:",
             f"live {f'{tau_live:.2f}' if tau_live is not None else '—'}   cum {tau_cum:.2f}"
             f"   k {f'{k_live:.1f}' if k_live is not None else '—'}", width)
    y += 1
    draw_row(stdscr, y, x, "Per-position:", "  ".join(pos_parts) if pos_parts else "—", width)
    y += 1
    draw_box_end(stdscr, y, x, width)
    y += 1

    # ---- Live cache ----
    kv = cur.get("kv_usage", 0.0) * 100
    cache_color = COLOR_GREEN if kv < 70 else COLOR_WARN if kv < 90 else COLOR_RED
    ci = cur.get("cache_info", {})
    pool_tokens = _to_float(ci.get("kv_cache_size_tokens"))
    used_tokens = pool_tokens * cur.get("kv_usage", 0.0) if pool_tokens else None
    d_cached = _delta(cur, prev, "vllm:prompt_tokens_cached_total")
    d_prompt = _delta(cur, prev, "vllm:prompt_tokens_total")
    hit_live = d_cached / d_prompt * 100 if d_cached is not None and d_prompt and d_prompt > 0 else None
    c_cached = cur.get("vllm:prompt_tokens_cached_total", 0.0)
    c_prompt = cur.get("vllm:prompt_tokens_total", 0.0)
    hit_cum = c_cached / c_prompt * 100 if c_prompt > 0 else 0.0
    d_hits = _delta(cur, prev, "vllm:prefix_cache_hits_total")
    d_queries = _delta(cur, prev, "vllm:prefix_cache_queries_total")
    qhit_live = d_hits / d_queries * 100 if d_hits is not None and d_queries and d_queries > 0 else None
    c_hits = cur.get("vllm:prefix_cache_hits_total", 0.0)
    c_queries = cur.get("vllm:prefix_cache_queries_total", 0.0)
    qhit_cum = c_hits / c_queries * 100 if c_queries > 0 else 0.0
    d_preempt = _delta(cur, prev, "vllm:num_preemptions_total")
    src_parts = []
    for key, label in (("local_compute", "computed"), ("local_cache_hit", "cache-hit"), ("external_kv_transfer", "ext-KV")):
        sd = _series_delta(cur, prev, "prompt_source", key)
        if sd is None or sd <= 0:
            continue
        src_parts.append(f"{label} {format_num(sd)}")

    draw_box_start(stdscr, y, x, width, "Live Cache")
    y += 1
    draw_bar_row(stdscr, y, x, "KV usage:", kv, width, cache_color)
    y += 1
    pool_str = (f"{format_num(used_tokens)} / {format_num(pool_tokens)} tokens"
                if pool_tokens else "pool size unknown")
    draw_row(stdscr, y, x, "Pool:", pool_str, width)
    y += 1
    hit_color = COLOR_GREEN if hit_cum >= 70 else COLOR_WARN
    draw_row(stdscr, y, x, "Prefix hit (tokens):",
             f"live {fmt_pct(hit_live)}   cum {hit_cum:.1f}%  ({format_num(c_cached)}/{format_num(c_prompt)})",
             width, value_color=hit_color)
    y += 1
    draw_row(stdscr, y, x, "Prefix hit (queries):",
             f"live {fmt_pct(qhit_live)}   cum {qhit_cum:.1f}%  ({format_num(c_hits)}/{format_num(c_queries)})",
             width)
    y += 1
    preempt = cur.get("vllm:num_preemptions_total", 0.0)
    draw_row(stdscr, y, x, "Preemptions:",
             f"{preempt:.0f}  (window {fmt_cnt(d_preempt)})" + ("   ⚠ KV pressure" if d_preempt else ""),
             width, value_color=COLOR_RED if d_preempt else COLOR_DIM)
    y += 1
    if src_parts:
        draw_row(stdscr, y, x, "Prompt source Δ:", "  ".join(src_parts), width)
        y += 1
    draw_box_end(stdscr, y, x, width)
    y += 1

    # ---- Latency (window averages + percentiles) ----
    def _live_avg(h):
        ds = _delta(cur, prev, h + "_sum")
        dc = _delta(cur, prev, h + "_count")
        if ds is None or dc is None or dc <= 0:
            return None
        return ds / dc

    ttft = _live_avg("vllm:time_to_first_token_seconds")
    itl = _live_avg("vllm:inter_token_latency_seconds")
    tpt = _live_avg("vllm:request_time_per_output_token_seconds")
    e2e = _live_avg("vllm:e2e_request_latency_seconds")
    queue = _live_avg("vllm:request_queue_time_seconds")
    itl_p50, itl_p90, itl_p99 = (_pct_delta(cur, prev, "vllm:inter_token_latency_seconds", q) for q in (0.5, 0.9, 0.99))
    ttft_p99 = _pct_delta(cur, prev, "vllm:time_to_first_token_seconds", 0.99)
    e2e_p99 = _pct_delta(cur, prev, "vllm:e2e_request_latency_seconds", 0.99)

    draw_box_start(stdscr, y, x, width, "Latency (window avg + percentiles)")
    y += 1
    draw_row(stdscr, y, x, "TTFT:",
             f"{_fmt_ms(ttft)}   p99 {_fmt_ms(ttft_p99)}", width)
    y += 1
    draw_row(stdscr, y, x, "ITL (inter-token):",
             f"{_fmt_ms(itl)}   p50 {_fmt_ms(itl_p50)}  p90 {_fmt_ms(itl_p90)}  p99 {_fmt_ms(itl_p99)}", width)
    y += 1
    draw_row(stdscr, y, x, "Time/token:",
             f"{_fmt_ms(tpt)}   E2E {_fmt_ms(e2e)} (p99 {_fmt_ms(e2e_p99)})   queue {_fmt_ms(queue)}", width)
    y += 1
    draw_box_end(stdscr, y, x, width)
    y += 1

    # ---- Scheduler & fairness ----
    wr = cur.get("waiting_reason", {})
    draw_box_start(stdscr, y, x, width, "Scheduler & Fairness")
    y += 1
    draw_row(stdscr, y, x, "Running / Waiting:",
             f"{int(cur.get('running', 0))} / {int(cur.get('waiting', 0))}"
             f"   (capacity {int(wr.get('capacity', 0))}, deferred {int(wr.get('deferred', 0))})", width)
    y += 1
    fair = (cur.get("fair_share", 0.0), cur.get("fair_pressure", 0.0), cur.get("fair_backlog", 0.0))
    fcolor = COLOR_DIM if not any(fair) else COLOR_WARN
    draw_row(stdscr, y, x, "Fairness:",
             f"prefill share {fair[0]:.2f}  pressure {fair[1]:.2f}  backlog {fair[2]:,.0f}",
             width, value_color=fcolor)
    y += 1
    cc = cur.get("compute_class", {})
    if cc:
        parts = []
        for cls in ("prefill", "decode"):
            cd = _series_delta(cur, prev, "compute_class", cls)
            if cd:
                parts.append(f"{cls} {cd:.2f}s")
        if parts:
            draw_row(stdscr, y, x, "Engine time Δ:", "  ".join(parts), width)
            y += 1
    draw_box_end(stdscr, y, x, width)
    y += 1

    # ---- Totals & traffic ----
    fin_parts = []
    for reason in FINISH_REASONS:
        fd = _series_delta(cur, prev, "finish", reason)
        if fd:
            fin_parts.append(f"{reason} +{fd:.0f}")
    tool = cur.get("tool", {})
    d_http = _delta(cur, prev, "http_total")
    http_rate = d_http / (cur["t"] - prev["t"]) if d_http is not None and prev is not None and cur["t"] > prev["t"] else None

    draw_box_start(stdscr, y, x, width, "Totals & Traffic")
    y += 1
    draw_row(stdscr, y, x, "Tokens P / G / cached:",
             f"{format_num(c_prompt)} / {format_num(cur.get('vllm:generation_tokens_total', 0.0))} / {format_num(c_cached)}", width)
    y += 1
    draw_row(stdscr, y, x, "Finished (window):",
             "  ".join(fin_parts) if fin_parts else "—", width)
    y += 1
    draw_row(stdscr, y, x, "Tool calls (stream):",
             f"tool_call {format_num(tool.get('tool_call', 0.0))}  no_tool_call {format_num(tool.get('no_tool_call', 0.0))}", width)
    y += 1
    draw_row(stdscr, y, x, "Process:",
             f"RSS {cur.get('proc_rss', 0.0) / 1e6:.0f}MB  CPU {cur.get('proc_cpu', 0.0):.0f}s"
             f"  HTTP {format_speed(http_rate)} req/s", width)
    y += 1
    draw_box_end(stdscr, y, x, width)
    y += 1

    # ---- Model card ----
    if ci:
        conc = _to_float(ci.get("kv_cache_max_concurrency"))
        card = (f"block {_g(ci, 'block_size')}  KV {_g(ci, 'cache_dtype')}"
                f"  prefix {'on' if _g(ci, 'enable_prefix_caching') == 'True' else 'off'}"
                f"  match {_g(ci, 'prefix_match_unit')}"
                f"  GMU {_g(ci, 'gpu_memory_utilization')}"
                f"  conc {f'{conc:.2f}' if conc is not None else '—'}")
        draw_row(stdscr, y, x, "Model card:", card, width, label_color=COLOR_DIM, value_color=COLOR_DIM)
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


def _series_delta(cur, prev, map_key, sub_key):
    """Delta of one labeled sub-series between samples; None on restart/first."""
    if prev is None:
        return None
    c = cur.get(map_key, {}).get(sub_key, 0.0)
    p = prev.get(map_key, {}).get(sub_key, 0.0)
    if c < p:
        return None
    return c - p


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _g(d, key, default=""):
    return d.get(key, default) if isinstance(d, dict) else default


def fmt_pct(v):
    return f"{v:.1f}%" if v is not None else "—"


def fmt_cnt(v):
    return "—" if v is None else f"{v:.0f}"
