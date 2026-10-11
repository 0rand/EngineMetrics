"""Headless render test: one clean frame per engine backend (live where reachable).

Usage: python3 tests/render_test.py [omlx|mlx-serve|vllm|splash|splash-synthetic] [host [port]]
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
import engines.splash as splash  # noqa: E402
import engines.vllm as vllm  # noqa: E402


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
    elif which == "splash":
        # live: two samples apart against the engine, then render with deltas
        host = sys.argv[2] if len(sys.argv) > 2 else "localhost"
        port = int(sys.argv[3]) if len(sys.argv) > 3 else 8001
        cur = splash.sample(host, port)
        if not cur.get("_error") and not cur.get("ready"):
            pass
        time.sleep(3)
        cur2 = splash.sample(host, port)
        render("splash live :%d" % port, lambda scr: splash.draw(
            scr, cur2, cur, [cur, cur2], time.time() - 60, host, port, 2, cur2.get("model_name"), rate_hist=None))
    elif which == "splash-synthetic":
        # offline: fabricated sample pair exercising every panel with data
        def mk(t, pre_rows, pre_ms, dec_rows, dec_ms, dec_b1, dec_b2, dec_batches,
               dec, pre, queued, kv_alloc, kv_cap, hits, cold, mem_charged, mem_limit,
               drafted, accepted, ttft95, busy):
            s = {
                "t": t, "_error": None, "ready": True, "pressure": "normal",
                "thermal": "fair", "max_ctx": 262144,
                "model_name": "mlx-community/Qwen3.8-27B-8bit",
                "engine_uptime": 3725.0,
                "sched_queued": queued, "sched_wait_res": 0, "sched_wait_prefix": 0,
                "sched_prefilling": pre, "sched_decoding": dec, "sched_wait_mask": 0,
                "sched_terminal": 0, "prefill_batches": 937, "decode_batches": dec_batches,
                "max_batch_width": 4, "decode_b1": dec_b1, "decode_b2": dec_b2,
                "decode_b3": 0, "decode_b4": 0,
                "adm_waiting": 0, "adm_mem": 0, "adm_conc": 0, "adm_oldest_ms": 0.0,
                "fe_active": dec + pre, "fe_waiting": 0, "fe_cap": 4,
                "loop_tick_ms": 120.5,
                "kv_pages_allocated": kv_alloc, "kv_pages_active": dec * 130 if dec else 130,
                "kv_pages_cache": 6557, "kv_pages_free": 97, "kv_capacity_pages": kv_cap,
                "kv_alloc_bytes": kv_alloc * 1064960, "kv_reclaim_bytes": 0,
                "kv_block_tokens": 32, "kv_extent_alloc": 52, "kv_extent_rel": 0,
                "kv_extent_compact": 0, "kv_extent_alloc_ms": 31.3, "kv_format": "int8",
                "cache_hits": hits, "cache_cold": cold, "cache_hit_rate": hits / max(1, hits + cold),
                "cache_reused": 595904, "cache_kv_hit_tokens": 1206080,
                "state_entries": 314, "state_in_use": dec, "state_bytes": 50252218368,
                "state_lanes": 1 if dec else 0, "state_evictions": 0,
                "disk_capacity": 0, "disk_kv_blocks": 0,
                "mem_limit": mem_limit, "mem_charged": mem_charged,
                "mem_headroom": mem_limit - mem_charged, "mem_current": mem_charged,
                "mem_peak": mem_charged + 2**30, "host_available": 32 * 2**30,
                "denied_res": 0, "weights_released": False, "idle_release_s": 600,
                "weight_restores": 1, "weight_failures": 0,
                "device_name": "Apple M5 Max", "gpu_cores": 40, "physical_mem": 128 * 2**30,
                "macos": "27.0.1",
                "drafted": drafted, "accepted": accepted,
                "draft_rate": accepted / drafted if drafted else 0.0,
                "draft_ctx_avoided": 53664, "draft_ctx_resets": 39,
                "ttft_p50_ms": 946.5, "ttft_p95_ms": ttft95, "ttft_n": 318,
                "itl_p50_ms": 16.1, "itl_p95_ms": 76.3, "itl_n": 4096,
                "http_avg_s": 6.88, "http_n": 318, "prep_avg_ms": 3.6, "queue_avg_ms": 37.9,
                "prefill_rows": pre_rows, "prefill_wall_s": pre_ms / 1000.0,
                "decode_rows": dec_rows, "decode_wall_s": dec_ms / 1000.0,
                "decode_cycle_s": dec_ms / 1000.0,
                "pp_lifetime_tps": 549.9, "tg_lifetime_tps": 47.3,
                "req_submitted": 319, "req_completed": 318, "req_cancelled": 0, "req_failed": 0,
                "capacity_failures": 0, "metal_failures": 0,
                "last_ppb_valid": True, "last_ppb_width": 1, "last_ppb_tok": 2048,
                "last_ppb_tps": 2311.4,
                "last_tgb_valid": bool(dec), "last_tgb_width": 1, "last_tgb_out": 3,
                "last_tgb_draft": 7, "last_tgb_acc": 1, "last_tgb_tps": 39.2,
                "metal_healthy": True, "transport_ready": True, "transport_pending": 0,
                "transport_pending_limit": 32, "transport_restarts": 0,
                "ane_state": "split", "ane_share": 0.206,
                "warmup_prefill": True, "warmup_b": True, "warmup_restore": True,
                "busy": busy,
            }
            return s
        a = mk(1000.0, 222000, 420000.0, 70000, 1500000.0, 15000, 42, 15042,
               1, 1, 0, 6656, 75008, 310, 3, 90 * 2**30, 113 * 2**30,
               100000, 55000, 6400.0, True)
        b = mk(1002.0, 224000, 430000.0, 70300, 1505000.0, 15040, 42, 15082,
               1, 0, 0, 6656, 75008, 312, 3, 90.5 * 2**30, 113 * 2**30,
               101000, 55500, 6500.0, True)
        render("splash synthetic", lambda scr: splash.draw(
            scr, b, a, [a, b], time.time() - 60, "localhost", 8001, 2, b["model_name"], rate_hist=None))
    elif which == "splash-fresh":
        # regression: /status with almost nothing in it (as on a just-started engine);
        # sample() must tolerate missing sections and empty histograms (avg -> None)
        splash.fetch_json = lambda host, port, path: {
            "schema_version": 6, "ready": False,
            "latency": {"http_request": {"count": 0, "sum": 0.0}}}
        cur = splash.sample("localhost", 8001)
        render("splash fresh", lambda scr: splash.draw(
            scr, cur, None, [cur], time.time() - 60, "localhost", 8001, 2, cur.get("model_name"), rate_hist=None))
    elif which == "splash-robust":
        # regressions: (1) NaN/Infinity accepted by json.loads must not crash sample();
        # (2) a counter reset (engine restart) must not yield negative rates
        import json as _json
        from collections import deque
        from types import SimpleNamespace
        splash.fetch_json = lambda h, p, path: _json.loads(
            '{"ready": true, "scheduler": {"queued": NaN, "decoding": Infinity},'
            ' "kv": {"pages_allocated": -Infinity}}')
        cur = splash.sample("localhost", 8001)
        assert cur["sched_queued"] == 0 and cur["sched_decoding"] == 0, cur
        print("nan/inf: OK")

        def st(pt, pms, dt, dms):
            return {"ready": True, "metrics": {"prefill_input_tokens": pt, "prefill_wall_ms": pms,
                                               "decode_output_tokens": dt, "decode_wall_ms": dms}}
        state = {"start_time": time.time(), "prev": None, "history": deque(maxlen=15),
                 "rate_hist": deque(), "model_name": None}
        args = SimpleNamespace(host="localhost", port=8001, interval=2)
        for i, s in enumerate([st(1000, 2000.0, 500, 10000.0), st(1100, 2200.0, 600, 12000.0),
                               st(10, 20.0, 5, 100.0)]):   # last one = restarted engine
            splash.fetch_json = lambda h, p, path, s=s: s
            scr = FakeScr()
            splash.tick(scr, args, state)
            time.sleep(0.2)
            line = next(l for l in scr.buf if "TG live" in l)
            assert "-" not in line, line      # em dash is fine, an ASCII minus is not
        assert len(state["history"]) == 1, len(state["history"])   # window dropped at restart
        print("restart: OK")
    elif which == "vllm":
        # live: two samples 3s apart against the engine, then render with deltas
        host, port = "localhost", 8100
        if len(sys.argv) > 3:
            host, port = sys.argv[2], int(sys.argv[3])
        cur = vllm.sample(host, port)
        time.sleep(3)
        cur2 = vllm.sample(host, port)
        render("vllm live :%d" % port, lambda scr: vllm.draw(
            scr, cur2, cur, [cur, cur2], time.time() - 60, host, port, 2, cur2.get("_model"), rate_hist=None))
    elif which == "vllm-synthetic":
        # offline: fabricated sample pair exercising every panel with data
        def mk(t, gen, prompt, cached, drafts, dtok, acc, dec_s, dec_c, pre_s, pre_c,
               ttft_s, ttft_c, itl_s, itl_c, preempt, hits, queries, running):
            s = {"t": t, "_error": None, "_model": "glm-5.3-flash",
                 "vllm:generation_tokens_total": gen, "vllm:prompt_tokens_total": prompt,
                 "vllm:prompt_tokens_cached_total": cached,
                 "vllm:spec_decode_num_drafts_total": drafts,
                 "vllm:spec_decode_num_draft_tokens_total": dtok,
                 "vllm:spec_decode_num_accepted_tokens_total": acc,
                 "vllm:request_decode_time_seconds_sum": dec_s,
                 "vllm:request_decode_time_seconds_count": dec_c,
                 "vllm:request_prefill_time_seconds_sum": pre_s,
                 "vllm:request_prefill_time_seconds_count": pre_c,
                 "vllm:time_to_first_token_seconds_sum": ttft_s,
                 "vllm:time_to_first_token_seconds_count": ttft_c,
                 "vllm:inter_token_latency_seconds_sum": itl_s,
                 "vllm:inter_token_latency_seconds_count": itl_c,
                 "vllm:inter_token_latency_seconds_buckets": {"0.05": itl_c * 0.2, "0.1": itl_c * 0.7, "+Inf": float(itl_c)},
                 "vllm:time_to_first_token_seconds_buckets": {"0.3": ttft_c * 0.5, "0.5": ttft_c * 0.9, "+Inf": float(ttft_c)},
                 "vllm:num_preemptions_total": preempt,
                 "vllm:prefix_cache_hits_total": hits, "vllm:prefix_cache_queries_total": queries,
                 "pos_accepted": {0: acc * 0.4, 1: acc * 0.33, 2: acc * 0.27},
                 "pos_drafted": {0: dtok / 3, 1: dtok / 3, 2: dtok / 3},
                 "finish": {"stop": 100.0}, "tool": {"tool_call": 6750.0, "no_tool_call": 5478.0},
                 "waiting_reason": {"capacity": 0.0, "deferred": 0.0},
                 "compute_class": {"prefill": 0.0, "decode": 0.0},
                 "prompt_source": {"local_compute": 524599.0, "local_cache_hit": 36486923.0, "external_kv_transfer": 0.0},
                 "http_total": 12345.0,
                 "running": running, "waiting": 0.0, "kv_usage": 0.335,
                 "fair_share": 0.0, "fair_pressure": 0.0, "fair_backlog": 0.0,
                 "proc_rss": 631656448.0, "proc_cpu": 131.9,
                 "cache_info": {"block_size": "4096", "cache_dtype": "fp8", "enable_prefix_caching": "True",
                                "prefix_match_unit": "128", "gpu_memory_utilization": "0.87",
                                "kv_cache_size_tokens": "1606127", "kv_cache_max_concurrency": "1.53"}}
            for h in ("vllm:request_time_per_output_token_seconds", "vllm:e2e_request_latency_seconds",
                      "vllm:request_queue_time_seconds", "vllm:request_prefill_kv_computed_tokens",
                      "vllm:iteration_tokens_total"):
                s[h + "_sum"] = 0.0
                s[h + "_count"] = 0.0
                s[h + "_buckets"] = {}
            return s
        a = mk(1000.0, 100000, 50000, 40000, 1000, 3000, 2000, 50.0, 1000, 10.0, 20,
               5.0, 20, 3.0, 2000, 0, 40000, 41000, 1)
        b = mk(1010.0, 100450, 50100, 40080, 1010, 3030, 2020, 51.4, 1010, 10.2, 21,
               5.1, 21, 3.4, 2020, 0, 40080, 41010, 1)
        render("vllm synthetic", lambda scr: vllm.draw(
            scr, b, a, [a, b], time.time() - 60, "localhost", 8100, 2, "glm-5.3-flash", rate_hist=None))
