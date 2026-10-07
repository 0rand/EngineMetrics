"""Headless render test: one clean frame per engine backend (live where reachable).

Usage: python3 tests/render_test.py [omlx|mlx-serve]
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
