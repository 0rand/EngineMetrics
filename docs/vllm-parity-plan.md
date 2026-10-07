# vLLM Backend Parity Plan — replicating the TensorFold / mlx-serve dashboards

Date: 2026-10-07 · Source: live GLM-5.3-Flash-NVFP4 stack (vllm 0.1.dev1+ga7b41c45a, 2× DGX Spark TP2, :8100)
Method: full `/metrics` dump + delta-sampling probe with a live streaming request. All "proven" numbers
below were measured on the live engine, not theorized.

## Executive summary

The current `engines/vllm.py` uses ~15 of ~60 available metric families and only the
cumulative-histogram view. The Prometheus surface is rich enough to replicate **every panel**
of the TensorFold dashboard (live delta-sampled throughput, sparklines, cache occupancy, MTP
acceptance, latency averages) **plus** things TF and mlx-serve cannot show (ITL percentiles,
per-position decay bars, finished-reason accounting, scheduler fairness gauges, tool-call
traffic profile). Three implementation changes unlock it:

1. **Delta-sampling state** — vllm.py needs a `prev` sample + history like `tf_sample()` already
   does in tensorfold.py. Counters are cumulative and never reset; deltas between polls give
   live rates. Proven: Δ`generation_tokens_total`/Δt = 44.8 tok/s wall, Δgen/Δ`decode_time_sum`
   = 55.2 tok/s engine-time.
2. **Label-aware parsing** — `common.fetch_metrics()` collapses labeled series (last wins).
   Per-position spec counters, `prompt_tokens_by_source`, `waiting_by_reason`,
   `scheduler_compute_seconds{class=}`, `tool_call_parser_invocations{outcome=}` and histogram
   `_bucket` lines all need label-keyed parsing (probe used `name@posN` suffixing — works).
3. **Keep `_bucket` lines** — common.py currently drops them; they are the only source of
   p50/p90/p99 latency percentiles (335 bucket series available).

## The table

Legend: **[D]** = delta-sampled counter (needs prev sample) · **[G]** = direct gauge ·
**[H]** = histogram (sum/count delta for avg, buckets for percentiles) ·
✅ = proven live on this engine · ⚠️ = present but zero/not populated in this build

### Panel 1 — Live Throughput (delta-sampled)  ← replicates TF "Live Throughput" + mlx-serve TG live

| Dashboard row | Metric(s) | Math | Status |
|---|---|---|---|
| TG live (wall) | `vllm:generation_tokens_total` [D] | Δgen / Δwall | ✅ 44.8 tok/s |
| TG (engine) | `vllm:generation_tokens_total` + `vllm:request_decode_time_seconds_sum` [D,H] | Δgen / Δdecode_time | ✅ 55.2 tok/s |
| PP (wall) | `vllm:prompt_tokens_total` [D] | Δprompt / Δwall | ✅ (0 when idle — honest gap) |
| PP (engine) | `vllm:prompt_tokens_total` + `vllm:request_prefill_time_seconds_sum` [D,H] | Δprompt / Δprefill_time | derivable, same pattern |
| PP computed (cache-restores excluded) | `vllm:request_prefill_kv_computed_tokens` [H] | Δsum / Δt | new — mlx-serve parity for "prefill tokens actually computed" |
| 120s sparklines | same counters | per-poll rates → sparkline | ✅ pattern proven (TF uses identical code) |
| Batch shape | `vllm:iteration_tokens_total` [H] | Δsum/Δcount = avg tokens per engine step | new — batch-size distribution over time |

### Panel 2 — Live Cache  ← replicates TF "Live Cache"

| Dashboard row | Metric(s) | Math | Status |
|---|---|---|---|
| KV occupancy bar | `vllm:kv_cache_usage_perc` [G] | ×100 | ✅ 33.5% under load |
| Pool tokens | `vllm:cache_config_info` [G] | `kv_cache_size_tokens` = 1,606,127; used = usage_perc × pool | ✅ → "538K / 1.6M tokens" (TF pool-row parity) |
| Prefix cache hit (tokens, live) | `vllm:prompt_tokens_cached_total` / `vllm:prompt_tokens_total` [D] | Δcached / Δprompt | ✅ cumulative 98.6% (36.5M/37.0M) |
| Prefix cache hit (queries, live) | `vllm:prefix_cache_hits_total` / `vllm:prefix_cache_queries_total` [D] | Δhits / Δqueries | ✅ cumulative 96.7% |
| Cache source split | `vllm:prompt_tokens_by_source_total{source=local_compute\|local_cache_hit\|external_kv_transfer}` [D] | per-source deltas | new — shows external KV transfer (multi-node) activity |
| Preemptions | `vllm:num_preemptions_total` [D] | Δ > 0 = KV pressure events | ✅ 0 (healthy) |

### Panel 3 — Spec Decode / MTP  ← replicates TF "MTP Acceptance" + tool-eval-bench spec-live gauges

| Dashboard row | Metric(s) | Math | Status |
|---|---|---|---|
| Live acceptance % | `vllm:spec_decode_num_accepted_tokens_total` / `..._draft_tokens_total` [D] | Δacc / Δdraft_tok | ✅ 92.1% live |
| τ acceptance length | `..._accepted_tokens_total` + `..._num_drafts_total` [D] | 1 + Δacc/Δdrafts | ✅ 3.76 (k=3) |
| k (draft window) | `..._draft_tokens_total` / `..._num_drafts_total` [D] | Δdraft_tok / Δdrafts | ✅ exactly 3.0 |
| Per-position bars | `..._accepted_tokens_per_pos_total{position=N}` / `..._draft_tokens_per_pos_total{position=N}` [D] | Δacc@pos / Δdraft@pos | ✅ 95.2 / 90.5 / 90.5% |
| Decay summary | same | pos0→posN drop | new vs TF |
| Rolling 30s window | pooled deltas | spec-live `apply_rolling_rates` pattern | design adopted from tool-eval-bench |
| Sticky gauges | n/a | counters never reset — NOT needed for vLLM counters (spec-live needs it only for its deprecated avg gauges) | simplification vs spec-live |

### Panel 4 — Latency  ← replicates TF "Averages", exceeds it

| Dashboard row | Metric(s) | Math | Status |
|---|---|---|---|
| TTFT (live avg) | `vllm:time_to_first_token_seconds` [H] | Δsum/Δcount | ✅ 325ms |
| ITL (live avg) | `vllm:inter_token_latency_seconds` [H] | Δsum/Δcount | ✅ 69.1ms — **TF and mlx-serve have no ITL metric at all** |
| Time/token | `vllm:request_time_per_output_token_seconds` [H] | Δsum/Δcount | available |
| E2E latency | `vllm:e2e_request_latency_seconds` [H] | Δsum/Δcount | available |
| Queue time | `vllm:request_queue_time_seconds` [H] | Δsum/Δcount | new — scheduler wait visibility |
| Prefill/decode/inference time | `vllm:request_{prefill,decode,inference}_time_seconds` [H] | Δsum/Δcount | available |
| **Percentiles p50/p90/p99** | `_bucket` lines of ITL/TTFT/E2E [H] | bucket interpolation per window | new — 335 bucket series live; ITL buckets show real shape (13,845 tokens ≤100ms of 15,640) |

### Panel 5 — Queue & Scheduler (new panel, no TF/mlx-serve equivalent)

| Dashboard row | Metric(s) | Math | Status |
|---|---|---|---|
| Running / Waiting | `vllm:num_requests_running` / `vllm:num_requests_waiting` [G] | direct | ✅ |
| Waiting by reason | `vllm:num_requests_waiting_by_reason{reason=capacity\|deferred}` [G] | direct | new |
| Fairness: prefill compute share | `vllm:scheduler_prefill_compute_share` [G] | direct | ⚠️ gauge present (0 = disabled this boot) |
| Fairness: compute pressure | `vllm:scheduler_compute_pressure` [G] | direct | ⚠️ same |
| Fairness: prefill backlog | `vllm:scheduler_local_prefill_backlog_tokens` [G] | direct | ⚠️ same |
| Engine time by class | `vllm:scheduler_compute_seconds_total{class=decode\|prefill}` [D] | Δ per class | ⚠️ 0.0 this boot (populates when fairness engine active) |

### Panel 6 — Totals, Model Card & Traffic Profile

| Dashboard row | Metric(s) | Math | Status |
|---|---|---|---|
| Requests by finish reason | `vllm:request_success_total{finished_reason=stop\|length\|abort\|error\|repetition}` [D] | deltas | new — error/abort accounting TF lacks |
| Token totals | `vllm:prompt_tokens_total` / `generation_tokens_total` / `prompt_tokens_cached_total` | cumulative | ✅ |
| Model card | `vllm:cache_config_info` [G] labels | block_size 4096, KV fp8, prefix on, match_unit 128, pool 1.6M tok, max_concurrency 1.53, GMU 0.87, mamba align | ✅ one gauge = whole model card |
| Tool-call traffic | `vllm:tool_call_parser_invocations_total{mode,outcome,request_type}` [D] | Δ by outcome | new — Governor traffic profile (6,750 tool_calls / 5,478 no_tool_call streaming) |
| HTTP layer | `http_requests_total{handler,status}` + `http_request_duration_highr_seconds` [H] | per-handler rates + p99 | new |
| Process health | `process_resident_memory_bytes`, `process_cpu_seconds_total`, `process_open_fds` [G/D] | direct + Δcpu/Δt | new |
| MFU | `vllm:estimated_flops_per_gpu_total` + `estimated_{read,write}_bytes_per_gpu_total` [D] | Δflops / (peak_flops × Δt) | ⚠️ all 0.0 in this build — would be free MFU/MBW if populated; upstream gap |

## What vLLM cannot give us (TF/mlx-serve features with no metric)

| TF / mlx-serve feature | vLLM status | Workaround |
|---|---|---|
| Per-stream states (decoding/prefilling/filling/paused) | none — only running/waiting counts | running gauge is the closest proxy |
| Native prefill progress (processed/total + ETA) | none | `request_prefill_kv_computed_tokens` histogram deltas approximate it |
| GPU util % + phys footprint (`mlx_serve:gpu_utilization_pct`, `memory_mb`) | none in /metrics | DCGM exporter sidecar or `nvidia-smi` poller (out of scope for zero-dep TUI, or optional) |
| Multi-prefill chunk/piece stats | TF-specific | n/a |
| Pool free/used rows | derivable | usage_perc × kv_cache_size_tokens |

## Implementation notes

- `common.fetch_metrics()` needs a label-aware mode: keep `name{labels}` series keyed by
  label tuple (or `name@key` suffixes) instead of last-wins. Keep `_bucket` lines behind a flag.
- vllm.py: add `prev`/`history` state exactly like tensorfold.py's `tf_sample()`; reuse
  `rate()`, `wall_rate()`, `avg_rate()`, `sparkline()` from common.py unchanged.
- Counters never reset on the Prometheus registry (no sticky-gauge problem). The only
  zero-flicker risk is gauges (`num_requests_running` legitimately hits 0 when idle) — show
  real 0, don't stick.
- Optional server flag `--per-request-spec-decode-metrics` (used by tool-eval-bench) embeds
  exact request-scoped spec metrics in responses — not needed for the dashboard, but useful
  for bench runs.
- Honest-gap rule (TF parity): show `—` when a window has no finished requests / no progress,
  never a fake 0.

## Verification performed (2026-10-07, live engine)

- Idle-window delta: all counters stable, rates honestly `—`/0.
- One 80-token streaming probe: TTFT 752ms client / 325ms engine-hist, TG 44.8 wall /
  55.2 engine, acceptance 92.1% (τ=3.76), per-pos 95.2/90.5/90.5, ITL 69.1ms.
- Cumulative cross-checks: token hit 98.6%, query hit 96.7%, k=3.0, pool 1.6M tokens.
