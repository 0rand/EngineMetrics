# EngineMetrics

A zero-dependency, single-file curses TUI that shows **real-time inference-engine
metrics** for **vLLM** and **TensorFold** servers. Like `htop` for your LLM engine.

```
╔══════════════════════════════════════════════════════════════════════╗
║                    TensorFold Performance Dashboard                  ║
║  Backend: tensorfold  Model: glm-5.3-flash  Host: 192.168.1.8:8100   ║
╚══════════════════════════════════════════════════════════════════════╝
  Status: OK (BUSY)
┌─ Live Throughput (delta-sampled) ──────────────────────────────────┐
│ Prefill PP (engine): 1,331 tok/s   avg 1,290                       │
│ Decode TG (engine):  62.7 tok/s   avg 58.4                         │
│ TG live (wall):      100.9 tok/s   avg 96.2                        │
│ Streams: decoding 1  prefilling 0  filling 0  paused 0  / max 4    │
│ PP 120s [1.33K]: [····▁▁▂▄█▆▅▄▃▂▁·]                                │
│ TG 120s [100.9]: [·····▄▅▆▇█▇▆▅▄▃·]                                │
└────────────────────────────────────────────────────────────────────┘
┌─ Live Cache ───────────────────────────────────────────────────────┐
│ Pool occupancy:  11.6%  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]           │
│ Pool: 226K / 1.9M rows free 1.7M                                   │
│ Stream KV ratio: 0.0% (per-stream / ctx window)                    │
│ Kept prompts: 32   context window 1.0M                             │
└────────────────────────────────────────────────────────────────────┘
┌─ Averages (cumulative) ────────────────────────────────────────────┐
│ Avg Latency: 10.21s  (n=588)                                       │
│ Avg TTFT: 1.85s  (n=588)                                           │
│ MTP Acceptance: 48.9%  (70,755/144,736)                            │
│ Cache Hit (tokens): 75.4%  (1.4M/1.9M)                             │
└────────────────────────────────────────────────────────────────────┘
```

## Why

vLLM ships a full Prometheus `/metrics` endpoint. **TensorFold does not (yet)** — it
exposes a partial `tensorfold:*` metric surface plus a rich JSON `/health`. This
dashboard works with **both**, auto-detected, so you keep one monitoring tool across
engines. For TensorFold it delta-samples the cumulative counters between polls to
derive live rates — no agent, no exporter, no dependencies.

## Supported backends

| Backend | Detection | What you get |
|---|---|---|
| **vLLM** | `vllm:*` keys in `/metrics` | Latency breakdown (E2E/prefill/decode/TTFT/time-per-token), KV cache %, running/waiting, token totals, prefix-cache hit rate |
| **TensorFold** | `backend: "tensorfold"` in `/health` | Live PP/TG (engine-time + wall-clock), 120s sparklines, stream states, shared-pool occupancy, kept prompts, MTP acceptance, TTFT/latency averages, cache-hit %, multi-prefill stats |

## Install & run

Python 3.8+, stdlib only (curses, urllib, json, re). No pip installs.

```bash
python3 vllm-stats.py --host 192.168.1.8 --port 8100
```

| Flag | Default | |
|---|---|---|
| `--host` | `localhost` | Engine host |
| `--port` | `8100` | Engine port |
| `--interval` | `2` | Refresh seconds |
| `--window` | `15` | Rate-averaging window (samples) |

Keys: `q` quit · `r` reset uptime · `c` clear rate history/sparklines.

## How the TensorFold numbers are derived

TensorFold's counters are **cumulative** (finished requests), so the dashboard
delta-samples between polls:

- **PP (engine)** = Δ`prompt_tokens_total` / Δ`prefill_seconds_total` — engine-time
  prefill throughput of requests that *finished* in the window. Shows `—` when no
  request finished (honest gap, not a fake 0).
- **TG live (wall)** = Δ`completion_tokens_total` / Δwall-clock — the health
  completion counter **includes in-flight replies**, so this is the true live
  decode rate you feel in the terminal.
- **TG (engine)** = Δ`generation_tokens_total` / Δ`decode_seconds_total`.
- **Live cache** = (`pool_tokens` − `pool_free_tokens`) / `pool_tokens` — shared
  cache-pool occupancy — plus per-stream `kv_cache_usage_ratio` and `kept_prompts`.
- **MTP acceptance** = `accepted_total` / `drafted_total` — TensorFold counts
  drafted *tokens* (not rounds), so no vLLM-style ×k multiplier.
- **Averages** = `request_latency_seconds` / `time_to_first_token_seconds`
  histogram sum ÷ count.

## Notes

- Works over plain HTTP to any reachable engine endpoint (LAN or SSH-tunneled).
- Sparklines are session-local (last 120 s while the TUI runs), `·` marks gaps.
- Handles engine down/restart gracefully (auto re-detects backend, waits).
- Tested against: vLLM (OpenAI-compatible serving), TensorFold on 2× DGX Spark
  (GB10) serving GLM-5.3-Flash NVFP4 with native MTP3.

## License

MIT
