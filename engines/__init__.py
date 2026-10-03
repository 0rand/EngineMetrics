"""Engine registry + backend auto-detection.

Each engine module exposes:
    tick(stdscr, args, state)  — sample + draw one frame; mutates state
State dict (owned by the entrypoint): prev, history, rate_hist, start_time, model_name.
"""

from . import mlx_serve, omlx, tensorfold, vllm

ENGINES = {
    "vllm": vllm,
    "tensorfold": tensorfold,
    "mlx-serve": mlx_serve,
    "omlx": omlx,
}


def detect_backend(host, port):
    """Return an ENGINES key or None."""
    from .common import fetch_json, fetch_metrics

    h = fetch_json(host, port, "/health") or {}
    # TensorFold declares itself
    if isinstance(h, dict) and h.get("backend"):
        return h["backend"]
    # oMLX /health: {"status": "healthy", "engine_pool": {...}, ...}
    if isinstance(h, dict) and "engine_pool" in h:
        return "omlx"
    m, _ = fetch_metrics(host, port)
    # mlx-serve also emits vllm:* compat counters, so check its native prefix FIRST
    if any(k.startswith("mlx_serve:") for k in m):
        return "mlx-serve"
    if any(k.startswith("vllm:") for k in m):
        return "vllm"
    if any(k.startswith("tensorfold") for k in m):
        return "tensorfold"
    # oMLX fallback probe (no /metrics): /api/status has models_discovered
    s = fetch_json(host, port, "/api/status")
    if isinstance(s, dict) and "models_discovered" in s:
        return "omlx"
    return None
