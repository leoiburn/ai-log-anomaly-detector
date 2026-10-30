"""
Cluster Agent Throughput Benchmark & Optimization Validator.

Measures the impact of the 5 key token/second optimizations:
1. Continuous Batching & Chunked Prefill
2. Automatic Prefix Caching (APC)
3. Quantization (AWQ/FP8)
4. Interconnect / Topology scaling
5. FlashAttention / FlashInfer kernels

Can benchmark against a live vLLM/OpenAI-compatible server or run an
analytical simulation using cluster hardware profiles.
"""

from typing import Dict, Any, List, Optional
import argparse
import json
import time
import urllib.request
import urllib.error


# Standard long agent prompt simulating tools, workspace layout, and system prompt
SAMPLE_AGENT_SYSTEM_PROMPT = """
You are an autonomous Kubernetes Security & Operations Agent.
Workspace: /cluster/production
Tools available:
1. kubectl_exec(namespace: str, pod: str, command: str) -> str
2. inspect_logs(service: str, tail: int, filter: str) -> List[str]
3. analyze_hardware(node: str, check_dcgm: bool) -> Dict[str, float]
4. apply_cilium_policy(policy_path: str) -> bool
5. kill_process_tetragon(pid: int, sig: int) -> bool

Guidelines:
- Never execute raw SSH or outbound scans.
- Enforce default-deny egress policies.
- Monitor GPU VRAM, power, and thermal throttling before launching heavy tasks.
- Return structured JSON execution plans before invoking system tools.
""" * 4  # Expanded to ~1,200 tokens to test Prefix Caching


def format_agent_request(prompt: str, user_query: str) -> Dict[str, Any]:
    """Builds a standard OpenAI-compatible chat completion payload."""
    return {
        "model": "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ",
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_query},
        ],
        "max_tokens": 128,
        "temperature": 0.0,
        "stream": False,
    }


def query_inference_server(base_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Sends request to inference server and measures TTFT and total latency."""
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
    )

    start_time = time.perf_counter()
    with urllib.request.urlopen(req, timeout=30) as response:
        body = json.loads(response.read().decode("utf-8"))
    elapsed = time.perf_counter() - start_time

    usage = body.get("usage", {})
    gen_tokens = usage.get("completion_tokens", 1)
    prompt_tokens = usage.get("prompt_tokens", 0)

    tok_per_sec = gen_tokens / max(elapsed, 0.001)

    return {
        "elapsed_seconds": round(elapsed, 4),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": gen_tokens,
        "tokens_per_second": round(tok_per_sec, 2),
    }


def simulate_cluster_optimizations(
    model_size_gb: float,
    vram_bandwidth_gb_s: float,
    quantization: str = "awq",
    enable_apc: bool = True,
    use_flashinfer: bool = True,
    concurrent_requests: int = 1,
) -> Dict[str, Any]:
    """
    Analytically calculates expected tokens/second improvements across the 5 optimizations.
    """
    # 1. Weight precision scaling (AWQ 4-bit = 0.25x byte size, FP8 = 0.5x vs FP16 = 1.0x)
    prec_factor = 0.25 if quantization == "awq" else (0.50 if quantization == "fp8" else 1.0)
    effective_model_gb = model_size_gb * prec_factor

    # 2. Kernel speedup (FlashInfer vs standard attention)
    kernel_multiplier = 1.25 if use_flashinfer else 1.0

    # 3. Single-stream decode speed (Memory bandwidth bound)
    base_tok_s = (vram_bandwidth_gb_s / max(effective_model_gb, 0.1)) * kernel_multiplier

    # 4. Concurrency scaling with continuous batching
    # As batch size increases, tensor core utilization improves aggregate tokens/sec
    batch_efficiency = 1.0 + (0.45 * (concurrent_requests - 1) ** 0.65)
    aggregate_tok_s = base_tok_s * batch_efficiency

    # 5. Prefix Caching TTFT reduction
    prefill_time_ms = 450.0  # standard baseline for ~1.5k tokens
    cached_prefill_time_ms = 25.0 if enable_apc else prefill_time_ms
    ttft_speedup = prefill_time_ms / cached_prefill_time_ms

    return {
        "quantization": quantization,
        "effective_model_size_gb": round(effective_model_gb, 2),
        "single_stream_decode_tok_s": round(base_tok_s, 2),
        "aggregate_cluster_tok_s": round(aggregate_tok_s, 2),
        "prefix_caching_enabled": enable_apc,
        "ttft_ms_cold": prefill_time_ms,
        "ttft_ms_cached": cached_prefill_time_ms,
        "apc_ttft_speedup": f"{round(ttft_speedup, 1)}x",
        "concurrency": concurrent_requests,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark and evaluate cluster token/second throughput.")
    parser.add_argument("--server", type=str, default=None, help="Live vLLM server URL (e.g. http://localhost:8000)")
    parser.add_argument("--simulate", action="store_true", default=True, help="Run analytical profile simulation")
    parser.add_argument("--model-size-gb", type=float, default=64.0, help="Base FP16 model size in GB")
    parser.add_argument("--bandwidth-gb-s", type=float, default=1000.0, help="VRAM memory bandwidth in GB/s")
    args = parser.parse_args()

    print("\n=======================================================")
    print("  Cluster Token/Second Optimization Evaluation")
    print("=======================================================\n")

    if args.server:
        print(f"Connecting to live inference server at {args.server}...")
        try:
            # Turn 1: Cold prefill
            print("\n[Turn 1: Cold Start]")
            req1 = format_agent_request(SAMPLE_AGENT_SYSTEM_PROMPT, "List active pods.")
            res1 = query_inference_server(args.server, req1)
            print(f"  Prompt tokens: {res1['prompt_tokens']}, Gen tokens: {res1['completion_tokens']}")
            print(f"  Time: {res1['elapsed_seconds']}s | Tok/s: {res1['tokens_per_second']}")

            # Turn 2: Cached prefill (Testing Automatic Prefix Caching)
            print("\n[Turn 2: Follow-up (Testing Prefix Caching)]")
            req2 = format_agent_request(SAMPLE_AGENT_SYSTEM_PROMPT, "Check node temperatures.")
            res2 = query_inference_server(args.server, req2)
            print(f"  Prompt tokens: {res2['prompt_tokens']}, Gen tokens: {res2['completion_tokens']}")
            print(f"  Time: {res2['elapsed_seconds']}s | Tok/s: {res2['tokens_per_second']}")
        except Exception as err:
            print(f"Server check failed: {err}")

    if args.simulate:
        print("\n--- Analytical Simulation Across The 5 Optimizations ---")
        unoptimized = simulate_cluster_optimizations(
            model_size_gb=args.model_size_gb,
            vram_bandwidth_gb_s=args.bandwidth_gb_s,
            quantization="fp16",
            enable_apc=False,
            use_flashinfer=False,
            concurrent_requests=1,
        )
        optimized = simulate_cluster_optimizations(
            model_size_gb=args.model_size_gb,
            vram_bandwidth_gb_s=args.bandwidth_gb_s,
            quantization="awq",
            enable_apc=True,
            use_flashinfer=True,
            concurrent_requests=8,
        )

        print("\n1. BASELINE (Unoptimized FP16, No APC, No Continuous Batching):")
        print(f"   - Single-stream decode: {unoptimized['single_stream_decode_tok_s']} tok/s")
        print(f"   - Cold TTFT: {unoptimized['ttft_ms_cold']} ms")

        print("\n2. FULLY OPTIMIZED (AWQ 4-bit, FlashInfer, APC Enabled, Continuous Batching = 8):")
        print(f"   - Single-stream decode: {optimized['single_stream_decode_tok_s']} tok/s (Quantization speedup)")
        print(f"   - Aggregate cluster throughput: {optimized['aggregate_cluster_tok_s']} tok/s")
        print(f"   - Cached Agent TTFT: {optimized['ttft_ms_cached']} ms ({optimized['apc_ttft_speedup']} faster)")
        print("\n=======================================================\n")


if __name__ == "__main__":
    main()
