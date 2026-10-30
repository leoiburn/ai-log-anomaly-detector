import yaml
from pathlib import Path
from benchmarks.agent_throughput_benchmark import (
    simulate_cluster_optimizations,
    format_agent_request,
    SAMPLE_AGENT_SYSTEM_PROMPT,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "inference_cluster_config.yaml"
K8S_PATH = Path(__file__).resolve().parents[1] / "deploy" / "k8s" / "vllm-high-throughput.yaml"


def test_inference_cluster_config_structure():
    assert CONFIG_PATH.exists()
    content = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    assert "profiles" in content
    profiles = content["profiles"]
    assert "enterprise_nvlink" in profiles
    assert "multi_gpu_pcie" in profiles
    assert "single_gpu_workstation" in profiles

    # Verify all 5 optimizations are present in the profiles
    nvlink_opts = profiles["enterprise_nvlink"]["optimizations"]
    assert nvlink_opts["continuous_batching"]["enable_chunked_prefill"] is True
    assert nvlink_opts["prefix_caching"]["enable_prefix_caching"] is True
    assert nvlink_opts["quantization"]["format"] == "fp8"
    assert nvlink_opts["topology"]["tensor_parallel_size"] == 4
    assert nvlink_opts["attention"]["backend"] == "FLASHINFER"


def test_vllm_k8s_manifest_flags():
    assert K8S_PATH.exists()
    manifest_text = K8S_PATH.read_text(encoding="utf-8")

    # Verify the 5 flags are explicitly injected into the vLLM container args
    assert "--enable-chunked-prefill" in manifest_text
    assert "--enable-prefix-caching" in manifest_text
    assert "--quantization" in manifest_text
    assert "--tensor-parallel-size" in manifest_text
    assert "--attention-backend" in manifest_text
    assert "FLASHINFER" in manifest_text
    # Shared memory IPC volume mount for Tensor Parallelism
    assert "mountPath: /dev/shm" in manifest_text


def test_analytical_simulation_quantization_and_apc_speedup():
    base = simulate_cluster_optimizations(
        model_size_gb=64.0,
        vram_bandwidth_gb_s=1000.0,
        quantization="fp16",
        enable_apc=False,
        use_flashinfer=False,
        concurrent_requests=1,
    )
    awq_opt = simulate_cluster_optimizations(
        model_size_gb=64.0,
        vram_bandwidth_gb_s=1000.0,
        quantization="awq",
        enable_apc=True,
        use_flashinfer=True,
        concurrent_requests=8,
    )

    # AWQ + FlashInfer should yield at least 4x decode speed over unquantized FP16
    assert awq_opt["single_stream_decode_tok_s"] > base["single_stream_decode_tok_s"] * 4.0
    # Continuous batching with 8 concurrent agents should scale aggregate throughput
    assert awq_opt["aggregate_cluster_tok_s"] > awq_opt["single_stream_decode_tok_s"]
    # Prefix caching should reduce TTFT significantly
    assert awq_opt["ttft_ms_cached"] < base["ttft_ms_cold"]


def test_agent_request_formatting():
    req = format_agent_request(SAMPLE_AGENT_SYSTEM_PROMPT, "Check node temperatures.")
    assert "messages" in req
    assert len(req["messages"]) == 2
    assert req["messages"][0]["role"] == "system"
    assert req["messages"][1]["role"] == "user"
    assert req["temperature"] == 0.0
