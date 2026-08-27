# 9. Compare engines by objective

[Previous](08-rtx-5090.md) · [Index](README.md) · [Next](10-qwen-transfer.md)

This matrix describes design centers, not benchmark winners. Support changes
quickly; verify the exact model, quant and GPU commit before choosing.

| Engine/family | Design center | Batching/cache policy | Kernel/quant posture | Best comparison question |
|---|---|---|---|---|
| DwarfStar | narrow native DeepSeek/GLM deployments | sessions, prefix/KV checkpoints, microbatch; compressed model-specific state | C/Metal/CUDA/ROCm, specialized quant and streaming paths | Does specialization win for this exact model, hardware, context and concurrency? |
| llama.cpp / Ollama | broad local portability; Ollama packages/serves models | slot batching and prompt cache; policy varies by frontend | GGUF and many CPU/GPU backends/quants | What is the simplest portable local baseline with the same GGUF and offload? |
| vLLM | high-throughput accelerator serving | continuous batching, paged KV, prefix caching | framework/library kernels and broad quant ecosystem | What throughput/latency frontier appears under a realistic arrival process? |
| SGLang | structured generation and serving runtime | continuous batching, radix/prefix-oriented reuse | optimized accelerator backends | Do prefix-heavy/agent workloads benefit at equal memory and quality? |
| Unsloth-related runtimes/artifacts | efficient fine-tuning, conversion, and packaged inference paths | depends on selected runtime | aggressive model-specific quant artifacts | Is the artifact/runtime pair validated for this architecture and modality? |

Primary design references: [llama.cpp](https://github.com/ggml-org/llama.cpp),
[Ollama](https://github.com/ollama/ollama), [vLLM PagedAttention
paper](https://arxiv.org/abs/2309.06180), [vLLM docs](https://docs.vllm.ai/),
[SGLang paper](https://arxiv.org/abs/2312.07104), and
[SGLang docs](https://docs.sglang.ai/).

## Fair protocol

Pin engine/model/tokenizer/template/quant commits. Match context, max output,
sampling and stop rules. Include model load and steady-state separately. Fix GPU
power/clocks where allowed. Use identical prompts and arrival traces; report
quality parity, failed/OOM requests, TTFT/ITL distributions, aggregate tokens/s,
and peak host/device memory. For local single-user tests use concurrency one; for
serving, sweep concurrency. Do not compare DwarfStar batch-one decode to vLLM
aggregate throughput or compare different quant quality as “speed.”

## Check

Choose an engine for (a) a laptop offline chat, (b) eight-GPU multi-tenant API,
and (c) hacking one DeepSeek layout. Expected: portability, scheduler throughput,
and specialization lead to different answers; no row is universally best.

