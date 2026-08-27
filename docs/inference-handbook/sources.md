# Source and evidence ledger

[Index](README.md)

Baseline is repository commit [`c1d4597`](https://github.com/antirez/ds4/tree/c1d4597a80e300b803dc642519718f2c999589da), inspected 2026-08-27. Internal
claims below are **source-verified**, not runtime measurements.

## Internal anchors

| Claim area | Stable symbol/file | Evidence |
|---|---|---|
| supported shapes/constants | `ds4_shape`, `DS4_SHAPE_FLASH`, `DS4_SHAPE_PRO` in `ds4.c` | source-verified |
| compression schedule | `ds4_expected_layer_compress_ratio`, `validate_compress_ratio_metadata` | source-verified |
| quant layouts/sizes | `block_q2_K`, `block_q4_K`, `block_q8_K`, `block_iq2_xxs`, `block_mxfp4` | source-verified + static assertions |
| GGUF open/validation/binding | `model_open`, `config_validate_model`, `weights_bind` | source-verified |
| prompt rendering/tokenization | `ds4_chat_*`, `ds4_tokenize_rendered_chat` | source-verified |
| engine/session lifecycle | `ds4_engine_open_internal`, `ds4_session_create/free` | source-verified |
| CPU reference | `forward_token_raw_swa_cpu`, `prefill_layer_major_cpu` | source-verified |
| graph prefill/decode | `metal_graph_prefill_chunked`, `metal_graph_eval_token_raw_swa`, `metal_graph_encode_decode_layer_phase` | source-verified |
| sampling | `ds4_session_sample`, `sample_build_probabilities` | source-verified |
| persisted state | `ds4_session_save_payload`, `ds4_session_load_payload` | source-verified |
| CUDA graph capture | `ds4_gpu_decode_graph_begin/end`, `attention_decode_score_split_graph_launch` | source-verified |
| quant kernel dispatch | `cuda_use_mmq`, `cuda_use_mxfp4_mmq`, `ds4_mmq_mxfp4_moe*` | source-verified |
| streaming | `ds4_ssd.c`, `graph_stream_expert_table_make` | source-verified |
| benchmark policy | `CONTRIBUTING.md`, `QA_BEFORE_RELEASES.md`, `speed-bench/README.md` | repository protocol |

Measured repository numbers in README or QA tables apply only to their stated
hardware/model/configuration. This handbook does not promote them to 5090 or
cross-engine claims.

## External primary sources

- Qwen, [Qwen3.8-27B model card pinned at
  `1d4bf0f`](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/README.md)
  and [config](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/config.json): architecture,
  context, vision and MTP. **External.**
- NVIDIA, [RTX 5090 specification](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/)
  and [launch architecture article](https://www.nvidia.com/en-us/geforce/news/rtx-50-series-graphics-cards-gpu-laptop-announcements/): memory, bus,
  bandwidth, cores. **External.**
- NVIDIA, [Blackwell tuning guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/)
  and [CUDA release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/):
  compute capability and toolkit behavior. **External; version-sensitive.**
- Kwon et al., [PagedAttention/vLLM](https://arxiv.org/abs/2309.06180), and
  Zheng et al., [SGLang](https://arxiv.org/abs/2312.07104): serving design.
  **External.** Engine feature availability must be rechecked at benchmark time.
- [llama.cpp](https://github.com/ggml-org/llama.cpp),
  [Ollama](https://github.com/ollama/ollama), [vLLM](https://docs.vllm.ai/), and
  [SGLang](https://docs.sglang.ai/): current implementation documentation.
  **External; moving targets.**

## Estimates and proposed experiments

- Quant block bits/value: **Estimated** directly from pinned `sizeof` assertions.
- KV and weight arithmetic in Chapters 3 and 8: **Estimated**, excludes listed
  runtime overhead unless explicitly included.
- Any RTX 5090 DwarfStar/Qwen speed, fit, FP4 utilization, or quality outcome:
  **Proposed** until the Chapter 8 protocol produces raw results.
- `signalnine/q27`: external comparative case only; no claim in this handbook
  depends on it.
