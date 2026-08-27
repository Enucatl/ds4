# 4. Numerics and model preparation

[Previous](03-deepseek-v4.md) · [Index](README.md) · [Next](05-gpu-implementation.md)

GGUF records metadata, named tensor dimensions/types, and aligned payloads.
`model_open` bounds-checks the file; `weights_bind` maps exact names into layer
roles. Shape validation prevents a plausible-looking but incompatible file from
reaching a kernel.

## Formats in this checkout

The canonical C definitions and static sizes are in [`GGUF Quant Block
Formats`](https://github.com/antirez/ds4/blob/c1d4597a80e300b803dc642519718f2c999589da/ds4.c#L730).

| Format | Values/block | Bytes/block | Effective bits/value | Role |
|---|---:|---:|---:|---|
| F32 | 1 | 4 | 32 | norms, logits, sensitive state |
| F16 | 1 | 2 | 16 | caches/dense paths on some backends |
| Q8_K | 256 | 292 | 9.125 | activation/dot representation |
| Q2_K | 256 | 84 | 2.625 | routed down experts |
| Q4_K | 256 | 144 | 4.5 | higher-quality routed experts |
| IQ2_XXS | 256 | 66 | 2.0625 | routed gate/up experts |
| MXFP4 | 32 | 17 | 4.25 | preserved native routed weights |

“2-bit model” is shorthand, not a uniform representation. DwarfStar’s Q2 mix
keeps routing, projections, shared experts, norms, embeddings, and output at
higher precision while aggressively quantizing the dominant routed tensors.
That asymmetric allocation is a transferable idea: spend error where the model
tolerates it, not uniformly.

MXFP4 and NVFP4 are not interchangeable names. The former’s block uses an E8M0
scale and packed FP4 values. NVFP4 support in CUDA kernels has different block
and scaling assumptions. Blackwell FP4 Tensor Core use also requires compatible
activation quantization and matrix layout; merely storing 4-bit weights does not
guarantee native FP4 MMA.

## Calibration and quality

An importance matrix (imatrix) measures activation-weight sensitivity on a
representative corpus, informing quantization error allocation. The offline path
is in `gguf-tools/deepseek4-quantize.c`; the quality harness collects official
continuations and compares local outputs. Calibration data leakage or a tiny
domain can create misleading wins.

Use this acceptance ladder:

1. Tensor-level dequant/dot tests on known blocks.
2. CPU-versus-backend differential logits at layer boundaries.
3. Greedy continuation equality or explained floating-point divergence.
4. A fixed multi-domain official-output score set.
5. Long-context and tool-call behavior.

## Memory worksheet

For each tensor: `elements * effective_bits / 8`, including block metadata and
alignment. Then add duplicated device mappings, derived/repacked weights, KV,
compressor state, logits, temporary activations, CUDA libraries/graphs, server
sessions, and a safety reserve. File size is not peak VRAM.

**Estimated DeepSeek example.** The published Flash Q2 download target is
described as suitable for 96/128 GB system-memory machines, so it cannot be a
resident single-5090 workload. SSD streaming may make execution possible but
changes the bottleneck to PCIe/storage and is not asserted to be practical here.

## Check and experiment

Run `make q4k-dot-test mxfp4-dot-test`. Expected: quantized dot products match
their scalar references within the test tolerance. Quantize a held-out tensor
with and without imatrix weighting; expected: byte size stays similar while
weighted error distribution changes. Model-quality improvement remains an
experiment, not a deduction.

