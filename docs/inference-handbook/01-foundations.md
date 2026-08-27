# 1. Inference fundamentals

[Index](README.md) · [Next](02-execution-path.md)

## Why this matters

Decode and prefill are different workloads, and Qwen has two different state
growth laws. Confusing either pair breaks performance or memory planning.

## Concepts and worked dataflow

**Prefill** evaluates prompt rows, normally with matrix-matrix work; measure
prompt tokens/s and TTFT. **Decode** advances one committed position per session,
normally with matrix-vector work; measure ITL. For `X[T,d] @ W[d,o]`:

```text
FLOPs = 2*T*d*o
BF16 weight bytes = 2*d*o
latency >= max(FLOPs/peak_FLOPs, bytes/bandwidth, launch_floor)
```

Qwen's FFN maps `5120 -> 17408` twice, multiplies after SiLU, then maps back:

```text
linear FLOPs = 3 * 2*T*5120*17408 = 534,773,760*T
BF16 matrix storage = 267,386,880*2 bytes = 510 MiB/layer
```

At `T=1`, weight traffic dominates; larger `T` reuses weights. Quantization
reduces traffic but adds metadata and unpacking.

Each full-attention token stores K and V for 4 heads of 256 values in 16 layers:

```text
16 * 2 * 4 * 256 * 2 BF16 bytes = 65,536 bytes/token
32,768 tokens = 2 GiB; 262,144 tokens = 16 GiB
```

GDN instead retains fixed recurrent matrices and short convolution history.

## DwarfStar transfer boundary

DwarfStar's engine/session lifetimes, MMV/MMQ split, allocation accounting, and
TTFT/ITL reporting transfer unchanged. Its compressed-attention rows, sparse
indexer, routed experts, expert streaming, and mHC streams must be discarded.

## Concrete work

Record prompt and output lengths, batch, context frontier, quant policy, GPU
power/clocks, TTFT, p50/p95 ITL, prompt and aggregate tokens/s, peak allocated
and reserved VRAM, and correctness fixture. Never report only “tokens/s.”

## Common failures

- Estimating fit from file size and omitting repacks, KV, graphs, and reserve.
- Predicting batch-one decode from advertised TOPS.
- Comparing different templates, samplers, quality, or prefix reuse.
- Applying a KV growth law to recurrent state.

## Exercise and expected result

For 1 GiB of weights and 1 TiB/s sustainable bandwidth, derive a 1 ms read
floor. Run copy and launch microbenchmarks. Expected: large copies approach a
bandwidth plateau and tiny operations approach a launch floor; neither alone
predicts inference.

