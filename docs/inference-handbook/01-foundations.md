# 1. Foundations: work, bytes, and time

[Index](README.md) · [Next](02-execution-path.md)

An autoregressive model maps a token prefix to one logit per vocabulary item.
Sampling chooses the next token; appending it makes the next invocation depend
on all prior state. “Inference” therefore includes tokenizer and prompt policy,
model evaluation, persistent state, sampling, and serving—not only GEMMs.

## Vocabulary

**Prefill** evaluates many prompt tokens, usually with matrix-matrix operations.
Its useful metric is prompt tokens/s and its user-visible result is time to first
token (TTFT). **Decode** evaluates one new token per session; matrix-vector work,
cache reads, launch overhead, and weight bandwidth dominate. **Continuous or
microbatch serving** combines ready decode steps across sessions: aggregate
throughput can rise while an individual request waits longer.

For hidden width `d`, batch/token rows `T`, and a linear weight `[d, o]`:

```text
X[T,d] @ W[d,o] -> Y[T,o]
FLOPs ~= 2*T*d*o
arithmetic intensity ~= FLOPs / bytes moved
```

At `T=1`, a weight used once has little reuse. At larger `T`, the same weight
serves many rows. That is the decode matvec versus prefill matmul distinction.

## A practical roofline

Let peak compute be `P` FLOP/s, sustainable memory bandwidth `B` byte/s, and
kernel arithmetic intensity `I` FLOP/byte. An optimistic bound is
`min(P, B*I)`. Add a launch/coordination floor `L`: a more useful lower bound on
latency is `max(F/P, bytes/B, L)`. This is a bound, not a prediction: irregular
expert access, imperfect occupancy, dependencies, quant unpacking, and caches
all reduce attainment.

DwarfStar Flash’s concrete symbols come from [`ds4_shape` and
`DS4_SHAPE_FLASH`](https://github.com/antirez/ds4/blob/c1d4597a80e300b803dc642519718f2c999589da/ds4.c#L490):

| Quantity | Symbol | Flash value |
|---|---:|---:|
| transformer layers | `DS4_N_LAYER` | 43 |
| residual width | `DS4_N_EMBD` | 4,096 |
| vocabulary | `DS4_N_VOCAB` | 129,280 |
| attention heads | `DS4_N_HEAD` | 64 |
| compressed head state | `DS4_N_HEAD_DIM` | 512 |
| query low-rank width | `DS4_N_LORA_Q` | 1,024 |
| routed experts / selected | `DS4_N_EXPERT` / `_USED` | 256 / 6 |
| expert inner width | `DS4_N_FF_EXP` | 2,048 |
| hyper-connections | `DS4_N_HC` | 4 |

Do not infer a standard `[heads, head_dim]` KV cache from those names. Chapter 3
shows DwarfStar’s raw/compressed storage and sparse indexer.

## Latency is a vector

Report TTFT, inter-token latency (ITL), prefill tokens/s, per-session decode
tokens/s, aggregate tokens/s, peak resident memory, context length, and quality
parity separately. A design can improve aggregate throughput and worsen TTFT;
compress memory and harm quality; or win at batch one and lose at batch 32.

## Check and experiment

1. A weight matrix is 1 GiB and bandwidth is 1 TiB/s. The best-case batch-one
   read floor is about 1 ms. At eight rows, bytes per generated token can fall
   nearly eightfold if the weight stays reusable.
2. Write a CUDA kernel-launch microbenchmark and an HBM copy benchmark. Expected:
   tiny kernels hit a launch floor; large copies approach a stable fraction of
   advertised bandwidth.
3. Explain why “TOPS” alone cannot predict decode. Expected: weight traffic,
   precision, supported instructions, and batch reuse are missing.

