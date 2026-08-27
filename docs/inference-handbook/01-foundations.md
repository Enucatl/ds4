# 1. Inference fundamentals

[Index](README.md) · [Next](02-execution-path.md)

## Why this matters

Decode and prefill are different workloads, and Qwen has two different state
growth laws. Confusing either pair breaks performance or memory planning.

## Concepts and worked dataflow

An inference engine repeatedly applies a trained function to a sequence of
tokens. A token is first converted to a vector of numbers (an **embedding**).
The model transforms those vectors through its layers and produces **logits**:
one unnormalized score for every possible next token. A sampler turns the
scores into an integer token ID, which is appended to the sequence and becomes
the input to the next step. The engine's job is to do this while retaining just
enough intermediate state to avoid recomputing the whole sequence.

**Prefill** is the first pass over the prompt. If the prompt has `T` tokens,
the engine can process all `T` rows together because none of their outputs are
being generated yet (the causal mask still prevents a row from looking into the
future). Prefill is therefore usually a matrix-matrix operation. Its user-facing
metric is **time to first token (TTFT)**, and its throughput is prompt tokens per
second.

**Decode** is the repeating generation step. Usually only one new token is
processed for one session, so `T=1`; the engine reads the session's saved state,
computes one new row, and updates that state. This is usually matrix-vector
work. Its user-facing metric is **inter-token latency (ITL)**: the time between
two generated tokens. A server may combine several sessions into one decode
batch, making the effective `T` larger, but each session still has its own
state and waiting time.

For a linear layer, let `d` be the input vector width, `o` the output width,
and `T` the number of rows processed together. `X[T,d]` is the activation matrix
and `W[d,o]` is the learned weight matrix. The result `Y[T,o]` is computed by
dotting each input row with each output column:

```text
X[T,d] @ W[d,o] -> Y[T,o]
```

```text
FLOPs = 2*T*d*o
BF16 weight bytes = 2*d*o
latency >= max(FLOPs/peak_FLOPs, bytes/bandwidth, launch_floor)
```

The FLOP estimate counts one multiply and one add for each term in each dot
product, hence the factor of two. It is a work estimate, not a stopwatch: GPU
instructions for loads, scaling, activation functions, and quantization are
additional work.

```text
FLOPs = 2*T*d*o
```

For BF16, each weight occupies two bytes. The estimate below counts reading the
weight matrix once. A real kernel may read scales, write outputs, reread tiles,
or keep some data in cache:

```text
linear FLOPs = 3 * 2*T*5120*17408 = 534,773,760*T
BF16 matrix storage = 267,386,880*2 bytes = 510 MiB/layer
```

Qwen's feed-forward network (FFN) is a small subprogram inside every layer. It
uses two input-to-hidden projections, one called `gate` and one called `up`.
The gate projection is passed through SiLU, then multiplied element-by-element
with the up projection. The product is projected back to the residual width.
There are three matrices, not one:

```text
gate = gate_proj(x)       # [T,5120] -> [T,17408]
up   = up_proj(x)         # [T,5120] -> [T,17408]
h    = SiLU(gate) * up    # elementwise product, still [T,17408]
y    = down_proj(h)       # [T,17408] -> [T,5120]
```

For one decode row, the three matrix multiplications perform about 535 million
multiply/add operations and the BF16 matrices occupy about 510 MiB. At `T=1`,
the same 510 MiB may be read for one output row, so bandwidth and launch costs
matter more than the arithmetic peak. At `T=128`, those weights serve 128 rows;
the arithmetic work grows with `T`, but the weight bytes need not grow by 128 if
the kernel tiles and reuses them. **Quantization** stores an approximate weight
using fewer bits plus scales/metadata. It lowers traffic, but the engine must
unpack or interpret those blocks and can lose accuracy.

The practical lower bound combines three possible limits. `peak_FLOPs` is the
hardware's useful compute rate, `bandwidth` is the sustained rate for the memory
being read, and `launch_floor` is the fixed cost of dispatching and coordinating
a kernel. The slowest of these floors wins:

Full attention needs a different kind of memory. At each full-attention layer,
the model creates a **key** vector (K) and a **value** vector (V) for the token.
Future queries compare themselves with all earlier keys, then use the matching
values. K and V are therefore appended to a **KV cache** and read again on every
later decode step. Qwen has 16 full-attention layers, four KV heads per layer,
and 256 values per head. With BF16 K and V:

```text
16 * 2 * 4 * 256 * 2 BF16 bytes = 65,536 bytes/token
32,768 tokens = 2 GiB; 262,144 tokens = 16 GiB
```

This is why full-attention memory grows linearly with context. **Gated
DeltaNet (GDN)** layers use a recurrent summary instead: each layer updates a
fixed `[48,128,128]` matrix and a short convolution ring as each token arrives.
They do not append one K/V row per token. That fixed state is not “free”—it must
be allocated, copied for multiple sessions, and serialized—but its size does
not grow with a 256K context.

### How to read a shape

`[T,5120]` means `T` rows, each with 5,120 scalar values. `[48,128,128]` means
48 independent value heads, each holding a 128-by-128 matrix. The order matters:
`[T,heads,dim]` is not interchangeable with `[heads,T,dim]` without a reshape or
transpose. When a chapter writes “conceptually,” it describes logical indexing;
an implementation may choose a tiled physical layout, but it must preserve the
same mapping.

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
