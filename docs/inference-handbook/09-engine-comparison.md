# 9. Gated text-engine implementation roadmap

[Previous](08-rtx-5090.md) · [Index](README.md) · [Next](10-qwen-transfer.md)

## Why this matters

Each milestone produces a runnable artifact and prevents performance work from
hiding a semantic error. Do not begin a later gate while an earlier one has
unexplained drift.

## Milestones and acceptance gates

| # | Deliverable | Input -> output | Gate |
|---:|---|---|---|
| 1 | pinned fixture | prompt/messages -> IDs, positions, logits | hashes and expected logits recorded |
| 2 | inventory/converter | checkpoint shards -> validated manifest/artifact | every tensor accounted; sample round trips |
| 3 | scalar text forward | IDs -> taps, state, logits | matches Transformers and llama.cpp |
| 4 | dense CUDA primitives | shape/dtype/tails -> projections/norm/FFN | scalar differential suite passes |
| 5 | one-token GDN | row + old state -> output + new state | warm-up and FP32 recurrence pass |
| 6 | hybrid scheduler | row -> 48 GDN + 16 attention layers | layers 3/4/63 and logits pass |
| 7 | chunked prefill | arbitrary chunks -> logits + final state | decode-equivalent final state |
| 8 | quantized path | calibrated artifact -> logits/text | high-precision comparison and quality suite pass |
| 9 | 32 GiB proof | artifact + 32K session -> allocation log | graphs included; reserve remains; run completes |
| 10 | measured optimization | baseline -> fused/graphed/scheduled path | profiler attribution, correctness, raw A/B data |
| 11 | persistence | arbitrary prefix -> saved/restored/forked sessions | all hybrid state continues identically |
| 12 | extensions | text core -> MTP, then vision boundary | Chapter 10 gates pass separately |

Milestone 1 includes one-token and multi-token prompts and the official chat
template. Milestone 6 tests the first attention layer and the return to GDN.
Milestone 7 includes chunk sizes around 4 and 64. Milestone 8 compares greedy
walks, held-out perplexity/NLL, long recurrence, and task fixtures; a short
perplexity win cannot excuse long-state degradation. Milestone 9 repeats at
larger contexts until the admitted maximum is established.

## Stable interfaces

`ModelSpec` and `ModelWeights` are created once; `SequenceInput` carries
embeddings and positions; `SessionState` contains all mutable prefix state;
`BackendOps` supplies reference or CUDA operations. Server and CLI code see
tokens/logits/session operations, never tensor names. This conceptual boundary
does not require changing DwarfStar's runtime API.

## DwarfStar lessons

Reuse lifetimes, allocation accounting, differential tests, MMV/MMQ, graph
discipline, benchmarking, and correctness gates. Adapt validation, binding,
quantization, serialization, hybrid scheduling, batching, and future MTP.
Reject compressed-attention schemas, indexers, MoE routing/streaming, mHC, and
DSpark semantics.

## Failure modes and exercise

Do not optimize before a high-precision oracle, accept only final token text,
introduce quantization and CUDA simultaneously, or call a nominal artifact size
a fit proof. Exercise: for each gate write the exact fixture, observable output,
tolerance, failure artifact, and rollback criterion. Expected: another engineer
can run every gate without an oral explanation.

