# 7. Correctness-first engineering

[Previous](06-system-optimization.md) · [Index](README.md) · [Next](08-rtx-5090.md)

An optimization is complete only when its semantic boundary and performance
effect are both measured.

## Oracles and gates

Use the scalar/CPU path for local numerical differentials and official model
continuations for end-to-end behavior. CPU parity alone can preserve a shared
bug; text equality alone misses compensating drift. The repository’s test and
release protocols are [CONTRIBUTING.md](../../CONTRIBUTING.md),
[QA_BEFORE_RELEASES.md](../../QA_BEFORE_RELEASES.md), and
[quality-testing](../../gguf-tools/quality-testing/README.md).

For each change:

1. State the exact hot path, supported shapes/formats, and fallback.
2. Add tensor tests including tails, alignment, zero scales, and extreme values.
3. Compare layer/logit outputs with the reference at short and compression
   boundary contexts.
4. Run fixed greedy continuations and scored official continuations.
5. Benchmark balanced A/B order with identical model, prompt, context, clocks,
   cache state, sampling, and concurrency.
6. Gate regression: repository release policy treats repeatable slowdown over
   10% in prefill, decode, or aggregate batch as a blocker.

## Benchmark record

```text
commit/model checksum/quant/backend/toolkit/driver/GPU/power+clocks
prompt hash; input/output tokens; context frontier; batch/sessions
warmups and repetitions; cold/warm storage; sampling seed/settings
TTFT p50/p95; ITL p50/p95; prefill tok/s; decode tok/s
aggregate tok/s; peak device+host bytes; quality parity result
```

Measure with events around GPU work and wall-clock around the request. Synchronize
only at intended boundaries. Median hides stalls; publish distributions and raw
CSV. A faster greedy result is invalid if it uses a different continuation.

## History as evidence

Read comments around CUDA decode graph capture, MMQ gating, exact split-score
graphs, and DSpark state reuse. They record recurring lessons: graph capture can
regress kernels when addresses/control vary; alternative reductions change
floating-point grouping; fast speculative paths need an exact sampling mode;
specialized kernels need explicit fallback. “Rejected” remains provisional for
a new GPU, batch size, or layout—rerun the experiment rather than cargo-culting.

## Reproducible suite

- TTFT: fixed 128, 2K, and 8K prompts; time request arrival to first emitted token.
- Decode: 256 generated tokens at context frontiers; report ITL distribution.
- Prefill: `ds4-bench` with fixed prompt file and chunk policy.
- Serving: 1/2/4/8 sessions, fixed arrival schedule, aggregate plus p95 latency.
- Memory: driver peak plus engine accounting at each context/session count.
- Quality: identical tokenizer/template and deterministic output fixture.
- Long context: cross raw/compression boundaries and checkpoint save/restore.

## Check

Given a 12% faster run with different output tokens, classify it. Expected: not
a valid speed comparison until semantics/quality are reconciled. Given 5% median
gain and 30% worse p95, decide from the serving goal, not the mean.

