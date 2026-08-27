# 4. Scalar reference and tensor oracles

[Previous](03-deepseek-v4.md) · [Index](README.md) · [Next](05-gpu-implementation.md)

## Why this matters

Optimization needs a trusted intermediate-tensor oracle. End-token equality is
too late to locate a bad layout, recurrence update, or rounding decision.

## Design and dataflow

An **oracle** is a slower implementation trusted enough to tell us what the
optimized implementation should produce. “Scalar” means that the reference
uses straightforward loops and explicit indexing rather than relying on a
specialized matrix kernel. It is not intended for serving; it is intended to
make one wrong transpose or update order obvious.

Implement `BackendOps` first as simple, typed host operations. It may be slow,
but must expose named taps. A tap is a copy of an intermediate tensor at a
named boundary, saved before the next operation overwrites its buffer:

```text
SequenceInput -> embed -> for layer 0..63:
  input norm -> mixer projections -> mixer state/output -> residual
  post norm -> gate/up -> product -> down -> residual
-> final norm -> lm_head -> F32 logits
```

The arrows describe data ownership as well as computation: embeddings and
activations are temporary, weights are read-only, and session state is the only
mutable input that survives a call. Keep `ModelSpec` independent of storage. It validates dimensions, the 64-entry
layer-kind schedule, exact tensor inventory, dtype, orientation, and quant
policy. `ModelWeights` owns immutable tensors; `SessionState` alone mutates.
Use FP32 accumulation for norms, softmax, GDN recurrence, and initial comparison.

## Oracle workflow

1. Pin checkpoint revision, Transformers revision, llama.cpp revision,
   tokenizer files, template, prompt bytes, and reference dtype.
2. Add hooks to save shapes, dtypes, and little-endian raw tensors at embedding;
   GDN post-convolution, recurrent output, and projection; attention Q/K after
   norm/RoPE and output; FFN output; final norm; logits.
3. Feed the same IDs and positions to the scalar engine. Compare max absolute,
   max relative, RMS error, cosine similarity, and first failing index.
4. Compare Transformers first, then llama.cpp as an independent oracle. A
   disagreement between authorities is investigated, not averaged away.

Use one-token fixtures at positions 0–5 for convolution warm-up; prompts of
length 1, 4, 5, 63, 64, and 65; and taps around layers 3/4 and 63. Check chunked
prefill against token-by-token decode by comparing final logits and every state
byte or tolerance-defined state value.

For a numerical comparison, **absolute error** catches a fixed-size offset,
**relative error** catches scale-dependent drift, RMS error summarizes the whole
tensor, and cosine similarity catches a changed direction. None is sufficient
alone: a nearly zero reference value makes relative error unstable, while a
cosine score can hide one bad channel. Set tolerances per boundary and dtype
before looking at the candidate result.

## DwarfStar transfer boundary

DwarfStar's CPU-only diagnostic path, tensor validation before binding, quant
dot tests, and CPU/GPU boundary comparisons transfer unchanged. Its tensor
roles, compression fixtures, and permissive fallback assumptions do not.

## Concrete work and acceptance gate

The scalar milestone passes only when one-token and multi-token prompts match
both semantic authorities at every available boundary, greedy continuations are
stable, and a save/restore at each tested position produces the uninterrupted
result. Store fixtures with source hashes and generation commands.

## Common failures

- Comparing logits only, or loosening tolerance until a structural bug passes.
- Transposing checkpoint matrices twice.
- Applying the convolution to unpacked Q/K/V independently.
- Updating session state while producing diagnostic retries.
- Allowing a GPU fallback into the reference path.

## Exercise and expected result

Implement one GDN head for five tokens with a four-entry convolution ring.
Compare streaming and whole-sequence execution. Expected: every output and the
final recurrent/ring state agree; corrupting the oldest warm-up row first causes
a difference at the predictable convolution boundary.
