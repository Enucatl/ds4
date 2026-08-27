# 7. Hybrid session state

[Previous](06-system-optimization.md) · [Index](README.md) · [Next](08-rtx-5090.md)

## Why this matters

Prefix reuse, batching, checkpointing, and CUDA Graphs all depend on precise
state ownership. Qwen's state is both fixed recurrent data and growing KV.

## Session layout and transitions

```text
SessionState
  gdn[48]: recurrent FP32[48,128,128], conv ring[10240,4], ring frontier
  attention[16]: K and V[positions,4,256], logical length/capacity
  position_frontier: committed text and future mRoPE coordinates
  committed_input_identity: token IDs plus template/tokenizer identity
  optional_mtp: absent in v1
```

`begin_step` obtains writable state, kernels produce logits and next state, and
`commit_step` advances every layer and the input frontier atomically. On error,
none advances. Prefill chunk boundaries are implementation details and may not
change the final state.

Checkpoint headers include magic/version, endianness, model/config/weights,
tokenizer/template and quant hashes, context/capacity, committed positions,
per-section shapes/dtypes/lengths/checksums, and feature flags. Write to a new
payload and publish atomically. Restore validates everything before mutation.

Prefix reuse compares rendered token IDs and position metadata. An exact saved
prefix can restore directly; a shorter common prefix needs a checkpoint at or
before it plus replay. Arbitrary rollback is not obtained by truncating KV:
GDN recurrence is not invertible. Maintain deliberate checkpoint intervals.

## Batching and CUDA Graph constraints

Batch only ready work with compatible kernel shapes. Per-session state remains
disjoint; weight reads may be shared. Record queue delay separately from kernel
time. Capacity planning multiplies the full session ledger, not only KV.

Graph capture requires stable addresses and allocation-free replay. A graph key
contains batch/row bucket, kernel and quant policy, KV layout/capacity class,
workspace addresses or generations, feature set, and any control path changing
topology. Logical lengths may be parameters only if every captured kernel reads
them safely. Instantiate graphs before declaring the VRAM fit.

## DwarfStar transfer boundary

Reuse engine/session lifetimes, token-prefix comparison, versioned checkpoint
validation, allocation guards, and graph discipline. Adapt the payload and keys.
Discard compressed row counts/windows and expert identities.

## Failure modes and exercise

Failures include KV-only saves, physical rather than logical ring order,
non-atomic commits, restoring across a quant/template mismatch, sharing mutable
state between batch slots, or capturing allocator calls.

Save/restore after positions 1, 3, 4, 5, 32K, and arbitrary chunk boundaries.
Expected: continuation logits and every state component equal uninterrupted
execution. Prefix forks share immutable weights but never mutable buffers.

