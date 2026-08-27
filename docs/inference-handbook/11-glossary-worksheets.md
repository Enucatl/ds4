# 11. Glossary, worksheets, and capstone

[Previous](10-qwen-transfer.md) · [Index](README.md) · [Sources](sources.md)

## Glossary

- **Arithmetic intensity:** operations per byte moved at the measured level.
- **Decode:** evaluation of newly committed token positions, often one/session.
- **Expert:** one MoE FFN selected by a router; absent in dense Qwen FFNs.
- **GGUF:** metadata plus named, typed, aligned tensor container.
- **ITL:** inter-token latency experienced by one request.
- **KV cache:** attention state retained from prior positions.
- **mHC:** DeepSeek’s multi-stream hyper-connection residual mechanism.
- **MMQ/MMVQ:** quantized matrix-matrix / matrix-vector kernel families.
- **MTP:** auxiliary multi-token prediction used for training or speculation.
- **Occupancy:** resident warps relative to hardware capacity; not utilization.
- **Prefill:** evaluation of a prompt suffix, normally many token rows.
- **RoPE/mRoPE:** rotary positional encoding / modality-aware variant.
- **TTFT:** request arrival to first emitted token.

## Tensor worksheet

| Tensor/state | Logical shape | Format | Lifetime | Bytes | Access pattern |
|---|---|---|---|---:|---|
| weights | | | engine | | decode/prefill |
| raw/recurrent state | | | session | | per layer/token |
| full KV | | | session | | append + attention read |
| logits | `[vocab]` | F32 | session | `4*vocab` | write then sample |
| workspace | | | session/shared | | kernel-specific |

For quant blocks use `ceil(elements/block_values) * block_bytes`, then tensor
alignment. Track host mapping, device copy and derived repack separately.

## Benchmark worksheet

| Goal | Controlled variables | Primary metric | Correctness gate |
|---|---|---|---|
| interactive | model/quant/prompt/context | TTFT, p95 ITL | identical template + quality |
| prefill | token count/chunk/cache state | prompt tok/s | final logits |
| serving | arrival trace/session cap | aggregate tok/s + p95 | no failed requests |
| memory | context/batch/features | peak host/device bytes | completes fixture |

## Small labs

1. **Roofline:** measure copy bandwidth and FMA throughput. Expected: kernels
   approach different ceilings as intensity rises.
2. **Quant dot:** inspect `block_q2_K`; implement scalar decode on 256 values.
   Expected: metadata makes storage exceed exactly 2 bits/value.
3. **Graph capture:** capture a fixed-address chain, then vary a pointer.
   Expected: replay requires updated nodes or a distinct cache key.
4. **KV boundary:** test positions around ratios 4 and 128. Expected: compressed
   row counts and partial states follow exact boundaries.
5. **Balanced A/B:** alternate variant order. Expected: order effects become
   visible rather than silently assigned to the second variant.

## Capstone questions

- Trace one Flash token through mHC, Q/KV, compression, index selection, shared
  and routed experts, logits, sampling, and checkpoint commit.
- Produce a 5090 allocation ledger for a specific Qwen artifact, including
  vision, MTP, recurrent state, 32K KV, workspace and reserve.
- Propose one fusion. State shapes, numerical oracle, fallback, graph impact,
  profiler counters, and rollback gate.
- Design a fair DwarfStar-versus-general-engine study at concurrency one and
  eight. Explain why the conclusions may differ.

