# 11. Glossary, worksheets, and review

[Previous](10-qwen-transfer.md) · [Index](README.md) · [Sources](sources.md)

## Glossary

- **Decode:** advance newly committed positions, often one per session.
- **Gated DeltaNet (GDN):** recurrent token mixer with decay and delta update.
- **ITL / TTFT:** inter-token latency / time to first token.
- **KV cache:** growing full-attention key/value rows; not GDN state.
- **MMV / MMQ:** matrix-vector / matrix-matrix kernel families (often quantized).
- **MTP:** multi-token prediction used for speculative proposals after v1.
- **mRoPE:** modality-aware temporal/height/width rotary positions.
- **Prefill:** evaluate an unmatched prompt suffix, usually many rows.
- **RoPE:** rotary position encoding; Qwen text attention rotates 64 dimensions.
- **State frontier:** last atomically committed sequence position.

## Tensor and state worksheet

| Item | Logical shape | Format | Lifetime | Bytes | Oracle tap |
|---|---|---|---|---:|---|
| embeddings / LM head | `[248320,5120]` each | | engine | | embedding/logits |
| GDN recurrence | `48 layers * [48,128,128]` | FP32 v1 | session | 144 MiB | each GDN output |
| convolution rings | `48 * [10240,4]` | FP32 v1 | session | 7.5 MiB | post-convolution |
| full K and V | `16 * 2 * [C,4,256]` | BF16 v1 | session | `65536*C` | attention output |
| logits | `[248320]` | FP32 | session | 993,280 B | final logits |
| workspace/repack/graphs | measured plan | | engine/session | | allocation log |

For each blank, record source key, orientation, block metadata, alignment,
owner, address stability, and checksum. Totals must reconcile with runtime logs.

## Verification worksheet

| Scenario | Compare | Expected result |
|---|---|---|
| positions 0–5 | GDN convolution and recurrence | oracle agreement through warm-up |
| prompt lengths 1/4/5/63/64/65 | full tap set | tolerance gates pass |
| layers 2/3/4/63 | scheduler and mixer taps | correct kind and residual order |
| arbitrary prefill chunks | final state and logits vs decode | equivalent result |
| arbitrary save/restore | uninterrupted continuation | all hybrid state agrees |
| 32K and larger context | allocation ledger and completion | admitted sizes retain reserve |
| quant vs high precision | logits, NLL/PPL, long/task suite | declared quality budget passes |
| MTP accept/partial/reject | target-only run | same committed sequence/state |
| future visual rows | Transformers embeddings/positions/logits | multimodal boundary agrees |

## Reader labs

1. Derive the 10,240 packed GDN width and label every slice. Expected:
   `16*128 + 16*128 + 48*128`.
2. State the recurrence update without consulting code. Expected: decay state,
   predict at K, beta-scaled correction, outer update, query readout.
3. Fill an allocation ledger from an artifact and runtime log. Expected: explain
   every byte category and prove reserve after graph creation.
4. Specify scalar milestone steps. Expected: pin, inventory, embed, 64 scheduled
   blocks, final norm/head, intermediate taps, two independent comparisons.
5. Review every claim label and link. Expected: measured, external, estimated,
   and proposed statements cannot be mistaken for one another.

The handbook passes reader review when someone unfamiliar with GDN can complete
labs 1–4 and state the implementation sequence without undocumented assumptions.

