# 5. Weight conversion, quantization, and 32 GiB fit

[Previous](04-numerics.md) · [Index](README.md) · [Next](06-system-optimization.md)

## Why this matters

BF16 language weights alone are about 54 GB decimal. A 5090 build therefore
requires quantization, but quantizing recurrence-sensitive tensors before the
high-precision path passes makes errors hard to attribute.

## Inventory before conversion

Emit one manifest row per checkpoint tensor: canonical role, source name, shape,
dtype, elements, source bytes, destination format/block size, payload bytes,
alignment, checksum, and transpose/repack. Reject missing, duplicate, extra
mandatory, or shape-incompatible tensors. The manifest, not filename folklore,
is the input to `ModelSpec` validation.

For block format `(N values, B bytes)`, storage is
`ceil(elements/N)*B`, rounded for tensor alignment. Record device copies and
derived layouts separately. Conversion must round-trip a sample block through a
scalar dequantizer and compare each converted tensor's checksum.

## Quantization policy

Bring up BF16/FP16 weights with FP32 sensitive state first. Then calibrate on a
representative corpus and change one tensor class at a time. A reasonable
**Proposed** order is FFN gate/up, FFN down, mixer projections, embeddings/LM
head, and finally recurrence-sensitive GDN paths only if evidence permits.
Keep norms, biases, `A_log`, `dt_bias`, and recurrent state high precision at v1.
q27's published per-tensor recipes are useful external evidence that sensitivity
is non-uniform, not proof that its choices transfer to this converter.

## Reproducible memory ledger

| Allocation | Formula / source | 32K scenario |
|---|---|---:|
| quantized resident weights | converter manifest | measured at load |
| repacked/derived weights | runtime allocation log | measured at load |
| GDN recurrent matrices | `48*48*128*128*4` | 144 MiB/session |
| GDN convolution rings | `48*10240*4*4` | 7.5 MiB/session |
| full-attention BF16 KV | `65536*context` | 2 GiB/session |
| logits | `248320*4` | 0.95 MiB/session |
| activation/workspace | maximum live plan | measured |
| CUDA libraries + graphs | allocation delta | measured |
| fragmentation/reserve | explicit policy | never zero |

The 4-bit arithmetic lower bound is `27e9*0.5 = 13.5 GB` decimal (12.6 GiB),
not an artifact size or fit proof. The allocation log must show current, peak,
and reserved bytes by owner and demonstrate headroom after graph instantiation.
At native 256K, BF16 KV alone is 16 GiB, so a 32K fit does not prove native
context fit. Larger context requires a measured lower-precision KV policy or a
smaller weight/workspace plan and its own quality gate.

## DwarfStar transfer boundary

Reuse exact-name binding, block-size arithmetic, calibration, and allocation
guards. Adapt the quant policy to dense Qwen tensors. Reject routed-expert
formats and SSD expert streaming: every dense FFN weight is used every token.

## Failures and exercise

Do not omit alignment/scales, count mmap file bytes as VRAM, overwrite the only
high-precision oracle, or claim quality from perplexity alone. Exercise: fill
the ledger from an actual artifact and context 32K. Expected: logged allocation
totals reproduce the sum within allocator accounting, the process completes a
fixture, and free reserve remains after graphs; otherwise the fit gate fails.

