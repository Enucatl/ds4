# 3. Exact Qwen3.8-27B text model contract

[Previous](02-execution-path.md) · [Index](README.md) · [Next](04-numerics.md)

## Why this matters

A plausible transformer is wrong if a packed slice, norm, residual, or state
update differs. This is the executable forward-pass specification.

## Validated constants

| Field | Value |
|---|---:|
| vocabulary / residual / FFN | 248,320 / 5,120 / 17,408 |
| layers | 64: `(GDN,GDN,GDN,attention) * 16` |
| GDN QK / value heads / dimensions | 16 / 48 / 128 |
| convolution width | 4 |
| attention Q / KV heads / dimension | 24 / 4 / 256 |
| rotary dimensions | 64 per head (`partial_rotary_factor=0.25`) |
| RMS epsilon / recurrence dtype | `1e-6` / FP32 |
| maximum positions | 262,144 |

Full-attention layers are zero-based `3,7,...,63`. Test transitions `2 -> 3`,
`3 -> 4`, and layer 63 explicitly.

## Common block and weights

RMSNorm computes in FP32 and scales by `(1 + weight)`. Each layer runs:

```text
x <- x + Mixer(RMSNorm(x))
x <- x + down_proj(SiLU(gate_proj(RMSNorm(x))) * up_proj(RMSNorm(x)))
```

Every layer binds two `[5120]` norms and FFN weights `[17408,5120]`,
`[17408,5120]`, `[5120,17408]`. Global weights include embeddings and an
untied LM head, both `[248320,5120]`, plus final norm. Treat `[out,in]` as the
checkpoint convention and inventory actual keys before conversion.

| Layer kind | Learned tensor | Checkpoint shape `[out,in]` or vector |
|---|---|---:|
| GDN | `in_proj_qkv` / `in_proj_z` | `[10240,5120]` / `[6144,5120]` |
| GDN | `in_proj_a` / `in_proj_b` | `[48,5120]` each |
| GDN | depthwise convolution | `[10240,1,4]` |
| GDN | `A_log` / `dt_bias` / gated-norm weight | `[48]` / `[48]` / `[128]` |
| GDN | `out_proj` | `[5120,6144]` |
| attention | packed query+gate projection | `[12288,5120]` |
| attention | K / V projections | `[1024,5120]` each |
| attention | per-head Q / K norms | `[256]` each |
| attention | output projection | `[5120,6144]` |

Biases are disabled for these linear projections and the GDN convolution in the
pinned config. Tensor key spelling remains a manifest concern because wrapper
prefixes can differ across checkpoint tooling.

## Gated DeltaNet

From normalized `u[B,T,5120]`:

```text
packed = in_proj_qkv(u) -> [B,T,10240]
Q = packed[...,0:2048]       -> [B,T,16,128]
K = packed[...,2048:4096]    -> [B,T,16,128]
V = packed[...,4096:10240]   -> [B,T,48,128]
Z = in_proj_z(u)             -> [B,T,48,128]
b = in_proj_b(u)             -> [B,T,48]
a = in_proj_a(u)             -> [B,T,48]
```

Apply depthwise causal width-4 convolution and SiLU to packed QKV. L2-normalize
Q/K and repeat each Q/K head three times. Let `beta=sigmoid(b)`,
`g=-exp(A_log)*softplus(a+dt_bias)`, and `q=Q/sqrt(128)`. Per head, with FP32
`S[128,128]`:

```text
S <- exp(g_t) * S
prediction <- k_t^T S
delta <- beta_t * (v_t - prediction)
S <- S + k_t outer delta
y_t <- q_t^T S
```

Apply headwise RMSNorm to `y`, multiply by `SiLU(Z)`, flatten 48 heads to 6,144,
and apply `out_proj[5120,6144]`. Persistent state is FP32 `[48,128,128]` and
the official cache's last four packed QKV rows `[10240,4]` per GDN layer.

## Full attention and positions

`q_proj` emits `[T,24,512]`, split into query and gate halves of 256. K and V
are `[T,4,256]`. Apply per-head Q/K RMSNorm, RoPE to 64 dimensions, append K/V,
repeat KV heads sixfold conceptually, compute causal attention scaled by
`1/sqrt(256)`, multiply output by `sigmoid(gate)`, flatten to 6,144, and apply
`o_proj[5120,6144]`.

Text constructs four identical scalar-position channels: channel 0 controls the
causal mask and channels 1–3 supply rotary positions. A future multimodal
processor supplies temporal/height/width positions and visual embeddings.

## State and transfer ledger

**Estimated, batch one:** recurrent matrices use
`48*48*128*128*4 = 144 MiB`; convolution state uses
`48*10240*4*4 = 7.5 MiB`; attention KV uses 64 KiB/token (2 GiB at 32K).

Reuse DwarfStar validation, ownership, allocation accounting, and differential
testing. Adapt serialization and hybrid scheduling. Reject compressed attention,
sparse indexing, MoE routing/streaming, mHC, and DSpark equations.

## Failure modes and exercise

Common errors are wrong head expansion, gate order, conventional RMS weights,
BF16 recurrence, convolution warm-up, rotating 256 dimensions, and misclassifying
layer 3. Trace embedding, GDN convolution/recurrence/output, layer-3 attention,
FFN, final norm, and logits. Expected: scalar results match pinned Transformers
and an independent llama.cpp trace at every boundary within declared tolerances.
