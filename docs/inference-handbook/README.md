# DwarfStar Inference Engineering Handbook

Baseline: DwarfStar [`c1d4597`](https://github.com/antirez/ds4/tree/c1d4597a80e300b803dc642519718f2c999589da), 2026-08-27.

This is a self-paced path for a systems programmer who knows C and memory
hierarchies but is new to GPU language-model inference. DwarfStar is the
worked example, not a universal winner: it deliberately specializes for a few
model layouts. The final chapters ask which ideas survive when the target is a
dense multimodal Qwen model on one RTX 5090.

## How to read

1. [Foundations](01-foundations.md) — vocabulary, shapes, rooflines.
2. [From prompt to token](02-execution-path.md) — a complete call and state trace.
3. [DeepSeek V4](03-deepseek-v4.md) — compressed attention, indexer, mHC, MoE, DSpark.
4. [Numerics and preparation](04-numerics.md) — formats, calibration, quality.
5. [GPU implementation](05-gpu-implementation.md) — dispatch, fusion, lifetimes, graphs.
6. [System optimization](06-system-optimization.md) — streaming, caching, parallelism.
7. [Engineering method](07-engineering-method.md) — oracles, tests, benchmarks.
8. [RTX 5090 lab](08-rtx-5090.md) — Blackwell constraints and experiments.
9. [Engine comparison](09-engine-comparison.md) — choose goals before engines.
10. [Qwen3.8-27B transfer study](10-qwen-transfer.md) — direct transfers and redesigns.
11. [Glossary and worksheets](11-glossary-worksheets.md) — review and capstone.
12. [Source and evidence ledger](sources.md) — claims, anchors, confidence.

Every performance statement uses one label:

- **Measured:** produced by a named repository fixture on named hardware.
- **External:** reported by a linked primary source; not reproduced here.
- **Estimated:** arithmetic from stated inputs; not a benchmark.
- **Proposed:** a reproducible experiment whose result is not yet known.

Stable symbols plus the pinned commit are authoritative. Line numbers in source
links are navigation aids because later commits move them.

## Update checklist

- Record the new commit and date; keep this baseline available in history.
- Diff `ds4_shape`, `config_validate_model`, graph allocation, cache serialization,
  quant block definitions, and CUDA dispatch.
- Re-run the internal-link checker and Mermaid rendering.
- Recalculate shape and memory tables; relabel any stale measurements.
- Recheck primary Qwen, NVIDIA, CUDA, and engine documentation.
- Audit for unqualified “faster”, “fits”, and “supported” claims.

