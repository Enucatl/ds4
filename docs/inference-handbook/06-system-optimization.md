# 6. System-level optimization

[Previous](05-gpu-implementation.md) · [Index](README.md) · [Next](07-engineering-method.md)

Once weights exceed fast memory, placement and scheduling become model code.

## SSD expert streaming

DwarfStar keeps dense/shared tensors resident where possible and streams routed
expert slices. A selected-expert cache and hotlist exploit routing locality.
Reads for missing experts overlap work on the shared expert and already resident
experts; prefill can load the next layer while the current one computes.
`ds4_ssd.c`, `graph_stream_expert_table_make`, and
`metal_graph_decode_selected_readahead_override` are the reading path.

The latency bound becomes roughly `max(compute, missing_bytes/storage_rate)` only
when asynchronous reads begin early enough; otherwise costs add. Page cache and
mmap residency can make repeated tests look like SSD wins. Record cold/warm
state, cache hit rate, bytes read, queue depth, and tail latency.

## KV compression and checkpoints

Raw recent rows protect local fidelity; compressed append-only rows bound
long-context growth; state tensors preserve unfinished windows. Disk checkpoints
turn a long prefix into reusable state, but are version/layout-specific and must
be committed atomically by the surrounding KV store. API prefix reuse compares
token IDs—not source strings—then resumes at the common prefix.

## Batching and parallelism

Microbatching collects one decode token from several sessions. It raises matrix
rows and weight reuse, at the cost of queueing and more per-session KV. Report
aggregate and per-request latency.

| Method | Partition | Main cost/risk |
|---|---|---|
| pipeline parallel | contiguous layer ranges | bubbles, activation transport |
| tensor parallel | matrix/expert dimensions | collective latency each layer |
| expert parallel | routed experts | load imbalance, token all-to-all |
| data parallel | whole replicas | duplicated weights, easy scaling |

DwarfStar’s two-rank TP maps contiguous expert halves and replicates dense
weights; CUDA TP requires an even GPU count. Distributed layer slices can combine
machines to aggregate memory. These policies fit its layouts and are not general
proofs about the best sharding for another model.

## Scheduler policy

Prefer short bounded prefill chunks so a long prompt does not starve decode.
Schedule ready decode microbatches; cap memory by session; group compatible graph
shapes; account for cancellation. Prefix-cache admission should weigh bytes
saved, expected reuse, and eviction cost rather than prefix length alone.

## Check and experiment

Replay a fixed expert-ID trace against LRU and hotlist-seeded caches. Expected:
hotlists help only if the workload resembles profiling. Run cold and warm SSD
trials after documenting OS cache control. Expected: warm results are faster and
must not be labeled storage throughput.

