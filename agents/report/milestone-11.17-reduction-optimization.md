# M11.17 — Fused Timestamp Check + Disable Kernel

## Summary

Eliminated ~1346 `commit_and_wait()` syncs per Whisper inference by fusing the timestamp probability check and token disabling into a single encode-only GPU kernel. The previous implementation required a GPU→CPU readback (sync) to decide which tokens to disable, then a second sync for the `indexed_fill` that applied the disables. The fused kernel performs both operations entirely on the GPU with no CPU readback.

## Changes

| File | Change |
|------|--------|
| `src/metal/primitives_reduction.mm` | New MSL kernel `fuse_ts_disable_<T>` + C++ function `fuse_timestamp_check_and_disable_metal<T>()` |
| `src/metal/utils.h` | Declared `fuse_timestamp_check_and_disable_metal<T>()` |
| `src/models/whisper.cc` | Metal path uses fused kernel instead of `should_sample_timestamps_metal` + `disable_tokens.add()` loop |

## Architecture

### Previous Flow (M11.14)
1. `disable_tokens.apply()` → `indexed_fill` with `CT2_COMMIT_AND_WAIT()` — **sync #1**
2. `LogSoftMax(logits, log_probs)` — GPU encode
3. `should_sample_timestamps_metal(log_probs)` — GPU encode + `CT2_COMMIT_AND_WAIT()` — **sync #2**
4. CPU reads boolean results, calls `disable_tokens.add()` for batches where should_sample=true
5. Returns from logits processor
6. Framework calls `disable_tokens.apply()` → `indexed_fill` with `CT2_COMMIT_AND_WAIT()` — **sync #3**

### New Flow (M11.17)
1. `disable_tokens.apply()` → `indexed_fill` with `CT2_COMMIT_AND_WAIT()` — **sync #1** (unchanged)
2. `LogSoftMax(logits, log_probs)` — GPU encode
3. `fuse_timestamp_check_and_disable_metal(log_probs, logits)` — GPU encode-only, **no sync**
4. Returns from logits processor
5. Framework calls `disable_tokens.apply()` → no tokens accumulated → **no-op, no sync**

Syncs 2 and 3 are eliminated. The fused kernel writes `-inf` directly to `logits` on the GPU, bypassing the CPU readback entirely.

### Fused GPU Kernel

The kernel reuses the same 3-pass threadgroup reduction as `should_sample_ts`:
1. **Pass 1**: Max over text tokens `[0, num_text)`
2. **Pass 2**: Max over timestamp tokens `[num_text, num_text+num_ts)`
3. **Pass 3**: `sum(exp(ts[i] - max_ts))`

Then thread 0 computes `logsumexp_ts > max_text` and broadcasts the decision via shared memory. If true, all threads cooperatively write `-inf` to `logits[bid][0..num_text)`.

One threadgroup per batch_id. Encode-only — no `CT2_COMMIT_AND_WAIT()`.

## Results

### Sync Count (whisper-large-v3-turbo, beam_size=5, 60s audio)

| Source | Before (M11.16) | After (M11.17) | Delta |
|--------|-----------------|-----------------|-------|
| `primitives_reduction.mm:345` (should_sample_ts) | ~1176 | 0 | **-1176** |
| `primitives_memory.mm:80` (indexed_fill) | ~1205 | 1035 | **-170** |
| **Total** | ~2381 | 1035 | **-1346** |

### Test Results

| Test | Result |
|------|--------|
| `test_beam_search.py` | 39/39 PASS |
| `test_translation.py` | 90/90 PASS |
| `test_whisper.py` | 13/13 PASS |
| `test_faster_whisper.py` | 8/8 PASS |

## Design Decisions

### Why Not GPU `indexed_fill`?

The original plan included making `indexed_fill` a GPU kernel (encode-only). This was implemented and tested but had to be reverted because:

1. **Buffer lifetime issue**: `DisableTokens::apply()` creates a temporary `StorageView` for indices that is freed immediately after `indexed_fill` returns. With encode-only dispatch, the MetalAllocator can reuse the underlying buffer before the GPU kernel executes, causing use-after-free.

2. **Sync dependency**: Even with a copied temp buffer, the encode-only approach fails. The `indexed_fill` sync serves as a fence that ensures prior GPU writes to logits are visible before subsequent CPU code (or GPU kernels in a new command buffer) reads them. Removing this fence breaks the data flow.

3. **CPU loop is fast enough**: For the typical index counts in `DisableTokens` (~50-500 indices), the CPU scatter loop on shared memory is sub-microsecond and dominates neither latency nor throughput.

The remaining ~1035 `indexed_fill` syncs come from the first `disable_tokens.apply()` inside `ApplyTimestampRules::apply()` (line 819 of whisper.cc), which accumulates suppress/ordering tokens that require the fence. Eliminating these would require restructuring `DisableTokens` to be fully deferred — a larger architectural change for a future milestone.

### Why Fuse Instead of Just Skipping CPU Readback?

The fused kernel combines the check AND the disable in one GPU dispatch. An alternative would be to keep `should_sample_timestamps_metal` but write results to a GPU buffer, then have a second kernel read the results and write -inf. However:
- Two kernels = two encoder dispatches (more overhead)
- The fused approach is simpler and eliminates ALL intermediate state
- One threadgroup per batch_id is sufficient for both the reduction and the scatter
