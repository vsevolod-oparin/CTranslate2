# Decode Pipeline Profile — whisper-large-v3-turbo (2026-03-15)

## Setup
- Model: whisper-large-v3-turbo (4 decoder layers, d_model=1280, 12 heads)
- Audio: FLEURS en_us sample #0 (~10.6s)
- Beam size: 5, 23 decode steps
- Hardware: Apple M4
- `CT2_DECODE_PROFILE=1`

## Results (timed run, warmup excluded)

### f16 flash=True (primary target)

| Component | Time (ms) | Per Step | % |
|-----------|-----------|----------|---|
| decoder_call | 131 | 5.7 ms | 66% |
| sampler (TopK + sync + memcpy) | 64 | 2.8 ms | 33% |
| logits_process | 1.2 | 0.05 ms | 0.6% |
| state_update | 0.5 | 0.02 ms | 0.3% |
| everything else | 0.7 | 0.03 ms | 0.1% |
| **total_loop** | **197** | **8.6 ms** | **100%** |

Wall time: 1.33s, RTF: 0.126

### f16 flash=False (standard MHA)

| Component | Time (ms) | Per Step | % |
|-----------|-----------|----------|---|
| decoder_call | 60–79 | 2.6–3.4 ms | 38–44% |
| sampler | 96 | 4.2 ms | 54–60% |
| **total_loop** | **160–178** | **7.0–7.8 ms** | **100%** |

Wall time: 1.33s, RTF: 0.125

### f32 flash=True

| Component | Time (ms) | Per Step | % |
|-----------|-----------|----------|---|
| decoder_call | 185–243 | 8.0–10.6 ms | 61–66% |
| logits_process | 100–105 | 4.3–4.6 ms | 29–33% |
| sampler | 16 | 0.7 ms | 5% |
| **total_loop** | **301–367** | **13.1–16.0 ms** | **100%** |

Wall time: 1.54s, RTF: 0.145

## Key Findings

### 1. Sampler is 33% of f16 flash decode time

The sampler (TopK + GPU sync + memcpy) takes **64ms / 2.8ms per step** for f16 flash. This is the single largest optimization target after the decoder itself. The sync here is unavoidable — CPU must read sampled token IDs for beam search decisions.

### 2. Flash vs standard MHA: decoder is slower, but sampler is faster

Counterintuitively, flash MHA has a *slower* decoder_call (131ms vs 60-79ms) but *faster* sampler (64ms vs 96ms). This is because:
- Flash path: single fused SDPA kernel per layer → longer GPU batch → decoder_call includes more GPU compute time within the commit
- Standard path: per-head MPS GEMMs → GPU work is more spread out → sampler sync waits longer for GPU to finish prior work

Net result: flash and standard have nearly identical wall time (1.33s vs 1.33s) for this model. The benefit of flash is more apparent at longer sequences.

### 3. f32 logits_process is 33% (!) of total

For f32, `logits_process` (disable_tokens + processors) takes ~100ms — this is likely due to the indexed_fill f32 pre-sync (MPS ordering issue, kept for correctness). This is a known f32-specific overhead.

### 4. CPU bookkeeping is negligible

beam_bookkeep + beam_gather + step_overhead + state_update = **< 3ms total** (< 1.5%). No further optimization needed here.

### 5. Where time goes in decoder_call (f16 flash)

Per step (5.7ms), the decoder_call executes 4 layers × (self-attn + cross-attn + FFN):
- 4× QKV GEMM (fused, 1 MPS dispatch each)
- 4× fused SDPA decode kernel (1 dispatch each)
- 4× output projection GEMM
- 4× cross-attention (per-head MPS GEMMs — not fused, flash disabled for cross-attn)
- 4× FFN: LayerNorm + Linear1+activation + Linear2+residual
- Plus LayerNorms, residual adds, etc.

The cross-attention uses standard per-head MPS GEMMs (flash cross-attention is disabled due to padding quality issues). This is likely the largest single contributor within the 5.7ms decoder_call.

## Optimization Potential

| Target | Current Cost | Potential Saving | How |
|--------|-------------|-----------------|-----|
| **Sampler sync** | 2.8ms/step (33%) | Eliminate entirely | GPU-side EOS check + token readback batching |
| **Cross-attention SDPA** | ~1-2ms/step (est.) | 30-50% of cross-attn | Fused decode kernel for cross-attention (currently per-head MPS) |
| **FFN activation** | ~0.1ms/step (est.) | Negligible | Already encode-only |
| **LayerNorm+Linear f16** | ~0.1ms/step (est.) | Negligible | Already encode-only |

The two meaningful targets are:
1. **Sampler sync elimination** — requires GPU-side beam search, very high complexity
2. **Cross-attention fused decode kernel** — extend `fused_sdpa_decode` to cross-attention (currently disabled due to padding; could be enabled for batch_size=1)
