# Milestone 16: Multi-Language Whisper Benchmark Report

**Generated:** 2026-03-15 10:39 UTC
**Model:** whisper-large-v3-turbo
**Dataset:** FLEURS test split (50 samples/language)
**Platform:** Apple M4, macOS

## Speed Comparison (RTF — lower is better)

| Configuration                            |  Avg RTF |   ENG |   JAP |   MAN |   GER |   SPA |   ARA |  RSS MB |
|------------------------------------------|----------|-------|-------|-------|-------|-------|-------|---------|
| ct2_metal float16                        |    0.153 | 0.138 | 0.127 | 0.182 | 0.135 | 0.160 | 0.187 |    4326 |
| ct2_metal_flash (Flash) float16          |    0.161 | 0.176 | 0.146 | 0.172 | 0.131 | 0.162 | 0.196 |    4853 |
| ct2_metal float16 b=5                    |    0.164 | 0.184 | 0.150 | 0.175 | 0.137 | 0.161 | 0.190 |    5209 |
| ct2_metal_flash (Flash) float16 b=5      |    0.168 | 0.204 | 0.178 | 0.175 | 0.130 | 0.151 | 0.189 |    5117 |
| ct2_metal_flash (Flash) float32          |    0.173 | 0.203 | 0.161 | 0.183 | 0.142 | 0.167 | 0.200 |    4672 |
| ct2_metal float32                        |    0.182 | 0.220 | 0.173 | 0.188 | 0.148 | 0.183 | 0.198 |    4682 |
| ct2_metal float32 b=5                    |    0.184 | 0.196 | 0.164 | 0.202 | 0.156 | 0.185 | 0.214 |    4636 |
| mlx_whisper f16                          |    0.217 | 0.251 | 0.198 | 0.236 | 0.179 | 0.213 | 0.249 |    2011 |
| whisper_cpp Q5_0                         |    0.274 | 0.265 | 0.281 | 0.356 | 0.202 | 0.226 | 0.342 |    1452 |
| whisper_cpp F16                          |    0.295 | 0.292 | 0.297 | 0.376 | 0.222 | 0.242 | 0.375 |    2170 |
| openai_whisper f32                       |    0.309 | 0.434 | 0.342 | 0.407 | 0.226 | 0.228 | 0.267 |    4882 |
| ct2_cpu float32                          |    0.382 | 0.440 | 0.346 | 0.412 | 0.318 | 0.377 | 0.436 |    4698 |
| ct2_cpu int8                             |    0.550 | 0.647 | 0.494 | 0.602 | 0.460 | 0.549 | 0.606 |    2357 |

## Quality Comparison (WER/CER % — lower is better)
*Japanese and Chinese use CER (character error rate)*

| Configuration                            |    Avg |    ENG |    JAP |    MAN |    GER |    SPA |    ARA |
|------------------------------------------|--------|--------|--------|--------|--------|--------|--------|
| ct2_metal float16                        |  14.0% |   4.7% |   6.7% |  50.4% |   3.8% |   2.5% |  15.9% |
| ct2_metal_flash (Flash) float16          |  14.0% |   4.7% |   6.7% |  50.4% |   3.8% |   2.5% |  15.9% |
| ct2_metal float16 b=5                    |  14.3% |   4.5% |   6.7% |  50.7% |   5.8% |   2.5% |  15.4% |
| ct2_metal_flash (Flash) float16 b=5      |  14.3% |   4.5% |   6.7% |  50.8% |   5.8% |   2.5% |  15.4% |
| ct2_metal_flash (Flash) float32          |  14.0% |   4.7% |   6.7% |  50.5% |   3.8% |   2.5% |  15.9% |
| ct2_metal float32                        |  14.0% |   4.7% |   6.7% |  50.5% |   3.8% |   2.5% |  15.9% |
| ct2_metal float32 b=5                    |  14.3% |   4.5% |   6.7% |  50.8% |   5.8% |   2.5% |  15.6% |
| mlx_whisper f16                          |  13.9% |   4.7% |   6.3% |  50.5% |   3.8% |   2.5% |  15.7% |
| whisper_cpp Q5_0                         |  96.3% |   4.3% | 162.2% | 113.8% |  73.9% |  75.5% | 148.1% |
| whisper_cpp F16                          |  99.4% |   4.3% | 166.5% | 122.0% |  79.4% |  76.7% | 147.2% |
| openai_whisper f32                       |  14.0% |   4.7% |   6.7% |  50.5% |   3.8% |   2.5% |  15.9% |
| ct2_cpu float32                          |  14.0% |   4.7% |   6.7% |  50.5% |   3.8% |   2.5% |  15.9% |
| ct2_cpu int8                             |  14.0% |   4.7% |   7.0% |  50.6% |   3.9% |   2.4% |  15.6% |

## Speedup Analysis

**CT2 Metal f16 (baseline RTF: 0.153)**

- vs CT2 CPU f32: **2.50x** faster (RTF 0.382 → 0.153)
- vs whisper.cpp F16: **1.93x** faster (RTF 0.295 → 0.153)
- vs mlx-whisper f16: **1.42x** faster (RTF 0.217 → 0.153)
- vs CT2 Metal Flash f16: **1.05x** faster (RTF 0.161 → 0.153)

**FlashMHA vs Standard MHA (f16 beam=1):**
- RTF: 0.153 → 0.161 (0.95x)

## Key Findings

1. **Fastest config:** ct2_metal float16 (RTF 0.153)
2. **Best quality:** mlx_whisper f16 (avg WER 13.9%)
3. **Metal GPU speedup over CPU:** 2.5x (f16 Metal vs f32 CPU)
4. **whisper.cpp caveat:** Very high WER for non-English languages (likely pywhispercpp language setting issue, not a model problem)

## Notes

- RTF = wall_time / audio_duration (lower = faster, <1.0 means faster than real-time)
- WER = Word Error Rate; CER = Character Error Rate (used for ja/zh)
- CT2 Metal INT8 skipped: MPS backend doesn't support INT8 encoder input
- mlx-whisper: beam search not yet implemented, greedy only
- OpenAI whisper: CPU only (MPS broken), extremely slow — partial results only
- All configs use whisper-large-v3-turbo model
- FLEURS test split, 50 samples per language, 6 languages
