# Milestone 16: Multi-Language Whisper Benchmark Report

**Generated:** 2026-03-14 21:03 UTC
**Model:** whisper-large-v3-turbo
**Dataset:** FLEURS test split (50 samples/language)
**Platform:** Apple M4, macOS

## Speed Comparison (RTF — lower is better)

| Configuration                            |  Avg RTF |   ENG |   JAP |   MAN |   GER |   SPA |   ARA |  RSS MB |
|------------------------------------------|----------|-------|-------|-------|-------|-------|-------|---------|
| ct2_metal_flash (Flash) float16 b=5      |    0.154 | 0.185 | 0.143 | 0.162 | 0.131 | 0.146 | 0.170 |    4794 |
| ct2_metal_flash (Flash) float16          |    0.157 | 0.147 | 0.137 | 0.210 | 0.130 | 0.143 | 0.186 |    4279 |
| ct2_metal float16                        |    0.172 | 0.149 | 0.153 | 0.233 | 0.140 | 0.162 | 0.212 |    4956 |
| ct2_metal float16 b=5                    |    0.174 | 0.157 | 0.185 | 0.205 | 0.148 | 0.166 | 0.190 |    4588 |
| ct2_metal_flash (Flash) float32          |    0.200 | 0.210 | 0.193 | 0.294 | 0.150 | 0.164 | 0.214 |    4967 |
| ct2_metal float32 b=5                    |    0.204 | 0.264 | 0.194 | 0.219 | 0.166 | 0.191 | 0.213 |    4832 |
| mlx_whisper f16                          |    0.217 | 0.251 | 0.198 | 0.236 | 0.179 | 0.213 | 0.249 |    2011 |
| ct2_metal float32                        |    0.218 | 0.215 | 0.242 | 0.280 | 0.169 | 0.184 | 0.235 |    5004 |
| whisper_cpp Q5_0                         |    0.274 | 0.265 | 0.281 | 0.356 | 0.202 | 0.226 | 0.342 |    1452 |
| whisper_cpp F16                          |    0.295 | 0.292 | 0.297 | 0.376 | 0.222 | 0.242 | 0.375 |    2170 |
| openai_whisper f32                       |    0.309 | 0.434 | 0.342 | 0.407 | 0.226 | 0.228 | 0.267 |    4882 |
| ct2_cpu float32                          |    0.488 | 0.485 | 0.476 | 0.746 | 0.344 | 0.397 | 0.542 |    4878 |
| ct2_cpu int8                             |    1.029 | 0.985 | 0.955 | 1.294 | 0.847 | 1.005 | 1.169 |    2626 |

## Quality Comparison (WER/CER % — lower is better)
*Japanese and Chinese use CER (character error rate)*

| Configuration                            |    Avg |    ENG |    JAP |    MAN |    GER |    SPA |    ARA |
|------------------------------------------|--------|--------|--------|--------|--------|--------|--------|
| ct2_metal_flash (Flash) float16 b=5      |  24.3% |  18.6% |  26.6% |  61.9% |   8.7% |   5.6% |  24.7% |
| ct2_metal_flash (Flash) float16          |  27.3% |  15.0% |  34.0% |  64.6% |   9.9% |   9.4% |  30.8% |
| ct2_metal float16                        |  27.3% |  17.5% |  34.1% |  65.2% |   9.0% |   8.5% |  29.4% |
| ct2_metal float16 b=5                    |  25.5% |  16.0% |  33.8% |  62.4% |   8.5% |   7.4% |  24.9% |
| ct2_metal_flash (Flash) float32          |  26.2% |  13.3% |  29.4% |  66.3% |  10.2% |   9.2% |  28.9% |
| ct2_metal float32 b=5                    |  23.8% |  14.9% |  27.2% |  61.9% |   8.6% |   5.4% |  25.0% |
| mlx_whisper f16                          |  13.9% |   4.7% |   6.3% |  50.5% |   3.8% |   2.5% |  15.7% |
| ct2_metal float32                        |  26.3% |  15.3% |  30.7% |  62.1% |  10.8% |   7.5% |  31.4% |
| whisper_cpp Q5_0                         |  96.3% |   4.3% | 162.2% | 113.8% |  73.9% |  75.5% | 148.1% |
| whisper_cpp F16                          |  99.4% |   4.3% | 166.5% | 122.0% |  79.4% |  76.7% | 147.2% |
| openai_whisper f32                       |  14.0% |   4.7% |   6.7% |  50.5% |   3.8% |   2.5% |  15.9% |
| ct2_cpu float32                          |  26.1% |   9.2% |  37.4% |  64.8% |   8.9% |   7.2% |  29.3% |
| ct2_cpu int8                             |  27.3% |  10.2% |  41.4% |  63.9% |   9.2% |   6.9% |  32.3% |

## Speedup Analysis

**CT2 Metal f16 (baseline RTF: 0.172)**

- vs CT2 CPU f32: **2.83x** faster (RTF 0.488 → 0.172)
- vs whisper.cpp F16: **1.71x** faster (RTF 0.295 → 0.172)
- vs mlx-whisper f16: **1.26x** faster (RTF 0.217 → 0.172)
- vs CT2 Metal Flash f16: **0.91x** faster (RTF 0.157 → 0.172)

**FlashMHA vs Standard MHA (f16 beam=1):**
- RTF: 0.172 → 0.157 (1.10x)

## Key Findings

1. **Fastest config:** ct2_metal_flash (Flash) float16 b=5 (RTF 0.154)
2. **Best quality:** mlx_whisper f16 (avg WER 13.9%)
3. **Metal GPU speedup over CPU:** 2.8x (f16 Metal vs f32 CPU)
4. **whisper.cpp caveat:** Very high WER for non-English languages (likely pywhispercpp language setting issue, not a model problem)

## Notes

- RTF = wall_time / audio_duration (lower = faster, <1.0 means faster than real-time)
- WER = Word Error Rate; CER = Character Error Rate (used for ja/zh)
- CT2 Metal INT8 skipped: MPS backend doesn't support INT8 encoder input
- mlx-whisper: beam search not yet implemented, greedy only
- OpenAI whisper: CPU only (MPS broken), extremely slow — partial results only
- All configs use whisper-large-v3-turbo model
- FLEURS test split, 50 samples per language, 6 languages
