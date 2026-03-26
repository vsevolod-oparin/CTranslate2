#!/usr/bin/env python3
"""M19 A/B: run with current dylib, print results."""
import ctranslate2, numpy as np, time, resource

MODEL = "/Users/smileijp/projects/branch/data/whisper-large-v3-turbo"

def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)

def make_mel(n):
    np.random.seed(42)
    return ctranslate2.StorageView.from_array(
        np.random.randn(1, 128, n).astype(np.float32))

def bench(model, prompt, n_frames, beam, max_len, n_iter):
    # Warmup
    for _ in range(2):
        model.generate(make_mel(n_frames), [prompt], beam_size=beam, max_length=max_len)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        model.generate(make_mel(n_frames), [prompt], beam_size=beam, max_length=max_len)
    return (time.perf_counter() - t0) / n_iter * 1000

def main():
    model = ctranslate2.models.Whisper(MODEL, device="auto")
    prompt = [50258, 50259, 50359, 50363]

    r = {}
    r["greedy_3000"] = bench(model, prompt, 3000, 1, 20, 5)
    r["greedy_500"]  = bench(model, prompt,  500, 1, 20, 5)
    r["greedy_205"]  = bench(model, prompt,  205, 1, 20, 10)
    r["beam5_3000"]  = bench(model, prompt, 3000, 5, 20, 3)
    r["beam5_500"]   = bench(model, prompt,  500, 5, 20, 3)

    # Mixed cycle
    for _ in range(2):
        model.generate(make_mel(3000), [prompt], beam_size=5, max_length=10)
        model.generate(make_mel(205), [prompt], beam_size=1, max_length=10)
    t0 = time.perf_counter()
    for _ in range(5):
        model.generate(make_mel(3000), [prompt], beam_size=5, max_length=10)
        model.generate(make_mel(205), [prompt], beam_size=1, max_length=10)
    r["mixed_cycle"] = (time.perf_counter() - t0) / 5 * 1000

    r["rss_mb"] = rss_mb()

    for k, v in r.items():
        print(f"  {k}: {v:.0f}")

if __name__ == "__main__":
    main()
