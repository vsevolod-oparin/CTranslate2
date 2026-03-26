#!/usr/bin/env python3
"""M19 quick perf/memory check: no-pooling vs pooling."""
import ctranslate2, numpy as np, time, os, resource

MODEL = "/Users/smileijp/projects/branch/data/whisper-large-v3-turbo"

def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)

def make_mel(n):
    np.random.seed(42)
    return ctranslate2.StorageView.from_array(
        np.random.randn(1, 128, n).astype(np.float32))

def bench(label, model, prompt, n_frames, beam, max_len, n_iter):
    # Warmup
    model.generate(make_mel(n_frames), [prompt], beam_size=beam, max_length=max_len)

    t0 = time.perf_counter()
    for _ in range(n_iter):
        model.generate(make_mel(n_frames), [prompt], beam_size=beam, max_length=max_len)
    elapsed = (time.perf_counter() - t0) / n_iter * 1000
    print(f"  {label}: {elapsed:.0f} ms/call  (n={n_iter})")
    return elapsed

def main():
    print(f"RSS before model load: {rss_mb():.0f} MB")
    model = ctranslate2.models.Whisper(MODEL, device="auto")
    print(f"RSS after model load:  {rss_mb():.0f} MB")

    prompt = [50258, 50259, 50359, 50363]

    print("\n--- Greedy (beam=1) ---")
    bench("3000 frames", model, prompt, 3000, 1, 20, 5)
    bench(" 500 frames", model, prompt,  500, 1, 20, 5)
    bench(" 205 frames", model, prompt,  205, 1, 20, 10)

    print("\n--- Beam search (beam=5) ---")
    bench("3000 frames", model, prompt, 3000, 5, 20, 3)
    bench(" 500 frames", model, prompt,  500, 5, 20, 3)

    print("\n--- Mixed: beam→greedy cycles ---")
    t0 = time.perf_counter()
    for _ in range(5):
        model.generate(make_mel(3000), [prompt], beam_size=5, max_length=10)
        model.generate(make_mel(205), [prompt], beam_size=1, max_length=10)
    elapsed = (time.perf_counter() - t0) / 5 * 1000
    print(f"  beam(3000)+greedy(205) cycle: {elapsed:.0f} ms/cycle  (n=5)")

    print(f"\nRSS at end: {rss_mb():.0f} MB")

if __name__ == "__main__":
    main()
