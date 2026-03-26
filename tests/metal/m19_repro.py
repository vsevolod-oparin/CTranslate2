#!/usr/bin/env python3
"""M19 reproduction: beam search → greedy with different frame counts.

Reproduces the bug where:
  encode(3000 frames) + generate(beam=5) → encode(201 frames) + generate(beam=1)
  produces all token-ID-0 output.

Usage:
  conda run -n ct2 python tests/metal/m19_repro.py
"""
import ctranslate2
import numpy as np
import sys

MODEL_PATH = "/Users/smileijp/projects/branch/data/whisper-base"

def make_mel(n_frames, n_mels=80):
    """Create synthetic mel spectrogram features [1, n_mels, n_frames]."""
    np.random.seed(42)
    return ctranslate2.StorageView.from_array(
        np.random.randn(1, n_mels, n_frames).astype(np.float32))

def tokens_are_valid(tokens):
    """Check that tokens are not all-zero (token ID 0 = corruption)."""
    if not tokens:
        return False
    return not all(t == 0 for t in tokens)

def run_test(model, label, n_frames, beam_size, prompt):
    """Run encode + generate and return (tokens, valid)."""
    features = make_mel(n_frames)
    results = model.generate(features, [prompt], beam_size=beam_size, max_length=10)
    tokens = results[0].sequences_ids[0]
    valid = tokens_are_valid(tokens)
    status = "OK" if valid else "ALL-ZERO"
    print(f"  {label}: frames={n_frames} beam={beam_size} → tokens={tokens[:8]}{'...' if len(tokens)>8 else ''} [{status}]")
    return tokens, valid

def main():
    print(f"Loading model: {MODEL_PATH}")
    model = ctranslate2.models.Whisper(MODEL_PATH, device="cuda" if False else "auto")
    print(f"Device: {model.device}")

    prompt = [50258, 50259, 50359, 50363]  # <|startoftranscript|><|en|><|transcribe|><|notimestamps|>

    passed = 0
    failed = 0

    # --- Test 1: Baseline greedy (no prior beam search) ---
    print("\n--- Test 1: Baseline greedy (no prior beam search) ---")
    _, v = run_test(model, "greedy_3000", 3000, 1, prompt)
    if v: passed += 1
    else: failed += 1

    _, v = run_test(model, "greedy_201", 201, 1, prompt)
    if v: passed += 1
    else: failed += 1

    # --- Test 2: Beam search then greedy (THE BUG) ---
    print("\n--- Test 2: Beam search (3000) then greedy (201) ---")
    _, v = run_test(model, "beam5_3000", 3000, 5, prompt)
    if v: passed += 1
    else: failed += 1

    _, v = run_test(model, "greedy_201_after_beam", 201, 1, prompt)
    if v: passed += 1
    else: failed += 1

    # --- Test 3: Beam search then greedy with medium frames ---
    print("\n--- Test 3: Beam search (3000) then greedy (1691) ---")
    _, v = run_test(model, "beam5_3000_b", 3000, 5, prompt)
    if v: passed += 1
    else: failed += 1

    _, v = run_test(model, "greedy_1691_after_beam", 1691, 1, prompt)
    if v: passed += 1
    else: failed += 1

    # --- Test 4: Beam search then greedy with various small frames ---
    print("\n--- Test 4: Beam search (3000) then greedy (various small) ---")
    _, v = run_test(model, "beam5_3000_c", 3000, 5, prompt)
    if v: passed += 1
    else: failed += 1

    for n_frames in [500, 300, 200, 100, 50]:
        _, v = run_test(model, f"greedy_{n_frames}_after_beam", n_frames, 1, prompt)
        if v: passed += 1
        else: failed += 1

    # --- Test 5: Multiple beam→greedy cycles ---
    print("\n--- Test 5: Multiple beam→greedy cycles ---")
    for i in range(3):
        _, v = run_test(model, f"cycle{i}_beam5_3000", 3000, 5, prompt)
        if v: passed += 1
        else: failed += 1

        _, v = run_test(model, f"cycle{i}_greedy_201", 201, 1, prompt)
        if v: passed += 1
        else: failed += 1

    print(f"\n=== Results: {passed} passed, {failed} failed ===")
    return 1 if failed > 0 else 0

if __name__ == "__main__":
    sys.exit(main())
