#!/usr/bin/env python3
"""Test FlashMultiHeadAttention on MPS Generator with TinyLlama.

Compares standard path (flash_attention=False) vs flash path (flash_attention=True).
"""
import ctranslate2
import ctypes
import os
import sys

MODEL_PATH = "/Users/smileijp/projects/branch/data/tinyllama-ct2"
PROMPT_TOKENS = [["<s>", "Hello", ",", " world"]]

def get_commit_count():
    """Get Metal commit count via ctypes."""
    lib_path = os.environ.get("CT2_LIB_PATH",
        "/opt/anaconda3/envs/ct2/lib/libctranslate2.4.7.1.dylib")
    try:
        lib = ctypes.CDLL(lib_path)
        lib._ZN12ctranslate25metal12commit_countEv.restype = ctypes.c_uint64
        return lib._ZN12ctranslate25metal12commit_countEv()
    except:
        return -1

def reset_commit_count():
    lib_path = os.environ.get("CT2_LIB_PATH",
        "/opt/anaconda3/envs/ct2/lib/libctranslate2.4.7.1.dylib")
    try:
        lib = ctypes.CDLL(lib_path)
        lib._ZN12ctranslate25metal18reset_commit_countEv()
    except:
        pass

def test_generator(flash_attention, max_length=20):
    """Run Generator with the given flash_attention flag."""
    label = "FLASH" if flash_attention else "STANDARD"
    print(f"\n--- {label} path (flash_attention={flash_attention}) ---")

    try:
        gen = ctranslate2.Generator(
            MODEL_PATH,
            device="mps",
            compute_type="float16",
            flash_attention=flash_attention,
        )
    except Exception as e:
        print(f"  ERROR loading model: {e}")
        return None

    reset_commit_count()
    try:
        results = gen.generate_batch(
            PROMPT_TOKENS,
            max_length=max_length,
            beam_size=1,
            sampling_topk=1,
        )
    except Exception as e:
        print(f"  ERROR during generation: {e}")
        return None

    commits = get_commit_count()
    tokens = results[0].sequences[0]
    num_tokens = len(tokens)
    commits_per_token = commits / max(num_tokens, 1) if commits >= 0 else -1

    print(f"  Tokens generated: {num_tokens}")
    print(f"  Commits: {commits} ({commits_per_token:.1f}/token)")
    print(f"  Output: {' '.join(tokens[:30])}")

    return tokens

def main():
    if not os.path.isdir(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        sys.exit(1)

    print("=== FlashMHA Generator Test (TinyLlama-1.1B, MPS, float16) ===")

    # Standard path (known correct)
    standard_tokens = test_generator(flash_attention=False)

    # Flash path (possibly broken)
    flash_tokens = test_generator(flash_attention=True)

    if standard_tokens and flash_tokens:
        match = standard_tokens == flash_tokens
        print(f"\n--- Comparison ---")
        print(f"  Match: {match}")
        if not match:
            # Find first divergence
            for i in range(max(len(standard_tokens), len(flash_tokens))):
                st = standard_tokens[i] if i < len(standard_tokens) else "<end>"
                ft = flash_tokens[i] if i < len(flash_tokens) else "<end>"
                if st != ft:
                    print(f"  First divergence at token {i}: standard='{st}' flash='{ft}'")
                    break

if __name__ == "__main__":
    main()
