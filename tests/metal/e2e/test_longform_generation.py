#!/usr/bin/env python3
"""T3 — Long-form generation test (100+ tokens) to stress-test KV-cache growth.

Generates sequences of 100, 200, and 300 tokens using GPT-2 on Metal
and verifies exact match with CPU output. This exercises the KV-cache
update path (offset > 0) over many decode steps.

Requires: CT2_TEST_DATA pointing to a directory containing gpt2-ct2/
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import model_path

import ctranslate2


def main():
    gpt2_path = model_path("gpt2-ct2")
    if not os.path.isdir(gpt2_path):
        print(f"SKIP: model not found at {gpt2_path}")
        print("Set CT2_TEST_DATA to the directory containing gpt2-ct2/")
        return 0

    gen_cpu = ctranslate2.Generator(gpt2_path, device="cpu")
    gen_metal = ctranslate2.Generator(gpt2_path, device="mps")

    prompts = [
        ["Hello", ",", "Ġmy", "Ġname", "Ġis"],
        ["The", "Ġquick", "Ġbrown", "Ġfox"],
        ["In", "Ġthe", "Ġbeginning"],
    ]

    lengths = [100, 200, 300]

    passes = 0
    total = 0

    for max_len in lengths:
        print(f"\n=== max_length={max_len} (greedy) ===")
        for i, prompt in enumerate(prompts):
            cpu_r = gen_cpu.generate_batch([prompt], max_length=max_len, sampling_topk=1)
            met_r = gen_metal.generate_batch([prompt], max_length=max_len, sampling_topk=1)
            cpu_t = cpu_r[0].sequences[0]
            met_t = met_r[0].sequences[0]
            match = cpu_t == met_t
            total += 1
            if match:
                passes += 1
            status = "PASS" if match else "FAIL"
            print(f"  [{status}] Prompt {i} (generated {len(cpu_t)} tokens)")
            if not match:
                # Show first divergence point
                for j, (c, m) in enumerate(zip(cpu_t, met_t)):
                    if c != m:
                        print(f"         First diff at token {j}: cpu={c} metal={m}")
                        print(f"         CPU  [{j-2}:{j+3}]: {cpu_t[max(0,j-2):j+3]}")
                        print(f"         Metal[{j-2}:{j+3}]: {met_t[max(0,j-2):j+3]}")
                        break
                if len(cpu_t) != len(met_t):
                    print(f"         Length mismatch: cpu={len(cpu_t)} metal={len(met_t)}")

    # Also test beam search with long output
    print(f"\n=== max_length=150 (beam=2) ===")
    for i, prompt in enumerate(prompts[:2]):
        cpu_r = gen_cpu.generate_batch([prompt], max_length=150, beam_size=2)
        met_r = gen_metal.generate_batch([prompt], max_length=150, beam_size=2)
        cpu_t = cpu_r[0].sequences[0]
        met_t = met_r[0].sequences[0]
        match = cpu_t == met_t
        total += 1
        if match:
            passes += 1
        status = "PASS" if match else "FAIL"
        print(f"  [{status}] Beam2 prompt {i} (generated {len(cpu_t)} tokens)")
        if not match:
            print(f"         CPU  : ...{cpu_t[-10:]}")
            print(f"         Metal: ...{met_t[-10:]}")

    print(f"\n{passes}/{total} passed")
    if passes == total:
        print("ALL PASS")
        return 0
    else:
        print("SOME FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
