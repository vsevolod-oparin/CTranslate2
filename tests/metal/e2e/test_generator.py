#!/usr/bin/env python3
"""M10.3 — GPT-2 (decoder-only / Generator) end-to-end test on Metal.

Verifies that the Metal backend produces identical output to CPU for:
  - Greedy decoding (sampling_topk=1)
  - Beam search (beam_size 2 and 4)
  - Batch decoding (multiple prompts in one call)

Requires: CT2_TEST_DATA pointing to a directory containing gpt2-ct2/
(a CTranslate2-converted GPT-2 model).
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
        ["Once", "Ġupon", "Ġa", "Ġtime"],
    ]

    passes = 0
    total = 0

    # --- Greedy decoding ---
    for i, prompt in enumerate(prompts):
        cpu_r = gen_cpu.generate_batch([prompt], max_length=20, sampling_topk=1)
        met_r = gen_metal.generate_batch([prompt], max_length=20, sampling_topk=1)
        cpu_t = cpu_r[0].sequences[0]
        met_t = met_r[0].sequences[0]
        match = cpu_t == met_t
        total += 1
        if match:
            passes += 1
        status = "PASS" if match else "FAIL"
        print(f"[{status}] Greedy prompt {i} (len={len(cpu_t)})")
        if not match:
            print(f"  CPU  : {cpu_t}")
            print(f"  Metal: {met_t}")

    # --- Beam search ---
    for beam in [2, 4]:
        for i, prompt in enumerate(prompts[:2]):
            cpu_r = gen_cpu.generate_batch([prompt], max_length=15, beam_size=beam)
            met_r = gen_metal.generate_batch([prompt], max_length=15, beam_size=beam)
            cpu_t = cpu_r[0].sequences[0]
            met_t = met_r[0].sequences[0]
            match = cpu_t == met_t
            total += 1
            if match:
                passes += 1
            status = "PASS" if match else "FAIL"
            print(f"[{status}] Beam{beam} prompt {i} (len={len(cpu_t)})")
            if not match:
                print(f"  CPU  : {cpu_t}")
                print(f"  Metal: {met_t}")

    # --- Batch decoding ---
    cpu_r = gen_cpu.generate_batch(prompts, max_length=10, sampling_topk=1)
    met_r = gen_metal.generate_batch(prompts, max_length=10, sampling_topk=1)
    for i in range(len(prompts)):
        cpu_t = cpu_r[i].sequences[0]
        met_t = met_r[i].sequences[0]
        match = cpu_t == met_t
        total += 1
        if match:
            passes += 1
        status = "PASS" if match else "FAIL"
        print(f"[{status}] Batch prompt {i} (len={len(cpu_t)})")
        if not match:
            print(f"  CPU  : {cpu_t}")
            print(f"  Metal: {met_t}")

    print(f"\n{passes}/{total} passed")
    if passes == total:
        print("ALL PASS")
        return 0
    else:
        print("SOME FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
