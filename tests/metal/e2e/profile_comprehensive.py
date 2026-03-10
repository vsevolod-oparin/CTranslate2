#!/usr/bin/env python3
"""
Comprehensive Metal inference profiler — M11.23+

Measures ALL possible bottleneck categories, not just sync counts:
  1. Wall-clock time (end-to-end, CPU vs Metal)
  2. GPU execution time (GPUStartTime/GPUEndTime via Metal API)
  3. Commit/sync counts and overhead
  4. GPU utilization % (IOKit PerformanceStatistics)
  5. Memory: RSS, VSIZE, Metal allocator live/pool bytes, system GPU memory
  6. CPU profiling (cProfile) — find CPU-side hotspots
  7. Per-op profiling (CT2_ENABLE_PROFILING) — PROFILE macro output
  8. Thread activity — are we CPU-bound or GPU-bound?

Usage:
  python profile_comprehensive.py [model_name] [beam_size]
  python profile_comprehensive.py whisper-large-v3-turbo 5
"""
import os, sys, time, ctypes, subprocess, threading, resource, gc, io
import contextlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import model_path, audio_path

# ---------------------------------------------------------------------------
# GPU stats reader (IOKit PerformanceStatistics)
# ---------------------------------------------------------------------------
def read_gpu_stats():
    """Read GPU utilization and memory from IOKit AGXAccelerator."""
    try:
        r = subprocess.run(['/tmp/test_gpu_perf'], capture_output=True, text=True, timeout=2)
        stats = {}
        for line in r.stdout.strip().split('\n'):
            if '=' in line:
                k, v = line.rsplit('=', 1)
                stats[k.strip()] = int(v.strip())
        return stats
    except Exception:
        return {}

# ---------------------------------------------------------------------------
# Process memory reader
# ---------------------------------------------------------------------------
def get_process_memory():
    """Get RSS and VSIZE in MB via ps."""
    try:
        pid = os.getpid()
        r = subprocess.run(['ps', '-o', 'rss=,vsz=', '-p', str(pid)],
                           capture_output=True, text=True, timeout=2)
        parts = r.stdout.strip().split()
        rss_kb = int(parts[0])
        vsz_kb = int(parts[1])
        return {'rss_mb': rss_kb / 1024, 'vsz_mb': vsz_kb / 1024}
    except Exception:
        return {'rss_mb': 0, 'vsz_mb': 0}

# ---------------------------------------------------------------------------
# Background sampler — samples GPU utilization and memory at interval
# ---------------------------------------------------------------------------
class BackgroundSampler:
    def __init__(self, interval=0.1):
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self):
        while not self._stop.is_set():
            t = time.monotonic()
            gpu = read_gpu_stats()
            mem = get_process_memory()
            self.samples.append({
                'time': t,
                'gpu_util': gpu.get('Device Utilization %', -1),
                'renderer_util': gpu.get('Renderer Utilization %', -1),
                'gpu_mem_alloc_mb': gpu.get('Alloc system memory', 0) / (1024*1024),
                'gpu_mem_inuse_mb': gpu.get('In use system memory', 0) / (1024*1024),
                'rss_mb': mem['rss_mb'],
            })
            self._stop.wait(self.interval)

    def summary(self):
        if not self.samples:
            return {}
        gpu_utils = [s['gpu_util'] for s in self.samples if s['gpu_util'] >= 0]
        rss_vals = [s['rss_mb'] for s in self.samples]
        gpu_mem = [s['gpu_mem_inuse_mb'] for s in self.samples]
        return {
            'gpu_util_avg': sum(gpu_utils)/len(gpu_utils) if gpu_utils else -1,
            'gpu_util_max': max(gpu_utils) if gpu_utils else -1,
            'gpu_util_min': min(gpu_utils) if gpu_utils else -1,
            'gpu_util_samples': len(gpu_utils),
            'rss_max_mb': max(rss_vals),
            'rss_min_mb': min(rss_vals),
            'gpu_mem_max_mb': max(gpu_mem) if gpu_mem else -1,
        }

# ---------------------------------------------------------------------------
# CTranslate2 Metal API bindings
# ---------------------------------------------------------------------------
def bind_metal_api():
    """Bind ctranslate2 internal Metal profiling functions via ctypes."""
    lib = ctypes.CDLL("/opt/anaconda3/envs/ct2/lib/libctranslate2.dylib")

    api = {}
    bindings = {
        'commit_count':       ('_ZN11ctranslate25metal12commit_countEv', ctypes.c_uint64),
        'reset_commit_count': ('_ZN11ctranslate25metal18reset_commit_countEv', None),
        'gpu_time_elapsed':   ('_ZN11ctranslate25metal16gpu_time_elapsedEv', ctypes.c_double),
        'reset_gpu_time':     ('_ZN11ctranslate25metal14reset_gpu_timeEv', None),
        'pso_hit_count':      ('_ZN11ctranslate25metal13pso_hit_countEv', ctypes.c_uint64),
        'pso_miss_count':     ('_ZN11ctranslate25metal14pso_miss_countEv', ctypes.c_uint64),
        'reset_pso_stats':    ('_ZN11ctranslate25metal15reset_pso_statsEv', None),
    }

    for name, (sym, rtype) in bindings.items():
        try:
            fn = getattr(lib, sym)
            if rtype:
                fn.restype = rtype
            fn.argtypes = []
            api[name] = fn
        except AttributeError:
            print(f"  [WARN] Symbol not found: {sym}")
            api[name] = lambda: 0

    return api

# ---------------------------------------------------------------------------
# Run a single profiled transcription
# ---------------------------------------------------------------------------
def profile_transcription(model, audio_file, beam_size, api, label,
                          sampler_interval=0.05, enable_cprofile=True,
                          enable_bg_sampling=False):
    """Run transcription with full profiling. Returns dict of all metrics."""
    gc.collect()
    time.sleep(0.5)  # let GC settle

    # Reset counters
    api['reset_commit_count']()
    api['reset_gpu_time']()
    api['reset_pso_stats']()

    mem_before = get_process_memory()
    gpu_before = read_gpu_stats()

    # Start background sampling (OUTSIDE the timed section to avoid subprocess overhead)
    sampler = BackgroundSampler(interval=sampler_interval)
    if enable_bg_sampling:
        sampler.start()

    # CPU profiling (optional — cProfile itself adds ~5-10% overhead)
    import cProfile, pstats
    profiler = cProfile.Profile() if enable_cprofile else None
    if profiler:
        profiler.enable()

    t0 = time.monotonic()
    segments = list(model.transcribe(
        audio_file, language="ru", beam_size=beam_size,
        without_timestamps=True
    )[0])
    wall_ms = (time.monotonic() - t0) * 1000

    if profiler:
        profiler.disable()
    if enable_bg_sampling:
        sampler.stop()

    # Collect metrics
    mem_after = get_process_memory()
    gpu_after = read_gpu_stats()

    commits = api['commit_count']()
    gpu_time_s = api['gpu_time_elapsed']()
    pso_hits = api['pso_hit_count']()
    pso_misses = api['pso_miss_count']()

    # CPU profile top functions
    cpu_profile_text = ""
    cpu_self_time_text = ""
    if profiler:
        stream = io.StringIO()
        ps = pstats.Stats(profiler, stream=stream)
        ps.sort_stats('cumulative')
        ps.print_stats(30)
        cpu_profile_text = stream.getvalue()

        stream2 = io.StringIO()
        ps2 = pstats.Stats(profiler, stream=stream2)
        ps2.sort_stats('tottime')
        ps2.print_stats(20)
        cpu_self_time_text = stream2.getvalue()

    bg_summary = sampler.summary()

    result = {
        'label': label,
        'wall_ms': wall_ms,
        'commits': commits,
        'gpu_time_ms': gpu_time_s * 1000,
        'sync_overhead_ms': commits * 0.4,  # ~0.4ms per commit_and_wait
        'cpu_time_ms': wall_ms - gpu_time_s * 1000,  # approximate
        'pso_hits': pso_hits,
        'pso_misses': pso_misses,
        'rss_before_mb': mem_before['rss_mb'],
        'rss_after_mb': mem_after['rss_mb'],
        'rss_delta_mb': mem_after['rss_mb'] - mem_before['rss_mb'],
        'gpu_mem_before_mb': gpu_before.get('In use system memory', 0) / (1024*1024),
        'gpu_mem_after_mb': gpu_after.get('In use system memory', 0) / (1024*1024),
        'gpu_alloc_before_mb': gpu_before.get('Alloc system memory', 0) / (1024*1024),
        'gpu_alloc_after_mb': gpu_after.get('Alloc system memory', 0) / (1024*1024),
        'bg_sampling': bg_summary,
        'cpu_profile_cumulative': cpu_profile_text,
        'cpu_profile_self_time': cpu_self_time_text,
        'num_segments': len(segments),
        'transcript_chars': sum(len(s.text) for s in segments),
    }
    return result

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "whisper-large-v3-turbo"
    beam_size = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    whisper_path = model_path(model_name)
    audio_file = audio_path("sample.mp3")

    if not os.path.isdir(whisper_path):
        print(f"SKIP: {model_name} not found at {whisper_path}")
        sys.exit(0)

    import librosa
    audio, _ = librosa.load(audio_file, sr=16000, mono=True)
    dur = len(audio) / 16000
    print(f"=" * 70)
    print(f"COMPREHENSIVE METAL PROFILER")
    print(f"Model: {model_name}  beam_size={beam_size}  audio={dur:.0f}s")
    print(f"=" * 70)

    api = bind_metal_api()

    # -----------------------------------------------------------------------
    # Phase 1: CPU baseline
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("PHASE 1: CPU BASELINE")
    print(f"{'='*70}")

    from faster_whisper import WhisperModel

    model_cpu = WhisperModel(whisper_path, device="cpu", compute_type="float32")
    # Warmup
    list(model_cpu.transcribe(audio_file, language="ru", beam_size=beam_size, without_timestamps=True)[0])

    # Timed run
    mem0 = get_process_memory()
    t0 = time.monotonic()
    segs = list(model_cpu.transcribe(audio_file, language="ru", beam_size=beam_size, without_timestamps=True)[0])
    cpu_ms = (time.monotonic() - t0) * 1000
    mem1 = get_process_memory()

    print(f"  Wall time:     {cpu_ms:.0f} ms")
    print(f"  RSS:           {mem0['rss_mb']:.0f} → {mem1['rss_mb']:.0f} MB (delta: {mem1['rss_mb']-mem0['rss_mb']:.0f})")
    print(f"  Segments:      {len(segs)}")

    del model_cpu
    gc.collect()

    # -----------------------------------------------------------------------
    # Phase 2: Metal profiled run
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("PHASE 2: METAL PROFILED RUN")
    print(f"{'='*70}")

    # Enable commit trace
    os.environ['CT2_MPS_TRACE'] = '1'

    model_metal = WhisperModel(whisper_path, device="mps", compute_type="float32")
    # Warmup (also triggers PSO compilation)
    print("  Warmup run...")
    api['reset_pso_stats']()
    list(model_metal.transcribe(audio_file, language="ru", beam_size=beam_size, without_timestamps=True)[0])
    warmup_pso_misses = api['pso_miss_count']()
    warmup_pso_hits = api['pso_hit_count']()
    print(f"  Warmup PSO: {warmup_pso_misses} compilations, {warmup_pso_hits} cache hits")

    # Clean timed run (no cProfile, no background sampling — pure timing)
    print("\n  Clean timed run (no cProfile overhead)...")
    result = profile_transcription(model_metal, audio_file, beam_size, api, "mps",
                                   enable_cprofile=False, enable_bg_sampling=False)

    # cProfile run (separate — to see CPU hotspots without polluting timing)
    print("  cProfile run...")
    cprofile_result = profile_transcription(model_metal, audio_file, beam_size, api, "metal_cprofile",
                                            enable_cprofile=True, enable_bg_sampling=False)

    # Background sampling run (separate — to see GPU utilization)
    print("  GPU utilization sampling run...")
    bg_result = profile_transcription(model_metal, audio_file, beam_size, api, "metal_bg",
                                      enable_cprofile=False, enable_bg_sampling=True)

    # Use bg_result for GPU utilization, cprofile_result for CPU profile, result for timing
    result['bg_sampling'] = bg_result['bg_sampling']
    result['cpu_profile_cumulative'] = cprofile_result['cpu_profile_cumulative']
    result['cpu_profile_self_time'] = cprofile_result['cpu_profile_self_time']

    # -----------------------------------------------------------------------
    # Phase 3: Results
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")

    speedup = cpu_ms / result['wall_ms'] if result['wall_ms'] > 0 else 0

    print(f"\n--- Timing ---")
    print(f"  CPU wall time:           {cpu_ms:.0f} ms")
    print(f"  Metal wall time:         {result['wall_ms']:.0f} ms")
    print(f"  Speedup:                 {speedup:.2f}x")
    print(f"  GPU execution time:      {result['gpu_time_ms']:.0f} ms")
    print(f"  Non-GPU time:            {result['wall_ms'] - result['gpu_time_ms']:.0f} ms")
    print(f"  GPU fraction:            {result['gpu_time_ms']/result['wall_ms']*100:.1f}%")

    print(f"\n--- Sync Analysis ---")
    print(f"  commit_and_wait() calls: {result['commits']}")
    print(f"  Est. sync overhead:      {result['sync_overhead_ms']:.0f} ms ({result['sync_overhead_ms']/result['wall_ms']*100:.1f}% of wall time)")
    if result['commits'] > 0:
        avg_gpu_per_commit = result['gpu_time_ms'] / result['commits']
        print(f"  Avg GPU work per commit: {avg_gpu_per_commit:.2f} ms")

    print(f"\n--- PSO Cache (profiled run) ---")
    print(f"  Cache hits:              {result['pso_hits']}")
    print(f"  Compilations:            {result['pso_misses']}")

    print(f"\n--- Memory ---")
    print(f"  RSS before:              {result['rss_before_mb']:.0f} MB")
    print(f"  RSS after:               {result['rss_after_mb']:.0f} MB")
    print(f"  RSS delta:               {result['rss_delta_mb']:.0f} MB")
    print(f"  GPU mem (in use) before: {result['gpu_mem_before_mb']:.0f} MB")
    print(f"  GPU mem (in use) after:  {result['gpu_mem_after_mb']:.0f} MB")
    print(f"  GPU mem (alloc) before:  {result['gpu_alloc_before_mb']:.0f} MB")
    print(f"  GPU mem (alloc) after:   {result['gpu_alloc_after_mb']:.0f} MB")

    bg = result['bg_sampling']
    if bg:
        print(f"\n--- GPU Utilization (sampled at {1/0.05:.0f} Hz, {bg.get('gpu_util_samples', 0)} samples) ---")
        print(f"  Device Utilization:      avg={bg.get('gpu_util_avg', -1):.1f}%  min={bg.get('gpu_util_min', -1)}%  max={bg.get('gpu_util_max', -1)}%")
        print(f"  RSS range:               {bg.get('rss_min_mb', 0):.0f} - {bg.get('rss_max_mb', 0):.0f} MB")
        print(f"  GPU mem (in use) peak:   {bg.get('gpu_mem_max_mb', 0):.0f} MB")

    # Time breakdown analysis
    print(f"\n--- Time Breakdown Analysis ---")
    gpu_ms = result['gpu_time_ms']
    wall_ms_metal = result['wall_ms']
    non_gpu_ms = wall_ms_metal - gpu_ms
    sync_est_ms = result['sync_overhead_ms']
    cpu_compute_ms = max(0, non_gpu_ms - sync_est_ms)

    print(f"  Total wall time:         {wall_ms_metal:.0f} ms (100%)")
    print(f"  ├─ GPU execution:        {gpu_ms:.0f} ms ({gpu_ms/wall_ms_metal*100:.1f}%)")
    print(f"  ├─ Sync overhead (~):    {sync_est_ms:.0f} ms ({sync_est_ms/wall_ms_metal*100:.1f}%)")
    print(f"  └─ CPU/other:            {cpu_compute_ms:.0f} ms ({cpu_compute_ms/wall_ms_metal*100:.1f}%)")

    if cpu_compute_ms > wall_ms_metal * 0.1:
        print(f"\n  ⚠ CPU/other time is {cpu_compute_ms/wall_ms_metal*100:.1f}% of wall time!")
        print(f"    This includes: ObjC overhead, Python overhead, memory management,")
        print(f"    autorelease pools, MPS object creation, buffer binding, etc.")

    # CPU profile (self-time — shows where CPU is actually spending time)
    print(f"\n--- CPU Profile (top 20 by self-time) ---")
    # Filter to show only interesting lines
    lines = result['cpu_profile_self_time'].strip().split('\n')
    for line in lines[:25]:
        print(f"  {line}")

    print(f"\n--- CPU Profile (top 20 by cumulative time) ---")
    lines = result['cpu_profile_cumulative'].strip().split('\n')
    for line in lines[:25]:
        print(f"  {line}")

    # -----------------------------------------------------------------------
    # Phase 4: Stability check (3 additional runs)
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("PHASE 4: STABILITY CHECK (3 runs)")
    print(f"{'='*70}")

    run_times = [result['wall_ms']]
    run_gpu_times = [result['gpu_time_ms']]
    for i in range(3):
        ri = profile_transcription(model_metal, audio_file, beam_size, api,
                                   f"metal_run{i+2}",
                                   enable_cprofile=False, enable_bg_sampling=False)
        run_times.append(ri['wall_ms'])
        run_gpu_times.append(ri['gpu_time_ms'])
        print(f"  Run {i+2}: wall={ri['wall_ms']:.0f}ms  gpu={ri['gpu_time_ms']:.0f}ms  commits={ri['commits']}  rss_delta={ri['rss_delta_mb']:.0f}MB")

    avg_wall = sum(run_times) / len(run_times)
    avg_gpu = sum(run_gpu_times) / len(run_gpu_times)
    min_wall = min(run_times)
    max_wall = max(run_times)
    print(f"\n  Wall time: avg={avg_wall:.0f}ms  min={min_wall:.0f}ms  max={max_wall:.0f}ms  spread={max_wall-min_wall:.0f}ms ({(max_wall-min_wall)/avg_wall*100:.1f}%)")
    print(f"  GPU time:  avg={avg_gpu:.0f}ms  min={min(run_gpu_times):.0f}ms  max={max(run_gpu_times):.0f}ms")
    if max_wall > min_wall * 1.3:
        print(f"\n  ⚠ HIGH VARIANCE ({(max_wall-min_wall)/avg_wall*100:.0f}%) — possible thermal throttling or memory pressure!")

    # Cleanup
    del model_metal
    gc.collect()
    import ctranslate2; ctranslate2.clear_device_cache("mps")

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
