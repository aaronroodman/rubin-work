#!/usr/bin/env python3
"""Probe the RSP terminal's CPU/thread allocation and whether >1 thread helps.

Answers two questions:
  1. Does multi-threading even work here (the numexpr 'NUMEXPR_MAX_THREADS (1)'
     error), and
  2. Do you actually get more than one *usable* core (JupyterHub often pins a
     low cgroup CPU quota even if many cores are visible), so we know whether
     `snakemake -j N` buys anything.

Run in the RSP terminal:
    python code/check_threads.py
"""
import os
import time

# Record what the environment set BEFORE we touch anything.
_THREAD_VARS = ('NUMEXPR_MAX_THREADS', 'NUMEXPR_NUM_THREADS', 'OMP_NUM_THREADS',
                'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')
_ORIG = {v: os.environ.get(v) for v in _THREAD_VARS}

# Force a high numexpr cap (override JupyterHub's =1) so set_num_threads() can
# be tested — this is the knob that fixes the
# 'nthreads cannot be larger than NUMEXPR_MAX_THREADS (1)' error.
os.environ['NUMEXPR_MAX_THREADS'] = '16'


def report_cpu():
    print("=== CPU / core detection ===")
    print(f"  os.cpu_count():                 {os.cpu_count()}")
    try:
        print(f"  sched_getaffinity (schedulable): {len(os.sched_getaffinity(0))}")
    except AttributeError:
        print("  sched_getaffinity:               n/a (non-Linux)")
    # cgroup CPU quota — the real limit JupyterHub usually imposes.
    try:
        txt = open('/sys/fs/cgroup/cpu.max').read().split()      # cgroup v2
        q = ('unlimited' if txt[0] == 'max'
             else f"~{int(txt[0]) / int(txt[1]):.2f} cores")
        print(f"  cgroup v2 cpu.max:               {' '.join(txt)}  -> {q}")
    except FileNotFoundError:
        try:
            qv = int(open('/sys/fs/cgroup/cpu/cpu.cfs_quota_us').read())
            pv = int(open('/sys/fs/cgroup/cpu/cpu.cfs_period_us').read())
            q = 'unlimited' if qv < 0 else f"~{qv / pv:.2f} cores"
            print(f"  cgroup v1 quota/period:          {qv}/{pv} -> {q}")
        except Exception:
            print("  cgroup quota:                    n/a")


def report_env():
    print("\n=== thread env vars (as JupyterHub set them) ===")
    for v in _THREAD_VARS:
        print(f"  {v:22s} = {_ORIG[v]!r}")


def test_numexpr():
    print("\n=== numexpr: does set_num_threads(>1) work + speed up? ===")
    try:
        import numpy as np
        import numexpr as ne
    except Exception as e:
        print(f"  numexpr unavailable: {type(e).__name__}: {e}")
        return
    print(f"  numexpr.detect_number_of_cores(): {ne.detect_number_of_cores()}")
    a = np.random.rand(40_000_000)
    for nt in (1, 4):
        try:
            ne.set_num_threads(nt)
        except Exception as e:
            print(f"  set_num_threads({nt}) FAILED: {type(e).__name__}: {e}")
            continue
        t = time.perf_counter()
        for _ in range(6):
            ne.evaluate('sin(a) * cos(a) + a**2')
        print(f"  {nt} thread(s): {time.perf_counter() - t:.3f}s")


def _cpu_chunk(n):
    s = 0.0
    for i in range(n):
        s += (i ** 0.5)
    return s


def test_process_parallel(n_tasks=8, chunk=6_000_000):
    print("\n=== CPU-bound speedup: fixed work split across N processes ===")
    from concurrent.futures import ProcessPoolExecutor
    base = None
    for nw in (1, 2, 4):
        t = time.perf_counter()
        with ProcessPoolExecutor(max_workers=nw) as ex:
            list(ex.map(_cpu_chunk, [chunk] * n_tasks))
        dt = time.perf_counter() - t
        base = base or dt
        print(f"  max_workers={nw}: {dt:5.2f}s   speedup x{base / dt:.2f}")
    print("  -> speedup ~N means N usable cores; ~1.0 (flat) means a "
          "single-core/quota-limited allocation")


def test_blas():
    print("\n=== numpy / BLAS matmul (auto-threaded) ===")
    import numpy as np
    A = np.random.rand(2500, 2500)
    B = np.random.rand(2500, 2500)
    A @ B  # warm up
    t = time.perf_counter()
    for _ in range(3):
        A @ B
    print(f"  3x 2500^2 matmul: {time.perf_counter() - t:.3f}s")
    try:
        from threadpoolctl import threadpool_info
        for p in threadpool_info():
            print(f"  BLAS pool: {p.get('internal_api')} "
                  f"num_threads={p.get('num_threads')}")
    except Exception:
        print("  (threadpoolctl not available — BLAS thread count unknown)")


if __name__ == '__main__':
    report_cpu()
    report_env()
    test_numexpr()
    test_blas()
    test_process_parallel()
    print("\nDone.")
