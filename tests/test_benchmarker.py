import time
from typing import List

import pytest
from benchdas.benchmarking import Benchmarker, CustomProfiler, Profiler, RuntimeProfiler

def _custom_profile_fn(func, fargs, fkwargs, setup, cleanup):
    setup()
    start = time.perf_counter()
    func(*fargs, **fkwargs)
    end = time.perf_counter()
    cleanup()
    return end - start

@pytest.mark.parametrize("profilers", [
    [RuntimeProfiler(results_key="runtime1"), RuntimeProfiler(results_key="runtime2")],
    [CustomProfiler(_custom_profile_fn, results_key="custom1"), CustomProfiler(_custom_profile_fn, results_key="custom2")]
])
def test_benchmarker(profilers: List[Profiler]):
    benchmarker = Benchmarker(profilers)

    setup_tracker = 0
    cleanup_tracker = 0
    def _setup():
        nonlocal setup_tracker
        setup_tracker += 1
    def _cleanup():
        nonlocal cleanup_tracker
        cleanup_tracker += 1
    benchmarker.run(lambda: time.sleep(.001), setup=_setup, cleanup=_cleanup, variables={"sleep": 1})

    n_calls = sum([profiler._warmup + profiler._repeat for profiler in profilers])
    assert setup_tracker == n_calls
    assert cleanup_tracker == n_calls

    benchmarker.run(lambda: time.sleep(.002), variables={"sleep": 2})
    benchmarker.run(lambda: time.sleep(.003), variables={"sleep": 3})

    assert benchmarker.get_results().shape == (3, len(profilers) + 2)
