import time

import pytest
from benchdas.benchmarking import Benchmarker, CustomProfiler, RuntimeProfiler

def _custom_profile_fn(func, fargs, fkwargs, cleanup):
    start = time.perf_counter()
    func(*fargs, **fkwargs)
    end = time.perf_counter()
    cleanup()
    return end - start

@pytest.mark.parametrize("profilers", [
    [RuntimeProfiler(results_key="runtime1"), RuntimeProfiler(results_key="runtime2")],
    [CustomProfiler(_custom_profile_fn, results_key="custom1"), CustomProfiler(_custom_profile_fn, results_key="custom2")]
])
def test_benchmarker(profilers):
    benchmarker = Benchmarker(profilers)
    benchmarker.run(lambda: time.sleep(.001), variables={"sleep": 1})
    benchmarker.run(lambda: time.sleep(.002), variables={"sleep": 2})
    benchmarker.run(lambda: time.sleep(.003), variables={"sleep": 3})

    assert benchmarker.get_results().shape == (3, len(profilers) + 2)