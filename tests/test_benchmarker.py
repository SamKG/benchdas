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

ERR_MSG = "This is an error"
def _custom_profile_fn_err(func, fargs, fkwargs, setup, cleanup):
    setup()
    func(*fargs, **fkwargs)
    cleanup()
    raise ValueError(ERR_MSG)

@pytest.mark.parametrize("profilers", [
    [RuntimeProfiler(results_key="runtime1"), RuntimeProfiler(results_key="runtime2")],
    [CustomProfiler(_custom_profile_fn, results_key="custom1", catch_exceptions=False), CustomProfiler(_custom_profile_fn, results_key="custom2", catch_exceptions=False)]
])
def test_profilers(profilers: List[Profiler]):
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


def test_error_catch():
    benchmarker = Benchmarker([CustomProfiler(_custom_profile_fn_err, results_key="custom1", catch_exceptions=True)])

    benchmarker.run(lambda: time.sleep(.001), variables={"sleep": 1})

    assert benchmarker.get_results().shape == (1, 3)

    benchmarker.run(lambda: time.sleep("a"), variables={"sleep": 1})

    # disable error catching
    benchmarker = Benchmarker([CustomProfiler(_custom_profile_fn_err, results_key="custom1", catch_exceptions=False)])

    error_message = None
    try:
        benchmarker.run(lambda: time.sleep(.002), variables={"sleep": 2})
    except ValueError as e:
        error_message = str(e)
    
    assert error_message == ERR_MSG