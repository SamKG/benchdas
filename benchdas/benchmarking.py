from abc import abstractmethod
import functools
import time
from typing import Any, Dict, List, Callable, NewType
import numpy as np
import pandas as pd


NO_OP = lambda: True

def pcall_or_value(func, fallback, *args, **kwargs):
    """Call a function with the given arguments or return None if it fails"""
    try:
        return func(*args, **kwargs)
    except:
        return fallback 

def call_func(func, *args, **kwargs):
    """Call a function with the given arguments and return the result"""
    return func(*args, **kwargs)



RUN_ID_KEY = "PROFILER_RUN_ID"
class Profiler:
    """Base class for all profilers"""

    def __init__(self, warmup=3, repeat=5, catch_exceptions=False, reduction_func=np.mean, results_key="Result"):
        self._results: Dict[str, Any] = []
        self._run_id: int = 0
        self._warmup: int = warmup
        self._repeat: int = repeat
        self._catch_exceptions: bool = catch_exceptions
        self._reduction_func = reduction_func
        self._results_key = results_key

    def _log(self, result, variables: Dict[str, Any], run_id: int):
        """Log the result and variables of a function call"""
        self._results.append({self._results_key: result, **variables, RUN_ID_KEY: run_id})

    @abstractmethod
    def _run(self, func, fargs=[], fkwargs={}, setup=NO_OP, cleanup=NO_OP):
        pass

    def run(self, func, fargs=[], fkwargs={}, setup=NO_OP, cleanup=NO_OP, variables: Dict[str, Any]={}):
        """Run the function and log the metric"""
        for _ in range(self._warmup):
            self._run(func, fargs, fkwargs, setup, cleanup)

        for _ in range(self._repeat):
            value = self._run(func, fargs, fkwargs, setup, cleanup)
            self._log(value, variables, self._run_id)

        self._run_id += 1
    
    def get_results(self):
        """Return the results as a multi-index DataFrame"""
        df = pd.DataFrame(self._results)

        # apply reduction function to results based on run_id groups to dataframe
        # do not discard the variables, as they should all be the same within a run_id
        df = df.groupby([RUN_ID_KEY], dropna=False).agg({self._results_key: self._reduction_func, **{k: "first" for k in df.columns.difference([self._results_key])}})

        # manually make the multi-index
        variables = df.columns.difference([self._results_key]).to_list()
        df.index = pd.MultiIndex.from_arrays([df[v] for v in variables], names=variables)
        return df


class RuntimeProfiler(Profiler):
    """Log the runtime of a function call"""

    def __init__(self, warmup=3, repeat=5, catch_exceptions=False, reduction_func=np.mean, results_key="Runtime"):
        super().__init__(warmup=warmup, repeat=repeat, catch_exceptions=catch_exceptions, reduction_func=reduction_func, results_key=results_key)
    
    def _run(self, func, fargs=[], fkwargs={}, setup=NO_OP, cleanup=NO_OP):
        def _loop():
            setup()
            start = time.perf_counter()
            func(*fargs, **fkwargs)
            end = time.perf_counter()
            cleanup()
            return end - start
        _runner = functools.partial(pcall_or_value, fallback=float("nan")) if self._catch_exceptions else call_func 
        return _runner(_loop)

ProfileFn = Callable[[Callable, List[Any], Dict[str, Any], Callable, Callable], Any]

class CustomProfiler(Profiler):
    def __init__(self, profile_fn: ProfileFn, warmup=3, repeat=5, catch_exceptions=False, reduction_func=np.mean, results_key="Custom"):
        super().__init__(warmup=warmup, repeat=repeat, catch_exceptions=catch_exceptions, reduction_func=reduction_func, results_key=results_key)
        self._profile_fn = profile_fn
    
    def _run(self, func, fargs=[], fkwargs={}, setup=NO_OP, cleanup=NO_OP):
        def _loop():
            val = self._profile_fn(func, fargs, fkwargs, setup=setup, cleanup=cleanup)
            return val
        _runner = functools.partial(pcall_or_value, fallback=float("nan")) if self._catch_exceptions else call_func 
        return _runner(_loop)

class Benchmarker:
    def __init__(self, profilers: List[Profiler] = [RuntimeProfiler()]):
        self._profilers = profilers

    def run(self, func, fargs=[], fkwargs={}, setup=NO_OP, cleanup=NO_OP, variables: Dict[str, Any]={}):
        for profiler in self._profilers:
            profiler.run(func, fargs, fkwargs, setup, cleanup, variables)
    
    def get_results(self) -> pd.DataFrame:
        """Return the results as a multi-index DataFrame"""
        df = pd.concat([profiler.get_results() for profiler in self._profilers], axis=1)
        # remove duplicate columns (normally, variables are counted multiple times across profilers)
        df = df.loc[:,~df.columns.duplicated()]
        return df


if __name__ == "__main__":
    # test the profiler
    benchmarker = Benchmarker([RuntimeProfiler(results_key="runtime1"), RuntimeProfiler(results_key="runtime2")])
    benchmarker.run(lambda: time.sleep(.001), variables={"sleep": 1})
    benchmarker.run(lambda: time.sleep(.002), variables={"sleep": 2})
    benchmarker.run(lambda: time.sleep(.003), variables={"sleep": 3})
    print(benchmarker.get_results())