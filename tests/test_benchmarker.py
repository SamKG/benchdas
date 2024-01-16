import time
from benchdas.benchmarking import Benchmarker, RuntimeProfiler


def test_benchmarker():
    benchmarker = Benchmarker([RuntimeProfiler(results_key="runtime1"), RuntimeProfiler(results_key="runtime2")])
    benchmarker.run(lambda: time.sleep(.001), variables={"sleep": 1})
    benchmarker.run(lambda: time.sleep(.002), variables={"sleep": 2})
    benchmarker.run(lambda: time.sleep(.003), variables={"sleep": 3})
    print(benchmarker.get_results())