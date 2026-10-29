import json
import os
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]
LIMITS = ROOT / "robot_sim/scripts/resource_limits.sh"


class ResourceTests(unittest.TestCase):
    def test_one_cpu_and_inherited_thread_limits(self):
        environment = os.environ.copy()
        environment.pop("ROBOT_SIM_CPU", None)
        environment.update(OMP_NUM_THREADS="40", LP_NUM_THREADS="40", OPENBLAS_NUM_THREADS="40")
        probe = "import os,json; print(json.dumps([sorted(os.sched_getaffinity(0)), [os.environ[name] for name in (\"OMP_NUM_THREADS\", \"LP_NUM_THREADS\", \"OPENBLAS_NUM_THREADS\")]]))"
        completed = subprocess.run(["bash", "-c", 'source "$1" && python3 -c "$2"', "bash", str(LIMITS), probe],
                                   env=environment, text=True, capture_output=True, check=True)
        affinity, threads = json.loads(completed.stdout)
        self.assertEqual(affinity, [min(os.sched_getaffinity(0))])
        self.assertEqual(threads, ["1", "1", "1"])

    def test_cpu_lock_prevents_descendant_affinity_expansion(self):
        environment = os.environ.copy()
        environment["ROBOT_SIM_CPU"] = str(min(os.sched_getaffinity(0)))
        probe = """import errno,os
expected = {int(os.environ["ROBOT_SIM_CPU"])}
assert os.sched_getaffinity(0) == expected
try:
    os.sched_setaffinity(0, expected)
except OSError as error:
    assert error.errno == errno.EPERM
else:
    raise AssertionError("affinity changes were not blocked")
"""
        completed = subprocess.run(["python3", "-m", "robot_sim.resource_limits", "python3", "-c", probe],
                                   cwd=ROOT, env=environment, text=True, capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_multiple_cpus_rejected(self):
        for selection in ("0,1", "0-3", "all"):
            environment = os.environ.copy()
            environment["ROBOT_SIM_CPU"] = selection
            completed = subprocess.run(["bash", "-c", 'source "$1"', "bash", str(LIMITS)],
                                       env=environment, text=True, capture_output=True)
            self.assertEqual(completed.returncode, 2)

    def test_multiple_llm_gpus_rejected_before_model_start(self):
        environment = os.environ.copy()
        environment.pop("ROBOT_SIM_CPU", None)
        environment["ROBOT_LLM_GPU"] = "0,1"
        completed = subprocess.run(["bash", str(ROOT / "robot_sim/scripts/run_llm.sh")],
                                   env=environment, text=True, capture_output=True)
        self.assertEqual(completed.returncode, 2)
        self.assertIn("exactly one GPU", completed.stderr)


if __name__ == "__main__":
    unittest.main()
