"""Real POSIX lock ownership regression for the candidate build runner."""
import errno
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

POSIX_TOOLS = os.name == "posix" and shutil.which("bash") and shutil.which("flock")
if os.name == "posix":
    import fcntl

RUNNER = Path(__file__).resolve().parents[1] / "validation/validate-muse-spark-final-candidate.sh"


def runner_function():
    matches = re.findall(r"(?ms)^run\(\) \{\n.*?^\}", RUNNER.read_text())
    if len(matches) != 1:
        raise AssertionError("Expected exactly one production run() function")
    function = matches[0]
    if function.count('"$@" 9>&-') != 1:
        raise AssertionError("Production child invocation must close lock descriptor 9")
    return function


def wait_until(predicate, description, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("Timed out waiting for " + description)


def lock_available(path):
    with path.open("a") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        return True


HELPER = r'''
import json, os, pathlib, subprocess, sys, time
root = pathlib.Path(sys.argv[1])
def publish_json(name, value):
    target = root / name
    temporary = root / (name + ".tmp")
    temporary.write_text(json.dumps(value))
    temporary.replace(target)
def has_lock_fd():
    try:
        os.fstat(9)
        return True
    except OSError as ex:
        if ex.errno != 9: raise
        return False
if len(sys.argv) == 2:
    inherited = has_lock_fd()
    child = subprocess.Popen([sys.executable, __file__, str(root), "child"],
                             close_fds=False, stdin=subprocess.DEVNULL)
    publish_json("helper.json", {"fd9_open": inherited, "child_pid": child.pid})
else:
    # Self-terminating and stopped only by this test's private marker. No process
    # enumeration or signalling of PIDs read from files is required for cleanup.
    publish_json("child.json", {"fd9_open": has_lock_fd(), "pid": os.getpid()})
    deadline = time.monotonic() + 20
    while not (root / "stop-child").exists() and time.monotonic() < deadline:
        time.sleep(.02)
    (root / "child-exiting").touch()
'''


@unittest.skipUnless(POSIX_TOOLS, "Requires POSIX bash/flock descriptor inheritance; unavailable on Windows.")
class BuildLockTests(unittest.TestCase):
    def exercise(self, original):
        function = runner_function()
        if original:
            # The exact old invocation, with every other production line intact.
            function = function.replace('"$@" 9>&-', '"$@"')
        with tempfile.TemporaryDirectory(prefix="muse-spark-build-lock-") as directory:
            root = Path(directory)
            helper = root / "helper.py"
            helper.write_text(HELPER)
            script = root / "parent.sh"
            script.write_text("\n".join([
                "set -euo pipefail", 'proof=$1', 'python=$2', 'helper=$3',
                'idle() { :; }', function,
                'exec 9>"$proof/build.lock"', 'flock -n 9',
                'run child.log "$python" "$helper" "$proof"',
                'touch "$proof/parent-ready"',
                'while [[ ! -e "$proof/release-parent" ]]; do sleep .02; done', "",
            ]))
            parent = subprocess.Popen([shutil.which("bash"), str(script), str(root), sys.executable, str(helper)],
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                wait_until(lambda: (root / "parent-ready").exists() and (root / "child.json").exists(), "parent and child readiness")
                helper_state = json.loads((root / "helper.json").read_text())
                child_state = json.loads((root / "child.json").read_text())
                self.assertEqual(helper_state["child_pid"], child_state["pid"])
                self.assertEqual(helper_state["fd9_open"], original)
                self.assertEqual(child_state["fd9_open"], original)
                self.assertIsNone(parent.poll(), "Owner must still be alive")
                self.assertFalse(lock_available(root / "build.lock"), "Parent must retain its exclusive lock")
                (root / "release-parent").touch()
                self.assertEqual(parent.wait(timeout=4), 0)
                self.assertFalse((root / "child-exiting").exists(), "Separate child must outlive the owner")
                os.kill(child_state["pid"], 0)  # Read-only existence check, never used to terminate a PID.
                available_after_parent = lock_available(root / "build.lock")
                self.assertEqual(available_after_parent, not original)
                (root / "stop-child").touch()
                wait_until(lambda: (root / "child-exiting").exists() and lock_available(root / "build.lock"), "child exit and lock release")
                print(json.dumps({"mode": "original-negative-control" if original else "fixed-production-function",
                                  "helper_fd9_open": helper_state["fd9_open"], "child_fd9_open": child_state["fd9_open"],
                                  "parent_held_lock": True, "lock_free_after_parent_while_child_alive": available_after_parent}), flush=True)
            finally:
                (root / "release-parent").touch()
                (root / "stop-child").touch()
                try:
                    parent.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    # Only the Popen handle we created, never an unrelated process.
                    parent.terminate()
                    try:
                        parent.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        parent.kill()
                        parent.wait(timeout=2)
                if (root / "helper.json").exists():
                    wait_until(lambda: (root / "child-exiting").exists(), "bounded helper cleanup", timeout=4)
                if parent.stdout:
                    parent.stdout.close()
                if parent.stderr:
                    parent.stderr.close()

    def test_production_child_cannot_retain_owner_lock(self):
        self.exercise(original=False)

    def test_original_invocation_negative_control_retains_lock(self):
        self.exercise(original=True)


if __name__ == "__main__":
    unittest.main()
