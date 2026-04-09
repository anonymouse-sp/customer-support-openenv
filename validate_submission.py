import json
import subprocess
import sys
import time
from pathlib import Path

import httpx
import yaml

from app import graders


ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
BASE_URL = "http://127.0.0.1:7860"


def check_file(path: Path) -> tuple[bool, str]:
    if path.exists():
        return True, f"PASS file exists: {path.name}"
    return False, f"FAIL missing file: {path.name}"


def check_openenv_yaml() -> tuple[bool, str]:
    yaml_path = ROOT / "openenv.yaml"
    if not yaml_path.exists():
        return False, "FAIL openenv.yaml missing"

    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    required_keys = ["name", "version", "entrypoint", "models", "tasks"]
    missing = [k for k in required_keys if k not in data]
    if missing:
        return False, f"FAIL openenv.yaml missing keys: {missing}"

    task_count = len(data.get("tasks", []))
    if task_count < 3:
        return False, f"FAIL openenv.yaml has {task_count} tasks (<3)"

    return True, f"PASS openenv.yaml valid with {task_count} tasks"


def start_server() -> subprocess.Popen:
    return subprocess.Popen(
        [str(PYTHON), "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "7860"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def wait_server_ready(timeout_sec: int = 30) -> bool:
    deadline = time.time() + timeout_sec
    with httpx.Client() as client:
        while time.time() < deadline:
            try:
                resp = client.get(f"{BASE_URL}/", timeout=2)
                if resp.status_code == 200:
                    return True
            except Exception:
                pass
            time.sleep(0.5)
    return False


def validate_endpoints() -> tuple[bool, str]:
    with httpx.Client() as client:
        root = client.get(f"{BASE_URL}/", timeout=10)
        if root.status_code != 200:
            return False, f"FAIL root status {root.status_code}"

        tasks_resp = client.get(f"{BASE_URL}/tasks", timeout=10)
        tasks_resp.raise_for_status()
        tasks = tasks_resp.json().get("tasks", [])
        if len(tasks) < 3:
            return False, f"FAIL tasks count {len(tasks)} (<3)"

        for task_id in tasks:
            reset = client.post(f"{BASE_URL}/reset", json={"task_id": task_id}, timeout=10)
            step = client.post(
                f"{BASE_URL}/step",
                json={"action": "I am sorry for the issue. I will help resolve it. Please share your order ID or reference so we can proceed."},
                timeout=10,
            )
            if reset.status_code != 200 or step.status_code != 200:
                return False, f"FAIL task endpoint call for {task_id}"

            reward = step.json().get("reward")
            if not isinstance(reward, (float, int)):
                return False, f"FAIL reward type invalid for {task_id}"
            if not (0.0 < float(reward) < 1.0):
                return False, f"FAIL reward must be strictly between 0 and 1 for {task_id}: {reward}"

    return True, "PASS endpoints/reset/step and reward range checks"


def validate_task_graders() -> tuple[bool, str]:
    yaml_path = ROOT / "openenv.yaml"
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    tasks = data.get("tasks", [])
    if not tasks:
        return False, "FAIL no tasks found in openenv.yaml"

    for task in tasks:
        task_id = task["id"]
        grader_name = task["grader"].split(":")[-1]
        grader = getattr(graders, grader_name, None)
        if grader is None:
            return False, f"FAIL missing grader function for {task_id}: {grader_name}"

        samples = [
            "I am sorry about this. I will help fix it and review the issue right away. Please share your order ID or reference number so I can proceed with the next steps.",
            "",
            {"action": "Sorry for the inconvenience. I will investigate this and help resolve it. Please send the relevant reference details."},
            {"messages": [{"role": "assistant", "content": "I am sorry for the issue. I will help resolve this and review the next steps. Please share your order or transaction reference."}]},
        ]

        for sample in samples:
            try:
                score = float(grader(sample))
            except Exception as exc:
                return False, f"FAIL grader runtime error for {task_id}: {exc}"
            if not (0.0 < score < 1.0):
                return False, f"FAIL grader score must be strictly between 0 and 1 for {task_id}: {score}"

    return True, "PASS task graders return strict unit-interval scores"


def validate_inference_runtime(max_sec: int = 1200) -> tuple[bool, str]:
    start = time.time()
    proc = subprocess.run(
        [str(PYTHON), "inference.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=max_sec,
    )
    elapsed = time.time() - start

    if proc.returncode != 0:
        return False, f"FAIL inference.py exit {proc.returncode}\n{proc.stdout}\n{proc.stderr}"

    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    has_start = any(ln.startswith("[START]") for ln in lines)
    has_step = any(ln.startswith("[STEP]") for ln in lines)
    has_end = any(ln.startswith("[END]") for ln in lines)
    if not (has_start and has_step and has_end):
        return False, "FAIL inference logging format missing START/STEP/END"

    return True, f"PASS inference runtime {elapsed:.2f}s with START/STEP/END logs"


def main() -> None:
    checks = []

    for rel in ["inference.py", "Dockerfile", "README.md", "openenv.yaml"]:
        checks.append(check_file(ROOT / rel))

    checks.append(check_openenv_yaml())

    server = start_server()
    try:
        if not wait_server_ready():
            checks.append((False, "FAIL local server did not start in time"))
        else:
            checks.append(validate_endpoints())
            checks.append(validate_task_graders())
            checks.append(validate_inference_runtime())
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()

    passed = all(ok for ok, _ in checks)
    report = {
        "passed": passed,
        "results": [{"ok": ok, "message": msg} for ok, msg in checks],
    }
    print(json.dumps(report, indent=2))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
