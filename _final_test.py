"""Final verification of the SSE fix."""
import sys, os, subprocess, time, json, socket

# Use the current Python interpreter (must be run in the auto_tune conda env)
conda_python = sys.executable
project_root = os.path.dirname(os.path.abspath(__file__))

# Start server
server_proc = subprocess.Popen(
    [conda_python, "-m", "auto_tune.main"],
    cwd=project_root,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
time.sleep(6)

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
if sock.connect_ex(('127.0.0.1', 8000)) != 0:
    print("Server failed to start!")
    sys.exit(1)
sock.close()

import requests

ok = True

# Test dry_run SSE
print("1. Dry-run SSE test...", end=" ")
r = requests.post(
    "http://127.0.0.1:8000/tuning/start",
    json={"mode": "dry_run", "max_retries": 1, "reference_run": ""},
    timeout=30, stream=True,
)
events = []
for line in r.iter_lines(decode_unicode=True):
    if line and line.startswith("data: "):
        events.append(json.loads(line[6:]))

print(f"{len(events)} events received")
has_running = any(e.get("status") == "running" for e in events)
has_done = any(e.get("status") == "done" for e in events)

print(f"   Events by status:")
for e in events:
    print(f"      [{e['status']}] {e.get('message','')[:60]}")
if has_running and has_done:
    print(f"   PASS")
else:
    print(f"   FAIL: running={has_running}, done={has_done}")
    ok = False

server_proc.terminate()
server_proc.wait(timeout=5)

print(f"\n{'ALL PASS' if ok else 'SOME FAILED'}")
