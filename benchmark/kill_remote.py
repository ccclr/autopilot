#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import os
import sys
from pathlib import Path

from fabric import Connection, ThreadingGroup as Group
from fabric.exceptions import GroupException
from paramiko import RSAKey
from paramiko.ssh_exception import PasswordRequiredException, SSHException

sys.path.append(os.path.join(os.path.dirname(__file__), "benchmark"))

from benchmark.gcp_instance import InstanceManager
from benchmark.utils import BenchError

# Runs on each host. Match /proc cmdline, never pkill -f (that kills the SSH
# wrapper because the pattern appears in the remote command line).
_REMOTE_KILL_PY = r"""
import os
import signal

self_pid = os.getpid()
ppid = os.getppid()
needles = (
    "controllers/controller.py",
    "train_cmab_continuous.py",
    "train_xgboost.py",
    "train_gp_bo.py",
    "train_kernel_ucb.py",
    "metrics_collector.py",
    "benchmark_client",
)
killed = []
for pid in os.listdir("/proc"):
    if not pid.isdigit():
        continue
    ipid = int(pid)
    if ipid in (self_pid, ppid):
        continue
    try:
        raw = open(f"/proc/{pid}/cmdline", "rb").read()
    except OSError:
        continue
    cmdline = raw.replace(b"\x00", b" ").decode("utf-8", "replace")
    hit = any(n in cmdline for n in needles)
    if (not hit) and ("--keys" in cmdline) and (
        "./node" in cmdline or "/node " in cmdline or cmdline.lstrip().startswith("node ")
    ):
        hit = True
    if not hit:
        continue
    try:
        os.kill(ipid, signal.SIGKILL)
        killed.append(ipid)
    except OSError:
        pass
print("killed_pids", killed)
"""


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _load_ssh_connect_kwargs(settings) -> dict:
    try:
        password = getattr(settings, "ssh_key_password", None) or os.environ.get("SSH_KEY_PASSWORD")
        if password:
            pkey = RSAKey.from_private_key_file(settings.key_path, password=password)
        else:
            pkey = RSAKey.from_private_key_file(settings.key_path)
        return {"pkey": pkey}
    except (IOError, PasswordRequiredException, SSHException) as e:
        raise RuntimeError(f"Failed to load SSH key {settings.key_path}: {e}") from e


def _remote_kill_command() -> str:
    payload = base64.b64encode(_REMOTE_KILL_PY.encode()).decode()
    return (
        f"echo {payload} | base64 -d | python3 ; "
        "tmux kill-server >/dev/null 2>&1 || true"
    )


def kill_all(settings_path: Path) -> None:
    try:
        manager = InstanceManager.make(str(settings_path))
    except BenchError as e:
        raise RuntimeError(f"Failed to load settings {settings_path}: {e}") from e

    settings = manager.settings
    hosts = manager.hosts(flat=True)
    if not hosts:
        raise RuntimeError(f"No running instances for {settings_path}")

    connect = _load_ssh_connect_kwargs(settings)
    cmd = _remote_kill_command()

    print(f"Killing remote processes on {len(hosts)} host(s): {', '.join(hosts)}")

    try:
        g = Group(*hosts, user=settings.username, connect_kwargs=connect)
        results = g.run(cmd, hide=True, warn=True)
    except GroupException as e:
        raise RuntimeError(f"SSH group failure: {e}") from e

    for host, result in results.items():
        name = host.host if isinstance(host, Connection) else str(host)
        out = ((result.stdout or "") + (result.stderr or "")).strip()
        if result.ok:
            print(f"  OK  {name} {out}")
        else:
            print(f"  FAIL {name}: exit={result.exited} {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Kill autopilot processes on all remote hosts")
    parser.add_argument(
        "--settings",
        default=str(_script_dir() / "settings.json"),
        help="Path to GCP settings JSON (default: ./settings.json)",
    )
    args = parser.parse_args()

    try:
        kill_all(Path(args.settings))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
