#!/usr/bin/env python3
"""Fix leftover-owner permissions so `fab remote` / FPT sweep can boot.

Logs, repos, and /tmp sockets may be owned by another user. This user then
cannot overwrite logs or delete sticky-bit sockets in /tmp.

Usage (from benchmark/):
  python3 fix_permissions.py
  python3 fix_permissions.py --settings settings.json
"""

from __future__ import annotations

import argparse
import os
import shlex
import stat
import sys
from pathlib import Path

from fabric import Connection, ThreadingGroup as Group
from fabric.exceptions import GroupException
from paramiko import RSAKey
from paramiko.ssh_exception import PasswordRequiredException, SSHException

sys.path.append(os.path.join(os.path.dirname(__file__), "benchmark"))

from benchmark.gcp_instance import InstanceManager
from benchmark.utils import BenchError

# Applied on every host. HOME_DIR is injected by the Python wrapper.
_REMOTE_FIX = r"""
set -euo pipefail
HOME_DIR="${HOME_DIR:-/home/ccclr0302}"

fix_git_safe() {
  local path
  for path in "$HOME_DIR/autopilot" "$HOME_DIR/autopilot-test" "*"; do
    sudo git config --system --get-all safe.directory 2>/dev/null | grep -Fxq "$path" \
      || sudo git config --system --add safe.directory "$path"
  done
}

chmod_if_exists() {
  local p
  for p in "$@"; do
    if [ -e "$p" ]; then
      sudo chmod -R g+w "$p"
    fi
  done
}

mkdir -p "$HOME_DIR/logs" "$HOME_DIR/results"
sudo chmod -R g+w "$HOME_DIR/logs" "$HOME_DIR/results" || true

chmod_if_exists \
  "$HOME_DIR/autopilot" \
  "$HOME_DIR/autopilot-test" \
  "$HOME_DIR/.cargo" \
  "$HOME_DIR/.rustup" \
  "$HOME_DIR/autopilot-venv" \
  "$HOME_DIR/autopilot-control-venv"

shopt -s nullglob
chmod_if_exists "$HOME_DIR"/.db-* "$HOME_DIR"/metrics-*
shopt -u nullglob

# /tmp is sticky: only the owner (or root) can unlink leftover sockets/scripts.
sudo rm -f \
  /tmp/autopilot_rl_param_*.sock \
  /tmp/autopilot_rl_param_abandon_*.signal \
  /tmp/autopilot_core_*.sock \
  /tmp/autopilot_controller_*.sock \
  /tmp/tc_latency.sh

fix_git_safe

# Prove this user can tee primary logs (boot path: ./node |& tee $HOME_DIR/logs/...).
touch "$HOME_DIR/logs/.perm_check" && rm -f "$HOME_DIR/logs/.perm_check"

echo "logs_writable=yes git_safe=yes sockets_cleared=yes"
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


def _tighten_ssh_key(key_path: str) -> None:
    path = Path(key_path)
    if not path.is_file():
        print(f"  skip ssh key (missing): {key_path}")
        return
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        path.chmod(0o600)
        print(f"  chmod 600 {key_path} (was {mode:o})")
    else:
        print(f"  ssh key already 600: {key_path}")


def fix_all(settings_path: Path) -> None:
    try:
        manager = InstanceManager.make(str(settings_path))
    except BenchError as e:
        raise RuntimeError(f"Failed to load settings {settings_path}: {e}") from e

    settings = manager.settings
    hosts = manager.hosts(flat=True)
    if not hosts:
        raise RuntimeError(f"No running instances for {settings_path}")

    print(f"Tightening local SSH key {settings.key_path}")
    _tighten_ssh_key(settings.key_path)

    connect = _load_ssh_connect_kwargs(settings)
    home = settings.home.rstrip("/")
    remote = f"HOME_DIR={shlex.quote(home)} bash -c {shlex.quote(_REMOTE_FIX)}"

    print(f"Fixing permissions on {len(hosts)} host(s): {', '.join(hosts)}")
    try:
        g = Group(*hosts, user=settings.username, connect_kwargs=connect)
        results = g.run(remote, hide=True, warn=True)
    except GroupException as e:
        raise RuntimeError(f"SSH group failure: {e}") from e

    failed = False
    for host, result in results.items():
        name = host.host if isinstance(host, Connection) else str(host)
        out = ((result.stdout or "") + (result.stderr or "")).strip()
        if result.ok:
            print(f"  OK  {name} {out}")
        else:
            failed = True
            print(f"  FAIL {name}: exit={result.exited} {out}")
    if failed:
        raise RuntimeError("Permission fix failed on at least one host")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fix leftover file/socket permissions on all GCP hosts"
    )
    parser.add_argument(
        "--settings",
        default=str(_script_dir() / "settings.json"),
        help="Path to GCP settings JSON (default: ./settings.json)",
    )
    args = parser.parse_args()

    try:
        fix_all(Path(args.settings))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
