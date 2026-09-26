#!/usr/bin/env python3
"""从远程节点下载 benchmark 日志（对齐 remote.py 的 _logs 逻辑）。

用法（在 benchmark 目录下）:
  python3 download_log.py
  python3 download_log.py --settings settings.json --faults 0
  python3 download_log.py --extra --parse
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections import OrderedDict
from json import JSONDecodeError, load
from pathlib import Path

from fabric import Connection
from paramiko import RSAKey
from paramiko.ssh_exception import PasswordRequiredException, SSHException

# 与 analyze_logs.py 一样，保证可导入 benchmark.*
sys.path.append(os.path.join(os.path.dirname(__file__), "benchmark"))

from benchmark.settings import Settings, SettingsError
from benchmark.commands import CommandMaker
from benchmark.config import Committee, Key
from benchmark.logs import LogParser, ParseError
from benchmark.utils import BenchError, PathMaker, Print, progress_bar


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _load_ssh_connect_kwargs(settings: Settings) -> dict:
    try:
        password = getattr(settings, "ssh_key_password", None) or os.environ.get("SSH_KEY_PASSWORD")
        if password:
            pkey = RSAKey.from_private_key_file(settings.key_path, password=password)
        else:
            pkey = RSAKey.from_private_key_file(settings.key_path)
        return {"pkey": pkey}
    except (IOError, PasswordRequiredException, SSHException) as e:
        raise BenchError("Failed to load SSH key", e)


def _load_committee(committee_file: Path, key_dir: Path) -> Committee:
    """从 .committee.json 加载，并尽量按 .node-i.json 恢复运行时节点顺序。"""
    try:
        with open(committee_file, "r", encoding="utf-8") as f:
            data = load(f)
    except (OSError, JSONDecodeError) as e:
        raise BenchError(f"Failed to load committee file {committee_file}", e)

    authorities = data.get("authorities")
    if not isinstance(authorities, dict) or not authorities:
        raise BenchError(
            "Invalid committee file",
            ValueError("missing authorities"),
        )

    # committee.print(..., sort_keys=True) 会打乱顺序；用 key 文件还原 index。
    ordered = OrderedDict()
    i = 0
    while True:
        # PathMaker.key_file returns '.node-{i}.json'
        key_path = key_dir / PathMaker.key_file(i)
        if not key_path.exists():
            break
        name = Key.from_file(str(key_path)).name
        if name not in authorities:
            raise BenchError(
                "Committee/key mismatch",
                KeyError(f"{name} from {key_path.name} not in committee"),
            )
        ordered[name] = authorities[name]
        i += 1

    if not ordered:
        # 回退：按 primary 端口排序（与 Committee 构造时的端口递增一致）
        Print.warn("No .node-*.json found; ordering authorities by primary port")
        items = sorted(
            authorities.items(),
            key=lambda kv: int(
                kv[1]["primary"]["primary_to_primary"].rsplit(":", 1)[1]
            ),
        )
        ordered = OrderedDict(items)

    if len(ordered) != len(authorities):
        missing = set(authorities) - set(ordered)
        Print.warn(f"Committee has authorities not covered by key files: {missing}")

    committee = object.__new__(Committee)
    committee.json = {"authorities": ordered}
    return committee


def _safe_get(conn: Connection, remote: str, local: str) -> bool:
    """下载单个文件；远端不存在时跳过并返回 False。"""
    local_path = Path(local)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # 先探测，避免 Fabric 对缺失文件抛一长串异常。
        result = conn.run(f"test -f {remote}", hide=True, warn=True)
        if result.exited != 0:
            Print.warn(f"Missing remote file {conn.host}:{remote}")
            return False
        conn.get(remote, local=local)
        return True
    except Exception as e:
        Print.warn(f"Failed to get {conn.host}:{remote} -> {local}: {e}")
        return False


def download_logs(
    settings: Settings,
    connect_kwargs: dict,
    committee: Committee,
    faults: int = 0,
    clean_local: bool = True,
    extra: bool = False,
) -> str:
    """对齐 Bench._logs 的下载逻辑。日志位于 settings.home。"""
    CommandMaker.set_home(settings.home)

    if clean_local:
        cmd = CommandMaker.clean_logs()
        subprocess.run([cmd], shell=True, stderr=subprocess.DEVNULL)

    Path(PathMaker.logs_path()).mkdir(parents=True, exist_ok=True)

    workers_addresses = committee.workers_addresses(faults)
    progress = progress_bar(workers_addresses, prefix="Downloading workers logs:")
    for i, addresses in enumerate(progress):
        for worker_id, address in addresses:
            worker_id = int(worker_id)
            host = Committee.ip(address)
            conn = Connection(
                host, user=settings.username, connect_kwargs=connect_kwargs
            )
            with conn:
                _safe_get(
                    conn,
                    f"{settings.home}/{PathMaker.client_log_file(i, worker_id)}",
                    PathMaker.client_log_file(i, worker_id),
                )
                _safe_get(
                    conn,
                    f"{settings.home}/{PathMaker.worker_log_file(i, worker_id)}",
                    PathMaker.worker_log_file(i, worker_id),
                )

    primary_addresses = committee.primary_addresses(faults)
    progress = progress_bar(primary_addresses, prefix="Downloading primaries logs:")
    for i, address in enumerate(progress):
        host = Committee.ip(address)
        conn = Connection(host, user=settings.username, connect_kwargs=connect_kwargs)
        with conn:
            _safe_get(
                conn,
                f"{settings.home}/{PathMaker.primary_log_file(i)}",
                PathMaker.primary_log_file(i),
            )
            if extra:
                for remote in (
                    f"logs/controller-{i}.log",
                    f"logs/metrics_collector-{i}.log",
                    f"logs/continuous_training-{i}.log",
                    f"logs/metrics-{i}.log",
                ):
                    _safe_get(conn, f"{settings.home}/{remote}", remote)

                # 同步 metrics-{i}/ 目录（若存在，位于 settings.home 下）
                metrics_dir = f"metrics-{i}"
                result = conn.run(
                    f"test -d {settings.home}/{metrics_dir}", hide=True, warn=True
                )
                if result.exited == 0:
                    local_metrics = Path(metrics_dir)
                    local_metrics.mkdir(parents=True, exist_ok=True)
                    # 用 tar 流更稳，避免逐文件 get。
                    try:
                        conn.run(
                            f"tar czf /tmp/{metrics_dir}.tgz -C {settings.home} {metrics_dir}",
                            hide=True,
                        )
                        local_tgz = Path(f"/tmp/{metrics_dir}.tgz")
                        conn.get(f"/tmp/{metrics_dir}.tgz", local=str(local_tgz))
                        subprocess.run(
                            ["tar", "xzf", str(local_tgz), "-C", "."],
                            check=True,
                        )
                        local_tgz.unlink(missing_ok=True)
                        conn.run(f"rm -f /tmp/{metrics_dir}.tgz", hide=True, warn=True)
                        Print.info(f"Downloaded {host}:{metrics_dir}/")
                    except Exception as e:
                        Print.warn(f"Failed to download {host}:{metrics_dir}/: {e}")

    return PathMaker.logs_path()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从远程 GCP 节点下载 primary/worker/client 日志"
    )
    parser.add_argument(
        "--settings",
        default=str(_script_dir() / "settings.json"),
        help="GCP settings 路径，默认 ./settings.json",
    )
    parser.add_argument(
        "--committee",
        default=str(_script_dir() / PathMaker.committee_file()),
        help="committee 文件路径，默认 ./.committee.json",
    )
    parser.add_argument(
        "--faults",
        type=int,
        default=0,
        help="故障节点数（与 bench 参数一致），默认 0",
    )
    parser.add_argument(
        "--keep-local",
        action="store_true",
        help="不清空本地 logs/（默认会先 clean_logs）",
    )
    parser.add_argument(
        "--extra",
        action="store_true",
        help="额外下载 controller/metrics_collector/continuous_training/metrics 日志与 metrics-i/",
    )
    parser.add_argument(
        "--parse",
        action="store_true",
        help="下载后用 LogParser 解析并打印 SUMMARY",
    )
    parser.add_argument(
        "--workdir",
        default=str(_script_dir()),
        help="工作目录（应能看到 .committee.json / logs/），默认脚本所在目录",
    )
    args = parser.parse_args()

    workdir = Path(args.workdir).resolve()
    os.chdir(workdir)

    try:
        settings = Settings.load(args.settings)
        connect_kwargs = _load_ssh_connect_kwargs(settings)
        committee = _load_committee(Path(args.committee), workdir)

        Print.heading("Downloading remote logs")
        Print.info(f"settings={args.settings}")
        Print.info(f"committee={args.committee}")
        Print.info(f"nodes={committee.size()} faults={args.faults} workdir={workdir}")

        logs_dir = download_logs(
            settings=settings,
            connect_kwargs=connect_kwargs,
            committee=committee,
            faults=args.faults,
            clean_local=not args.keep_local,
            extra=args.extra,
        )
        Print.info(f"Logs saved under: {workdir / logs_dir}")

        if args.parse:
            Print.info("Parsing logs and computing performance...")
            result = LogParser.process(logs_dir, faults=args.faults)
            print(result.result())
    except SettingsError as e:
        Print.error(BenchError("Failed to load GCP settings", e))
        sys.exit(1)
    except ParseError as e:
        Print.error(BenchError("Failed to parse logs", e))
        sys.exit(1)
    except BenchError as e:
        Print.error(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
