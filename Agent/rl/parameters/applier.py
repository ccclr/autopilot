from __future__ import annotations

import fcntl
import json
import os
import socket
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping


class ParameterApplyError(RuntimeError):
    """Raised when an action cannot be written or signalled to the local core."""


class ParameterApplier:
    """Atomically apply one Autobahn action on the local node.

    The Rust core owns a node-local Unix socket.  Updating the JSON file alone
    is not sufficient: the epoch signal is what makes the core capture and
    schedule the new values for ``applied_begin``.
    """

    REQUIRED_KEYS = (
        "batch_size",
        "header_size",
        "cut_condition_type",
        "fast_path_timeout",
        "k",
    )

    def __init__(
        self,
        parameters_file: str,
        node_index: int,
        socket_attempts: int = 10,
        socket_retry_delay: float = 0.2,
    ) -> None:
        if node_index < 0:
            raise ValueError("node_index must be non-negative")
        if socket_attempts <= 0:
            raise ValueError("socket_attempts must be positive")
        if socket_retry_delay < 0:
            raise ValueError("socket_retry_delay must be non-negative")
        self.parameters_file = Path(parameters_file)
        self.node_index = int(node_index)
        self.socket_attempts = int(socket_attempts)
        self.socket_retry_delay = float(socket_retry_delay)
        self.lock_file = self.parameters_file.with_suffix(
            self.parameters_file.suffix + ".lock"
        )

    @property
    def socket_path(self) -> str:
        return f"/tmp/autopilot_rl_param_{self.node_index}.sock"

    @classmethod
    def validate_params(cls, params: Mapping[str, Any]) -> dict[str, int]:
        missing = [key for key in cls.REQUIRED_KEYS if key not in params]
        if missing:
            raise ValueError(f"missing action parameters: {missing}")

        normalized: dict[str, int] = {}
        for key in cls.REQUIRED_KEYS:
            value = params[key]
            if isinstance(value, bool):
                raise ValueError(f"{key} must be an integer, not bool")
            try:
                normalized[key] = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{key} must be an integer: {value!r}") from exc

        positive = ("batch_size", "header_size", "cut_condition_type", "k")
        for key in positive:
            if normalized[key] <= 0:
                raise ValueError(f"{key} must be positive")
        if normalized["fast_path_timeout"] < 0:
            raise ValueError("fast_path_timeout must be non-negative")
        return normalized

    def apply(self, params: Mapping[str, Any], signal_epoch: int) -> None:
        if isinstance(signal_epoch, bool) or int(signal_epoch) < 0:
            raise ValueError("signal_epoch must be a non-negative integer")
        signal_epoch = int(signal_epoch)
        normalized = self.validate_params(params)

        self.parameters_file.parent.mkdir(parents=True, exist_ok=True)
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: str | None = None

        try:
            with open(self.lock_file, "a+", encoding="utf-8") as lock_handle:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                current = self._read_current_parameters()
                current.update(normalized)
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    suffix=".json",
                    delete=False,
                    dir=str(self.parameters_file.parent),
                ) as temporary:
                    json.dump(current, temporary, indent=2, sort_keys=True)
                    temporary.write("\n")
                    temporary.flush()
                    os.fsync(temporary.fileno())
                    temporary_path = temporary.name
                os.replace(temporary_path, self.parameters_file)
                temporary_path = None
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        except Exception as exc:
            if temporary_path:
                try:
                    os.unlink(temporary_path)
                except OSError:
                    pass
            raise ParameterApplyError(
                f"failed to update {self.parameters_file}: {exc}"
            ) from exc

        self._signal_core(signal_epoch)

    def _read_current_parameters(self) -> dict[str, Any]:
        try:
            with open(self.parameters_file, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if not isinstance(data, dict):
                raise ValueError("parameter file root must be a JSON object")
            return data
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError as exc:
            raise ParameterApplyError(
                f"invalid JSON in {self.parameters_file}: {exc}"
            ) from exc

    def _signal_core(self, signal_epoch: int) -> None:
        payload = json.dumps({"epoch": signal_epoch}) + "\n"
        last_error: OSError | None = None
        for attempt in range(1, self.socket_attempts + 1):
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                client.settimeout(2.0)
                client.connect(self.socket_path)
                client.sendall(payload.encode("utf-8"))
                return
            except OSError as exc:
                last_error = exc
                if attempt < self.socket_attempts:
                    time.sleep(min(self.socket_retry_delay * attempt, 1.0))
            finally:
                client.close()

        raise ParameterApplyError(
            f"failed to signal local core socket {self.socket_path} after "
            f"{self.socket_attempts} attempts: {last_error}"
        )
