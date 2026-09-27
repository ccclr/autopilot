from __future__ import annotations

import json
import logging
import socket
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Iterable

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = 1
MAX_MESSAGE_BYTES = 64 * 1024


@dataclass(frozen=True)
class ActionEndpoint:
    node_index: int
    host: str
    port: int

    @classmethod
    def parse(cls, value: str) -> "ActionEndpoint":
        """Parse ``NODE@HOST:PORT`` used by the benchmark command line."""
        try:
            node_text, address = value.split("@", 1)
            host, port_text = address.rsplit(":", 1)
            endpoint = cls(int(node_text), host.strip(), int(port_text))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid DQN action endpoint {value!r}; expected NODE@HOST:PORT"
            ) from exc
        if endpoint.node_index < 0 or not endpoint.host or endpoint.port <= 0:
            raise ValueError(f"invalid DQN action endpoint: {value!r}")
        return endpoint

    def format(self) -> str:
        return f"{self.node_index}@{self.host}:{self.port}"


@dataclass(frozen=True)
class ActionAck:
    node_index: int
    ok: bool
    status: str
    decision_id: str
    signal_epoch: int
    error: str | None = None
    attempts: int = 1


@dataclass(frozen=True)
class BroadcastResult:
    decision_id: str
    signal_epoch: int
    acknowledgements: tuple[ActionAck, ...]

    @property
    def success(self) -> bool:
        return bool(self.acknowledgements) and all(
            acknowledgement.ok for acknowledgement in self.acknowledgements
        )

    @property
    def failed_nodes(self) -> list[int]:
        return [ack.node_index for ack in self.acknowledgements if not ack.ok]


class ActionBroadcaster:
    """Send one node0 decision to every node concurrently and collect ACKs."""

    def __init__(
        self,
        endpoints: Iterable[ActionEndpoint],
        timeout: float = 2.0,
        retries: int = 2,
        retry_delay: float = 0.1,
    ) -> None:
        endpoints = tuple(sorted(endpoints, key=lambda item: item.node_index))
        if not endpoints:
            raise ValueError("at least one DQN action endpoint is required")
        node_indices = [endpoint.node_index for endpoint in endpoints]
        if len(set(node_indices)) != len(node_indices):
            raise ValueError(f"duplicate DQN endpoint node indices: {node_indices}")
        if timeout <= 0:
            raise ValueError("action broadcast timeout must be positive")
        if retries < 0:
            raise ValueError("action broadcast retries must be non-negative")
        self.endpoints = endpoints
        self.timeout = float(timeout)
        self.retries = int(retries)
        self.retry_delay = max(0.0, float(retry_delay))

    @classmethod
    def from_csv(
        cls,
        value: str,
        timeout: float = 2.0,
        retries: int = 2,
    ) -> "ActionBroadcaster":
        entries = [entry.strip() for entry in value.split(",") if entry.strip()]
        return cls(
            [ActionEndpoint.parse(entry) for entry in entries],
            timeout=timeout,
            retries=retries,
        )

    def broadcast(
        self,
        *,
        decision_id: str,
        signal_epoch: int,
        action_id: int,
        arm: str,
        params: dict[str, Any],
    ) -> BroadcastResult:
        message = {
            "protocol_version": PROTOCOL_VERSION,
            "type": "apply_action",
            "decision_id": str(decision_id),
            "signal_epoch": int(signal_epoch),
            "action_id": int(action_id),
            "arm": str(arm),
            "params": params,
        }
        acknowledgements: list[ActionAck] = []
        with ThreadPoolExecutor(max_workers=len(self.endpoints)) as executor:
            futures = {
                executor.submit(self._send_one, endpoint, message): endpoint
                for endpoint in self.endpoints
            }
            for future in as_completed(futures):
                endpoint = futures[future]
                try:
                    acknowledgements.append(future.result())
                except Exception as exc:  # Defensive: always report every node.
                    acknowledgements.append(
                        ActionAck(
                            node_index=endpoint.node_index,
                            ok=False,
                            status="client_error",
                            decision_id=str(decision_id),
                            signal_epoch=int(signal_epoch),
                            error=str(exc),
                            attempts=self.retries + 1,
                        )
                    )
        acknowledgements.sort(key=lambda item: item.node_index)
        result = BroadcastResult(
            decision_id=str(decision_id),
            signal_epoch=int(signal_epoch),
            acknowledgements=tuple(acknowledgements),
        )
        logger.info(
            "DQN_ACTION_BROADCAST decision=%s epoch=%d success=%s acks=%s",
            decision_id,
            signal_epoch,
            result.success,
            [ack.__dict__ for ack in result.acknowledgements],
        )
        return result

    def _send_one(
        self, endpoint: ActionEndpoint, message: dict[str, Any]
    ) -> ActionAck:
        encoded = (json.dumps(message, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        if len(encoded) > MAX_MESSAGE_BYTES:
            raise ValueError("DQN action message exceeds maximum size")

        last_error: Exception | None = None
        for attempt in range(1, self.retries + 2):
            try:
                with socket.create_connection(
                    (endpoint.host, endpoint.port), timeout=self.timeout
                ) as client:
                    client.settimeout(self.timeout)
                    client.sendall(encoded)
                    response = self._receive_line(client)
                data = json.loads(response.decode("utf-8"))
                if not isinstance(data, dict):
                    raise ValueError("DQN action ACK must be a JSON object")
                ack_node = int(data.get("node_index", -1))
                ack_decision = str(data.get("decision_id", ""))
                ack_epoch = int(data.get("signal_epoch", -1))
                if (
                    ack_node != endpoint.node_index
                    or ack_decision != str(message["decision_id"])
                    or ack_epoch != int(message["signal_epoch"])
                ):
                    raise ValueError(
                        "mismatched DQN action ACK: "
                        f"node={ack_node} decision={ack_decision!r} "
                        f"epoch={ack_epoch}"
                    )
                ok = bool(data.get("ok", False))
                return ActionAck(
                    node_index=endpoint.node_index,
                    ok=ok,
                    status=str(data.get("status", "invalid_ack")),
                    decision_id=ack_decision,
                    signal_epoch=ack_epoch,
                    error=str(data["error"]) if data.get("error") else None,
                    attempts=attempt,
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt <= self.retries:
                    time.sleep(self.retry_delay * attempt)

        return ActionAck(
            node_index=endpoint.node_index,
            ok=False,
            status="unreachable",
            decision_id=str(message["decision_id"]),
            signal_epoch=int(message["signal_epoch"]),
            error=str(last_error),
            attempts=self.retries + 1,
        )

    @staticmethod
    def _receive_line(client: socket.socket) -> bytes:
        chunks = bytearray()
        while len(chunks) <= MAX_MESSAGE_BYTES:
            chunk = client.recv(4096)
            if not chunk:
                break
            chunks.extend(chunk)
            newline = chunks.find(b"\n")
            if newline >= 0:
                return bytes(chunks[:newline])
        if len(chunks) > MAX_MESSAGE_BYTES:
            raise ValueError("DQN action ACK exceeds maximum size")
        if not chunks:
            raise ValueError("empty DQN action ACK")
        return bytes(chunks)
