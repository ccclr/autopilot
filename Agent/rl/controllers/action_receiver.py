#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import socketserver
import sys
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

RL_ROOT = Path(__file__).resolve().parent.parent
if str(RL_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_ROOT))

from parameters import ParameterApplier  # noqa: E402
from controllers.action_transport import (  # noqa: E402
    MAX_MESSAGE_BYTES,
    PROTOCOL_VERSION,
)

logger = logging.getLogger(__name__)


class ActionReceiver:
    """Validate, de-duplicate, and apply node0 decisions on one local node."""

    def __init__(self, node_index: int, applier: ParameterApplier) -> None:
        self.node_index = int(node_index)
        self.applier = applier
        self._lock = threading.Lock()
        self._successful: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._decision_by_epoch: dict[int, str] = {}
        self._last_signal_epoch = -1
        self._cache_limit = 1024

    def handle(self, message: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(message, dict):
            return self._response(
                ok=False,
                status="rejected",
                decision_id="",
                signal_epoch=-1,
                error="message must be a JSON object",
            )
        decision_id = str(message.get("decision_id", "")).strip()
        try:
            self._validate_envelope(message, decision_id)
            signal_epoch = int(message["signal_epoch"])
            action_id = int(message["action_id"])
            params = message["params"]
        except Exception as exc:
            return self._response(
                ok=False,
                status="rejected",
                decision_id=decision_id,
                signal_epoch=message.get("signal_epoch", -1),
                error=str(exc),
            )

        with self._lock:
            cached = self._successful.get(decision_id)
            if cached is not None:
                logger.info(
                    "DQN_ACTION_DUPLICATE node=%d decision=%s epoch=%d action=%d",
                    self.node_index,
                    decision_id,
                    signal_epoch,
                    action_id,
                )
                return dict(cached)

            existing_decision = self._decision_by_epoch.get(signal_epoch)
            if existing_decision is not None and existing_decision != decision_id:
                return self._response(
                    ok=False,
                    status="conflict",
                    decision_id=decision_id,
                    signal_epoch=signal_epoch,
                    error=(
                        f"epoch {signal_epoch} already accepted decision "
                        f"{existing_decision}"
                    ),
                )
            if signal_epoch < self._last_signal_epoch:
                return self._response(
                    ok=False,
                    status="stale",
                    decision_id=decision_id,
                    signal_epoch=signal_epoch,
                    error=(
                        f"signal epoch {signal_epoch} is older than "
                        f"{self._last_signal_epoch}"
                    ),
                )

            try:
                self.applier.apply(params, signal_epoch)
            except Exception as exc:
                logger.exception(
                    "DQN_ACTION_APPLY_FAILED node=%d decision=%s epoch=%d",
                    self.node_index,
                    decision_id,
                    signal_epoch,
                )
                return self._response(
                    ok=False,
                    status="apply_failed",
                    decision_id=decision_id,
                    signal_epoch=signal_epoch,
                    error=str(exc),
                )

            response = self._response(
                ok=True,
                status="signalled",
                decision_id=decision_id,
                signal_epoch=signal_epoch,
                action_id=action_id,
            )
            self._successful[decision_id] = response
            self._successful.move_to_end(decision_id)
            while len(self._successful) > self._cache_limit:
                self._successful.popitem(last=False)
            self._decision_by_epoch[signal_epoch] = decision_id
            self._last_signal_epoch = max(self._last_signal_epoch, signal_epoch)
            # Only a small recent epoch map is needed for conflict protection.
            stale_epochs = [
                epoch
                for epoch in self._decision_by_epoch
                if epoch < self._last_signal_epoch - self._cache_limit
            ]
            for epoch in stale_epochs:
                self._decision_by_epoch.pop(epoch, None)

            logger.info(
                "DQN_ACTION_SIGNALLED node=%d decision=%s epoch=%d action=%d params=%s",
                self.node_index,
                decision_id,
                signal_epoch,
                action_id,
                params,
            )
            return response

    def _validate_envelope(
        self, message: dict[str, Any], decision_id: str
    ) -> None:
        if not isinstance(message, dict):
            raise ValueError("message must be a JSON object")
        if message.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError("unsupported action protocol version")
        if message.get("type") != "apply_action":
            raise ValueError("unsupported action message type")
        if not decision_id or len(decision_id) > 256:
            raise ValueError("decision_id must contain 1..256 characters")
        signal_epoch = message.get("signal_epoch")
        action_id = message.get("action_id")
        if isinstance(signal_epoch, bool) or int(signal_epoch) < 0:
            raise ValueError("signal_epoch must be non-negative")
        if isinstance(action_id, bool) or int(action_id) < 0:
            raise ValueError("action_id must be non-negative")
        if not isinstance(message.get("params"), dict):
            raise ValueError("params must be a JSON object")
        ParameterApplier.validate_params(message["params"])

    def _response(
        self,
        *,
        ok: bool,
        status: str,
        decision_id: str,
        signal_epoch: Any,
        action_id: int | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        response: dict[str, Any] = {
            "protocol_version": PROTOCOL_VERSION,
            "node_index": self.node_index,
            "ok": bool(ok),
            "status": status,
            "decision_id": decision_id,
            "signal_epoch": signal_epoch,
        }
        if action_id is not None:
            response["action_id"] = int(action_id)
        if error:
            response["error"] = error
        return response


class ThreadingActionServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, server_address, receiver: ActionReceiver):
        self.receiver = receiver
        super().__init__(server_address, ActionRequestHandler)


class ActionRequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        raw = self.rfile.readline(MAX_MESSAGE_BYTES + 1)
        # CloudLab readiness probes only establish and close a TCP connection.
        if not raw:
            return
        if len(raw) > MAX_MESSAGE_BYTES:
            response = {
                "protocol_version": PROTOCOL_VERSION,
                "node_index": self.server.receiver.node_index,
                "ok": False,
                "status": "rejected",
                "decision_id": "",
                "signal_epoch": -1,
                "error": "message too large",
            }
        else:
            try:
                message = json.loads(raw.decode("utf-8"))
                response = self.server.receiver.handle(message)
            except Exception as exc:
                response = {
                    "protocol_version": PROTOCOL_VERSION,
                    "node_index": self.server.receiver.node_index,
                    "ok": False,
                    "status": "rejected",
                    "decision_id": "",
                    "signal_epoch": -1,
                    "error": str(exc),
                }
        self.wfile.write(
            (json.dumps(response, separators=(",", ":")) + "\n").encode("utf-8")
        )
        self.wfile.flush()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply centralized node0 DQN actions on one Autobahn node"
    )
    parser.add_argument("--node-index", type=int, required=True)
    parser.add_argument("--parameters-file", required=True)
    parser.add_argument("--bind-host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=19100)
    parser.add_argument("--socket-attempts", type=int, default=10)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    applier = ParameterApplier(
        parameters_file=args.parameters_file,
        node_index=args.node_index,
        socket_attempts=args.socket_attempts,
    )
    receiver = ActionReceiver(args.node_index, applier)
    with ThreadingActionServer((args.bind_host, args.port), receiver) as server:
        logger.info(
            "DQN_ACTION_RECEIVER_READY node=%d address=%s:%d parameters=%s socket=%s",
            args.node_index,
            args.bind_host,
            args.port,
            args.parameters_file,
            applier.socket_path,
        )
        server.serve_forever(poll_interval=0.2)


if __name__ == "__main__":
    main()
