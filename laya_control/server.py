"""Local Unix RPC with immutable model/schema state and lifetime socket ownership."""

import argparse
import json
import os
import signal
import socket
import socketserver
import subprocess
import time
from pathlib import Path

from .jobs import artifact_hashes, identity, lock_file
from .store import ControlError, absolute, atomic_json

LIMIT = 1024 * 1024


def rpc(path, request, timeout=1):
    with socket.socket(socket.AF_UNIX) as s:
        s.settimeout(timeout)
        s.connect(str(path))
        s.sendall((json.dumps(request) + "\n").encode())
        with s.makefile("rb") as f:
            data = f.readline(LIMIT + 1)
        if not data.endswith(b"\n") or len(data) > LIMIT:
            raise ControlError("invalid RPC response")
        return json.loads(data)


def load_agent(release):
    if artifact_hashes(Path(release["model_path"])) != release["artifacts"]:
        raise ControlError("release artifact mismatch")
    if release["runtime"] == "mlx":
        import laya_mlx

        return laya_mlx.load(release["model_path"])
    if release["runtime"] == "pytorch":
        import laya

        return laya.load(release["model_path"], device="cpu")
    raise ControlError("unsupported serving runtime")


def serve(release, path, loader=load_agent):
    path = absolute(path)
    if len(os.fsencode(path)) > 100:
        raise ControlError("Unix socket path too long; select a shorter state root")
    with lock_file(Path(str(path) + ".lock")) as fd:
        if fd is None:
            raise ControlError("server already running or starting")
        # A legacy/unmanaged listener may not use our lock: never unlink an answering socket.
        if path.exists():
            try:
                with socket.socket(socket.AF_UNIX) as probe:
                    probe.settimeout(0.3)
                    probe.connect(str(path))
            except (ConnectionRefusedError, FileNotFoundError):
                path.unlink(missing_ok=True)
            else:
                raise ControlError("socket already owned by a listener")
        agent = loader(release)

        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                self.connection.settimeout(5)
                try:
                    line = self.rfile.readline(LIMIT + 1)
                    if not line.endswith(b"\n") or len(line) > LIMIT:
                        raise ControlError("request too large")
                    req = json.loads(line)
                    if req.get("kind") == "health":
                        answer = {k: release[k] for k in ("release_id", "generation", "snapshot_id")}
                        answer["ready"] = True
                    elif req.get("kind") == "predict" and isinstance(req.get("state"), str):
                        answer = agent.predict(req["state"], release["schema"])
                    else:
                        raise ControlError("unknown request")
                except Exception as exc:  # noqa: BLE001 - isolate one private RPC failure
                    answer = {"error": type(exc).__name__}
                self.wfile.write((json.dumps(answer) + "\n").encode())

        with socketserver.UnixStreamServer(str(path), Handler) as srv:
            os.chmod(path, 0o600)
            atomic_json(
                Path(str(path) + ".owner.json"), {"pid": os.getpid(), "identity": identity(os.getpid())}
            )
            try:
                srv.serve_forever(poll_interval=0.1)
            finally:
                path.unlink(missing_ok=True)


class ManagedService:
    def __init__(self, socket_path, python):
        self.path = absolute(socket_path)
        self.python = absolute(python)

    def stop(self):
        owner = Path(str(self.path) + ".owner.json")
        if not owner.exists():
            if self.path.exists():
                raise ControlError("unmanaged socket; cannot stop")
            return
        info = json.loads(owner.read_text())
        pid = info["pid"]
        actual = identity(pid)
        if actual is None:
            return
        if actual != info["identity"]:
            raise ControlError("server PID identity changed")
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + 10
        while identity(pid) == actual and time.monotonic() < deadline:
            time.sleep(0.05)
        if identity(pid) == actual:
            raise ControlError("server did not stop")

    def switch(self, pointer):
        self.stop()
        release = Path(pointer["model_path"]).parent / "release.json"
        log = Path(str(self.path) + ".log")
        log.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with log.open("ab") as out:
            process = subprocess.Popen(
                [
                    str(self.python),
                    "-m",
                    "laya_control.server",
                    "--release",
                    str(release),
                    "--socket",
                    str(self.path),
                ],
                stdout=out,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=dict(os.environ, HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1"),
            )
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise ControlError("server exited during startup")
            try:
                h = rpc(self.path, {"kind": "health"}, 0.3)
                if (
                    h.get("release_id") == pointer["release_id"]
                    and h.get("generation") == pointer["generation"]
                ):
                    return h
            except (OSError, ValueError):
                pass
            time.sleep(0.1)
        # Only our still-owned child is stopped on a timeout.
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise ControlError("server startup timed out")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", required=True)
    parser.add_argument("--socket", required=True)
    args = parser.parse_args()
    serve(json.loads(absolute(args.release).read_text()), absolute(args.socket))


if __name__ == "__main__":
    main()
