from fabric import Connection
from pathlib import Path
from shlex import quote

try:
    # Package execution: python -m autopilot.simulate_latency.deploy_latency
    from .latency_config import NODES
except ImportError:
    # Direct execution: python deploy_latency.py
    from latency_config import NODES


USER = "AlanXiao"
KEY_FILENAME = "/users/AlanXiao/.ssh/cloudlab"
SCRIPT_DIR = Path(__file__).resolve().parent


def deploy(node):

    conn = Connection(
        host=NODES[node],
        user=USER,
        connect_kwargs={
            "key_filename": KEY_FILENAME,
        },
    )


    script = SCRIPT_DIR / f"{node}_tc.sh"


    try:
        # A shared fixed filename may belong to another user from an earlier run.
        remote_script = conn.run(
            "mktemp /tmp/autopilot-tc-XXXXXXXX.sh", hide=True
        ).stdout.strip()
        if not remote_script:
            raise RuntimeError("Remote mktemp returned an empty path")
        remote_path = quote(remote_script)
        try:
            conn.put(str(script), remote_script)
            conn.run(f"chmod 700 {remote_path}")
            conn.run(f"sudo bash {remote_path}")
        finally:
            conn.run(f"rm -f -- {remote_path}", warn=True)
    finally:
        conn.close()


    print(
        f"{node} configured"
    )



def main():
    for node in NODES:
        deploy(node)


if __name__ == "__main__":
    main()
