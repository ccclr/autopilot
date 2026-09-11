from fabric import Connection
from pathlib import Path

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


    conn.put(
        str(script),
        "/tmp/tc_latency.sh"
    )


    conn.run(
        "chmod +x /tmp/tc_latency.sh"
    )


    conn.run(
        "sudo /tmp/tc_latency.sh"
    )


    print(
        f"{node} configured"
    )



def main():
    for node in NODES:
        deploy(node)


if __name__ == "__main__":
    main()
