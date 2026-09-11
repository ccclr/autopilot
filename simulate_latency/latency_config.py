# latency_config.py

NODES = {
    "node0": "10.10.1.1",
    "node1": "10.10.1.2",
    "node2": "10.10.1.3",
    "node3": "10.10.1.4",
}

LATENCY = {
    "node0": {
        "node1": 25,
        "node2": 40,
        "node3": 35,
    },

    "node1": {
        "node0": 25,
        "node2": 40,
        "node3": 50,
    },

    "node2": {
        "node0": 40,
        "node1": 40,
        "node3": 20,
    },

    "node3": {
        "node0": 35,
        "node1": 50,
        "node2": 20,
    },
}
