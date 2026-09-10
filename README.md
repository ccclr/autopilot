# AutoPilot

AutoPilot integrates the **Autobahn Byzantine Fault Tolerant (BFT) consensus protocol** with **online Contextual Multi-Armed Bandit (CMAB) learning** for adaptive consensus parameter tuning. The system combines Rust-based consensus execution with Python-based metrics collection and intelligent parameter optimization.

The consensus layer is based on [Autobahn](https://github.com/neilgiri/autobahn-artifact). For benchmark deployment details inherited from Narwhal/Bullshark, see [benchmark/README.md](benchmark/README.md).

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Key Components](#key-components)
4. [Project Layout](#project-layout)
5. [Installation & Setup](#installation--setup)
6. [Running Experiments](#running-experiments)
7. [Consensus Protocol Details](#consensus-protocol-details)
8. [Learning & Adaptation (CMAB)](#learning--adaptation-cmab)
9. [License](#license)

## Overview

AutoPilot enables **adaptive parameter choosing optimization** through a tight coupling of Byzantine Fault Tolerant consensus with online reinforcement learning. Rather than using static and fixed parameter values, AutoPilot continuously observes system performance metrics and uses a CMAB algorithm to select optimal consensus parameters at runtime.

### Key Features
- **Autobahn Consensus**: High-performance BFT protocol with configurable fast-path optimization
- **Online CMAB Learning**: Random Forest Thompson Sampling policy that adapts parameters every epoch

## Architecture

The system comprises two main layers that interact through Unix sockets:

```
┌─────────────────────────────────────────────────────────────────┐
│ Consensus Layer (Rust)                                          │
├─────────────────────────────────────────────────────────────────┤
│ • worker: batches incoming transactions into mini-batches       │
│ • primary: maintains DAG, runs Autobahn consensus (core.rs)     │
│ • core: event-driven consensus engine with fast/slow paths      │
│ • coordination: aggregate state reports robustly per epoch      |
└─────────────────────────────────────────────────────────────────┘
              ↓ Unix socket (metrics, state)
┌─────────────────────────────────────────────────────────────────┐
│ Learning & Control Layer (Python)                               │
├─────────────────────────────────────────────────────────────────┤
| • metrics_collector: parses logs, aggregates StateReports       │
│ • CMAB policy: selects consensus parameter arms                 │
│ • parameter encoder: maps discrete arms to parameter settings   │
│ • socket notifier: communicates parameter updates to core       │
└─────────────────────────────────────────────────────────────────┘
```

**Data Flow:**
```
global_state_epoch_t  
  → ContextBuilder (extract features from metrics)
  → CMABPolicy.select_arm (Random Forest & Thompson Sampling)
  → ArmCatalog.decode (arm → parameter settings)
  → write .parameters.json + notify via socket
  → core applies at applied_begin slot
  → observe execution, collect global_state_epoch_{t+1}
  → global_reward → policy.update
```

## Key Components

### Consensus (Autobahn)
- **Protocol**: Three-phase consensus (Prepare → Confirm → Commit) with fast-path optimization
- **Location**: [primary/src/core.rs](primary/src/core.rs)
- **Performance**: Achieves high throughput with configurable latency-throughput tradeoff
- **Metrics**: Collects fast-path ratios, lane growth rates, and per-transaction latencies

### Metrics Collection
- **Component**: [Agent/metrics_collector.py](Agent/metrics_collector.py)
- **Input**: Consensus logs and state snapshots
- **Output**: Aggregated `global_state_epoch_{N}.json` files
- **Reward Model**: Per-node reward = `1000 / (latency_ms + 1)`, global reward = committee median/mean

### Learning (CMAB)
- **Algorithm**: Random Forest Thompson Sampling
- **Location**: [Agent/rl/cmab/](Agent/rl/cmab/)
- **Selection**: Contextual arm selection based on system state features
- **Feedback**: Delayed reward (epoch t action credited with epoch t+1 reward)

### Benchmarking
- **Framework**: Fabric-based local/remote deployment
- **Location**: [benchmark/](benchmark/)
- **Features**: Automatic config generation, tmux-based process management
- **Deployment**: Supports local single-machine and distributed cloud runs

## Project Layout

```
AutoPilot/
├── node/                   # Primary, worker, benchmark_client implementations
│   ├── src/
│   │   ├── bin/
│   │   │   ├── primary/    # DAG + Autobahn consensus node
│   │   │   ├── worker/     # Transaction batching layer
│   │   │   └── benchmark_client/  # Load generation client
│   │   └── lib.rs
│   └── Cargo.toml
│
├── primary/                # Core consensus logic
│   ├── src/
│   │   ├── core.rs         # Main Autobahn consensus engine
│   │   ├── primary.rs      # Message handling
│   │   └── ...
│   └── Cargo.toml
│
├── worker/                 # Transaction batching
│   ├── src/
│   │   ├── core.rs         # Batching logic
│   │   └── ...
│   └── Cargo.toml
│
├── crypto/                 # Cryptographic primitives (ed25519-dalek)
│   └── ...
│
├── network/                # Network layer (TCP-based)
│   └── ...
│
├── consensus/              # Consensus interface
│   └── ...
│
├── config/                 # Parameter structures and defaults
│   └── ...
│
├── hotstuff/               # HotStuff baseline (if included)
│   └── ...
│
├── Agent/                  # Learning and metrics collection
│   ├── metrics_collector.py     # Aggregates consensus metrics
│   ├── requirements.txt
│   └── rl/
│       ├── cmab/
│       │   ├── context_builder.py    # Extract features from metrics
│       │   ├── arm_catalog.py        # Discrete action space (5 consensus params)
│       │   ├── policy.py             # RF Thompson Sampling
│       │   ├── trainer.py            # Main learning loop
│       │   └── ...
│       ├── actions/
│       │   └── action_encode.py      # Arm ↔ parameter mapping
│       ├── controllers/
│       │   └── controller.py         # Spawns CMAB subprocess
│       └── train_cmab_continuous.py  # Entry point
│
├── benchmark/              # Benchmarking & deployment
│   ├── fabfile.py          # Fabric tasks for local/remote experiments
│   ├── settings.json        # Cloud deployment config
│   ├── analyze_logs.py      # Result parsing
│   ├── plot_latencies.py    # Visualization
│   └── README.md            # Detailed benchmark guide
│
├── script/                 # Setup scripts & Python deps
│   ├── env.sh               # Environment bootstrap
│   └── requirements.txt     # Benchmark Python dependencies
│
├── store/                  # Persistent storage (RocksDB)
│   └── ...
│
├── target/                 # Compiled binaries
│   └── release/
│
└── README.md (this file)
```

## Installation & Setup

### Prerequisites

We recommend running on **Ubuntu 20.04 LTS**. The system requires:
- Python 3.8+
- Rust (recommend 1.80 stable)
- Clang version ≤ 14 (for building librocksdb—do NOT use v15 or higher)
- tmux (for local benchmarks)

### Automated Setup

An install script is provided for convenience:

```bash
./install_deps.sh
pip install -r script/requirements.txt
pip install numpy scikit-learn joblib psutil
```

This installs all Rust and Python dependencies.

### Manual Build

From the `AutoPilot` directory:

```bash
# Build with benchmark feature
cargo build --release --features benchmark
```

The binary will be located at `target/release/node`.

## Running Experiments

### Part 1: Local Deployment

#### Quick Start: Local Benchmark (Recommended)

Run a complete local benchmark with the full learning loop in one command:

```bash
cd benchmark
fab local
```

This will:
1. Build the `node` binary with benchmark feature
2. Generate configuration files for 4 replicas
3. Launch primaries, workers, metrics collector, RL controller, and clients via tmux
4. Display throughput and latency statistics

**Customization**: Edit `bench_params` and `node_params` in `benchmark/fabfile.py` before running.


#### Key Paths (Local)

| Path | Purpose |
|------|---------|
| `RUST_STATE_SOCKET_PATH` | Unix socket for core ↔ metrics_collector communication |
| `/tmp/autobahn_rl_param_{i}.sock` | Socket for CMAB → core parameter updates |
| `metrics-{i}/global_state_epoch_{N}.json` | CMAB input: aggregated metrics per epoch |
| `.parameters.json` | Hot-reloaded consensus parameters (applied at `applied_begin` slot) |

---

### Part 2: GCP Deployment

For distributed experiments on Google Cloud Platform (GCP), follow these steps. This deployment supports full-scale benchmarking across multiple regions with adaptive learning.

#### Prerequisites

1. **GCP Account** with sufficient quota:
   - New users: $300 free trial (recommended)
   - Existing users: Spot market recommended to save costs

2. **Local Setup**:
   - SSH key pair (RSA format)
   - `gcloud` CLI installed
   - Fabric (`pip install fabric`)

> **Note**: We recommend using GCP as experiments are optimized for cloud deployment. See [autobahn-artifact/README.md](https://github.com/neilgiri/autobahn-artifact) for comprehensive GCP setup instructions.

#### Step 1: Setup SSH Keys

Generate a new SSH key pair on your local machine (if not already present):

```bash
ssh-keygen -t rsa -f ~/.ssh/autopilot_key -C your_username -b 2048
```

Add the public key to GCP project metadata:
1. Go to [GCP Console](https://console.cloud.google.com)
2. Navigate to **Compute Engine** → **Metadata** → **SSH Keys**
3. Click **Add SSH Key**
4. Paste the contents of `~/.ssh/autopilot_key.pub`
5. Click **Save** and note your username

#### Step 2: Create Instance Templates

Create instance templates for each region (us-east5, us-east1, us-west1, us-west4):

1. Go to **Compute Engine** → **Instance templates** → **Create instance template**
2. **Name**: `autopilot-template-{region}` (e.g., `autopilot-template-us-east5`)
3. **Region**: Select from {asia-east2, us-central1}
4. **Machine type**: T2D series → `t2d-standard-4` (4 vCPU, 16 GB RAM)
5. **Provisioning model**: **Spot** (to save costs)
   - On VM termination: **Stop**
6. **Boot disk**:
   - OS: **Ubuntu 20.04 LTS**
   - Type: **Balanced persistent disk**
   - Size: **40 GB**
7. **Identity & API access**: Allow full access to Cloud APIs
8. **Security**: Enable **vTPM** and **Integrity Monitoring**
9. Click **Create**

#### Step 3: Create Control Machine Instance

1. Go to **VM Instances** → **Create instance**
2. Click **New VM instance from template**
3. Select `autopilot-control-template`
4. **Name**: `autopilot-control`
5. Wait for green status indicator
6. Note the **External IP address**

SSH into the control machine:

```bash
ssh -i ~/.ssh/autopilot_key USERNAME@EXTERNAL_IP_ADDRESS
```

#### Step 4: Setup Control Machine Environment

On the control machine:

```bash
# Clone repository
git clone https://github.com/ccclr/AutoPilot.git autopilot-repo
cd autopilot-repo
git checkout main  # or appropriate branch

# Install dependencies
./install_deps.sh
pip install -r script/requirements.txt
pip install numpy scikit-learn joblib psutil

# Generate SSH keys for GCP communication
ssh-keygen -t rsa -f /home/username/.ssh/gcp_key -C username -b 2048
# Add ~/.ssh/gcp_key.pub to GCP project metadata (Compute Engine → Metadata)
```

#### Step 5: Configure GCP Settings

Edit `benchmark/settings.json` (or copy from `settings-autopilot.json` if available):

```json
{
  "key": {
    "name": "gcp_key",
    "path": "/home/username/.ssh/gcp_key",
    "port": 5000
  },
  "repo": {
    "name": "autopilot-repo",
    "url": "https://github.com/ccclr/AutoPilot.git",
    "branch": "main"
  },
  "project_id": "YOUR_PROJECT_ID",
  "username": "your_username",
  "instances": {
    "type": "t2d-standard-4",
    "regions": ["asia-east2-a", " us-central1-c", "us-central1-f"],
    "templates": [
      "projects/YOUR_PROJECT_ID/regions/us-east1/instanceTemplates/autopilot-template-asia-east2",
      "projects/YOUR_PROJECT_ID/regions/us-east5/instanceTemplates/autopilot-template-us-central1-c",
      "projects/YOUR_PROJECT_ID/regions/us-west1/instanceTemplates/autopilot-template-us-central1",
      "projects/YOUR_PROJECT_ID/regions/us-west4/instanceTemplates/autopilot-template-us-central1"
    ]
  }
}
```

**Find your project ID**:
```bash
gcloud config get-value project
# Or in GCP console: top-left project dropdown → ID column
```

#### Step 6: Deploy and Run Experiments

From the control machine, navigate to the benchmark directory:

```bash
cd autopilot-repo/benchmark

# Create VM instances (first time only)
fab create

# Wait a few seconds, then install dependencies (first time only)
fab install

# Run the experiment
fab remote
```

**Monitoring**:
- `fab create` and `fab install` show progress in the terminal
- `fab remote` displays a progress bar
- Logs are downloaded automatically to `results/` folder on control machine

#### Troubleshooting GCP Deployment

| Issue | Solution |
|-------|----------|
| Spot instances pre-empted | Delete instances, re-run `fab create`, re-run `fab install` |
| Connection timeout (port 22) | Wait 30 seconds and retry `fab install` |
| SSH key errors | Verify key paths in `settings.json` and ensure keys are added to project metadata |
| Insufficient quota | Reduce machine type to `t2d-standard-4` (free tier) or use spot market |

#### Retrieving Results

Experiment results are automatically downloaded to `benchmark/results/`:

- **Filename format**: `bench-{faults}-{nodes}-{workers}-{collocate}-{rate}-{txsize}.txt`
- **Contents**: Throughput (tx/s) and End-to-End latency (ms)
- **Metrics files**: `metrics-{i}/global_state_epoch_{N}.json` contains detailed per-epoch data

For result analysis:

```bash
python3 analyze_logs.py results/bench-0-4-1-True-240000-512.txt
python3 plot_latencies.py results/
```

---

For comprehensive GCP setup documentation and troubleshooting, refer to [autobahn-artifact/README.md](https://github.com/neilgiri/autobahn-artifact/blob/main/README.md).

## Consensus Protocol Details

### Overview

Autobahn implements a three-phase Byzantine consensus protocol optimized for high throughput and adaptive latency. The protocol is implemented in [primary/src/core.rs](primary/src/core.rs).

### Phases

Each consensus instance follows:
1. **Prepare Phase**: Leader proposes value, replicas vote
2. **Confirm Phase** (optional): Skipped on fast path if all replicas voted Prepare in time
3. **Commit Phase**: Final agreement and execution

### Fast Path

If the leader receives 3f+1 Prepare votes within `fast_path_timeout`, the protocol skips Confirm and proceeds directly to Commit, reducing latency by one round trip.


## Learning & Adaptation (CMAB)

### Overview

AutoPilot uses a **Contextual Multi-Armed Bandit (CMAB)** approach to continuously optimize consensus parameters. The learning algorithm is **Random Forest Thompson Sampling**, implemented in [Agent/rl/cmab/](Agent/rl/cmab/).

### Action Space

The CMAB controls 5 consensus parameters, defining a discrete arm space:

| Parameter | Values | Count |
|-----------|--------|-------|
| `batch_size` | {100k, 500k} | 2 |
| `header_size` | {32, 64} | 2 |
| `cut_condition_type` | {1, 3, 4} | 3 |
| `fast_path_timeout` | {0, 100, 200, 300}ms | 4 |
| `k` (parallel instances) | {1, 4} | 2 |

**Total Arms**: 2 × 2 × 3 × 4 × 2 = **96 possible configurations**

The standard CMAB uses only its Global RF. Set
`enable_cmab_pairwise_residual_rf=true` to add the optional small RF over
`cut_condition_type × fast_path_timeout`; it learns the Global RF's pre-update
reward residual and its prediction is added directly to the Global RF score.

### Reward Calculation:**
- Per-node: `reward = 1000 / (latency_ms + 1)`
- Global: Median (or mean) across all replicas in the committee

### Learning Loop

```
Epoch t:
  1. ContextBuilder reads global_state_epoch_t
  2. Extract features: [fast_path_ratio, lane_growth_rates, ...]
  3. CMABPolicy.select_arm(context) → discrete arm
  4. ArmCatalog.decode(arm) → {batch_size, header_size, ...}
  5. Write .parameters.json, notify core via socket
  6. Core applies parameters at applied_begin slot

Epoch t+1:
  1. Metrics collected, global_state_epoch_{t+1} generated
  2. global_reward extracted
  3. Policy.update(arm, reward) → improve future selections
  4. Return to step 1 for epoch t+1
```

### Components

| Module | File | Purpose |
|--------|------|---------|
| **ContextBuilder** | [Agent/rl/cmab/context_builder.py](Agent/rl/cmab/context_builder.py) | Extract system state features |
| **ArmCatalog** | [Agent/rl/cmab/arm_catalog.py](Agent/rl/cmab/arm_catalog.py) | Discrete action space definition and encoding |
| **Policy** | [Agent/rl/cmab/policy.py](Agent/rl/cmab/policy.py) | Random Forest Thompson Sampling |
| **Trainer** | [Agent/rl/cmab/trainer.py](Agent/rl/cmab/trainer.py) | Main event loop |
| **ActionEncode** | [Agent/rl/actions/action_encode.py](Agent/rl/actions/action_encode.py) | Map arms to parameter vectors |
| **Controller** | [Agent/rl/controllers/controller.py](Agent/rl/controllers/controller.py) | Subprocess manager for benchmarks |

### Running CMAB Directly

```bash
python3 Agent/rl/train_cmab_continuous.py \
  --metrics-dir ./metrics-0 \
  --parameters-file ./.parameters.json \
  --policy rf_ts \
  --context-mode dynamic \
  --node-index 0
```

**Arguments:**
- `--metrics-dir`: Directory containing global state JSON files
- `--parameters-file`: Output file for selected parameters
- `--policy`: Policy type (only `rf_ts` currently supported)
- `--context-mode`: Feature extraction mode (`dynamic` or `static`)
- `--node-index`: Node identifier for socket path

```

## License

[Apache 2.0](LICENSE)
