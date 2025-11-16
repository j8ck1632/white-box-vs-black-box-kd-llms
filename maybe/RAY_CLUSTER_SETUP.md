# Ray Cluster Setup Guide for offline_teacher_data.py *(legacy)*

> **Heads up:** `offline_teacher_data.py` now streams data sequentially on a single workstation and no longer requires Ray. The instructions below are kept for archival purposes in case you want to adapt the script for a custom Ray deployment.

This guide explains how to set up and use a Ray cluster to run `offline_teacher_data.py` with pooled compute power across multiple machines.

## Overview

Running `offline_teacher_data.py` on a Ray cluster allows you to:
- **Distribute work across multiple GPUs/machines** - Process batches in parallel
- **Scale horizontally** - Add more worker nodes to speed up processing
- **Utilize pooled resources** - Share compute power across multiple machines

## Prerequisites

1. **Multiple machines with GPUs** (or a cloud cluster)
2. **Network connectivity** between machines
3. **Ray installed** on all machines: `pip install ray`
4. **Same Python environment** on all machines (same packages, versions)
5. **Shared access** to Hugging Face models (token configured)

## Step 1: Set Up the Ray Cluster

### Option A: Manual Setup (On-Premise or Cloud VMs)

#### On the Head Node (Main Machine)

1. Start the Ray head node:
```bash
ray start --head --port=6379 --dashboard-host=0.0.0.0
```

This will output something like:
```
Local node IP: 192.168.1.100
...
To connect to this Ray runtime from another node, run
  ray start --address='192.168.1.100:6379'
```

**Note the IP address and port** - you'll need this for worker nodes.

#### On Each Worker Node

1. Connect to the head node:
```bash
ray start --address='<HEAD_NODE_IP>:6379'
```

Replace `<HEAD_NODE_IP>` with the IP address from the head node output.

**Example:**
```bash
ray start --address='192.168.1.100:6379'
```

#### Verify Cluster Status

On the head node, check cluster status:
```bash
ray status
```

You should see all connected nodes with their resources (GPUs, CPUs).

### Option B: Using Ray Cluster Manager (Recommended for Cloud)

If you're using a cloud provider or Kubernetes, use their Ray cluster manager:

#### AWS (Ray on EC2)
```bash
# Install Ray cluster launcher
pip install ray[default] boto3

# Create cluster config
ray create-or-update-cluster-config.yaml

# Start cluster
ray up cluster-config.yaml
```

#### GCP (Ray on GKE)
```bash
# Use Ray KubeRay operator
kubectl apply -f ray-cluster.yaml
```

#### Azure (Ray on AKS)
Similar to GCP, use KubeRay operator.

## Step 2: Configure Your Script

You have **two options** to connect to the cluster:

### Option 1: Environment Variable (Recommended)

Set the `RAY_ADDRESS` environment variable before running the script:

```bash
# Linux/Mac
export RAY_ADDRESS="192.168.1.100:10001"
python offline_teacher_data.py

# Windows PowerShell
$env:RAY_ADDRESS="192.168.1.100:10001"
python offline_teacher_data.py

# Windows CMD
set RAY_ADDRESS=192.168.1.100:10001
python offline_teacher_data.py
```

### Option 2: Config File

Edit your `config.py` file:

```python
RAY_CLUSTER_ADDRESS = "192.168.1.100:10001"  # Head node IP:port
```

**Note:** The port is typically `10001` (Ray's default), but check your head node output.

## Step 3: Ensure All Nodes Have Access

### Model Access

All worker nodes need to:
1. **Have the Hugging Face token** - Either:
   - Set `HUGGING_FACE_TOKEN` in `config.py` (same file on all nodes)
   - Or set `HUGGING_FACE_HUB_TOKEN` environment variable

2. **Download models** - Each worker will download the model on first use (cached after that)

### Data Access

The script loads datasets from Hugging Face, so all nodes need:
- Internet connectivity
- Access to Hugging Face datasets

### Output Path

The output path (`OFFLINE_DATA_PATH`) should be:
- **Shared storage** (NFS, S3, etc.) if you want results in one place
- **Local to head node** if running from head node (default: `./offline_teacher_data`)

## Step 4: Run the Script

From the **head node** (or any node with access to the code):

```bash
python offline_teacher_data.py
```

The script will:
1. Connect to the Ray cluster
2. Detect available GPUs across all nodes
3. Distribute batches to workers
4. Each worker loads the model once and processes multiple batches
5. Save results to the output path

## Step 5: Monitor Progress

### Ray Dashboard

Access the Ray dashboard (usually on port 8265):
```
http://<HEAD_NODE_IP>:8265
```

You can see:
- Active tasks
- Resource utilization
- Worker status
- Task progress

### Command Line

Check cluster status:
```bash
ray status
```

## Troubleshooting

### Issue: "Connection refused" or "Cannot connect to cluster"

**Solutions:**
1. Verify head node is running: `ray status` on head node
2. Check firewall rules - Ray needs ports 6379, 10001, 8265 open
3. Verify IP address is correct
4. Ensure all nodes are on the same network

### Issue: "Model download fails on workers"

**Solutions:**
1. Verify Hugging Face token is set on all nodes
2. Check internet connectivity from worker nodes
3. Pre-download models on each node:
   ```python
   from transformers import AutoModelForCausalLM
   model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", token="YOUR_TOKEN")
   ```

### Issue: "Out of memory" errors

**Solutions:**
1. Reduce batch size in the script (currently `batch_size = 2`)
2. Ensure each worker has enough GPU memory (7B model needs ~14GB)
3. Use sequential mode: `export SEQUENTIAL_MODE=True`

### Issue: "Workers not using GPUs"

**Solutions:**
1. Verify GPUs are visible: `ray status` should show GPU count
2. Check CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
3. Ensure `num_gpus=1` in `map_batches()` call (already set in script)

## Performance Tips

1. **Pre-warm model cache**: Run a small batch first to download models
2. **Use shared model cache**: Set `HF_HOME` to a shared directory
3. **Optimize batch size**: Larger batches = fewer overhead, but more memory
4. **Monitor network**: Ensure low latency between nodes for best performance

## Example: Multi-Node Setup

**Scenario:** 4 machines, each with 1 GPU

1. **Head node** (192.168.1.100):
   ```bash
   ray start --head --port=6379
   ```

2. **Worker nodes** (192.168.1.101, 192.168.1.102, 192.168.1.103):
   ```bash
   ray start --address='192.168.1.100:6379'
   ```

3. **Run script** from head node:
   ```bash
   export RAY_ADDRESS="192.168.1.100:10001"
   python offline_teacher_data.py
   ```

**Result:** 4 GPUs processing batches in parallel = ~4x speedup (with overhead)

## Shutting Down the Cluster

### From Head Node
```bash
ray stop
```

### From Worker Nodes
```bash
ray stop
```

Or use Ray's cluster manager:
```bash
ray down cluster-config.yaml  # For managed clusters
```

## Additional Resources

- [Ray Cluster Documentation](https://docs.ray.io/en/latest/cluster/getting-started.html)
- [Ray Data Documentation](https://docs.ray.io/en/latest/data/data.html)
- [Ray Troubleshooting](https://docs.ray.io/en/latest/cluster/troubleshooting.html)

