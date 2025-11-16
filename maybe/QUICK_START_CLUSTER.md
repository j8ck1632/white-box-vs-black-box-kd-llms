# Quick Start: Using Ray Cluster for offline_teacher_data.py *(legacy)*

> `offline_teacher_data.py` now runs entirely on a single workstation without Ray. These quick-start directions remain available only if you need to adapt the script back to a Ray deployment.

## TL;DR - 3 Steps

### 1. Start Ray Cluster

**On head node:**
```bash
ray start --head --port=6379
```
Note the IP address shown (e.g., `192.168.1.100`)

**On each worker node:**
```bash
ray start --address='<HEAD_IP>:6379'
```

### 2. Connect Script to Cluster

**Option A - Environment variable (recommended):**
```bash
export RAY_ADDRESS="<HEAD_IP>:10001"
python offline_teacher_data.py
```

**Option B - Config file:**
Edit `config.py`:
```python
RAY_CLUSTER_ADDRESS = "<HEAD_IP>:10001"
```

### 3. Run Script

```bash
python offline_teacher_data.py
```

The script will automatically:
- Connect to the cluster
- Detect all available GPUs
- Distribute batches across workers
- Process in parallel

## Verify Setup

```bash
# Check cluster status
ray status

# View dashboard (if enabled)
# http://<HEAD_IP>:8265
```

## Common Issues

**"Connection refused"**
- Check head node is running: `ray status` on head node
- Verify firewall allows ports 6379, 10001, 8265
- Check IP address is correct

**"No GPUs detected"**
- Ensure GPUs are visible: `ray status` should show GPU count
- Check CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

**"Model download fails"**
- Set Hugging Face token in `config.py` or `HUGGING_FACE_HUB_TOKEN` env var
- Verify internet connectivity on worker nodes

## Full Documentation

See `RAY_CLUSTER_SETUP.md` for detailed instructions, troubleshooting, and advanced configuration.

