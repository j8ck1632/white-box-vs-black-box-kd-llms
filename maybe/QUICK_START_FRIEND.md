# Quick Start Guide for Friend (Worker Node)

**Just want to lend your GPU? Follow these simple steps!**

## Windows Users

### Option 1: Use the Script (Easiest) ⭐

1. **Download the script**: `start_worker_node.ps1`
2. **Right-click** the file and select **"Run with PowerShell"**
3. **Follow the prompts** - it will:
   - Check if everything is installed
   - Ask for the head node IP address
   - Connect automatically

That's it! Your GPU is now helping with the experiments.

### Option 2: Manual Setup

Open PowerShell and run:
```powershell
pip install ray[default]
ray start --address='<FRIEND_IP>:6379'
```

## Linux Users

### Option 1: Use the Script (Easiest) ⭐

1. **Make the script executable**:
   ```bash
   chmod +x start_worker_node.sh
   ```

2. **Run the script**:
   ```bash
   ./start_worker_node.sh
   ```

3. **Follow the prompts** - it will:
   - Check if everything is installed
   - Ask for the head node IP address
   - Connect automatically

That's it! Your GPU is now helping with the experiments.

### Option 2: Manual Setup

```bash
pip install ray[default]
ray start --address='<FRIEND_IP>:6379'
```

## Keep It Running

**Windows:** Leave the PowerShell window open. Your GPU is now part of the cluster!

**Linux:** Leave the terminal window open. Your GPU is now part of the cluster!

To stop later, press `Ctrl+C` or run:
- Windows: `ray stop`
- Linux: `ray stop`

## Troubleshooting

**"Connection refused"**
- Make sure your friend has started their head node
- Check the IP address is correct
- If using VPN (Tailscale), make sure it's running on both computers

**"Ray not found"**
- Install Ray: `pip install ray[default]`

**Linux: "Permission denied"**
- Make script executable: `chmod +x start_worker_node.sh`

**Need help?** Ask your friend or check `FRIEND_GPU_SETUP.md` for detailed instructions.

