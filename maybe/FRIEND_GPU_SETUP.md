# How to Share GPU Power with a Friend

This guide explains how your friend can lend their GPU to help with your machine learning experiments.

## Quick Overview

Your friend's computer will act as a **worker node** that connects to your **head node**. Together, they form a Ray cluster that can use both GPUs simultaneously.

## 🚀 Quick Start (Easiest Method)

### Windows Users

**For You (Head Node):**
1. Double-click `start_head_node.bat` - it will guide you through setup automatically!

**For Your Friend (Worker Node):**
1. Share `start_worker_node.ps1` and `start_worker_node.bat` with them
2. They double-click `start_worker_node.bat` and enter your IP address
3. Done!

### Linux Users

**For You (Head Node):**
1. Make executable: `chmod +x start_head_node.sh`
2. Run: `./start_head_node.sh` - it will guide you through setup automatically!

**For Your Friend (Worker Node):**
1. Share `start_worker_node.sh` with them
2. They make it executable: `chmod +x start_worker_node.sh`
3. They run: `./start_worker_node.sh` and enter your IP address
4. Done!

See `QUICK_START_FRIEND.md` for a simple guide you can share with your friend.

## 🏙️ Remote Connection Setup (For Friends Across the City/Internet)

**If your friend lives far away or on a different network**, you'll need a VPN to connect. This is the **recommended and easiest method**.

### Recommended Solution: Tailscale (Free & Easy)

Tailscale creates a secure VPN between your computers automatically. It's free for personal use and very easy to set up.

#### Step 1: Both Install Tailscale

**You and your friend:**
1. Go to https://tailscale.com/download
2. Download and install Tailscale for Windows
3. Sign in with Google, Microsoft, or GitHub account
4. **Important:** Make sure you're both in the same Tailscale network (same account or shared network)

#### Step 2: Get Your Tailscale IP

**On your computer (head node):**
1. Open Tailscale (it runs in the background)
2. Click the Tailscale icon in your system tray
3. Note your **Tailscale IP address** (looks like `100.x.x.x`)
4. Share this IP with your friend

**Or check via command line:**
```powershell
# On Windows PowerShell
Get-NetIPAddress -AddressFamily IPv4 | Where-Object {$_.IPAddress -like "100.*"}
```

#### Step 3: Start Ray with Tailscale IP

**On your computer:**

**Note:** On Windows, you need to enable multi-node cluster support first.

```powershell
# Set environment variable (required for Windows)
$env:RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1

# Start head node with Tailscale IP
ray start --head --port=6379 --dashboard-host=0.0.0.0 --node-ip-address=<YOUR_TAILSCALE_IP>
```

Replace `<YOUR_TAILSCALE_IP>` with your Tailscale IP (e.g., `100.64.1.2`)

**Example:**
```powershell
$env:RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1
ray start --head --port=6379 --dashboard-host=0.0.0.0 --node-ip-address=100.64.1.2
```

**Or run in one line:**
```powershell
$env:RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1; ray start --head --port=6379 --dashboard-host=0.0.0.0 --node-ip-address=100.64.1.2
```

#### Step 4: Friend Connects Using Tailscale IP

**On your friend's computer:**

**Note:** On Windows, you need to enable multi-node cluster support first.

```powershell
# Set environment variable (required for Windows)
$env:RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1

# Connect to the cluster
ray start --address='<YOUR_TAILSCALE_IP>:6379'
```

**Example:**
```powershell
$env:RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1
ray start --address='100.64.1.2:6379'
```

**Or run in one line:**
```powershell
$env:RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1; ray start --address='100.64.1.2:6379'
```

That's it! The VPN handles all the networking complexity.

### Alternative VPN Options

If you prefer other VPN solutions:

- **ZeroTier**: Similar to Tailscale, free for personal use
- **Hamachi**: Older but still works, free for up to 5 devices
- **WireGuard**: More technical but very fast

The setup process is similar - just use the VPN IP address instead of your local IP.

## Prerequisites

### What Your Friend Needs:
- A computer with a GPU (NVIDIA GPU recommended)
- Python installed (3.8+)
- Internet connection
- Same network access (same WiFi/LAN, or VPN/tunneling)

### What You Need:
- Your computer set up as the head node
- Your friend's IP address (or they need your IP address)
- Firewall configured to allow Ray connections

---

## Step-by-Step Setup

### Step 1: You Start the Head Node (On Your Computer)

1. Open PowerShell or Command Prompt
2. Navigate to your project directory
3. Start Ray head node:

```powershell
ray start --head --port=6379 --dashboard-host=0.0.0.0
```

**Important:** Note the IP address shown in the output. It will look like:
```
Local node IP: 192.168.1.100
```

**Share this IP address with your friend!**

The output will also show something like:
```
To connect to this Ray runtime from another node, run
  ray start --address='192.168.1.100:6379'
```

### Step 2: Your Friend Connects as Worker (On Their Computer)

Your friend needs to:

1. **Install Ray** (if not already installed):
```powershell
pip install ray[default]
```

2. **Install other dependencies** (optional, but recommended):
   - They can clone your repo and run `pip install -r requirements.txt`
   - Or just install Ray if they're only providing GPU power

3. **Connect to your cluster**:
```powershell
ray start --address='<YOUR_IP_ADDRESS>:6379'
```

Replace `<YOUR_IP_ADDRESS>` with the IP you shared (e.g., `192.168.1.100`)

**Example:**
```powershell
ray start --address='192.168.1.100:6379'
```

### Step 3: Verify Connection

**On your computer (head node), check cluster status:**
```powershell
ray status
```

You should see:
- Your node (head node)
- Your friend's node (worker node)
- Total GPUs available (should be sum of both GPUs)

**Example output:**
```
======== Autoscaler status: 2024-01-15 10:30:00 ========
Node status
--------------------------------------------------------
Active:
 1 node_1 (your computer)
    Resources: 1.0/1.0 CPU, 1.0/1.0 GPU
 1 node_2 (friend's computer)
    Resources: 1.0/1.0 CPU, 1.0/1.0 GPU

Total: 2.0/2.0 CPU, 2.0/2.0 GPU
```

### Step 4: Run Your Script

Now when you run your training script, it will automatically use both GPUs:

```powershell
python train_student.py
```

Or for offline teacher data generation:
```powershell
python offline_teacher_data.py
```

---

## Network Setup (Important!)

### If You're on the Same Network (Same WiFi/LAN)

This is the easiest case:
- Just use your local IP address (e.g., `192.168.1.100`)
- Make sure Windows Firewall allows Ray connections

**Allow Ray through Windows Firewall:**
1. Open Windows Defender Firewall
2. Click "Allow an app or feature through Windows Firewall"
3. Add Python and allow it for Private networks
4. Or temporarily disable firewall for testing

### If You're on Different Networks (Remote Connection)

**⚠️ For friends across the city or internet, use a VPN (see Remote Connection Setup section above).**

#### Option A: VPN (Strongly Recommended) ✅
- **Tailscale** (easiest, free) - See detailed instructions above
- **ZeroTier** (similar to Tailscale)
- **Hamachi** (older but works)
- Both connect to the same VPN network
- Use the VPN IP address instead of local IP
- **This is the safest and easiest method**

#### Option B: Port Forwarding (Not Recommended)
- Requires router access and configuration
- Exposes your computer to the internet
- Forward ports: 6379, 10001, 8265
- Your friend connects using your public IP
- **Security risk** - only use if you understand the risks

#### Option C: SSH Tunneling (Advanced)
- Set up SSH server on your computer
- Friend connects via SSH tunnel
- More complex but secure
- Requires SSH server setup

---

## Troubleshooting

### "Connection refused" Error

**Possible causes:**
1. **Firewall blocking connection**
   - Solution: Allow Python/Ray through Windows Firewall
   - Or temporarily disable firewall for testing

2. **Wrong IP address**
   - Solution: Double-check the IP address
   - On Windows, run `ipconfig` to see your IP
   - Make sure you're using the correct network interface

3. **Head node not running**
   - Solution: Make sure you started Ray with `ray start --head`

4. **Different networks (remote connection)**
   - Solution: Use VPN (Tailscale recommended - see Remote Connection Setup section above)
   - Make sure both computers have VPN running and are connected
   - Verify you're using the VPN IP address, not local IP

### "Failed to connect to GCS" or "Node timed out during startup" Error

**This error means your friend's worker node can't properly connect to your head node, even though ping works.**

**⚠️ If ping fails on friend's computer, Tailscale isn't connected properly. Fix that first (see section above).**

**If ping works but Ray connection times out, the issue is with Ray configuration:**

### Tailscale Ping Failed (Can't Reach Head Node)

**If `ping 100.101.255.2` fails on friend's computer, follow these steps:**

1. **Verify Tailscale is installed and running on both computers:**
   - **On your computer (head node):**
     - Open Tailscale app
     - Make sure it shows "Connected" status
     - Note your Tailscale IP (should be `100.101.255.2`)
   - **On friend's computer (worker node):**
     - Open Tailscale app
     - Make sure it shows "Connected" status
     - Verify they can see your computer in their Tailscale device list

2. **Check if both computers are in the same Tailscale network:**
   - **Option A: Same account** (easiest)
     - Both sign in with the same Google/Microsoft/GitHub account
   - **Option B: Shared network**
     - One person creates a shared network and invites the other
     - Both accept the invitation

3. **Verify Tailscale IP addresses:**
   - **On your computer (head node):**
   ```powershell
   # Windows
   Get-NetIPAddress -AddressFamily IPv4 | Where-Object {$_.IPAddress -like "100.*"} | Select-Object IPAddress
   ```
   ```bash
   # Linux
   ip addr show | grep "100\."
   ```
   - Confirm it shows `100.101.255.2` (or your actual Tailscale IP)
   - **On friend's computer:**
   ```bash
   # Linux
   ip addr show | grep "100\."
   ```
   ```powershell
   # Windows
   Get-NetIPAddress -AddressFamily IPv4 | Where-Object {$_.IPAddress -like "100.*"} | Select-Object IPAddress
   ```
   - They should have a different `100.x.x.x` IP

4. **Test connectivity from both sides:**
   - **On your computer**, try pinging friend's Tailscale IP:
   ```powershell
   # Windows
   ping <FRIEND_TAILSCALE_IP>
   ```
   ```bash
   # Linux
   ping <FRIEND_TAILSCALE_IP>
   ```
   - **On friend's computer**, try pinging your Tailscale IP:
   ```bash
   ping 100.101.255.2
   ```
   - If **both** ping fail, Tailscale routing isn't working
   - If **one** works but not the other, firewall might be blocking

5. **Restart Tailscale on both computers:**
   - **Windows:**
     - Right-click Tailscale icon in system tray
     - Click "Disconnect" then "Connect"
     - Or restart the Tailscale service
   - **Linux:**
   ```bash
   sudo systemctl restart tailscaled
   ```
   - Wait 30 seconds, then test ping again

6. **Check Tailscale admin panel:**
   - Go to https://login.tailscale.com/admin/machines
   - Verify **both** computers show as "Connected" (green dot)
   - If one shows "Offline" or "No recent activity", that computer isn't connected
   - Check if there are any access control restrictions

7. **Check firewall settings:**
   - **On your computer (head node):** Windows Firewall might block Tailscale
   
   **To temporarily disable Windows Firewall (for testing):**
   
   **Method 1: Using Windows Settings (Easiest)**
   1. Press `Win + I` to open Settings
   2. Go to **Privacy & Security** → **Windows Security**
   3. Click **Firewall & network protection**
   4. You'll see three networks: **Domain network**, **Private network**, and **Public network**
   5. Click on **Private network** (this is usually your Tailscale connection)
   6. Toggle **Windows Defender Firewall** to **Off**
   7. Also toggle **Public network** to **Off** if needed
   8. Click **Yes** when prompted
   9. Test ping again
   
   **Method 2: Using PowerShell (Faster)**
   ```powershell
   # Disable firewall for all profiles (Public, Private, Domain)
   Set-NetFirewallProfile -Profile Domain,Public,Private -Enabled False
   
   # Verify it's disabled
   Get-NetFirewallProfile | Select-Object Name,Enabled
   ```
   
   **To re-enable firewall later:**
   ```powershell
   # Re-enable firewall
   Set-NetFirewallProfile -Profile Domain,Public,Private -Enabled True
   ```
   
   **⚠️ Security Warning:** Only disable firewall temporarily for testing. Re-enable it after you're done or configure specific rules.
   
   **Better solution: Allow Tailscale through firewall (instead of disabling):**
   ```powershell
   # Allow Tailscale executable through firewall
   New-NetFirewallRule -DisplayName "Tailscale" -Direction Inbound -Program "C:\Program Files\Tailscale\tailscaled.exe" -Action Allow
   New-NetFirewallRule -DisplayName "Tailscale" -Direction Outbound -Program "C:\Program Files\Tailscale\tailscaled.exe" -Action Allow
   ```
   
   Or manually:
   1. Open Windows Defender Firewall
   2. Click "Allow an app or feature through Windows Defender Firewall"
   3. Click "Change settings" → "Allow another app"
   4. Browse to `C:\Program Files\Tailscale\tailscaled.exe`
   5. Check both "Private" and "Public" networks
   6. Click "Add" and "OK"
   
   - **On friend's computer:** Same check if they have firewall enabled

8. **Verify Tailscale subnet routing (if using custom routing):**
   - If you're using advanced Tailscale features, make sure subnet routing is enabled
   - Check Tailscale admin panel for routing settings

**Quick checklist:**
- ✅ Tailscale installed on both computers
- ✅ Both showing "Connected" in Tailscale app
- ✅ Both in same Tailscale account/network
- ✅ Both have `100.x.x.x` IP addresses
- ✅ Both can see each other in Tailscale admin panel
- ✅ Firewall allows Tailscale connections
- ✅ Test ping from both directions

**If ping still fails after these steps:**
- Try restarting both computers
- Check Tailscale logs for errors
- Consider using ZeroTier or another VPN as alternative

**Step-by-step troubleshooting:**

1. **⚠️ CRITICAL: Verify head node is running with Tailscale IP:**
   - On **your computer** (head node), check if Ray is currently running:
   ```powershell
   ray status
   ```
   - If Ray is running but **without** `--node-ip-address`, **you must restart it:**
   ```powershell
   # Stop current Ray instance
   ray stop
   
   # Wait a few seconds, then restart with Tailscale IP
   $env:RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1
   ray start --head --port=6379 --dashboard-host=0.0.0.0 --node-ip-address=100.101.255.2
   ```
   - ⚠️ **This is the #1 cause of timeout errors!** Ray must be listening on your Tailscale IP, not just your local IP.
   - Check the output - it should show something like:
   ```
   Local node IP: 100.101.255.2
   ```
   - If it shows a different IP (like `192.168.x.x`), Ray isn't using Tailscale IP.

2. **Verify Tailscale connectivity:**
   - On **your friend's computer**, test if they can reach your IP:
   ```bash
   ping 100.101.255.2
   ```
   - If ping fails, Tailscale isn't connected properly (see below)

3. **Check Tailscale status:**
   - Both computers must have Tailscale running
   - Both must be signed in to the same Tailscale account (or shared network)
   - Check Tailscale app shows both devices as "Connected"
   - Verify your head node's Tailscale IP is actually `100.101.255.2`:
   ```powershell
   # On Windows (head node)
   Get-NetIPAddress -AddressFamily IPv4 | Where-Object {$_.IPAddress -like "100.*"} | Select-Object IPAddress
   ```
   ```bash
   # On Linux (head node)
   ip addr show | grep "100\."
   ```

4. **Check firewall:**
   - On **your computer** (head node), Windows Firewall might be blocking:
   - Temporarily disable firewall to test, OR
   - Add Python/Ray to allowed apps for Private networks
   - Make sure port 6379 is not blocked

5. **Verify head node is listening on the right interface:**
   - On **your computer**, check Ray status:
   ```powershell
   ray status
   ```
   - It should show the correct IP address (your Tailscale IP)

6. **Try connecting from head node side:**
   - On **your computer**, check if Ray dashboard is accessible:
   - Open browser to: `http://100.101.255.2:8265`
   - If you can't access it from your own computer, Ray isn't listening on the Tailscale interface
   
7. **Check Ray uses multiple ports (not just 6379):**
   - Ray needs ports: 6379, 8265, 10001-10100 (approximately)
   - Since firewall is disabled, this shouldn't be the issue
   - But if you re-enable firewall later, you'll need to allow these ports

8. **Verify Ray can see both nodes:**
   - After your friend tries to connect, check on **your computer**:
   ```powershell
   ray status
   ```
   - You should see 2 nodes (head + worker)
   - If only 1 node shows, the worker connection failed

9. **"Deadline Exceeded" or "RPC Error: Deadline Exceeded" errors:**
   - This means the connection reaches the head node but times out during handshake
   - **Check Ray version compatibility:**
     - Both head and worker must have the **same Ray version**
     - On **your computer**: `python -c "import ray; print(ray.__version__)"`
     - On **friend's computer**: `python -c "import ray; print(ray.__version__)"`
     - If versions differ, update both to the same version:
     ```bash
     pip install --upgrade ray[default]
     ```
   - **Increase Ray timeout (if versions match):**
     - On **friend's computer** (Linux/Mac), try with increased timeout:
     ```bash
     RAY_OBJECT_TIMEOUT_S=300 ray start --address='100.101.255.2:6379'
     ```
     - On **Windows PowerShell** (your computer if you're running worker):
     ```powershell
     $env:RAY_OBJECT_TIMEOUT_S=300; ray start --address='100.101.255.2:6379'
     ```
     - Or set it permanently for the session:
     ```powershell
     $env:RAY_OBJECT_TIMEOUT_S=300
     ray start --address='100.101.255.2:6379'
     ```
   - **Check if head node is overloaded:**
     - On **your computer**, check Ray status and logs
     - If head node is busy, wait a few seconds and try again
   - **Restart both nodes:**
     - Sometimes Ray gets into a bad state
     - On **your computer**: `ray stop` then restart with Tailscale IP
     - On **friend's computer**: Wait 10 seconds, then try connecting again
   - **Try connecting with more verbose output:**
     - On **Linux/Mac** (friend's computer):
     ```bash
     RAY_BACKEND_LOG_LEVEL=debug ray start --address='100.101.255.2:6379'
     ```
     - On **Windows PowerShell**:
     ```powershell
     $env:RAY_BACKEND_LOG_LEVEL="debug"; ray start --address='100.101.255.2:6379'
     ```
     This will show more details about where it's failing

**Quick fix checklist (most common issues first):**
1. ✅ **Head node restarted with `--node-ip-address=100.101.255.2`** (MOST IMPORTANT!)
2. ✅ Both Tailscale apps running and connected
3. ✅ Can ping `100.101.255.2` from friend's computer
4. ✅ Firewall disabled (or allows Ray/Python)
5. ✅ **Ray versions match on both computers** (check with `python -c "import ray; print(ray.__version__)"`)
6. ✅ `ray status` on head node shows correct Tailscale IP
7. ✅ Ray dashboard accessible at `http://100.101.255.2:8265`

### "No GPUs detected" on Friend's Computer

**Check:**
1. GPU is installed and working:
```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

2. CUDA is installed:
```powershell
nvidia-smi
```

3. Ray can see the GPU:
```powershell
ray status
```

### Friend's Computer Can't Download Models

**Solution:** Your friend needs the Hugging Face token:
1. Share your token (or they can use their own)
2. They can set it as environment variable:
```powershell
$env:HUGGING_FACE_HUB_TOKEN="your_token_here"
```

Or they can create a `config.py` file with:
```python
HUGGING_FACE_TOKEN = "your_token_here"
```

### VPN/Remote Connection Issues

**Tailscale not connecting:**
1. Make sure both computers have Tailscale installed and running
2. Check that both are signed in to the same Tailscale account (or in a shared network)
3. Verify Tailscale shows both devices as "Connected" in the admin panel
4. Try restarting Tailscale on both computers

**Can't find Tailscale IP:**
```powershell
# PowerShell command to find Tailscale IP
Get-NetIPAddress -AddressFamily IPv4 | Where-Object {$_.IPAddress -like "100.*"} | Select-Object IPAddress
```

**Ray can't connect through VPN:**
1. Make sure you're using the VPN IP (starts with `100.` for Tailscale), not your local IP
2. Check Windows Firewall - you may need to allow Ray through the VPN network
3. Try temporarily disabling firewall to test if that's the issue
4. Verify VPN is working by pinging your friend's VPN IP:
```powershell
ping <FRIEND_TAILSCALE_IP>
```

**Slow performance over VPN:**
- This is normal - VPN adds latency
- GPU computation happens locally, so it's still very effective
- Data transfer between nodes will be slower than local network
- Consider running longer batches to minimize communication overhead

---

## Monitoring the Cluster

### Ray Dashboard

Access the dashboard from your browser:
```
http://<YOUR_IP>:8265
```

You can see:
- Active tasks
- GPU utilization
- Worker status
- Task progress

### Command Line

Check status anytime:
```powershell
ray status
```

---

## Stopping the Cluster

### When You're Done

**On your computer (head node):**
```powershell
ray stop
```

**On your friend's computer (worker node):**
```powershell
ray stop
```

---

## Security Considerations

⚠️ **Important Security Notes:**

1. **Ray cluster is not encrypted by default** - Anyone on your network can connect if they know the IP
2. **Use VPN for remote connections** - Don't expose Ray to the public internet
3. **Don't share sensitive data** - Only use this for experiments, not production
4. **Trust your friend** - They'll have access to your Ray cluster

---

## Example: Complete Setup Session

### For Remote Connection (VPN/Tailscale)

**On Your Computer:**
```powershell
# 1. Make sure Tailscale is running and note your Tailscale IP
# (Check system tray or use: Get-NetIPAddress | Where-Object {$_.IPAddress -like "100.*"})

# 2. Start head node with Tailscale IP
ray start --head --port=6379 --dashboard-host=0.0.0.0 --node-ip-address=100.64.1.2

# 3. Share your Tailscale IP (100.64.1.2) with your friend

# 4. Verify your node is running
ray status

# 5. Run your script (after friend connects)
python train_student.py
```

**On Your Friend's Computer:**
```powershell
# 1. Make sure Tailscale is running and connected

# 2. Install Ray (if needed)
pip install ray[default]

# 3. Connect to your cluster using your Tailscale IP
ray start --address='100.64.1.2:6379'

# 4. They're done! Their GPU is now available to you
```

### For Same Network (Local)

**On Your Computer:**
```powershell
# 1. Start head node
ray start --head --port=6379 --dashboard-host=0.0.0.0

# Output shows: Local node IP: 192.168.1.100
# Share this IP with your friend!

# 2. Verify your node is running
ray status

# 3. Run your script (after friend connects)
python train_student.py
```

**On Your Friend's Computer:**
```powershell
# 1. Install Ray (if needed)
pip install ray[default]

# 2. Connect to your cluster
ray start --address='192.168.1.100:6379'

# 3. They're done! Their GPU is now available to you
```

---

## Tips for Best Performance

1. **VPN adds some latency** - Remote connections will be slower than local network, but still very usable
2. **Stable connection** - Use wired Ethernet if possible for better reliability
3. **Pre-download models** - Have your friend download models first to avoid delays during training
4. **Monitor dashboard** - Watch the Ray dashboard to see if both GPUs are being used
5. **Test connection first** - Run `ray status` to verify both nodes are connected before starting long training jobs
6. **Keep VPN running** - Make sure Tailscale (or your VPN) stays connected on both computers

---

## Need More Help?

- See `RAY_CLUSTER_SETUP.md` for detailed documentation
- See `QUICK_START_CLUSTER.md` for quick reference
- Check Ray documentation: https://docs.ray.io/

