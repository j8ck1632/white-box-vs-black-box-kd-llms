#!/bin/bash
# Quick Start Script for Head Node (Your Computer)
# This script starts the Ray head node so your friend can connect
#
# Usage: ./start_head_node.sh

set -e  # Exit on error (but we'll handle errors manually)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
GRAY='\033[0;37m'
NC='\033[0m' # No Color

# Error handler
trap 'echo -e "\n${RED}Error occurred!${NC}"; echo "Press Enter to exit..."; read; exit 1' ERR

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  Ray Cluster Head Node Setup${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""
echo "Script starting..."
echo ""

# Check if Python is installed
echo -e "${YELLOW}Checking Python installation...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    PYTHON_VERSION=$(python3 --version 2>&1)
    echo -e "${GREEN}Found: $PYTHON_VERSION${NC}"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
    PYTHON_VERSION=$(python --version 2>&1)
    echo -e "${GREEN}Found: $PYTHON_VERSION${NC}"
else
    echo -e "${RED}Python not found! Please install Python first.${NC}"
    echo -e "${YELLOW}Install with: sudo apt install python3 python3-pip${NC} (Ubuntu/Debian)"
    echo -e "${YELLOW}Or: sudo yum install python3 python3-pip${NC} (RHEL/CentOS)"
    echo ""
    echo "Press Enter to exit..."
    read
    exit 1
fi

# Check if Ray is installed
echo ""
echo -e "${YELLOW}Checking Ray installation...${NC}"
if command -v ray &> /dev/null; then
    RAY_VERSION=$(ray --version 2>&1)
    echo -e "${GREEN}Ray is installed: $RAY_VERSION${NC}"
    RAY_INSTALLED=true
else
    RAY_INSTALLED=false
fi

if [ "$RAY_INSTALLED" = false ]; then
    echo -e "${YELLOW}Ray not found. Installing Ray...${NC}"
    echo "This may take a few minutes..."
    if $PYTHON_CMD -m pip install --upgrade pip && $PYTHON_CMD -m pip install ray[default]; then
        echo -e "${GREEN}Ray installed successfully!${NC}"
    else
        echo -e "${RED}Failed to install Ray. Please install manually:${NC}"
        echo -e "${YELLOW}pip install ray[default]${NC}"
        echo ""
        echo "Press Enter to exit..."
        read
        exit 1
    fi
fi

# Check if GPU is available (non-blocking, optional)
echo ""
echo -e "${YELLOW}Checking GPU availability...${NC}"
GPU_CHECK_CMD='import torch; print("CUDA available:", torch.cuda.is_available())'
if timeout 3 $PYTHON_CMD -c "$GPU_CHECK_CMD" 2>&1 | grep -q "True"; then
    echo -e "${GREEN}GPU detected and available!${NC}"
    GPU_COUNT_CMD='import torch; print(torch.cuda.device_count())'
    GPU_COUNT=$(timeout 3 $PYTHON_CMD -c "$GPU_COUNT_CMD" 2>&1 || echo "?")
    if [ "$GPU_COUNT" != "?" ]; then
        echo -e "${GREEN}Found $GPU_COUNT GPU(s)${NC}"
    fi
else
    echo -e "${YELLOW}GPU not detected or PyTorch not installed. Ray will still work but won't use GPU acceleration.${NC}"
fi

# Check for VPN/Tailscale
echo ""
echo -e "${YELLOW}Checking for VPN connection...${NC}"
TAILSCALE_IP=""
if command -v tailscale &> /dev/null; then
    TAILSCALE_IP=$(tailscale ip -4 2>/dev/null || echo "")
fi

# Also try checking network interfaces for Tailscale IP (100.x.x.x)
if [ -z "$TAILSCALE_IP" ]; then
    TAILSCALE_IP=$(ip -4 addr show 2>/dev/null | grep -oP '100\.\d+\.\d+\.\d+' | head -1 || echo "")
fi

USE_VPN=false
if [ -n "$TAILSCALE_IP" ]; then
    echo -e "${GREEN}Tailscale VPN detected! Your VPN IP: $TAILSCALE_IP${NC}"
    echo ""
    read -p "Use Tailscale VPN IP? (Y/n): " USE_VPN_CHOICE
    if [ -z "$USE_VPN_CHOICE" ] || [ "$USE_VPN_CHOICE" = "Y" ] || [ "$USE_VPN_CHOICE" = "y" ]; then
        USE_VPN=true
        NODE_IP="$TAILSCALE_IP"
    fi
fi

# Get local IP if not using VPN
if [ "$USE_VPN" = false ]; then
    echo ""
    echo -e "${YELLOW}Detecting local IP address...${NC}"
    
    # Try multiple methods to get local IP
    LOCAL_IPS=()
    
    # Method 1: Using ip command (modern)
    if command -v ip &> /dev/null; then
        while IFS= read -r ip; do
            if [[ ! "$ip" =~ ^127\. ]] && [[ ! "$ip" =~ ^169\.254\. ]] && [[ ! "$ip" =~ ^100\. ]]; then
                LOCAL_IPS+=("$ip")
            fi
        done < <(ip -4 addr show 2>/dev/null | grep -oP '(?<=inet\s)\d+(\.\d+){3}' || true)
    fi
    
    # Method 2: Using hostname (fallback)
    if [ ${#LOCAL_IPS[@]} -eq 0 ] && command -v hostname &> /dev/null; then
        HOSTNAME_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "")
        if [ -n "$HOSTNAME_IP" ] && [[ ! "$HOSTNAME_IP" =~ ^127\. ]]; then
            LOCAL_IPS+=("$HOSTNAME_IP")
        fi
    fi
    
    # Method 3: Using ifconfig (old method)
    if [ ${#LOCAL_IPS[@]} -eq 0 ] && command -v ifconfig &> /dev/null; then
        while IFS= read -r ip; do
            if [[ ! "$ip" =~ ^127\. ]] && [[ ! "$ip" =~ ^169\.254\. ]] && [[ ! "$ip" =~ ^100\. ]]; then
                LOCAL_IPS+=("$ip")
            fi
        done < <(ifconfig 2>/dev/null | grep -oP 'inet \K[\d.]+' || true)
    fi
    
    if [ ${#LOCAL_IPS[@]} -gt 1 ]; then
        echo "Multiple network interfaces found:"
        i=1
        for ip in "${LOCAL_IPS[@]}"; do
            echo -e "${GRAY}  $i. $ip${NC}"
            ((i++))
        done
        echo ""
        read -p "Select IP address (1-${#LOCAL_IPS[@]}): " CHOICE
        if [[ "$CHOICE" =~ ^[0-9]+$ ]] && [ "$CHOICE" -ge 1 ] && [ "$CHOICE" -le ${#LOCAL_IPS[@]} ]; then
            NODE_IP="${LOCAL_IPS[$((CHOICE-1))]}"
        else
            NODE_IP="${LOCAL_IPS[0]}"
            echo -e "${YELLOW}Using first IP: $NODE_IP${NC}"
        fi
    elif [ ${#LOCAL_IPS[@]} -eq 1 ]; then
        NODE_IP="${LOCAL_IPS[0]}"
        echo -e "${GREEN}Using local IP: $NODE_IP${NC}"
    else
        echo -e "${YELLOW}Could not detect local IP. You'll need to specify it manually.${NC}"
        read -p "Enter your IP address: " NODE_IP
        if [ -z "$NODE_IP" ]; then
            echo -e "${RED}No IP address provided. Exiting.${NC}"
            echo ""
            echo "Press Enter to exit..."
            read
            exit 1
        fi
    fi
fi

echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  Starting Ray Head Node${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# Check if Ray is already running (with timeout to prevent hanging)
echo -e "${YELLOW}Checking if Ray is already running...${NC}"
RAY_STATUS=""
if timeout 5 ray status 2>&1 | grep -vq "No cluster status\|error"; then
    RAY_STATUS="running"
fi

if [ -n "$RAY_STATUS" ]; then
    echo -e "${YELLOW}Ray is already running. Stopping existing instance...${NC}"
    timeout 5 ray stop 2>&1 > /dev/null || true
    sleep 2
else
    echo -e "${GRAY}Ray is not running (or check timed out). Starting fresh...${NC}"
fi

# Start Ray head node
echo -e "${YELLOW}Starting Ray head node...${NC}"
if [ "$USE_VPN" = true ]; then
    echo -e "${GRAY}  Using VPN IP: $NODE_IP${NC}"
    echo -e "${CYAN}  Share this IP with your friend: $NODE_IP${NC}"
    echo ""
    RAY_CMD="ray start --head --port=6379 --dashboard-host=0.0.0.0 --node-ip-address=$NODE_IP"
else
    echo -e "${GRAY}  Using local IP: $NODE_IP${NC}"
    echo -e "${CYAN}  Share this IP with your friend: $NODE_IP${NC}"
    echo ""
    RAY_CMD="ray start --head --port=6379 --dashboard-host=0.0.0.0"
fi

echo -e "${GRAY}Executing Ray command...${NC}"

# Execute Ray command
if eval "$RAY_CMD" 2>&1; then
    RAY_SUCCESS=true
else
    RAY_SUCCESS=false
fi

# Check if Ray started successfully by looking for success indicators
if ! $RAY_SUCCESS || ! timeout 3 ray status 2>&1 | grep -q "Local node IP\|Ray runtime started"; then
    # Try one more time to check
    sleep 1
    if timeout 3 ray status 2>&1 | grep -q "Local node IP\|Ray runtime started"; then
        RAY_SUCCESS=true
    else
        RAY_SUCCESS=false
    fi
fi

if [ "$RAY_SUCCESS" = true ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  Head Node Started Successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "${GREEN}Your Ray cluster is ready!${NC}"
    echo ""
    echo -e "${CYAN}Share this information with your friend:${NC}"
    echo -e "${YELLOW}  IP Address: $NODE_IP${NC}"
    echo -e "${YELLOW}  Port: 6379${NC}"
    echo ""
    echo -e "${YELLOW}Your friend should run: ./start_worker_node.sh${NC}"
    echo -e "${GRAY}  Or manually: ray start --address='$NODE_IP:6379'${NC}"
    echo ""
    
    echo -e "${CYAN}Ray Dashboard:${NC}"
    echo -e "${YELLOW}  http://$NODE_IP:8265${NC}"
    echo ""
    
    echo -e "${CYAN}Current cluster status:${NC}"
    ray status
    
    echo ""
    echo -e "${GREEN}Head node is running. Keep this window open!${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop when done.${NC}"
    echo ""
    echo -e "${GRAY}Waiting for worker nodes to connect...${NC}"
    echo -e "${GRAY}Press Ctrl+C to stop${NC}"
    echo ""
    
    # Keep script running and periodically show status
    ITERATION=0
    while true; do
        sleep 30
        ((ITERATION++))
        if [ $((ITERATION % 4)) -eq 0 ]; then
            echo ""
            echo -e "${CYAN}[Status Check] Current cluster:${NC}"
            ray status
            echo ""
        fi
    done
else
    echo ""
    echo -e "${RED}Failed to start Ray head node.${NC}"
    echo ""
    echo -e "${YELLOW}Troubleshooting:${NC}"
    echo -e "${GRAY}  1. Check if port 6379 is already in use: sudo lsof -i :6379${NC}"
    echo -e "${GRAY}  2. Check firewall settings: sudo ufw status${NC}"
    echo -e "${GRAY}  3. Try running as administrator: sudo ./start_head_node.sh${NC}"
    echo -e "${GRAY}  4. Make sure Ray is properly installed: pip install ray[default]${NC}"
    echo ""
    echo "Press Enter to exit..."
    read
    exit 1
fi

