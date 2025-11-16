#!/bin/bash
# Quick Start Script for Friend's Computer (Worker Node)
# This script helps your friend connect their GPU to your Ray cluster
#
# Usage: ./start_worker_node.sh

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
echo -e "${CYAN}  Ray Cluster Worker Node Setup${NC}"
echo -e "${CYAN}========================================${NC}"
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

# Check if GPU is available
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

if [ -n "$TAILSCALE_IP" ]; then
    echo -e "${GREEN}Tailscale VPN detected! Your VPN IP: $TAILSCALE_IP${NC}"
    echo -e "${YELLOW}Make sure the head node is also on the same VPN network.${NC}"
else
    echo -e "${YELLOW}No Tailscale VPN detected.${NC}"
    echo -e "${YELLOW}If connecting remotely, make sure VPN is running!${NC}"
fi

# Get head node address
echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  Connection Setup${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

echo -e "${YELLOW}Enter the head node address:${NC}"
echo -e "${GRAY}  - For local network: Use the local IP (e.g., 192.168.1.100)${NC}"
echo -e "${GRAY}  - For VPN/remote: Use the VPN IP (e.g., 100.64.1.2)${NC}"
echo ""

read -p "Head node IP address: " HEAD_NODE_IP

# Validate IP format (basic check)
if [[ ! "$HEAD_NODE_IP" =~ ^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}$ ]]; then
    echo -e "${YELLOW}Warning: IP address format looks unusual. Continuing anyway...${NC}"
fi

# Ask for port (default 6379)
echo ""
read -p "Port (press Enter for default: 6379): " PORT
if [ -z "$PORT" ]; then
    PORT="6379"
fi

RAY_ADDRESS="${HEAD_NODE_IP}:${PORT}"

echo ""
echo -e "${CYAN}Connecting to Ray cluster at: $RAY_ADDRESS${NC}"
echo ""

# Check if Ray is already running
echo -e "${YELLOW}Checking if Ray is already running...${NC}"
if timeout 5 ray status 2>&1 | grep -vq "No cluster status"; then
    echo -e "${YELLOW}Ray is already running. Stopping existing Ray instance...${NC}"
    timeout 5 ray stop 2>&1 > /dev/null || true
    sleep 2
fi

# Start Ray worker
echo -e "${YELLOW}Starting Ray worker node...${NC}"
echo -e "${GRAY}This will connect to: $RAY_ADDRESS${NC}"
echo ""

# Execute Ray command
if ray start --address="$RAY_ADDRESS" 2>&1; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  Successfully Connected!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "${GREEN}Your GPU is now available to the cluster!${NC}"
    echo ""
    echo -e "${YELLOW}To verify connection, the head node can run: ray status${NC}"
    echo ""
    echo -e "${YELLOW}To stop this worker node, press Ctrl+C or run: ray stop${NC}"
    echo ""
    
    echo -e "${CYAN}Current cluster status:${NC}"
    ray status
    
    echo ""
    echo -e "${GREEN}Worker node is running. Keep this window open!${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop when done.${NC}"
    echo ""
    echo -e "${GRAY}Waiting... (Press Ctrl+C to stop)${NC}"
    
    # Keep script running
    while true; do
        sleep 10
    done
else
    echo ""
    echo -e "${RED}Failed to connect to Ray cluster.${NC}"
    echo ""
    echo -e "${YELLOW}Troubleshooting:${NC}"
    echo -e "${GRAY}  1. Make sure the head node is running:${NC}"
    echo -e "${GRAY}     Head node should run: ray start --head${NC}"
    echo -e "${GRAY}  2. Check the IP address is correct${NC}"
    echo -e "${GRAY}  3. Check firewall settings: sudo ufw status${NC}"
    echo -e "${GRAY}  4. If using VPN, make sure both computers are connected${NC}"
    echo -e "${GRAY}  5. Test connectivity: ping $HEAD_NODE_IP${NC}"
    echo ""
    echo "Press Enter to exit..."
    read
    exit 1
fi

