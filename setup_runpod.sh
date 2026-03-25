#!/bin/bash
# Setup script for RunPod — Parameter Golf
# Run this once on a fresh RunPod instance

set -e

echo "=== Parameter Golf Setup ==="

# 1. Clone your fork
cd /workspace
git clone https://github.com/AalindTiwari/parameter-golf.git
cd parameter-golf

# 2. Download data (this takes a while)
python3 data/cached_challenge_fineweb.py --variant sp1024

# 3. Verify data
ls -la data/datasets/fineweb10B_sp1024/
ls -la data/tokenizers/

echo ""
echo "=== Setup Complete ==="
echo "Run with:"
echo "  cd /workspace/parameter-golf"
echo "  bash run_depth_recurrence.sh 1337   # seed 1"
echo "  bash run_depth_recurrence.sh 42     # seed 2"
echo "  bash run_depth_recurrence.sh 2025   # seed 3"
