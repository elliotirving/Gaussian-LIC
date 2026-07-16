#!/usr/bin/env bash
set -e

echo ">>> Creating conda env: spnet"
conda create -n spnet python=3.8 -y

echo ">>> Activating conda env: spnet"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate spnet

echo ">>> Installing PyTorch"
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

echo ">>> Installing ONNX & ONNXRuntime"
pip install onnx onnxruntime

echo ">>> Done. Environment spnet is ready."