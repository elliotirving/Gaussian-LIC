#!/usr/bin/env bash
# Build the SPNet depth-completion TensorRT engine ON THE ORIN.
# TensorRT engines are specific to the GPU arch (sm_87) AND the TensorRT version
# (10.x here), so the engine MUST be built on the device — you cannot ship the
# x86/TRT-8.6 engine from the README.
#
# Two stages:
#   1. ONNX export  — Python, needs torch + torchvision (SPNet uses
#      torchvision.ops.StochasticDepth). Done with the SYSTEM Python 3.10, where
#      the cp310 torch/torchvision wheels are installed (the conda ROS env is
#      Python 3.11 and deliberately has no torch — see Dockerfile).
#   2. Engine build — trtexec only, no Python/torch needed.
#
# If you already have spnet_512_640.onnx / spnet_480_640.onnx (e.g. exported on
# an x86 desktop with the original SPNet conda flow), drop them in ckpt/ and the
# export stage is skipped — only trtexec runs.
#
# Prereq for the export stage: Large_300.pth (Google Drive link in the README)
# in src/Gaussian-LIC/ckpt/.
#
# Run inside the container:
#   docker compose -f docker/docker-compose.yml run --rm \
#       gaussian-lic /usr/local/bin/build_engine.sh
set -euo pipefail

CKPT_DIR=/root/catkin_gaussian/src/Gaussian-LIC/ckpt
cd "${CKPT_DIR}"

PY=/usr/bin/python3                       # system Python 3.10 (has the cp310 torch/torchvision)

export_one() {
  local onnx="$1" script="$2"
  if [ -f "${onnx}" ]; then
    echo ">>> ${onnx} already present — skipping export."
    return
  fi
  if [ ! -f Large_300.pth ]; then
    echo "ERROR: ${CKPT_DIR}/Large_300.pth not found and ${onnx} missing."
    echo "Either place Large_300.pth here, or export ${onnx} elsewhere and drop it in."
    exit 1
  fi
  echo ">>> Exporting ${onnx} with system Python (torch/torchvision) ..."
  "${PY}" "${script}"
}

# torch + torchvision are already in the system Python (installed in the
# Dockerfile); make sure onnx is too.
"${PY}" -c 'import onnx' 2>/dev/null || "${PY}" -m pip install --no-cache-dir onnx

export_one spnet_512_640.onnx export_onnx_512_640.py
export_one spnet_480_640.onnx export_onnx_480_640.py
export_one spnet_640_800.onnx export_onnx_640_800.py   # odin1 half-res (1600x1296 -> 800x640)

# Engine build with the device's trtexec (TensorRT 10 from JetPack).
TRTEXEC=/usr/src/tensorrt/bin/trtexec
[ -x "${TRTEXEC}" ] || TRTEXEC=$(command -v trtexec)

"${TRTEXEC}" --onnx=spnet_512_640.onnx --saveEngine=spnet_512_640.engine --fp16 \
  --optShapes=rgb:1x3x512x640,depth:1x1x512x640,mask:1x1x512x640
"${TRTEXEC}" --onnx=spnet_480_640.onnx --saveEngine=spnet_480_640.engine --fp16 \
  --optShapes=rgb:1x3x480x640,depth:1x1x480x640,mask:1x1x480x640
"${TRTEXEC}" --onnx=spnet_640_800.onnx --saveEngine=spnet_640_800.engine --fp16 \
  --optShapes=rgb:1x3x640x800,depth:1x1x640x800,mask:1x1x640x800

echo ">>> Engines built: ${CKPT_DIR}/spnet_{512_640,480_640,640_800}.engine"
