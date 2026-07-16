#!/usr/bin/env bash
set -e

TENSORRT_ROOT=~/Software/TensorRT-8.6.1.6
TRT_BIN=$TENSORRT_ROOT/bin/trtexec
TRT_LIB=$TENSORRT_ROOT/targets/x86_64-linux-gnu/lib

echo ">>> Deactivating conda env (if any)"
conda deactivate || true

echo ">>> Setting TensorRT LD_LIBRARY_PATH"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TRT_LIB

echo ">>> Building TensorRT engine: 512x640"
$TRT_BIN \
  --onnx=spnet_512_640.onnx \
  --saveEngine=spnet_512_640.engine \
  --fp16 \
  --optShapes=rgb:1x3x512x640,depth:1x1x512x640,mask:1x1x512x640

echo ">>> Building TensorRT engine: 480x640"
$TRT_BIN \
  --onnx=spnet_480_640.onnx \
  --saveEngine=spnet_480_640.engine \
  --fp16 \
  --optShapes=rgb:1x3x480x640,depth:1x1x480x640,mask:1x1x480x640

echo ">>> TensorRT engine build finished."