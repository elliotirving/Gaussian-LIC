#!/usr/bin/env bash
# Build the SPNet depth-completion TensorRT engine(s) ON-DEVICE. Engines are
# specific to the GPU arch AND the TensorRT version, so they must be built on the
# target machine — you cannot ship an engine between x86/TRT-8.6 and Orin/TRT-10.
#
# IDENTICAL on both platforms — platform specifics come from image ENV vars:
#   SPNET_PYTHON       python with torch+torchvision for the ONNX export stage
#                      (x86: the `spnet` conda env; orin: system python3).
#   TRTEXEC            trtexec binary (x86: /opt/TensorRT-8.6.1.6/bin/trtexec;
#                      orin: JetPack's /usr/src/tensorrt/bin/trtexec).
#   SPNET_RESOLUTIONS  space-separated "<H>_<W>" list to build (default 512_640 480_640).
#
# Prereqs in ckpt/: the SPNet repo and Large_300.pth (see the root README's SPNet
# setup). If a spnet_<res>.onnx is already present, its export is skipped.
#
# Run inside the container:
#   docker compose -f docker/docker-compose.yml run --rm --name glic <service> \
#       /usr/local/bin/build_engine.sh
set -euo pipefail

CKPT_DIR="${GLIC_SRC_DIR:-/root/catkin_gaussian/src/Gaussian-LIC}/ckpt"
cd "${CKPT_DIR}"

PY="${SPNET_PYTHON:-python3}"
TRTEXEC="${TRTEXEC:-$(command -v trtexec || echo /usr/src/tensorrt/bin/trtexec)}"
RESOLUTIONS="${SPNET_RESOLUTIONS:-512_640 480_640}"

# onnx is needed by the export scripts; install into SPNET_PYTHON if absent.
"${PY}" -c 'import onnx' 2>/dev/null || "${PY}" -m pip install --no-cache-dir onnx

for res in ${RESOLUTIONS}; do
  H="${res%_*}"; W="${res#*_}"
  onnx="spnet_${res}.onnx"
  engine="spnet_${res}.engine"
  script="export_onnx_${res}.py"

  if [ -f "${onnx}" ]; then
    echo ">>> ${onnx} already present — skipping export."
  else
    if [ ! -f Large_300.pth ]; then
      echo "ERROR: ${CKPT_DIR}/Large_300.pth not found and ${onnx} missing."
      echo "Place Large_300.pth (+ the SPNet repo) here, or drop a pre-exported"
      echo "${onnx} in. See the root README SPNet setup."
      exit 1
    fi
    echo ">>> Exporting ${onnx} with ${PY} ..."
    "${PY}" "${script}"
  fi

  echo ">>> Building ${engine} (${W}x${H}, fp16) with ${TRTEXEC} ..."
  "${TRTEXEC}" --onnx="${onnx}" --saveEngine="${engine}" --fp16 \
    --optShapes=rgb:1x3x${H}x${W},depth:1x1x${H}x${W},mask:1x1x${H}x${W}
done

echo ">>> Engines built in ${CKPT_DIR} for: ${RESOLUTIONS}"
