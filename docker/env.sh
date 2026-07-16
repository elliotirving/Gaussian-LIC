# Sourceable environment for the Gaussian-LIC container. IDENTICAL on both
# platforms — every platform-specific value comes from image ENV vars set in the
# Dockerfile (the two Dockerfiles are the only platform-specific files).
#
# Sourced by entrypoint.sh (before exec-ing the container command) and appended
# to /root/.bashrc, so every interactive `docker exec ... bash` is ready to use.
#
# No `set -e` / no `exec` here — must be safe to source into any shell.
#
# Contract (see Dockerfile.<platform>):
#   GLIC_CONDA_ENV    conda env to micromamba-activate first (orin). Unset on x86.
#   GLIC_ROS_SETUP    path to the ROS setup.bash (apt path on x86, conda on orin).
#   GLIC_EXTRA_WS     space-separated extra devel/setup.bash to source (COCOLIC on x86).
#   GLIC_GAUSSIAN_WS  the built gaussian workspace setup.bash (sourced if present).
#   GLIC_LD_PREFIX    LD_LIBRARY_PATH prefix (order matters; baked per-platform).

# Activate the conda ROS env if this image uses one (orin/RoboStack). On the
# apt-ROS image (x86) GLIC_CONDA_ENV is unset, so this is skipped.
if [ -n "${GLIC_CONDA_ENV:-}" ]; then
  eval "$(micromamba shell hook --shell bash)"
  micromamba activate "${GLIC_CONDA_ENV}"
fi

# ROS environment.
[ -n "${GLIC_ROS_SETUP:-}" ] && [ -f "${GLIC_ROS_SETUP}" ] && source "${GLIC_ROS_SETUP}"

# Extra workspaces (e.g. COCOLIC on x86) before the gaussian workspace.
for _ws in ${GLIC_EXTRA_WS:-}; do
  [ -f "${_ws}" ] && source "${_ws}"
done

# The built gaussian workspace (present after build_ws.sh; lives in a named volume).
_gws="${GLIC_GAUSSIAN_WS:-/root/catkin_gaussian/devel/setup.bash}"
[ -f "${_gws}" ] && source "${_gws}"

# Runtime library path. GLIC_LD_PREFIX is baked per-platform: LibTorch first,
# then CUDA OpenCV, then (orin) the conda ros libs whose newer libstdc++ must
# precede the system one, then system TensorRT / CUDA.
export LD_LIBRARY_PATH="${GLIC_LD_PREFIX:-}:${LD_LIBRARY_PATH:-}"
