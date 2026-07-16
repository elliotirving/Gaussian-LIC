#!/usr/bin/env bash
# Shared entrypoint (identical on both platforms). Activates the ROS env +
# built workspace + library paths (env.sh), then runs the container command.
set -e

source /usr/local/bin/gaussian_lic_env.sh

# Make interactive `docker exec ... bash` sessions pick up the same env with no
# wrapper. Idempotent; re-run each container start (the container fs resets on
# recreate).
if ! grep -q gaussian_lic_env /root/.bashrc 2>/dev/null; then
  echo 'source /usr/local/bin/gaussian_lic_env.sh' >> /root/.bashrc
fi

cd "${GLIC_SRC_DIR:-/root/catkin_gaussian/src/Gaussian-LIC}"

exec "$@"
