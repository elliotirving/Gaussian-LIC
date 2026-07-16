#!/usr/bin/env bash
# Activate the ROS env + built workspace + library paths (see env.sh), then run
# whatever was passed (the compose default is `sleep infinity`, i.e. idle —
# you exec in and drive things by hand).
set -e

source /usr/local/bin/gaussian_lic_env.sh

# Make interactive `docker compose exec gaussian-lic bash` sessions pick up the
# same env automatically — no entrypoint wrapper needed. Idempotent; re-run on
# every container start (the container fs resets on recreate).
if ! grep -q gaussian_lic_env /root/.bashrc 2>/dev/null; then
  echo 'source /usr/local/bin/gaussian_lic_env.sh' >> /root/.bashrc
fi

exec "$@"
