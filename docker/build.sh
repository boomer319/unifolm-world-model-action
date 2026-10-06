#!/usr/bin/env bash
# Build the UnifoLM-WMA image once and record the host identity for compose.
#
# Unlike dreamzero_docker_env/build.sh this script does NOT create a venv or run
# pip: all dependencies are baked into the image (see docker/Dockerfile), so the
# image is self-contained and `docker compose run <svc>` works immediately.
#
# Prereq: the repo must live at
#   /data/docker-services/world_action_models/unifolm_wma   (on .240)
# because docker-compose.yml bind-mounts that path as /workspace.
set -euo pipefail
cd "$(dirname "$0")"

USERNAME=$(whoami)
tmp_UID=$(id -u)
tmp_GID=$(id -g)

cat > .env <<EOF
tmp_UID=$tmp_UID
tmp_GID=$tmp_GID
USERNAME=$USERNAME
EOF

echo ">>> building unifolm-wma:py310-cu124 as $USERNAME (uid=$tmp_UID gid=$tmp_GID)"
docker compose build
echo ">>> verifying the environment (CPU service)"
docker compose run --rm wma-env
echo ">>> done. Next:"
echo "    docker compose run --rm wma-convert   # LeRobot v2 -> WMA format"
echo "    docker compose run --rm wma-load      # load Base ckpt, report mismatches"
echo "    docker compose run --rm wma-smoke     # 100-step timing measurement"