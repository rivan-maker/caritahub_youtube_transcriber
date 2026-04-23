#!/usr/bin/env bash
# One-shot installer for CaritaHub YouTube Transcriber on Ubuntu 22.04+ EC2.
# Usage (on the EC2 box, inside the cloned repo):
#   sudo API_KEY=xxx YOUTUBE_COOKIES=yyy bash deploy/install.sh
#
# Required env: API_KEY
# Optional env: YOUTUBE_COOKIES (base64), DEFAULT_MODEL (default: base),
#               JOB_TTL_SECONDS (default: 86400)

set -euo pipefail

: "${API_KEY:?API_KEY is required. Example: sudo API_KEY=xxx bash deploy/install.sh}"
YOUTUBE_COOKIES_FILE="${YOUTUBE_COOKIES_FILE:-}"
YOUTUBE_COOKIES="${YOUTUBE_COOKIES:-}"
DEFAULT_MODEL="${DEFAULT_MODEL:-base}"
JOB_TTL_SECONDS="${JOB_TTL_SECONDS:-86400}"
APP_PORT="${APP_PORT:-8001}"

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${APP_DIR}/data"
ENV_FILE="/etc/caritahub-transcriber.env"
SERVICE_NAME="caritahub-transcriber"
RUN_USER="${SUDO_USER:-ubuntu}"

log() { printf '\n\033[1;34m▶ %s\033[0m\n' "$*"; }

if [[ $EUID -ne 0 ]]; then
    echo "Run with sudo: sudo API_KEY=... bash deploy/install.sh"
    exit 1
fi

if [[ ! -f "${APP_DIR}/server.py" ]]; then
    echo "server.py not found at ${APP_DIR} — run this from the repo root."
    exit 1
fi

log "Installing system packages (Python 3.11, ffmpeg, git, unzip)"
apt-get update
apt-get install -y software-properties-common curl unzip
if ! command -v python3.11 &>/dev/null; then
    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get update
fi
apt-get install -y python3.11 python3.11-venv ffmpeg git

if ! command -v deno &>/dev/null; then
    log "Installing Deno (required by yt-dlp for YouTube JS runtime)"
    curl -fsSL https://github.com/denoland/deno/releases/latest/download/deno-x86_64-unknown-linux-gnu.zip -o /tmp/deno.zip
    unzip -oq /tmp/deno.zip -d /tmp/
    mv /tmp/deno /usr/local/bin/
    chmod +x /usr/local/bin/deno
    rm /tmp/deno.zip
fi
deno --version | head -1

log "Creating data dir at ${DATA_DIR}"
mkdir -p "${DATA_DIR}"
chown -R "${RUN_USER}:${RUN_USER}" "${APP_DIR}"

log "Configuring yt-dlp to allow remote-components (required for YouTube JS challenges)"
RUN_HOME="$(getent passwd "${RUN_USER}" | cut -d: -f6)"
sudo -u "${RUN_USER}" mkdir -p "${RUN_HOME}/.config/yt-dlp"
sudo -u "${RUN_USER}" tee "${RUN_HOME}/.config/yt-dlp/config" > /dev/null <<'YTDLP_CFG'
--remote-components ejs:github
YTDLP_CFG

log "Setting up Python venv and installing dependencies"
sudo -u "${RUN_USER}" bash <<EOF
set -e
cd "${APP_DIR}"
if [[ ! -d .venv ]]; then
    python3.11 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -c "from faster_whisper import WhisperModel; WhisperModel('${DEFAULT_MODEL}', device='cpu', compute_type='int8'); print('whisper model ready')"
EOF

log "Writing ${ENV_FILE}"
cat > "${ENV_FILE}" <<EOF
API_KEY=${API_KEY}
YOUTUBE_COOKIES_FILE=${YOUTUBE_COOKIES_FILE}
YOUTUBE_COOKIES=${YOUTUBE_COOKIES}
DATA_DIR=${DATA_DIR}
DB_PATH=${DATA_DIR}/jobs.db
JOB_TTL_SECONDS=${JOB_TTL_SECONDS}
LOG_LEVEL=INFO
DEFAULT_MODEL=${DEFAULT_MODEL}
EOF
chmod 600 "${ENV_FILE}"
chown "${RUN_USER}:${RUN_USER}" "${ENV_FILE}"

log "Installing systemd service (port ${APP_PORT})"
sed "s|__APP_DIR__|${APP_DIR}|g; s|__RUN_USER__|${RUN_USER}|g; s|__APP_PORT__|${APP_PORT}|g" \
    "${APP_DIR}/deploy/caritahub-transcriber.service" \
    > /etc/systemd/system/${SERVICE_NAME}.service
systemctl daemon-reload
systemctl enable "${SERVICE_NAME}"
systemctl restart "${SERVICE_NAME}"

sleep 3
if ! systemctl is-active --quiet "${SERVICE_NAME}"; then
    echo "Service failed to start. Check: sudo journalctl -u ${SERVICE_NAME} -n 80"
    exit 1
fi

log "Health check"
curl -fsS http://127.0.0.1:${APP_PORT}/api/v1/health
echo

PUBLIC_IP="$(curl -fsS https://checkip.amazonaws.com 2>/dev/null | tr -d '[:space:]' || echo '<your-eip>')"
cat <<SUMMARY

─────────────────────────────────────────────────────────────────────
✅ Deployment complete
─────────────────────────────────────────────────────────────────────
Public URL:    http://${PUBLIC_IP}:${APP_PORT}
API docs:      http://${PUBLIC_IP}:${APP_PORT}/docs
Health:        http://${PUBLIC_IP}:${APP_PORT}/api/v1/health

Send X-API-Key header on every POST/GET/DELETE to /api/v1/transcribe*
  e.g. curl -H "X-API-Key: ${API_KEY}" http://${PUBLIC_IP}:${APP_PORT}/api/v1/health

⚠️  Make sure your EC2 security group allows inbound TCP ${APP_PORT} from the
    IPs that will call this API.

Logs:          sudo journalctl -u ${SERVICE_NAME} -f
Restart:       sudo systemctl restart ${SERVICE_NAME}
Update code:   cd ${APP_DIR} && git pull && sudo systemctl restart ${SERVICE_NAME}
─────────────────────────────────────────────────────────────────────
SUMMARY
