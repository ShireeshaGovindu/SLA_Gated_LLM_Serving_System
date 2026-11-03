#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="llm-serving"
ENGINE_HOST="127.0.0.1"
ENGINE_PORT="${ENGINE_PORT:-30000}"
GATEWAY_HOST="127.0.0.1"
GATEWAY_PORT="${GATEWAY_PORT:-8081}"
READY_FILE="/tmp/${SESSION}-setup-ready"
FAIL_FILE="/tmp/${SESSION}-setup-failed"

# Install tmux if not already installed
if ! command -v tmux >/dev/null 2>&1; then
    sudo apt-get update && sudo apt-get install -y tmux
fi

# Verify tmux installation
command -v tmux >/dev/null 2>&1 || {
    echo "ERROR: tmux installation failed."
    exit 1
}

# Prevent duplicate sessions
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session already exists. Attach using: tmux attach -t $SESSION"
    exit 1
fi

rm -f "$READY_FILE" "$FAIL_FILE"

# Create four terminal windows
tmux new-session -d -s "$SESSION" -n setup
tmux new-window -t "$SESSION" -n engine
tmux new-window -t "$SESSION" -n gateway
tmux new-window -t "$SESSION" -n health

# ==================================================
# STEP 1: ENVIRONMENT SETUP
# ==================================================

SETUP_COMMAND="cd '$PROJECT_DIR'; set -Eeuo pipefail; nvidia-smi && python -m pip install --upgrade pip setuptools wheel && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && export PATH=\"\$HOME/.cargo/bin:\$PATH\" && python -m pip install 'sglang[all]==0.5.22' fastapi uvicorn httpx aiohttp numpy requests transformers && python -c \"import torch; print('PyTorch:',torch.__version__); print('CUDA available:',torch.cuda.is_available()); print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not detected')\" && python --version && sglang version && touch '$READY_FILE' || { touch '$FAIL_FILE'; echo 'SETUP FAILED'; }; exec bash"

tmux send-keys -t "$SESSION:setup" "$SETUP_COMMAND" C-m

tmux select-window -t "$SESSION:setup"

echo "STEP 1: Installing dependencies and validating environment..."

until [[ -f "$READY_FILE" || -f "$FAIL_FILE" ]]; do
    sleep 2
done

if [[ -f "$FAIL_FILE" ]]; then
    echo "Setup failed. Stopping startup sequence."
    echo "See: tmux attach -t $SESSION"
    exit 1
fi

# ==================================================
# STEP 2: START INFERENCE ENGINE
# ==================================================

echo "STEP 2: Starting SGLang engine..."

tmux send-keys -t "$SESSION:engine" "cd '$PROJECT_DIR' && ./scripts/start_engine.sh" C-m

until (echo >"/dev/tcp/$ENGINE_HOST/$ENGINE_PORT") >/dev/null 2>&1; do
    sleep 2
done

echo "Engine is ready."

# ==================================================
# STEP 3: START GATEWAY
# ==================================================

echo "STEP 3: Starting API gateway..."

tmux send-keys -t "$SESSION:gateway" "cd '$PROJECT_DIR' && ./scripts/start_gateway.sh" C-m

until (echo >"/dev/tcp/$GATEWAY_HOST/$GATEWAY_PORT") >/dev/null 2>&1; do
    sleep 2
done

echo "Gateway is ready."

# ==================================================
# STEP 4: HEALTH CHECKS
# ==================================================

echo "STEP 4: Running health checks..."

tmux send-keys -t "$SESSION:health" "cd '$PROJECT_DIR' && ./scripts/sglang_healthcheck.sh && ./scripts/gateway_healthcheck.sh; echo; echo 'Health checks completed.'; exec bash" C-m

# ==================================================
# COMPLETE
# ==================================================

tmux select-window -t "$SESSION:setup"

echo ""
echo "======================================"
echo "STARTUP SEQUENCE COMPLETED"
echo "======================================"
echo "Terminal 1: Environment Setup"
echo "Terminal 2: SGLang Engine"
echo "Terminal 3: API Gateway"
echo "Terminal 4: Health Checks"
echo "======================================"

[[ -t 1 ]] && tmux attach -t "$SESSION" || echo "Attach with: tmux attach -t $SESSION"