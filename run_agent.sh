#!/bin/bash
# Load API keys from environment
source ~/.zshenv 2>/dev/null
cd /Users/glasschen/projs/slay_all_the_spires
export PYTHONIOENCODING=utf-8
LOG_DIR="data/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/agent_$(date +%Y%m%d_%H%M).log"
ln -sf "$(pwd)/$LOG_FILE" /tmp/sts_agent.log
exec .venv/bin/python -m sts_agent.main 2>"$LOG_FILE"
