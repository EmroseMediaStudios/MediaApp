#!/bin/zsh
cd ~/Desktop/MediaApp

# Homebrew
eval "$(/opt/homebrew/bin/brew shellenv zsh)" 2>/dev/null

# Load API keys
set -a
source .env
set +a

# Activate venv
source .venv/bin/activate

echo ""
echo "⬡ Emrose Media Studio"
echo "  Starting on http://localhost:7749"
echo "  Press Ctrl+C to stop"
echo ""

# Open browser after server has time to start
(sleep 3 && open "http://localhost:7749") &

# Launch server
python -m app.app
