#!/bin/zsh
# Emrose Media Studios — double-click to launch
# Put this file anywhere (Desktop, Dock, etc.)

cd "$(dirname "$0")"

# Homebrew path (Apple Silicon Mac)
eval "$(/opt/homebrew/bin/brew shellenv zsh)" 2>/dev/null

# Load API keys from .env file
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "⚠  Missing .env file!"
    echo "   Create a file called .env in the MediaApp folder with:"
    echo ""
    echo '   OPENAI_API_KEY=your-key-here'
    echo '   ELEVENLABS_API_KEY=your-key-here'
    echo '   HF_TOKEN=your-token-here'
    echo ""
    echo "   Press any key to exit."
    read -k1
    exit 1
fi

# Activate venv
source .venv/bin/activate

echo ""
echo "⬡ Emrose Media Studios"
echo "  Starting on http://localhost:7749"
echo "  Press Ctrl+C to stop"
echo ""

# Open browser after a brief delay
(sleep 2 && open "http://localhost:7749") &

# Launch
python -m app.app
