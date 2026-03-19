#!/bin/bash
# Launch Video Studio
cd "$(dirname "$0")"

# Activate venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

# Load .env file if it exists
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

# Check for required env vars
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠  OPENAI_API_KEY not set"
    echo "   export OPENAI_API_KEY='your-key'"
fi
if [ -z "$ELEVENLABS_API_KEY" ]; then
    echo "⚠  ELEVENLABS_API_KEY not set"
    echo "   export ELEVENLABS_API_KEY='your-key'"
fi
if [ -z "$HF_TOKEN" ]; then
    echo "⚠  HF_TOKEN not set (HuggingFace — needed for image generation)"
    echo "   export HF_TOKEN='your-token'"
fi

echo ""
echo "⬡ Emrose Media Studios"
echo "  Open http://localhost:7749 in your browser"
echo ""

python -m app.app
