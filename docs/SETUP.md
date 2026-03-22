# Emrose Media Studios — Setup Guide

From zero to generating videos on a fresh Mac.

## Prerequisites

- **macOS** (tested on Sonoma/Sequoia — Linux works too)
- **Python 3.10+** — check with `python3 --version`
  - If missing: `brew install python`
- **ffmpeg** — required for video/audio processing
  - `brew install ffmpeg`
- **Git** — `brew install git` (or Xcode command line tools)

## 1. Clone the Repo

```bash
git clone https://github.com/EmroseMediaStudios/MediaApp.git
cd MediaApp
```

## 2. Create Python Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Set Up API Keys

Create a `.env` file in the project root:

```bash
cat > .env << 'EOF'
OPENAI_API_KEY=sk-your-openai-key-here
ELEVENLABS_API_KEY=your-elevenlabs-key-here
HF_TOKEN=hf_your-huggingface-token-here
EOF
```

### Where to get keys:

| Key | Where | Cost |
|-----|-------|------|
| `OPENAI_API_KEY` | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) | Pay-per-use (~$0.01/script + $0.08/image) |
| `ELEVENLABS_API_KEY` | [elevenlabs.io/app/settings/api-keys](https://elevenlabs.io/app/settings/api-keys) | Free tier or paid plan (per character) |
| `HF_TOKEN` | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | Free (used for fallback image generation) |

## 4. YouTube Upload Setup (Optional)

To upload directly to YouTube from the app:

### a. Create Google Cloud OAuth credentials

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a project (or use existing)
3. Enable **YouTube Data API v3**
4. Go to **Credentials** → **Create Credentials** → **OAuth 2.0 Client ID**
5. Application type: **Desktop app**
6. Download the JSON file

### b. Save credentials

```bash
# Copy the downloaded file to the project root:
cp ~/Downloads/client_secret_XXXXX.json ./client_secret.json
```

### c. Authenticate

1. Start the app (`./run.sh`)
2. Open `http://localhost:7749`
3. Click the "⚠ YouTube: Click to connect" link on the dashboard
4. Sign into your Google account and authorize
5. For Brand Account channels: switch to the correct channel in YouTube *before* clicking authorize

Tokens are saved locally in `youtube_tokens/` and auto-refresh.

## 5. Run the App

```bash
./run.sh
```

Open **http://localhost:7749** in your browser.

If `run.sh` isn't executable:
```bash
chmod +x run.sh
```

## 6. Generate Your First Video

1. Select a channel from the dashboard
2. Click **💡 Generate Topic Idea** (or type your own)
3. Click **✍️ Create Video** and paste/type the topic
4. Click **Generate Script** — review the scenes
5. Check the estimated duration (should be 8+ min for mid-roll eligibility)
6. Click **🚀 Approve & Generate Video**
7. Watch progress in real-time
8. Video + Short land in `~/Desktop/EmroseMedia/<ChannelName>/`

## File Structure

```
MediaApp/
├── .env                    # API keys (NOT in git)
├── client_secret.json      # Google OAuth (NOT in git)
├── youtube_tokens/          # Auth tokens (NOT in git)
├── run.sh                  # Launch script
├── requirements.txt        # Python dependencies
├── channels/               # Channel configs (JSON)
├── app/
│   ├── app.py              # Flask web app
│   ├── generator.py        # Video generation pipeline
│   ├── youtube_upload.py   # YouTube API upload
│   ├── youtube_metrics.py  # YouTube analytics
│   └── templates/          # HTML templates
├── output/                 # Generated videos (NOT in git)
└── topic_bank.json         # Used topics tracker
```

## Troubleshooting

### `No module named 'cv2'`
```bash
source .venv/bin/activate
pip install opencv-python-headless
```

### `ffmpeg: command not found`
```bash
brew install ffmpeg
```

### YouTube upload stays "private"
Your API project may need [audit verification](https://support.google.com/youtube/contact/yt_api_form). Until then, uploads default to private. You can change visibility manually in YouTube Studio.

### ElevenLabs rate limiting
The app auto-retries on 429 errors with backoff. If persistent, check your plan's character quota at [elevenlabs.io](https://elevenlabs.io).

### Script too short
The generator targets 1000-1300 words (8-10 min video). If it falls short, it auto-expands. You can also manually edit scenes on the script page — the estimated duration updates live.

### Images falling back to procedural
DALL-E 3 is primary. If it fails (quota/billing), it falls back to FLUX via HuggingFace (free but slower). Procedural dark gradients are the last resort.

## Updating

```bash
cd MediaApp
git pull
source .venv/bin/activate
pip install -r requirements.txt  # in case new deps were added
./run.sh
```
