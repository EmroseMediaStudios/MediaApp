# ⬡ Video Studio

Local multi-channel automated video generator. Generates narrated videos with AI imagery and Ken Burns animation.

## Quick Start (Mac)

### 1. Prerequisites
- Python 3.10+ (comes with macOS or `brew install python`)
- ffmpeg (`brew install ffmpeg`)

### 2. Setup
```bash
cd deadlight-codex
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Set API Keys
```bash
export OPENAI_API_KEY="your-openai-key"
export ELEVENLABS_API_KEY="your-elevenlabs-key"
export HF_TOKEN="your-huggingface-token"
```

Or add them to your `~/.zshrc` / `~/.bash_profile` to persist.

### 4. Run
```bash
./run.sh
```

Open **http://localhost:5000** in your browser.

## How It Works

1. **Select a channel** from the dashboard
2. **Generate a topic idea** (or enter your own)
3. **Review & edit the script** (narration + image prompts per scene)
4. **Generate the video** — watch real-time progress
5. **Play the result** directly in your browser
6. **Mark as uploaded** when you push it to YouTube

## Channels

| Channel | Narrator | Voice | Style |
|---|---|---|---|
| DeadlightCodex | The Keeper | Victor | Cosmic horror archive |
| ZeroTraceArchive | The Archivist | Daniel | Documentary case files |
| TheUnwrittenWing | The Storyteller | Lily | Emotional surreal fiction |
| RemnantsProject | The Observer | Adam | Post-human observation |
| SomnusProtocol | The Guide | River | Sleep-inducing calm |
| AutonomousStack | The Instructor | Eric | Tech tutorials |
| GrayMeridian | The Thinker | Bill | Psychology & behavior |

## Adding New Channels

1. Copy `channels/_template.json` to `channels/your_channel.json`
2. Fill in all fields (voice, narrator, visual theme, etc.)
3. Restart the app — it appears automatically

## Output Structure

```
output/<channel_id>/YYYYMMDD_HHMMSS_<title>/
├── video.mp4          # Main video (1080p, 30fps)
├── short.mp4          # YouTube Short (if generated)
├── metadata.json      # Title, duration, YouTube status
├── plan.json          # Scene plan
└── images/            # Generated scene images
```

## Costs

- **Images**: Free (FLUX via HuggingFace)
- **Narration**: ElevenLabs (paid, per character)
- **Script**: OpenAI GPT-4o (paid, per token)
- **Video assembly**: Free (local processing)
