# ⬡ Emrose Media Studios — Video Studio

Local multi-channel automated video generator. AI scripts → AI images → AI narration → Ken Burns animation → ffmpeg assembly → YouTube upload. All from a browser UI.

## Quick Start (Mac)

> **Full setup guide:** [docs/SETUP.md](docs/SETUP.md)

### 1. Prerequisites
- Python 3.10+ (`brew install python`)
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

Or add to `~/.zshrc` to persist.

### 4. Run
```bash
./run.sh
```
Open **http://localhost:7749**

## Pipeline

1. **Topic Generation** — GPT-4o generates topic ideas per channel (temperature 0.9, shuffled focus areas for variety)
2. **Script Generation** — GPT-4o writes narration + DALL-E image prompts per scene (temperature 0.7)
3. **Image Generation** — DALL-E 3 at 1792×1024
4. **Narration** — ElevenLabs TTS per scene (eleven_multilingual_v2)
5. **Ken Burns Animation** — Upscale images, apply zoom/pan effects
6. **Assembly** — ffmpeg composites to 1920×1080 @ 30fps MP4
7. **YouTube Metadata** — GPT-4o generates title, description, tags, hashtags
8. **YouTube Upload** — Scheduled or manual via API with per-channel OAuth credentials
9. **Scheduler** — Background thread checks every 60 seconds, uploads at scheduled times

## Channels

| Channel | Style | Days | Time (ET) | Notes |
|---|---|---|---|---|
| DeadlightCodex | Horror/cosmic horror + real-life horror | Thu–Sat | 6PM | |
| ZeroTraceArchive | Unsolved mysteries, cold cases | Wed–Fri | 3PM | |
| TheUnwrittenWing | Emotional surreal fiction | Wed–Fri | 3PM | |
| RemnantsProject | Post-human observation, lost history | Wed–Fri | 3PM | |
| SomnusProtocol | Sleep stories, ultra-calm | Any day | 8PM | scene_pause: 7s |
| SoftlightKingdom | Children's bedtime stories | Any day | 6PM | made_for_kids: true |
| GrayMeridian | Psychology, human behavior | Tue–Thu | 3PM | |
| EchelonVeil | Cryptids, UFOs, paranormal, conspiracy | Thu–Sat | 6PM | |
| Loreletics | Sports history & legends | Fri–Sun | 1PM | |

**Schedule**: 1 video/channel/week, minimum 6-day gap between uploads per channel.

## Ken Burns Animation

Images upscaled then animated with randomized zoom + pan per scene.

| Setting | Default | SoftlightKingdom | SomnusProtocol |
|---|---|---|---|
| Zoom range | 1.06–1.18 | 1.03–1.08 | 1.02–1.06 |
| Pan range | 0.10 | 0.05 | 0.03 |
| Upscale | 2688×1536 | 2304×1312 | 2304×1312 |
| Visible area | ~71% | ~83% | ~83% |

## Output Structure

```
~/Desktop/EmroseMedia/<channel_id>/YYYYMMDD_HHMMSS_<title>/
├── video.mp4          # Main video (1920×1080, 30fps)
├── short.mp4          # YouTube Short (if generated)
├── thumbnail.png      # Generated thumbnail
├── metadata.json      # Title, description, YouTube status, schedule
├── plan.json          # Scene plan (narration + image prompts)
└── images/            # DALL-E 3 generated scene images
```

## YouTube Integration

### Authentication
- Per-channel OAuth credentials stored in `youtube_tokens/`
- Each YouTube channel has its own OAuth token
- Authenticate via `/youtube/auth?channel=<youtube_channel_id>` in browser

### Channel Map
| App Channel | YouTube Channel ID |
|---|---|
| deadlight_codex | UCeR5uvuGWIQgxHsCg6KOYlg |
| zero_trace_archive | UC8Hy4GyP9T_qTp72c0kAYpg |
| gray_meridian | UC9bKvKkjAtoA9ysy2RnRPaQ |
| remnants_project | UC7tUZwk9l23qVwryihjBZSg |
| the_unwritten_wing | UCyjZXmPy4gG1xX2RJVJpfGA |
| somnus_protocol | UCukqYDdDPGG8G_jNvkb_2VQ |
| softlight_kingdom | UC7AI57ebEXWAXw6Uhku0hJQ |
| echelon_veil | UChiZ0p3OOyBfk8gWC9DJ0NQ |
| loreletics | UCkabQW3XUx83Cnj25LJcf3A |

### Verified Channels (phone verification complete)
- Loreletics
- Gray Meridian
- Softlight Kingdom

### Tag Handling
- Tags sanitized: alphanumeric + spaces + hyphens + apostrophes only
- Max 30 chars per tag, max 480 chars total
- On `invalidTags` error: drops all tags, retries with no tags
- Top 8 tags also injected as #hashtags at bottom of description (fallback)
- Set default tags per channel in YouTube Studio → Settings → Upload Defaults

### Upload Flow
1. Scheduler checks every 60 seconds for videos with `scheduled_upload` in metadata
2. When time arrives: uploads video, sets thumbnail, marks metadata as uploaded
3. Manual upload also available via UI button
4. Privacy: defaults to `private` (changeable in UI)
5. Made for Kids: auto-set based on channel config (SoftlightKingdom only)

## Video Specs

- **Target length**: 8-10 minutes (1000-1300 words, 12-16 scenes)
- **Resolution**: 1920×1080 @ 30fps
- **Source images**: DALL-E 3 at 1792×1024
- **Format**: MP4 (H.264 video, AAC audio)
- **Mid-roll ads**: Eligible at 8+ minutes

## Topic Generation

- Focus areas and examples shuffled randomly before prompt injection
- Temperature: 0.9 (high variety)
- Explicit "pick at RANDOM" instruction to avoid repetition
- Per-channel `CHANNEL_FOCUS` defines theme areas and example topics

## App Structure

```
deadlight-codex/
├── app/
│   ├── generator.py        # Core pipeline: topic → script → images → video
│   ├── scheduler.py        # Upload scheduler + posting schedule
│   ├── youtube_upload.py   # YouTube API upload + tag handling
│   ├── server.py           # Flask routes + Socket.IO events
│   └── standalone.py       # CLI entry point
├── channels/               # Channel config JSON files
│   └── _template.json      # Template for new channels
├── docs/
│   ├── SETUP.md            # Setup guide
│   └── POSTING_SCHEDULE.md # Posting times rationale
├── static/img/             # Channel logos, UI assets
├── templates/              # HTML templates
├── youtube_tokens/         # OAuth tokens per channel (gitignored)
├── requirements.txt
├── run.sh
└── README.md
```

## Costs

- **Scripts**: OpenAI GPT-4o (~$0.01-0.03 per script)
- **Images**: DALL-E 3 ($0.08 per image × 12-16 scenes = ~$1-1.30 per video)
- **Narration**: ElevenLabs ($0.30/1K chars, ~$0.30-0.40 per video)
- **YouTube metadata**: GPT-4o (~$0.01 per video)
- **Video assembly**: Free (local ffmpeg)
- **Estimated per video**: ~$1.50-1.80
- **ElevenLabs plan**: Pro $99/mo, 500K chars

## Boost Mode (Catalog Builder)

Doubles upload frequency from 1x/week to 2x/week per channel. Designed for rapidly building catalog, then switching back.

**File:** `app/scheduler.py` (top of file)

```python
BOOST_MODE = True   # 2x/week, any day, 3-day gap
BOOST_MODE = False  # 1x/week, preferred days only, 6-day gap (default)
```

Change the variable and restart the app. That's it.

- Boost: ~18 videos/week across all 9 channels
- Normal: ~9 videos/week across all 9 channels
- UI shows `[BOOST 2x/week]` or `[Normal 1x/week]` in schedule recommendations

## Things You Edit

Quick reference for files you'll commonly change:

| What | File | Notes |
|---|---|---|
| Boost mode on/off | `app/scheduler.py` → `BOOST_MODE` | `True` or `False`, restart after |
| Posting days/times | `app/scheduler.py` → `POSTING_SCHEDULE` | Restart after |
| Channel themes/focus | `app/generator.py` → `CHANNEL_FOCUS` | Restart after |
| Ken Burns overrides | `app/generator.py` → `CHANNEL_KB_OVERRIDES` | Restart after |
| Channel config | `channels/<channel_id>.json` | Voice, narrator, visual style |
| YouTube channel map | `app/youtube_upload.py` → `YOUTUBE_CHANNEL_MAP` | Restart after |
| Default tags | YouTube Studio → Settings → Upload Defaults | Per channel, no restart |
| API keys | `~/.zshrc` or environment | `OPENAI_API_KEY`, `ELEVENLABS_API_KEY`, `HF_TOKEN` |

## Adding New Channels

1. Copy `channels/_template.json` to `channels/your_channel.json`
2. Fill in: voice, narrator, visual theme, focus areas, YouTube config
3. Add channel logo to `static/img/<channel_id>.png`
4. Add YouTube channel mapping to `app/youtube_upload.py`
5. Add posting schedule to `app/scheduler.py`
6. Add Ken Burns overrides to `app/generator.py` if needed
7. Restart the app — channel appears automatically
