# How to Add a New Channel — Emrose Media Studios

## 1. Create the JSON file

Copy the template and rename it:
```
cp channels/_template.json channels/your_channel_id.json
```

The filename doesn't technically matter (it scans all `.json` files), but keep it matching the `channel_id` for sanity. **Don't start the filename with `_`** — those are ignored.

## 2. Fill in the required fields

Open `channels/your_channel_id.json` and edit:

### Identity
- `channel_id` — lowercase, underscores, no spaces (e.g., `night_signal`). This is the internal key used everywhere.
- `channel_name` — Display name, PascalCase preferred (e.g., `NightSignal`). This becomes the Desktop folder name.
- `description` — What the channel is about. GPT uses this to write scripts.

### Narrator
- `narrator.name` — Character name (e.g., "The Keeper", "The Voice")
- `narrator.description` — Personality and delivery style. Be specific — this directly shapes the script output.
- `narrator.rules` — Array of rules GPT follows when writing. More rules = more consistent tone.

### Voice (ElevenLabs)
- `elevenlabs_voice_id` — Get this from [elevenlabs.io/voices](https://elevenlabs.io/voices). Click a voice → the ID is in the URL or API section.
- `elevenlabs_voice_name` — Just for display/logging.
- `speed` — 0.25 (extremely slow) to 1.0 (normal). Most channels use 0.85–0.92.
- `settings.stability` — Higher = more consistent (0.0–1.0). Use 0.85+ for narration.
- `settings.similarity_boost` — Higher = closer to voice sample (0.0–1.0).
- `settings.style` — Higher = more expressive/emotional (0.0–1.0). 0.15 is neutral, 0.45+ is dramatic.
- `settings.use_speaker_boost` — `true` makes it louder/punchier. Usually `false` for narration.

### Optional Voice Settings (for sleep/meditation channels)
- `narration_volume` — Post-processing volume reduction (0.0–1.0, default 1.0). E.g., 0.55 reduces volume 45%.
- `sentence_pause_seconds` — Inserts silence between every sentence (0 = off). E.g., 3.0 adds 3s gaps. Burns more ElevenLabs API calls.

### Opening/Closing
- `opening_format.template` — Use `{subject}` as placeholder for the topic.
- `closing_format.templates` — Array of possible closing lines (one is picked randomly).

### Visual Theme
- `palette` — Color names that guide image generation.
- `image_prompt_suffix` — Appended to every DALL-E/FLUX prompt. Critical for visual consistency.
- `avoid` — Things the image generator should never include.
- `camera` — Camera movement style for the Ken Burns effect.

### Video Settings
- `target_duration_min/max` — Target video length in minutes.
- `scene_count_min/max` — Number of scenes (images + narration blocks).
- `target_word_count_min/max` — Total narration word count.
- `scene_pause_seconds` — Silence between scenes (0 for normal channels, 4–7 for slow/meditative).
- `crossfade_seconds` — Visual transition between scenes.
- `fade_in_seconds/fade_out_seconds` — Opening/closing fades.

### Ambient Audio
- `enabled` — `true`/`false`
- `volume` — 0.0–1.0 relative to narration. 0.3–0.5 is typical.
- `style` — Text description sent to ElevenLabs SFX API. Be descriptive.

### Shorts
- `enabled` — `true`/`false`

### YouTube
- `category` — Must match a key in the code: `Entertainment`, `Education`, `Science & Technology`, `Film & Animation`, `Music`, `People & Blogs`, `Gaming`, `Howto & Style`
- `default_tags` — Always included in every upload's tags.
- `default_hashtags` — (Optional) prepended to every description.

## 3. Add YouTube channel mapping (for uploads)

Edit `app/youtube_upload.py` and add your channel to `YOUTUBE_CHANNEL_MAP` at the bottom:

```python
YOUTUBE_CHANNEL_MAP = {
    ...existing entries...
    "your_channel_id": "UCxxxxxxxxxxxxxxxxxxxxxx",  # Your YouTube channel ID
}
```

Get the YouTube channel ID from YouTube Studio → Settings → Channel → Advanced settings, or from the channel URL.

**Skip this step if you're not uploading to YouTube yet** — videos will still generate fine.

## 4. (Optional) Add a channel avatar

Drop an image at `app/static/img/your_channel_id.png` (80×80px works). It shows on the dashboard.

## 5. That's it

The channel appears on the dashboard automatically on next page load. No restart needed — it scans the `channels/` directory on every request.

---

## Current Voice Settings Reference

| Channel | Voice | Speed | Style | Notes |
|---------|-------|:-----:|:-----:|-------|
| DeadlightCodex | Victor | 0.85 | 0.15 | Dark, measured |
| ZeroTraceArchive | Daniel | 0.92 | 0.30 | Analytical, documentary |
| RemnantsProject | Brian | — | — | Post-apocalyptic observer |
| GrayMeridian | Bill | — | — | Contemplative, psychological |
| TheUnwrittenWing | Lily | — | — | Warm, emotional storytelling |
| SomnusProtocol | Charlotte | 0.40 | 0.45 | Whisper-soft, narration_volume 0.55, sentence_pause 3s |

## Current YouTube Channel Map

| App Channel ID | YouTube Channel ID |
|---|---|
| gray_meridian | UC9bKvKkjAtoA9ysy2RnRPaQ |
| zero_trace_archive | UC8Hy4GyP9T_qTp72c0kAYpg |
| remnants_project | UC7tUZwk9l23qVwryihjBZSg |
| the_unwritten_wing | UCyjZXmPy4gG1xX2RJVJpfGA |
| autonomous_stack | UCzZU7Gn_5eQAfUz6a9rXPAA |
| somnus_protocol | UCukqYDdDPGG8G_jNvkb_2VQ |
| deadlight_codex | UCeR5uvuGWIQgxHsCg6KOYlg |
