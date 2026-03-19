"""
Video generation pipeline module.
Wraps the core pipeline logic for use by the Flask app.
"""
import os
import sys
import json
import math
import time
import random
import asyncio
import logging
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Callable

import numpy as np
from scipy.io import wavfile as scipy_wav
from PIL import Image
import httpx
import imageio

from moviepy import (
    VideoFileClip, AudioFileClip, ImageClip,
    concatenate_videoclips, ColorClip, CompositeVideoClip,
    CompositeAudioClip, vfx,
)

log = logging.getLogger("generator")

CHANNELS_DIR = Path(__file__).parent.parent / "channels"
OUTPUT_DIR = Path(__file__).parent.parent / "output"
TOPIC_BANK_PATH = Path(__file__).parent.parent / "topic_bank.json"

# Desktop output — videos land here for easy YouTube upload
DESKTOP_BASE = Path.home() / "Desktop" / "EmroseMedia"


def _get_desktop_channel_dir(channel_name):
    """Get the desktop output folder for a channel's full videos."""
    d = DESKTOP_BASE / channel_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get_desktop_shorts_dir(channel_name):
    """Get the desktop output folder for a channel's Shorts."""
    d = DESKTOP_BASE / f"{channel_name}_Shorts"
    d.mkdir(parents=True, exist_ok=True)
    return d

# Channel-specific focus areas for topic generation
CHANNEL_FOCUS = {
    "zero_trace_archive": {
        "focus": [
            "Real-world unexplained events",
            "Hidden structures, anomalies, forgotten discoveries",
            "Things that 'should not exist'",
        ],
        "avoid": ["Supernatural claims without grounded realism"],
        "examples": [
            "The Tunnel That Appeared Overnight Beneath a School",
            "The Case of the Missing Floor in a Government Building",
        ],
    },
    "gray_meridian": {
        "focus": [
            "Human behavior patterns",
            "Cognitive bias",
            "Subtle psychological truths",
        ],
        "avoid": ["Generic self-help topics"],
        "examples": ["Why People Defend Ideas They Know Are Wrong"],
    },
    "autonomous_stack": {
        "focus": [
            "Automation workflows",
            "AI integrations",
            "Real-world efficiency improvements",
        ],
        "avoid": ["Basic beginner coding tutorials"],
        "examples": ["The One Automation Layer Everyone Forgets"],
    },
    "the_unwritten_wing": {
        "focus": [
            "Emotional, character-driven concepts",
            "Subtle surreal elements",
            "Memory, time, identity",
            "Continuity within this universe and storytelling",
        ],
        "avoid": ["High-action or plot-heavy ideas"],
        "examples": ["The Day She Remembered a Life That Wasn't Hers"],
    },
    "remnants_project": {
        "focus": [
            "Post-human environments",
            "Systems continuing without humans",
            "Nature reclaiming civilization",
        ],
        "avoid": ["Action or survival narratives"],
        "examples": ["What Happens to Airports After Humans Disappear"],
    },
    "somnus_protocol": {
        "focus": [
            "Calm, repetitive, low-stimulation scenarios",
            "Safe environments",
            "Gentle descriptive experiences",
        ],
        "avoid": ["Conflict, tension, or mystery"],
        "examples": ["Drifting Through a Silent Forest at Night"],
    },
    "deadlight_codex": {
        "focus": [
            "Cosmic horror entities and anomalies",
            "Abstract existential concepts",
            "Things that defy comprehension",
        ],
        "avoid": ["Standard horror tropes or jumpscares"],
        "examples": ["The Observer Paradox Entity"],
    },
}


def _load_topic_bank():
    if TOPIC_BANK_PATH.exists():
        return json.loads(TOPIC_BANK_PATH.read_text())
    return {}


def _save_topic_to_bank(channel_id, topic_title):
    bank = _load_topic_bank()
    if channel_id not in bank:
        bank[channel_id] = []
    if topic_title not in bank[channel_id]:
        bank[channel_id].append(topic_title)
    TOPIC_BANK_PATH.write_text(json.dumps(bank, indent=2))

# Defaults
TARGET_FPS = 30
TARGET_RESOLUTION = (1920, 1080)
IMAGE_GEN_WIDTH = 1920
IMAGE_GEN_HEIGHT = 1080
KB_ZOOM_RANGE = (1.02, 1.08)
KB_PAN_RANGE = 0.04
CROSSFADE_DURATION = 1.5
FLUX_SPACES = [
    os.environ.get("FLUX_SPACE", "multimodalart/FLUX.1-merged"),
    "black-forest-labs/FLUX.1-schnell",
    "stabilityai/stable-diffusion-3.5-large-turbo",
]
FLUX_STEPS = int(os.environ.get("FLUX_STEPS", "8"))
FLUX_GUIDANCE = float(os.environ.get("FLUX_GUIDANCE", "3.5"))


def list_channels():
    channels = []
    for f in sorted(CHANNELS_DIR.glob("*.json")):
        if f.name.startswith("_"):
            continue
        try:
            data = json.loads(f.read_text())
            channels.append(data)
        except Exception:
            pass
    return channels


def load_channel(channel_id):
    for f in CHANNELS_DIR.glob("*.json"):
        if f.name.startswith("_"):
            continue
        try:
            data = json.loads(f.read_text())
            if data.get("channel_id") == channel_id:
                return data
        except Exception:
            pass
    return None


def list_videos(channel_id):
    vdir = OUTPUT_DIR / channel_id
    if not vdir.exists():
        return []
    videos = []
    for d in sorted(vdir.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        meta_path = d / "metadata.json"
        video_path = d / "video.mp4"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                meta["dir_name"] = d.name
                meta["has_video"] = video_path.exists()
                if video_path.exists():
                    meta["file_size_mb"] = round(video_path.stat().st_size / 1024 / 1024, 1)
                meta["has_short"] = (d / "short.mp4").exists()
                videos.append(meta)
            except Exception:
                pass
    return videos


def update_video_meta(channel_id, dir_name, updates):
    meta_path = OUTPUT_DIR / channel_id / dir_name / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        meta.update(updates)
        meta_path.write_text(json.dumps(meta, indent=2))
        return meta
    return None


# --- LLM calls ---

def _call_openai_sync(messages, api_key):
    import httpx as hx
    resp = hx.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": "gpt-4o", "messages": messages, "temperature": 0.7},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def generate_topic_idea(channel, api_key):
    channel_id = channel["channel_id"]
    focus = CHANNEL_FOCUS.get(channel_id, {})

    # Build focus section
    focus_text = ""
    if focus.get("focus"):
        focus_text += "\nFocus on:\n" + "\n".join(f"- {f}" for f in focus["focus"])
    if focus.get("avoid"):
        focus_text += "\n\nAvoid:\n" + "\n".join(f"- {a}" for a in focus["avoid"])
    if focus.get("examples"):
        focus_text += "\n\nExamples of good topics:\n" + "\n".join(f'• "{e}"' for e in focus["examples"])

    # Load topic bank
    bank = _load_topic_bank()
    used_topics = bank.get(channel_id, [])
    used_text = ""
    if used_topics:
        used_text = "\n\nPreviously Used Topics:\n" + "\n".join(f"- {t}" for t in used_topics)
        used_text += "\n\nDo NOT generate ideas similar to anything listed above. Each idea must be completely unique and distinct from all previous topics."

    prompt = f"""You are generating a content idea for a YouTube channel.

Channel Name: {channel['channel_name']}
Channel Description: {channel['description']}
{focus_text}
{used_text}

Your task: Generate ONE high-quality video concept suitable for this channel.

Requirements:
- Must be unique and not repetitive of common topics
- Must have strong viewer curiosity (click-worthy but not clickbait)
- Must align tightly with the channel's tone and identity
- Must be expandable into a 10-20 minute video
- Must feel like part of a long-term series, not a one-off

Output format (respond ONLY with valid JSON, no markdown fences):
{{
  "title": "A compelling, YouTube-ready title",
  "core_concept": "2-3 sentence explanation of the idea",
  "hook": "The opening line or premise that captures attention immediately",
  "why_it_works": "Why this would perform well",
  "visual_direction": "What the video would look like visually",
  "series_potential": "How this could become a recurring theme"
}}

Do not generate generic ideas. Prioritize originality and curiosity."""

    result = _call_openai_sync([{"role": "user", "content": prompt}], api_key)
    result = result.strip()
    if result.startswith("```"):
        result = result.split("\n", 1)[1]
    if result.endswith("```"):
        result = result.rsplit("```", 1)[0]
    idea = json.loads(result.strip())

    # Save to topic bank
    _save_topic_to_bank(channel_id, idea["title"])

    return idea


def _build_director_prompt(channel):
    c = channel
    n = c.get("narrator", {})
    o = c.get("opening_format", {})
    cl = c.get("closing_format", {})
    v = c.get("visual_theme", {})
    vs = c.get("video_settings", {})

    rules = "\n".join(f"- {r}" for r in n.get("rules", []))
    avoid = "\n".join(f"- {a}" for a in v.get("avoid", []))
    elements = "\n".join(f"- {e}" for e in v.get("elements", []))
    closing_templates = "\n".join(f'"{t}"' for t in cl.get("templates", []))

    # Special pacing instructions for channels that use elongated pauses
    scene_pause = vs.get("scene_pause_seconds", 0)
    pacing_note = ""
    if scene_pause > 0:
        pacing_note = f"""
IMPORTANT PACING NOTE:
This channel uses ELONGATED PAUSES between sentences and scenes to reach its target duration.
The video will be {vs.get('target_duration_min', 5)}-{vs.get('target_duration_max', 7)} minutes, but the word count may be lower
because the pacing is deliberately slow with {scene_pause:.0f}-second pauses between scenes.
Do NOT try to reach the duration through more words — reach it through slower, more deliberate pacing.
Write narration that BREATHES. Short sentences. Repetition. Space between thoughts.
The narrator is drifting off too. Each sentence should feel like it takes effort to say."""

    return f"""You are the creative director for {c['channel_name']}.
{c.get('description', '')}
{pacing_note}

NARRATOR: {n.get('name', 'Narrator')}
{n.get('description', '')}

NARRATION RULES:
{rules}

OPENING FORMAT:
Template: "{o.get('template', '')}"
{o.get('notes', '')}

CLOSING FORMAT:
Choose from:
{closing_templates}
{cl.get('notes', '')}

IMAGE PROMPT RULES (for each scene):
Write detailed prompts for generating a STILL IMAGE. Each prompt must describe:
- Style: {v.get('style', '')}
- Palette: {', '.join(v.get('palette', []))}
- Lighting: {v.get('lighting', '')}
- Elements to include:
{elements}
- Environment: {v.get('environment', '')}
- Camera feel: {v.get('camera', '')}
- Mood: {v.get('mood', '')}
- Avoid:
{avoid}
- End each prompt with: {v.get('image_prompt_suffix', '')}

OUTPUT FORMAT — respond with ONLY valid JSON, no markdown fences:
{{
  "title": "short title for the entry",
  "subject": "brief subject line used in the opening",
  "scenes": [
    {{
      "narration": "Narration text for this scene",
      "image_prompt": "Detailed image generation prompt",
      "duration_hint": 10
    }}
  ]
}}

TARGET LENGTH — THIS IS THE MOST IMPORTANT REQUIREMENT:
The final video MUST be between {vs.get('target_duration_min', 5)} and {vs.get('target_duration_max', 7)} minutes long.
- You MUST write EXACTLY {vs.get('scene_count_min', 10)} to {vs.get('scene_count_max', 14)} scenes. No fewer than {vs.get('scene_count_min', 10)}.
- Each scene's narration MUST be 5-7 sentences and 60-90 words. SHORT SCENES ARE NOT ACCEPTABLE.
- Total narration across ALL scenes MUST be {vs.get('target_word_count_min', 700)}-{vs.get('target_word_count_max', 900)} words.
- If your total word count is under {vs.get('target_word_count_min', 700)}, you have FAILED. Add more detail, atmosphere, and description.
- Do NOT write short 1-2 sentence narration. Every scene needs substance.
- Err on the side of LONGER, not shorter. A 3-minute video is a failure.
- duration_hint should be 15 for each scene."""


def generate_script(channel, topic, api_key):
    system = _build_director_prompt(channel)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Create an entry about:\n\n{topic}"},
    ]

    vs = channel.get("video_settings", {})
    min_words = vs.get("target_word_count_min", 700)
    min_scenes = vs.get("scene_count_min", 10)

    # Try up to 3 times to get a script that meets length requirements
    for attempt in range(3):
        result = _call_openai_sync(messages, api_key)
        result = result.strip()
        if result.startswith("```"):
            result = result.split("\n", 1)[1]
        if result.endswith("```"):
            result = result.rsplit("```", 1)[0]
        script = json.loads(result.strip())

        # Validate length
        total_words = sum(len(s.get("narration", "").split()) for s in script.get("scenes", []))
        scene_count = len(script.get("scenes", []))

        if total_words >= min_words and scene_count >= min_scenes:
            log.info(f"Script accepted: {total_words} words, {scene_count} scenes (attempt {attempt+1})")
            return script

        log.warning(f"Script too short: {total_words} words, {scene_count} scenes (attempt {attempt+1}). Retrying...")
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Create an entry about:\n\n{topic}"},
            {"role": "assistant", "content": result},
            {"role": "user", "content": f"This script is TOO SHORT. It has only {total_words} words and {scene_count} scenes. I need AT LEAST {min_words} words across at least {min_scenes} scenes. Each scene needs 60-90 words of narration. Rewrite the ENTIRE script longer and more detailed."},
        ]

    log.warning(f"Script still short after 3 attempts, using last result")
    return script


# --- Audio generation ---

def _generate_narration_sync(text, voice_id, model_id, voice_settings, speed, api_key, out_path):
    import httpx as hx
    payload = {"text": text, "model_id": model_id, "voice_settings": voice_settings}
    if speed != 1.0:
        payload["speed"] = speed
    for attempt in range(5):
        resp = hx.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
            headers={"xi-api-key": api_key, "Content-Type": "application/json", "Accept": "audio/mpeg"},
            json=payload, timeout=180,
        )
        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", str(5 * (attempt + 1))))
            log.warning(f"ElevenLabs 429 rate limit, retrying in {wait}s (attempt {attempt+1}/5)")
            time.sleep(wait)
            continue
        if resp.status_code == 401:
            # Log the actual error body to understand why
            log.error(f"ElevenLabs 401: {resp.text[:300]}")
            # Only retry once for 401 — if it persists, it's a real auth issue
            if attempt < 1:
                log.warning(f"ElevenLabs 401, retrying once in 3s...")
                time.sleep(3)
                continue
            raise RuntimeError(f"ElevenLabs authentication failed (401): {resp.text[:200]}")
        resp.raise_for_status()
        Path(out_path).write_bytes(resp.content)
        clip = AudioFileClip(str(out_path))
        dur = clip.duration
        clip.close()
        return dur
    raise RuntimeError("ElevenLabs rate limit exceeded after 5 attempts")


def generate_ambient_drone(duration, out_path, channel_id=None):
    """Generate channel-specific ambient audio.
    
    Each channel gets a unique sonic palette that matches its theme:
    - deadlight_codex: dark cosmic drone with dissonance
    - zero_trace_archive: tense investigative hum with static
    - the_unwritten_wing: warm ambient pad with gentle piano-like overtones
    - remnants_project: post-industrial hum with nature textures
    - somnus_protocol: ultra-calm sleep sounds, soft green noise, gentle waves
    - autonomous_stack: clean electronic hum, minimal digital texture
    - gray_meridian: neutral warm pad, subtle heartbeat-like pulse
    """
    sr = 44100
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    if channel_id == "somnus_protocol":
        # Sleep-specific: green noise variant + gentle ocean-like waves + warm pad
        # Green noise (mid-frequency filtered noise, soothing)
        noise = np.random.randn(len(t)) * 0.15
        # Band-pass around 500Hz (green noise range)
        ks = int(sr / 500)
        if ks > 1:
            noise = np.convolve(noise, np.ones(ks) / ks, mode="same")
        # Very slow breathing modulation
        breath = (np.sin(2 * np.pi * 0.04 * t) * 0.3 + 0.7)
        noise *= breath
        # Gentle ocean-like wave wash
        wave_lfo = (np.sin(2 * np.pi * 0.07 * t) * 0.5 + 0.5) ** 1.5
        wave_noise = np.random.randn(len(t)) * 0.08
        ks2 = int(sr / 300)
        if ks2 > 1:
            wave_noise = np.convolve(wave_noise, np.ones(ks2) / ks2, mode="same")
        wave_noise *= wave_lfo
        # Ultra-soft warm pad (low sine)
        pad = np.sin(2 * np.pi * 65.0 * t) * 0.04
        pad += np.sin(2 * np.pi * 97.5 * t) * 0.02
        pad *= (np.sin(2 * np.pi * 0.02 * t) * 0.3 + 0.7)
        ambient = noise + wave_noise + pad

    elif channel_id == "the_unwritten_wing":
        # Warm ambient pad with gentle piano-like overtones
        # Warm low pad
        pad1 = np.sin(2 * np.pi * 65.4 * t) * 0.15  # C2
        pad2 = np.sin(2 * np.pi * 98.0 * t) * 0.10  # ~G2
        pad_lfo = np.sin(2 * np.pi * 0.02 * t) * 0.3 + 0.7
        # Piano-like bell tones (sparse, decaying)
        bell = np.zeros(len(t))
        bell_notes = [261.6, 329.6, 392.0, 523.2]  # C4, E4, G4, C5
        num_bells = max(3, int(duration / 12))
        for _ in range(num_bells):
            pos = random.randint(0, len(t) - sr * 4)
            note = random.choice(bell_notes)
            length = min(sr * 4, len(t) - pos)
            env = np.exp(-np.linspace(0, 5, length))
            tone = np.sin(2 * np.pi * note * np.linspace(0, length / sr, length)) * env * 0.03
            bell[pos:pos + length] += tone
        # Soft warmth noise
        warmth = np.random.randn(len(t)) * 0.01
        ks = int(sr / 200)
        if ks > 1:
            warmth = np.convolve(warmth, np.ones(ks) / ks, mode="same")
        ambient = (pad1 + pad2) * pad_lfo + bell + warmth

    elif channel_id == "zero_trace_archive":
        # Tense investigative hum with subtle static crackle
        # Low tension drone
        drone = np.sin(2 * np.pi * 45.0 * t) * 0.2
        drone += np.sin(2 * np.pi * 47.5 * t) * 0.08  # slight detuning = tension
        # Intermittent static crackle
        crackle = np.zeros(len(t))
        num_crackles = max(2, int(duration / 8))
        for _ in range(num_crackles):
            pos = random.randint(0, len(t) - sr)
            length = random.randint(int(sr * 0.1), int(sr * 0.5))
            end = min(pos + length, len(t))
            burst = np.random.randn(end - pos) * 0.03
            env = np.hanning(end - pos)
            crackle[pos:end] += burst * env
        # Subtle high tension tone
        tension = np.sin(2 * np.pi * 220 * t) * 0.015
        tension *= (np.sin(2 * np.pi * 0.03 * t) * 0.5 + 0.5)
        ambient = drone + crackle + tension

    elif channel_id == "remnants_project":
        # Post-industrial hum with nature textures (wind, birds-like)
        # Industrial hum
        hum = np.sin(2 * np.pi * 60.0 * t) * 0.12
        hum += np.sin(2 * np.pi * 120.0 * t) * 0.04
        hum_lfo = np.sin(2 * np.pi * 0.015 * t) * 0.4 + 0.6
        hum *= hum_lfo
        # Wind-like broadband noise with slow sweep
        wind = np.random.randn(len(t)) * 0.06
        ks = int(sr / 400)
        if ks > 1:
            wind = np.convolve(wind, np.ones(ks) / ks, mode="same")
        wind_lfo = (np.sin(2 * np.pi * 0.05 * t) * 0.5 + 0.5) ** 2
        wind *= wind_lfo
        # Sparse high chirp (bird-like, nature reclaiming)
        chirps = np.zeros(len(t))
        num_chirps = max(2, int(duration / 20))
        for _ in range(num_chirps):
            pos = random.randint(0, len(t) - int(sr * 0.3))
            length = random.randint(int(sr * 0.05), int(sr * 0.2))
            freq = random.uniform(2000, 4000)
            chirp_t = np.linspace(0, length / sr, length)
            chirp = np.sin(2 * np.pi * freq * chirp_t) * np.exp(-chirp_t * 20) * 0.015
            end = min(pos + length, len(t))
            chirps[pos:end] += chirp[:end - pos]
        ambient = hum + wind + chirps

    elif channel_id == "autonomous_stack":
        # Clean electronic hum, minimal digital texture
        # Clean sine pad
        pad = np.sin(2 * np.pi * 110.0 * t) * 0.1
        pad += np.sin(2 * np.pi * 165.0 * t) * 0.05
        # Digital texture — subtle quantized noise
        digital = np.random.randn(len(t)) * 0.015
        # Quantize to create digital feel
        digital = np.round(digital * 10) / 10
        ks = int(sr / 600)
        if ks > 1:
            digital = np.convolve(digital, np.ones(ks) / ks, mode="same")
        # Minimal pulse
        pulse_env = (np.sin(2 * np.pi * 0.5 * t) > 0.95).astype(float) * 0.02
        pulse = np.sin(2 * np.pi * 440 * t) * pulse_env
        ambient = pad + digital + pulse

    elif channel_id == "gray_meridian":
        # Neutral warm pad with subtle heartbeat-like pulse
        pad1 = np.sin(2 * np.pi * 82.4 * t) * 0.12
        pad2 = np.sin(2 * np.pi * 123.5 * t) * 0.06
        pad_lfo = np.sin(2 * np.pi * 0.025 * t) * 0.3 + 0.7
        # Heartbeat-like low thump (very subtle)
        heartbeat = np.zeros(len(t))
        beat_interval = int(sr * 1.2)  # ~50 BPM, resting
        for pos in range(0, len(t) - int(sr * 0.2), beat_interval):
            length = int(sr * 0.15)
            end = min(pos + length, len(t))
            env = np.exp(-np.linspace(0, 8, end - pos))
            thump = np.sin(2 * np.pi * 35.0 * np.linspace(0, (end - pos) / sr, end - pos)) * env * 0.04
            heartbeat[pos:end] += thump
        ambient = (pad1 + pad2) * pad_lfo + heartbeat

    else:
        # Default / deadlight_codex: dark cosmic drone with dissonance
        detune = np.sin(2 * np.pi * 0.03 * t) * 0.5
        drone1 = np.sin(2 * np.pi * (38.0 + detune) * t) * 0.25
        drone2 = np.sin(2 * np.pi * (55.0 + detune * 0.7) * t) * 0.18
        harm_lfo = np.sin(2 * np.pi * 0.05 * t) * 0.5 + 0.5
        harm1 = np.sin(2 * np.pi * 82.4 * t) * 0.07 * harm_lfo
        harm2 = np.sin(2 * np.pi * 87.3 * t) * 0.05 * (1.0 - harm_lfo) * 0.8
        pad_lfo = np.sin(2 * np.pi * 0.015 * t) * 0.5 + 0.5
        pad1 = np.sin(2 * np.pi * 110.0 * t) * 0.04 * pad_lfo
        pad2 = np.sin(2 * np.pi * 146.8 * t) * 0.03 * (1.0 - pad_lfo)
        noise = np.random.randn(len(t)) * 0.02
        ks = int(sr / 180)
        if ks > 1:
            noise = np.convolve(noise, np.ones(ks) / ks, mode="same")
        breath_lfo = (np.sin(2 * np.pi * 0.08 * t) * 0.5 + 0.5) ** 2
        noise *= breath_lfo
        sweep_freq = np.linspace(25, 35, len(t))
        sweep = np.sin(2 * np.pi * sweep_freq * t / 2) * 0.08
        sweep_lfo = np.sin(2 * np.pi * 0.01 * t) * 0.5 + 0.5
        sweep *= sweep_lfo
        rumble_env = np.zeros(len(t))
        num_rumbles = max(1, int(duration / 30))
        for _ in range(num_rumbles):
            pos = random.randint(int(sr * 5), len(t) - int(sr * 3))
            width = random.randint(int(sr * 1.5), int(sr * 4))
            end = min(pos + width, len(t))
            rumble_env[pos:end] += np.hanning(end - pos) * 0.06
        rumble = np.sin(2 * np.pi * 28.0 * t) * rumble_env
        ambient = drone1 + drone2 + harm1 + harm2 + pad1 + pad2 + noise + sweep + rumble

    # Fade in/out
    fi = int(sr * 4.0)
    fo = int(sr * 5.0)
    if len(ambient) > fi:
        ambient[:fi] *= np.linspace(0, 1, fi) ** 2
    if len(ambient) > fo:
        ambient[-fo:] *= np.linspace(1, 0, fo) ** 2

    peak = np.max(np.abs(ambient))
    if peak > 0:
        ambient = ambient / peak * 0.95
    scipy_wav.write(str(out_path), sr, (ambient * 32767).astype(np.int16))


# --- Image generation ---

def _generate_image(prompt, out_path, hf_token, width=1344, height=768):
    """Generate image. Tries DALL-E 3 first (best quality), falls back to FLUX via HuggingFace."""

    # PRIMARY: OpenAI DALL-E 3 (consistent high quality)
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if openai_key:
        try:
            import httpx as hx
            # DALL-E 3 generates at fixed sizes — use 1792x1024 (landscape, closest to 16:9)
            dalle_size = "1792x1024"
            if width < height:
                dalle_size = "1024x1792"  # vertical for Shorts

            log.info(f"Generating image via DALL-E 3 ({dalle_size})...")
            r = hx.post(
                "https://api.openai.com/v1/images/generations",
                headers={"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"},
                json={
                    "model": "dall-e-3",
                    "prompt": prompt,
                    "n": 1,
                    "size": dalle_size,
                    "quality": "hd",
                    "response_format": "url",
                },
                timeout=120,
            )
            if r.status_code == 200:
                img_url = r.json()["data"][0]["url"]
                # Download the image
                img_resp = hx.get(img_url, timeout=60)
                if img_resp.status_code == 200:
                    with open(str(out_path), "wb") as f:
                        f.write(img_resp.content)
                    img = Image.open(str(out_path))
                    img.save(str(out_path), "PNG")
                    log.info(f"Image generated via DALL-E 3: {img.size[0]}x{img.size[1]}")
                    return True
                else:
                    log.warning(f"DALL-E 3 image download failed: {img_resp.status_code}")
            else:
                error_msg = r.text[:200]
                log.warning(f"DALL-E 3 failed ({r.status_code}): {error_msg}")
                # Don't fall through on billing/auth errors
                if r.status_code == 401:
                    log.error("OpenAI API key invalid for DALL-E 3")
                elif "billing" in error_msg.lower() or "quota" in error_msg.lower():
                    log.warning("OpenAI billing/quota issue — falling back to FLUX")
        except Exception as e:
            log.warning(f"DALL-E 3 failed: {str(e)[:150]}")

    # FALLBACK 1: HuggingFace Inference API (free with HF Pro)
    inference_configs = [
        {
            "model": "black-forest-labs/FLUX.1-dev",
            "params": {"width": width, "height": height, "num_inference_steps": 20, "guidance_scale": 3.5},
        },
        {
            "model": "black-forest-labs/FLUX.1-schnell",
            "params": {"width": width, "height": height, "num_inference_steps": 8},
        },
        {
            "model": "stabilityai/stable-diffusion-3.5-large-turbo",
            "params": {"width": width, "height": height, "num_inference_steps": 8},
        },
    ]

    if hf_token:
        for cfg in inference_configs:
            model = cfg["model"]
            try:
                import httpx as hx
                headers = {"Authorization": f"Bearer {hf_token}"}
                payload = {
                    "inputs": prompt,
                    "parameters": cfg["params"],
                }
                url = f"https://router.huggingface.co/hf-inference/models/{model}"
                log.info(f"Trying HF Inference API: {model} (steps={cfg['params'].get('num_inference_steps', 4)})")
                r = hx.post(url, headers=headers, json=payload, timeout=180)
                if r.status_code == 200 and "image" in r.headers.get("content-type", ""):
                    with open(str(out_path), "wb") as f:
                        f.write(r.content)
                    img = Image.open(str(out_path))
                    img.save(str(out_path), "PNG")
                    log.info(f"Image generated via HF Inference API ({model}): {img.size[0]}x{img.size[1]}")
                    return True
                else:
                    log.warning(f"HF Inference {model}: {r.status_code} - {r.text[:120]}")
            except Exception as e:
                log.warning(f"HF Inference {model} failed: {str(e)[:120]}")

    # FALLBACK 2: HuggingFace Spaces (ZeroGPU, quota-limited)
    from gradio_client import Client

    space_configs = [
        {
            "space": os.environ.get("FLUX_SPACE", "multimodalart/FLUX.1-merged"),
            "kwargs": {"prompt": prompt, "seed": 0, "randomize_seed": True, "width": width, "height": height,
                       "guidance_scale": FLUX_GUIDANCE, "num_inference_steps": FLUX_STEPS, "api_name": "/infer"},
        },
        {
            "space": "black-forest-labs/FLUX.1-schnell",
            "kwargs": {"prompt": prompt, "seed": 0, "randomize_seed": True, "width": width, "height": height,
                       "num_inference_steps": 4, "api_name": "/infer"},
        },
    ]

    for cfg in space_configs:
        space = cfg["space"]
        for attempt in range(2):
            try:
                client = None
                if hf_token:
                    try:
                        client = Client(space, hf_token=hf_token)
                    except TypeError:
                        try:
                            client = Client(space, token=hf_token)
                        except TypeError:
                            client = Client(space)
                else:
                    client = Client(space)

                result = client.predict(**cfg["kwargs"])
                img_result = result[0] if isinstance(result, tuple) else result
                src = img_result.get("path", img_result.get("url", "")) if isinstance(img_result, dict) else str(img_result)
                img = Image.open(src)
                img.save(str(out_path), "PNG")
                log.info(f"Image generated via {space}")
                return True
            except Exception as e:
                err = str(e)
                log.warning(f"FLUX {space} attempt {attempt+1}/2 failed: {err[:120]}")
                if "quota" in err.lower() or "limit" in err.lower():
                    log.info(f"Quota exhausted on {space}, trying next space...")
                    break
                if "RUNTIME_ERROR" in err or "invalid state" in err.lower():
                    log.info(f"Space {space} is down, trying next...")
                    break
                if attempt < 1:
                    time.sleep(3)

    log.warning("All image generation methods exhausted, will use fallback image")
    return False


def _generate_fallback_image(out_path, scene_index, width=1344, height=768):
    img = np.zeros((height, width, 3), dtype=np.float32)
    for y in range(height):
        v = 0.02 + 0.03 * math.exp(-((y - height * 0.4) ** 2) / (2 * (height * 0.3) ** 2))
        img[y, :] = v
    cy, cx = height * 0.45, width * 0.5
    Y, X = np.ogrid[:height, :width]
    dist = np.sqrt((X - cx) ** 2 / (width * 0.4) ** 2 + (Y - cy) ** 2 / (height * 0.4) ** 2)
    vignette = np.clip(1.0 - dist * 0.5, 0, 1)
    img *= vignette[:, :, None]
    tints = [(0.08, 0.02, 0.02), (0.04, 0.03, 0.01), (0.02, 0.02, 0.04)]
    tint = tints[scene_index % len(tints)]
    img[:, :, 0] += tint[0]
    img[:, :, 1] += tint[1]
    img[:, :, 2] += tint[2]
    img += np.random.randn(height, width, 3) * 0.015
    for _ in range(random.randint(20, 50)):
        px, py = random.randint(0, width - 1), random.randint(0, height - 1)
        size = random.randint(1, 3)
        brightness = random.uniform(0.08, 0.2)
        y1, y2 = max(0, py - size), min(height, py + size)
        x1, x2 = max(0, px - size), min(width, px + size)
        img[y1:y2, x1:x2] += brightness
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(str(out_path), "PNG")


# --- Ken Burns ---

def apply_ken_burns(image_path, duration, out_path, target_res=(1920, 1080)):
    img = Image.open(image_path)
    img_w, img_h = img.size
    target_w, target_h = target_res
    if img_w < img_h:
        target_w, target_h = 1080, 1920

    # Ensure source image is large enough for quality cropping
    min_dim = max(target_w, target_h) * 1.5
    if img_w < min_dim or img_h < min_dim:
        scale = min_dim / min(img_w, img_h)
        img = img.resize((int(img_w * scale), int(img_h * scale)), Image.LANCZOS)
        img_w, img_h = img.size

    motion = random.choice(["zoom_in", "zoom_out", "pan_left", "pan_right", "drift"])
    zoom_start = 1.0
    zoom_end = random.uniform(*KB_ZOOM_RANGE)
    if motion == "zoom_out":
        zoom_start, zoom_end = zoom_end, zoom_start

    max_pan_x = int(img_w * KB_PAN_RANGE)
    max_pan_y = int(img_h * KB_PAN_RANGE)
    if motion == "pan_left":
        start_x, end_x, start_y, end_y = max_pan_x, -max_pan_x, 0, 0
    elif motion == "pan_right":
        start_x, end_x, start_y, end_y = -max_pan_x, max_pan_x, 0, 0
    elif motion == "drift":
        start_x = random.randint(-max_pan_x, max_pan_x)
        end_x = random.randint(-max_pan_x, max_pan_x)
        start_y = random.randint(-max_pan_y, max_pan_y)
        end_y = random.randint(-max_pan_y, max_pan_y)
    else:
        start_x, start_y = 0, 0
        end_x = random.randint(-max_pan_x // 2, max_pan_x // 2)
        end_y = random.randint(-max_pan_y // 2, max_pan_y // 2)

    img_array = np.array(img)

    def make_frame(t_val):
        t_norm = t_val / max(duration - 0.001, 0.001)
        t_norm = max(0.0, min(1.0, t_norm))
        # Smoothstep easing
        t_smooth = t_norm * t_norm * (3.0 - 2.0 * t_norm)
        zoom = zoom_start + (zoom_end - zoom_start) * t_smooth
        pan_x = start_x + (end_x - start_x) * t_smooth
        pan_y = start_y + (end_y - start_y) * t_smooth
        crop_w = int(target_w / zoom)
        crop_h = int(target_h / zoom)
        crop_w = min(crop_w, img_w)
        crop_h = min(crop_h, img_h)
        cx = img_w // 2 + int(pan_x)
        cy = img_h // 2 + int(pan_y)
        x1 = max(0, min(cx - crop_w // 2, img_w - crop_w))
        y1 = max(0, min(cy - crop_h // 2, img_h - crop_h))
        frame = img_array[y1:y1+crop_h, x1:x1+crop_w]
        frame_img = Image.fromarray(frame).resize((target_w, target_h), Image.LANCZOS)
        return np.array(frame_img)

    from moviepy import VideoClip
    clip = VideoClip(make_frame, duration=duration)
    clip.write_videofile(
        str(out_path), fps=TARGET_FPS, codec="libx264",
        preset="medium", logger=None,
        ffmpeg_params=["-crf", "18", "-pix_fmt", "yuv420p"],
    )
    clip.close()
    return str(out_path)


# --- Title card generation ---

def _generate_title_card(channel, title, duration, out_path, api_keys, hf_token, res=(1920, 1080)):
    """Generate a cinematic title card with channel-specific FLUX background.
    Shows only the video title (no channel name) with a channel-specific font."""
    w, h = res
    channel_name = channel.get("channel_name", "")
    channel_id = channel.get("channel_id", "")
    visual = channel.get("visual_theme", {})
    palette = ", ".join(visual.get("palette", ["dark", "moody"]))
    style = visual.get("style", "cinematic")
    suffix = visual.get("image_prompt_suffix", "")
    mood = visual.get("mood", "atmospheric")

    # Channel-specific title card prompts
    channel_title_prompts = {
        "deadlight_codex": f"A vast ancient cosmic void with faint tentacle-like structures emerging from darkness, deep red nebula glow, eldritch archive atmosphere, centered dark space for title text. {suffix}",
        "zero_trace_archive": f"An abandoned investigation room with scattered classified documents under a single harsh overhead light, concrete walls, forensic evidence board in shadow, muted earth tones. {suffix}",
        "the_unwritten_wing": f"An infinite ethereal library with floating luminous pages and soft golden light streaming through impossible architecture, dreamy bokeh, warm surreal atmosphere. {suffix}",
        "remnants_project": f"A decaying overgrown control room of an abandoned facility, nature reclaiming technology, cracked monitors with faint green glow, ivy and moss, post-human silence. {suffix}",
        "somnus_protocol": f"Soft moonlit clouds drifting over a calm dark lake, gentle fog, deep blue and silver tones, extremely peaceful and meditative, starlight reflections on water. {suffix}",
        "autonomous_stack": f"A sleek futuristic command center with holographic data streams and circuit board patterns, cool blue and electric cyan lighting, clean minimal tech aesthetic. {suffix}",
        "gray_meridian": f"An abstract visualization of a human brain split in half, one side geometric and analytical, the other organic and emotional, dark background with subtle warm and cool contrast. {suffix}",
    }

    title_bg_prompt = channel_title_prompts.get(
        channel_id,
        f"Abstract cinematic title card background. Style: {style}. Colors: {palette}. Mood: {mood}. {suffix}"
    )
    title_bg_prompt += f" Relating to the concept of '{title}'. Dark, atmospheric, space for text overlay in center. No text, no letters, no words, no readable symbols."

    bg_path = str(out_path).replace(".png", "_bg.png")
    ok = _generate_image(title_bg_prompt, bg_path, hf_token, width=1344, height=768)

    if ok:
        pil_img = Image.open(bg_path)
        pil_img = pil_img.resize((w, h), Image.LANCZOS)
        # Darken the image for text readability
        img_array = np.array(pil_img).astype(np.float32)
        img_array *= 0.35  # darken significantly
        # Add vignette
        cy, cx = h // 2, w // 2
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt(((X - cx) / (w * 0.7)) ** 2 + ((Y - cy) / (h * 0.7)) ** 2)
        vignette = np.clip(1.0 - dist * 0.6, 0.2, 1.0)
        img_array *= vignette[:, :, None]
        pil_img = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
    else:
        # Fallback: dark gradient
        img = np.zeros((h, w, 3), dtype=np.uint8)
        cy, cx = h // 2, w // 2
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt(((X - cx) / (w * 0.6)) ** 2 + ((Y - cy) / (h * 0.6)) ** 2)
        gradient = np.clip(0.08 - dist * 0.04, 0.01, 0.08)
        for c_idx in range(3):
            img[:, :, c_idx] = (gradient * 255).astype(np.uint8)
        noise = np.random.normal(0, 3, (h, w, 3)).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img)

    # Draw text overlay — video title only (no channel name)
    try:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(pil_img)

        # Channel-specific font preferences
        # Each channel gets a distinct font family for brand differentiation
        channel_font_prefs = {
            "deadlight_codex": [
                "/System/Library/Fonts/Supplemental/Copperplate.ttc",
                "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
            ],
            "zero_trace_archive": [
                "/System/Library/Fonts/Supplemental/Courier New.ttf",
                "/System/Library/Fonts/Courier.dfont",
                "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
            ],
            "the_unwritten_wing": [
                "/System/Library/Fonts/Supplemental/Baskerville.ttc",
                "/System/Library/Fonts/Supplemental/Palatino.ttc",
                "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
            ],
            "remnants_project": [
                "/System/Library/Fonts/Supplemental/Futura.ttc",
                "/System/Library/Fonts/Helvetica.ttc",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            ],
            "somnus_protocol": [
                "/System/Library/Fonts/Supplemental/Didot.ttc",
                "/System/Library/Fonts/Supplemental/Georgia.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
            ],
            "autonomous_stack": [
                "/System/Library/Fonts/SFMono-Regular.otf",
                "/System/Library/Fonts/Supplemental/Menlo.ttc",
                "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
            ],
            "gray_meridian": [
                "/System/Library/Fonts/Supplemental/Avenir Next.ttc",
                "/System/Library/Fonts/Supplemental/Gill Sans.ttc",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-ExtraLight.ttf",
            ],
        }

        # Find the right font for this channel
        font_prefs = channel_font_prefs.get(channel_id, [])
        generic_fonts = [
            "/System/Library/Fonts/Supplemental/Georgia.ttf",
            "/System/Library/Fonts/Georgia.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        ]

        title_font = None
        for fp in font_prefs + generic_fonts:
            if Path(fp).exists():
                try:
                    title_font = ImageFont.truetype(fp, 62)
                    break
                except Exception:
                    continue

        if not title_font:
            title_font = ImageFont.load_default()

        gold = (212, 168, 84)

        # Thin divider line above title
        line_w = min(400, w - 200)
        line_y = h // 2 - 45
        draw.line([(w // 2 - line_w // 2, line_y), (w // 2 + line_w // 2, line_y)], fill=gold, width=1)

        # Title — wrap long titles (centered vertically)
        words = title.split()
        lines = []
        current = ""
        for word in words:
            test = f"{current} {word}".strip()
            bbox = draw.textbbox((0, 0), test, font=title_font)
            if bbox[2] - bbox[0] > w - 300:
                if current:
                    lines.append(current)
                current = word
            else:
                current = test
        if current:
            lines.append(current)

        total_text_height = len(lines) * 72
        y_start = h // 2 - 20
        for i, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=title_font)
            lw = bbox[2] - bbox[0]
            draw.text(((w - lw) // 2, y_start + i * 72), line, fill=gold, font=title_font)

        # Thin divider line below title
        line_y_bottom = y_start + total_text_height + 15
        draw.line([(w // 2 - line_w // 2, line_y_bottom), (w // 2 + line_w // 2, line_y_bottom)], fill=gold, width=1)

    except Exception as e:
        log.warning(f"Title card text rendering failed: {e}")

    pil_img.save(str(out_path), "PNG")
    return str(out_path)


def _generate_end_card(channel, duration, out_path, res=(1920, 1080)):
    """Generate an end card with subscribe CTA, using channel-specific font."""
    w, h = res
    channel_id = channel.get("channel_id", "")
    visual = channel.get("visual_theme", {})
    palette = visual.get("palette", ["dark"])

    # Dark background with subtle gradient (no FLUX call — keep it clean and fast)
    img = np.zeros((h, w, 3), dtype=np.float32)
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt(((X - cx) / (w * 0.6)) ** 2 + ((Y - cy) / (h * 0.6)) ** 2)
    gradient = np.clip(0.06 - dist * 0.03, 0.01, 0.06)
    for c_idx in range(3):
        img[:, :, c_idx] = gradient
    # Subtle noise
    img += np.random.randn(h, w, 3) * 0.008
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img)

    try:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(pil_img)

        # Same channel-specific font system as title card
        channel_font_prefs = {
            "deadlight_codex": ["/System/Library/Fonts/Supplemental/Copperplate.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"],
            "zero_trace_archive": ["/System/Library/Fonts/Supplemental/Courier New.ttf", "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"],
            "the_unwritten_wing": ["/System/Library/Fonts/Supplemental/Baskerville.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"],
            "remnants_project": ["/System/Library/Fonts/Supplemental/Futura.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
            "somnus_protocol": ["/System/Library/Fonts/Supplemental/Didot.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"],
            "autonomous_stack": ["/System/Library/Fonts/SFMono-Regular.otf", "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"],
            "gray_meridian": ["/System/Library/Fonts/Supplemental/Avenir Next.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans-ExtraLight.ttf"],
        }

        font_prefs = channel_font_prefs.get(channel_id, [])
        generic_fonts = [
            "/System/Library/Fonts/Supplemental/Georgia.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        ]

        main_font = None
        sub_font = None
        for fp in font_prefs + generic_fonts:
            if Path(fp).exists():
                try:
                    main_font = ImageFont.truetype(fp, 52)
                    sub_font = ImageFont.truetype(fp, 28)
                    break
                except Exception:
                    continue
        if not main_font:
            main_font = ImageFont.load_default()
            sub_font = main_font

        gold = (212, 168, 84)
        dim_gold = (160, 130, 70)

        # Main CTA text
        cta_text = "If you enjoyed this, please"
        cta_bbox = draw.textbbox((0, 0), cta_text, font=sub_font)
        cta_w = cta_bbox[2] - cta_bbox[0]
        draw.text(((w - cta_w) // 2, h // 2 - 80), cta_text, fill=dim_gold, font=sub_font)

        # Like, Comment & Subscribe
        action_text = "Like, Comment & Subscribe"
        action_bbox = draw.textbbox((0, 0), action_text, font=main_font)
        action_w = action_bbox[2] - action_bbox[0]
        draw.text(((w - action_w) // 2, h // 2 - 30), action_text, fill=gold, font=main_font)

        # Decorative lines
        line_w = min(action_w + 60, w - 200)
        draw.line([(w // 2 - line_w // 2, h // 2 - 95), (w // 2 + line_w // 2, h // 2 - 95)], fill=dim_gold, width=1)
        draw.line([(w // 2 - line_w // 2, h // 2 + 40), (w // 2 + line_w // 2, h // 2 + 40)], fill=dim_gold, width=1)

        # Small copyright line
        year = datetime.now().year
        copy_text = f"© {year} Emrose Media Studios"
        copy_bbox = draw.textbbox((0, 0), copy_text, font=sub_font)
        copy_w = copy_bbox[2] - copy_bbox[0]
        draw.text(((w - copy_w) // 2, h // 2 + 60), copy_text, fill=(100, 85, 60), font=sub_font)

    except Exception as e:
        log.warning(f"End card text rendering failed: {e}")

    pil_img.save(str(out_path), "PNG")
    return str(out_path)


def _generate_thumbnail(channel, title, scene_image_path, out_path, res=(1280, 720)):
    """Generate a YouTube thumbnail from a scene image + title text overlay.
    YouTube recommends 1280x720 (16:9), minimum 640x360."""
    w, h = res
    channel_id = channel.get("channel_id", "")

    # Load the best scene image as background
    try:
        bg = Image.open(scene_image_path)
        bg = bg.resize((w, h), Image.LANCZOS)
    except Exception:
        bg = Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))

    # Increase contrast and saturation for thumbnail pop
    from PIL import ImageEnhance
    bg = ImageEnhance.Contrast(bg).enhance(1.3)
    bg = ImageEnhance.Color(bg).enhance(1.2)
    bg = ImageEnhance.Brightness(bg).enhance(0.85)

    img_array = np.array(bg).astype(np.float32)

    # Strong bottom gradient for text readability
    for y in range(h):
        fade = max(0, (y - h * 0.5) / (h * 0.5))
        img_array[y] *= (1.0 - fade * 0.7)

    # Subtle vignette
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt(((X - cx) / (w * 0.7)) ** 2 + ((Y - cy) / (h * 0.7)) ** 2)
    vignette = np.clip(1.0 - dist * 0.3, 0.4, 1.0)
    img_array *= vignette[:, :, None]

    pil_img = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

    try:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(pil_img)

        # Channel-specific fonts — LARGER for thumbnails
        channel_font_prefs = {
            "deadlight_codex": ["/System/Library/Fonts/Supplemental/Copperplate.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"],
            "zero_trace_archive": ["/System/Library/Fonts/Supplemental/Courier New.ttf", "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"],
            "the_unwritten_wing": ["/System/Library/Fonts/Supplemental/Baskerville.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"],
            "remnants_project": ["/System/Library/Fonts/Supplemental/Futura.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
            "somnus_protocol": ["/System/Library/Fonts/Supplemental/Didot.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"],
            "autonomous_stack": ["/System/Library/Fonts/SFMono-Regular.otf", "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"],
            "gray_meridian": ["/System/Library/Fonts/Supplemental/Avenir Next.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans-ExtraLight.ttf"],
        }

        font_prefs = channel_font_prefs.get(channel_id, [])
        generic_fonts = [
            "/System/Library/Fonts/Supplemental/Georgia.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        ]

        title_font = None
        for fp in font_prefs + generic_fonts:
            if Path(fp).exists():
                try:
                    title_font = ImageFont.truetype(fp, 72)
                    break
                except Exception:
                    continue
        if not title_font:
            title_font = ImageFont.load_default()

        # UPPERCASE title for thumbnail impact
        words = title.upper().split()
        lines = []
        current = ""
        for word in words:
            test = f"{current} {word}".strip()
            bbox = draw.textbbox((0, 0), test, font=title_font)
            if bbox[2] - bbox[0] > w - 100:
                if current:
                    lines.append(current)
                current = word
            else:
                current = test
        if current:
            lines.append(current)

        # Max 3 lines for readability
        if len(lines) > 3:
            lines = lines[:3]
            lines[2] = lines[2][:20] + "..."

        # Position text in bottom third
        line_height = 82
        total_text_h = len(lines) * line_height
        y_start = h - total_text_h - 50

        # Channel-specific accent colors
        accent_colors = {
            "deadlight_codex": (200, 50, 50),
            "zero_trace_archive": (200, 200, 180),
            "the_unwritten_wing": (255, 220, 150),
            "remnants_project": (120, 180, 100),
            "somnus_protocol": (150, 170, 220),
            "autonomous_stack": (100, 200, 255),
            "gray_meridian": (200, 180, 200),
        }
        text_color = accent_colors.get(channel_id, (255, 255, 255))
        outline_color = (0, 0, 0)

        for i, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=title_font)
            lw = bbox[2] - bbox[0]
            x = (w - lw) // 2
            y = y_start + i * line_height

            # Thick outline (8 directions)
            for ox in range(-3, 4):
                for oy in range(-3, 4):
                    if ox != 0 or oy != 0:
                        draw.text((x + ox, y + oy), line, fill=outline_color, font=title_font)
            draw.text((x, y), line, fill=text_color, font=title_font)

    except Exception as e:
        log.warning(f"Thumbnail text rendering failed: {e}")

    pil_img.save(str(out_path), "PNG")
    log.info(f"Thumbnail generated: {out_path}")
    return str(out_path)


# --- Main video generation pipeline ---

def generate_video(channel, scenes, title, topic, api_keys, generate_short=False, progress=None):
    """
    Full pipeline. progress is a callable(step, message) for SSE updates.
    Returns dict with output paths and metadata.
    """
    def emit(step, msg):
        if progress:
            progress(step, msg)
        log.info(f"[{step}] {msg}")

    channel_id = channel["channel_id"]
    voice = channel.get("voice", {})
    voice_id = voice.get("elevenlabs_voice_id", "EXAVITQu4vr4xnSDxMaL")
    model_id = voice.get("model_id", "eleven_multilingual_v2")
    voice_settings = voice.get("settings", {"stability": 0.85, "similarity_boost": 0.80, "style": 0.15, "use_speaker_boost": False})
    speed = voice.get("speed", 0.85)

    # Pre-flight: verify ElevenLabs API key works before starting expensive pipeline
    import httpx as _hx
    try:
        _check = _hx.get(
            "https://api.elevenlabs.io/v1/user",
            headers={"xi-api-key": api_keys["elevenlabs"]},
            timeout=15,
        )
        if _check.status_code == 200:
            user_info = _check.json()
            char_used = user_info.get("subscription", {}).get("character_count", 0)
            char_limit = user_info.get("subscription", {}).get("character_limit", 0)
            char_remaining = char_limit - char_used

            # Estimate characters needed for this video
            total_chars_needed = sum(len(s.get("narration", "")) for s in scenes)
            pct_of_remaining = (total_chars_needed / max(char_remaining, 1)) * 100 if char_remaining > 0 else 999

            emit("preflight", f"ElevenLabs key valid. Characters remaining: {char_remaining:,} of {char_limit:,}")
            emit("preflight", f"This video needs ~{total_chars_needed:,} characters ({pct_of_remaining:.0f}% of remaining quota)")

            if total_chars_needed > char_remaining:
                emit("error", f"⚠ Not enough ElevenLabs characters! Need ~{total_chars_needed:,} but only {char_remaining:,} remaining.")
                raise RuntimeError(f"Not enough ElevenLabs characters. Need ~{total_chars_needed:,}, have {char_remaining:,}. Upgrade your plan or wait for quota reset.")
        else:
            emit("error", f"ElevenLabs API key check failed: HTTP {_check.status_code} — {_check.text[:200]}")
            raise RuntimeError(f"ElevenLabs API key invalid (HTTP {_check.status_code}). Check your .env file.")
    except httpx.ConnectError as e:
        emit("error", f"Cannot reach ElevenLabs API: {e}")
        raise RuntimeError(f"Cannot reach ElevenLabs: {e}")

    # Image cost estimate (DALL-E 3 HD: $0.080 per image, title card + scenes)
    num_images = len(scenes) + 1  # scenes + title card
    dalle_cost = num_images * 0.080
    emit("preflight", f"DALL-E 3 image estimate: {num_images} images × $0.08 = ~${dalle_cost:.2f}")

    vs = channel.get("video_settings", {})
    res = tuple(vs.get("resolution", [1920, 1080]))
    fps = vs.get("fps", 30)
    crossfade = vs.get("crossfade_seconds", 1.5)
    fade_in = vs.get("fade_in_seconds", 2.0)
    fade_out = vs.get("fade_out_seconds", 3.0)
    scene_pause = vs.get("scene_pause_seconds", 0)
    ambient_cfg = channel.get("ambient_audio", {})

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = "".join(c if c.isalnum() or c in "-_ " else "" for c in title)[:50].strip().replace(" ", "_")
    out_dir = OUTPUT_DIR / channel_id / f"{timestamp}_{safe_title}"
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="vidgen_"))

    # Save plan
    plan = {"title": title, "subject": topic, "scenes": scenes}
    (out_dir / "plan.json").write_text(json.dumps(plan, indent=2))

    # Track topic in bank
    _save_topic_to_bank(channel_id, title)

    emit("plan", f"Saved scene plan ({len(scenes)} scenes)")

    # Step 1: Generate narration
    emit("narration", "Generating narration...")
    audio_durations = []
    for i, scene in enumerate(scenes):
        emit("narration", f"Generating narration for scene {i+1}/{len(scenes)}...")
        audio_path = work_dir / f"narration_{i:03d}.mp3"
        dur = _generate_narration_sync(
            scene["narration"], voice_id, model_id, voice_settings, speed,
            api_keys["elevenlabs"], str(audio_path),
        )
        scene["audio_path"] = str(audio_path)
        scene["audio_duration"] = dur
        audio_durations.append(dur)
        time.sleep(1.5)  # Pause between TTS calls to avoid rate limits

    total_narration = sum(audio_durations)
    emit("narration", f"All narration complete ({total_narration:.0f}s total)")

    # Step 2: Generate images
    emit("images", "Generating scene images...")
    fallback_count = 0
    for i, scene in enumerate(scenes):
        emit("images", f"Generating image for scene {i+1}/{len(scenes)}...")
        img_path = images_dir / f"scene_{i:03d}.png"
        # Generate at FLUX-native resolution (1344x768) for best quality
        ok = _generate_image(
            scene["image_prompt"], str(img_path),
            api_keys.get("hf_token", ""),
            width=1344, height=768,
        )
        if not ok:
            fallback_count += 1
            emit("images", f"⚠ Scene {i+1}: Using fallback image")
            _generate_fallback_image(str(img_path), i, width=1344, height=768)

        # Upscale to ~3x for Ken Burns headroom (more visible scene, less cropping)
        try:
            raw_img = Image.open(str(img_path))
            upscaled = raw_img.resize((3360, 1920), Image.LANCZOS)
            upscaled.save(str(img_path), "PNG")
        except Exception as e:
            log.warning(f"Upscale failed for scene {i}: {e}")

        scene["image_path"] = str(img_path)

    if fallback_count > 0:
        emit("images", f"⚠ {fallback_count}/{len(scenes)} scenes used fallback images — consider re-generating when image quota resets")
    else:
        emit("images", "All images generated successfully")
    used_fallback = fallback_count > 0

    # Generate thumbnail from the most visually interesting scene (scene 2 or 3 — skip opener)
    emit("images", "Generating YouTube thumbnail...")
    thumb_scene_idx = min(2, len(scenes) - 1)
    thumb_scene_img = scenes[thumb_scene_idx].get("image_path", scenes[0].get("image_path", ""))
    thumb_path = out_dir / "thumbnail.png"
    if thumb_scene_img:
        _generate_thumbnail(channel, title, thumb_scene_img, str(thumb_path))
    emit("images", "Thumbnail ready")

    # Step 3: Ken Burns animation
    emit("kenburns", "Applying Ken Burns animation...")
    for i, scene in enumerate(scenes):
        emit("kenburns", f"Animating scene {i+1}/{len(scenes)}...")
        # Make the animation longer than the audio to ensure no cutoff
        audio_dur = scene.get("audio_duration", 10)
        extra_buffer = max(3.0, scene_pause + 1.0) if scene_pause > 0 else 3.0
        duration = audio_dur + extra_buffer  # buffer beyond narration
        kb_path = work_dir / f"kb_{i:03d}.mp4"
        apply_ken_burns(scene["image_path"], duration, str(kb_path), target_res=res)
        scene["video_path"] = str(kb_path)

    emit("kenburns", "All animations complete")

    # Step 4: Title card
    emit("assembly", "Creating title card...")
    title_img_path = work_dir / "title_card.png"
    _generate_title_card(channel, title, 5.0, str(title_img_path), api_keys, api_keys.get("hf_token", ""), res=res)
    title_clip = ImageClip(str(title_img_path)).with_duration(5.0)
    title_clip = title_clip.with_effects([vfx.FadeIn(1.5), vfx.FadeOut(1.5)])
    title_clip = title_clip.with_effects([vfx.Resize(res)])

    # Step 5: Assemble with fade-to-black transitions
    emit("assembly", "Assembling final video...")

    # Build each scene clip with its audio
    assembled_clips = []

    # Title card first (no audio — give it a silent audio track)
    silent_title = AudioFileClip.__new__(AudioFileClip)
    title_with_silent = title_clip.with_audio(None)
    assembled_clips.append(title_with_silent)

    for i, scene in enumerate(scenes):
        clip = VideoFileClip(scene["video_path"])
        clip = clip.with_effects([vfx.Resize(res)])

        if scene.get("audio_path") and scene.get("audio_duration", 0) > 0:
            audio = AudioFileClip(scene["audio_path"])
            # Scene duration = narration + breathing room (more for sleep channels)
            extra_buffer = max(2.0, scene_pause) if scene_pause > 0 else 2.0
            target_dur = scene["audio_duration"] + extra_buffer
            clip = clip.subclipped(0, min(target_dur, clip.duration))
            # Ensure audio matches clip duration
            if audio.duration > clip.duration:
                audio = audio.subclipped(0, clip.duration)
            clip = clip.with_audio(audio)
            log.info(f"Scene {i}: video={clip.duration:.1f}s, audio={audio.duration:.1f}s")
        else:
            log.warning(f"Scene {i}: NO AUDIO")

        # Apply fades BEFORE adding to list (after audio is attached)
        if i == 0:
            clip = clip.with_effects([vfx.FadeIn(fade_in), vfx.FadeOut(0.5)])
        elif i == len(scenes) - 1:
            clip = clip.with_effects([vfx.FadeIn(0.5), vfx.FadeOut(fade_out)])
        else:
            clip = clip.with_effects([vfx.FadeIn(0.5), vfx.FadeOut(0.5)])

        # Add brief black gap before this scene (except first)
        if i > 0:
            gap_duration = scene_pause if scene_pause > 0 else 0.2
            gap = ColorClip(res, color=(0, 0, 0)).with_duration(gap_duration)
            gap = gap.with_audio(None)
            assembled_clips.append(gap)

        assembled_clips.append(clip)

    # End card — "Like, Comment & Subscribe"
    emit("assembly", "Creating end card...")
    end_card_img_path = work_dir / "end_card.png"
    _generate_end_card(channel, 6.0, str(end_card_img_path), res=res)
    end_card_clip = ImageClip(str(end_card_img_path)).with_duration(6.0)
    end_card_clip = end_card_clip.with_effects([vfx.FadeIn(1.5), vfx.FadeOut(2.0)])
    end_card_clip = end_card_clip.with_effects([vfx.Resize(res)])

    # Black gap before end card
    end_gap = ColorClip(res, color=(0, 0, 0)).with_duration(0.5)
    end_gap = end_gap.with_audio(None)
    assembled_clips.append(end_gap)
    assembled_clips.append(end_card_clip)

    # Use method="chain" to avoid audio issues with "compose"
    # Strip all audio from clips — we'll build audio separately via ffmpeg
    for idx in range(len(assembled_clips)):
        assembled_clips[idx] = assembled_clips[idx].without_audio()

    final = concatenate_videoclips(assembled_clips, method="chain")
    log.info(f"Final video duration: {final.duration:.1f}s")

    # Render VIDEO ONLY (no audio — ffmpeg handles all audio)
    video_path_temp = out_dir / "video_noaudio.mp4"
    video_path_with_narration = out_dir / "video_temp.mp4"
    video_path = out_dir / "video.mp4"
    emit("assembly", "Rendering video frames...")
    final.write_videofile(
        str(video_path_temp), fps=fps, codec="libx264",
        audio=False, preset="slow",
        threads=4, logger=None,
    )

    video_duration = final.duration
    for c in assembled_clips:
        try:
            c.close()
        except Exception:
            pass
    title_clip.close()
    final.close()

    # Build complete narration audio track via ffmpeg
    # This avoids moviepy's unreliable audio concatenation entirely
    import subprocess

    emit("assembly", "Building narration audio track via ffmpeg...")

    # Create silent segments and narration concat list
    concat_list_path = work_dir / "audio_concat.txt"
    title_silence = work_dir / "title_silence.wav"
    gap_silence = work_dir / "gap_silence.wav"

    # Generate silence files
    title_dur = 5.0  # title card duration
    gap_dur = scene_pause if scene_pause > 0 else 0.2

    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=mono",
        "-t", str(title_dur), str(title_silence),
    ], capture_output=True)
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=mono",
        "-t", str(gap_dur), str(gap_silence),
    ], capture_output=True)

    # Build concat list: title_silence, then for each scene: [gap_silence +] narration + scene_tail_silence
    concat_entries = []
    concat_entries.append(f"file '{title_silence}'")

    for i, scene in enumerate(scenes):
        if i > 0:
            concat_entries.append(f"file '{gap_silence}'")

        if scene.get("audio_path"):
            # Convert MP3 narration to WAV for consistent concat
            wav_path = work_dir / f"narration_{i:03d}.wav"
            subprocess.run([
                "ffmpeg", "-y", "-i", scene["audio_path"],
                "-ar", "44100", "-ac", "1", str(wav_path),
            ], capture_output=True)
            concat_entries.append(f"file '{wav_path}'")

            # Add tail silence for breathing room after narration
            extra_buffer = max(2.0, scene_pause) if scene_pause > 0 else 2.0
            tail_silence = work_dir / f"tail_silence_{i:03d}.wav"
            subprocess.run([
                "ffmpeg", "-y", "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=mono",
                "-t", str(extra_buffer), str(tail_silence),
            ], capture_output=True)
            concat_entries.append(f"file '{tail_silence}'")

    # End card silence (0.5s gap + 6s card)
    end_card_silence = work_dir / "end_card_silence.wav"
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=mono",
        "-t", "6.5", str(end_card_silence),
    ], capture_output=True)
    concat_entries.append(f"file '{end_card_silence}'")

    concat_list_path.write_text("\n".join(concat_entries))

    # Concat all audio segments into one continuous narration track
    full_narration_path = work_dir / "full_narration.wav"
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_list_path),
        "-c:a", "pcm_s16le", str(full_narration_path),
    ], capture_output=True)

    # Merge narration audio with video
    emit("assembly", "Merging narration with video...")
    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(video_path_temp),
        "-i", str(full_narration_path),
        "-map", "0:v",
        "-map", "1:a",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "320k",
        "-shortest",
        str(video_path_with_narration),
    ], capture_output=True)

    # Mix ambient drone using ffmpeg
    if ambient_cfg.get("enabled", True):
        emit("assembly", "Mixing ambient audio via ffmpeg...")
        drone_path = work_dir / "ambient.wav"
        generate_ambient_drone(video_duration + 2, str(drone_path), channel_id=channel_id)
        vol = ambient_cfg.get("volume", 0.30)

        mix_cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path_with_narration),
            "-i", str(drone_path),
            "-filter_complex",
            f"[1:a]atrim=0:{video_duration:.2f},volume={vol}[drone];[0:a][drone]amix=inputs=2:duration=first:dropout_transition=3[out]",
            "-map", "0:v",
            "-map", "[out]",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "320k",
            str(video_path),
        ]
        result = subprocess.run(mix_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            log.info(f"Ambient mixed successfully at volume {vol}")
        else:
            log.warning(f"Ambient mix failed: {result.stderr[:200]}")
            # Fall back to the version without ambient
            import shutil as sh
            sh.move(str(video_path_with_narration), str(video_path))
    else:
        import shutil as sh
        sh.move(str(video_path_with_narration), str(video_path))
        log.info("Ambient audio disabled for this channel")

    # Clean up temp files
    video_path_temp.unlink(missing_ok=True)
    video_path_with_narration.unlink(missing_ok=True)

    emit("assembly", f"Video complete: {video_duration:.0f}s")

    # Step 5: Optional Short
    short_meta = None
    if generate_short:
        emit("short", "Generating YouTube Short...")
        try:
            short_meta = _generate_short(channel, scenes, work_dir, out_dir, api_keys, voice_id, model_id, voice_settings, speed, emit)
        except Exception as e:
            emit("short", f"Short generation failed: {e}")

    # Step 6: Generate YouTube metadata
    emit("youtube_meta", "Generating YouTube description, tags & hashtags...")
    youtube_meta = _generate_youtube_metadata(channel, title, topic, scenes, api_keys["openai"])

    # Save YouTube metadata as separate file for easy copy-paste
    (out_dir / "youtube.json").write_text(json.dumps(youtube_meta, indent=2))
    emit("youtube_meta", "YouTube metadata ready")

    # Save metadata
    meta = {
        "title": title,
        "topic": topic,
        "channel_id": channel_id,
        "channel_name": channel["channel_name"],
        "timestamp": timestamp,
        "duration": round(video_duration, 1),
        "scenes_count": len(scenes),
        "youtube_uploaded": False,
        "has_short": short_meta is not None,
        "has_thumbnail": thumb_path.exists(),
        "short": short_meta,
        "used_fallback_images": used_fallback,
        "fallback_image_count": fallback_count,
        "youtube_meta": youtube_meta,
        "created_at": datetime.now().isoformat(),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    # Copy to Desktop folders
    channel_name = channel["channel_name"]
    desktop_video_name = f"{safe_title}_{timestamp}.mp4"

    desktop_vid_dir = _get_desktop_channel_dir(channel_name)
    desktop_vid_path = desktop_vid_dir / desktop_video_name
    shutil.copy2(str(video_path), str(desktop_vid_path))
    emit("export", f"Video saved to Desktop/EmroseMedia/{channel_name}/")
    meta["desktop_video_path"] = str(desktop_vid_path)

    if short_meta is not None and (out_dir / "short.mp4").exists():
        desktop_short_dir = _get_desktop_shorts_dir(channel_name)
        desktop_short_path = desktop_short_dir / f"{safe_title}_{timestamp}_short.mp4"
        shutil.copy2(str(out_dir / "short.mp4"), str(desktop_short_path))
        emit("export", f"Short saved to Desktop/EmroseMedia/{channel_name}_Shorts/")
        meta["desktop_short_path"] = str(desktop_short_path)

    # Re-save metadata with desktop paths
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    # Cleanup work dir
    shutil.rmtree(work_dir, ignore_errors=True)

    if used_fallback:
        emit("warning", f"⚠ {fallback_count} of {len(scenes)} scenes used fallback images. Consider re-generating this video when image quota resets.")

    emit("done", f"Complete! Video saved to Desktop/EmroseMedia/{channel_name}/")
    return {
        "video_path": str(video_path),
        "desktop_video_path": str(desktop_vid_path),
        "output_dir": str(out_dir),
        "dir_name": out_dir.name,
        "metadata": meta,
        "used_fallback": used_fallback,
        "fallback_count": fallback_count,
    }


def _generate_youtube_metadata(channel, title, topic, scenes, openai_key):
    """Generate YouTube description, tags, and hashtags for the video."""
    year = datetime.now().year
    channel_name = channel.get("channel_name", "")
    channel_desc = channel.get("description", "")
    default_tags = channel.get("youtube", {}).get("default_tags", [])
    default_hashtags = channel.get("youtube", {}).get("default_hashtags", [])
    category = channel.get("youtube", {}).get("category", "Entertainment")

    full_narration = " ".join(s.get("narration", "") for s in scenes)
    word_count = len(full_narration.split())

    prompt = f"""You are writing YouTube upload metadata for a video.

Channel: {channel_name}
Channel Description: {channel_desc}
Video Title: {title}
Topic: {topic}
Video length: ~{word_count} words of narration, approximately {len(scenes)} scenes

Write the following in JSON format:

1. "description" — A YouTube description (150-300 words) that:
   - Opens with a compelling 1-2 sentence hook about the video
   - Briefly describes what the viewer will experience
   - Includes a line: "If you enjoyed this, please like, comment, and subscribe for more."
   - Includes a call to turn on notifications
   - Matches the channel's tone perfectly
   - Ends with exactly this copyright line: "© {year} Emrose Media Studios. All rights reserved."

2. "tags" — An array of 15-25 YouTube tags (individual words or short phrases) optimized for search discovery. Include a mix of broad and specific terms.

3. "hashtags" — An array of 5-8 hashtags for the video description (with # prefix). These appear above the title on YouTube.

4. "category" — YouTube category (default: "{category}")

Respond with ONLY valid JSON, no markdown fences:
{{"description": "...", "tags": ["tag1", "tag2"], "hashtags": ["#tag1", "#tag2"], "category": "..."}}"""

    try:
        result = _call_openai_sync([{"role": "user", "content": prompt}], openai_key)
        result = result.strip()
        if result.startswith("```"):
            result = result.split("\n", 1)[1]
        if result.endswith("```"):
            result = result.rsplit("```", 1)[0]
        yt_meta = json.loads(result.strip())

        # Merge default channel tags
        existing_tags = set(t.lower() for t in yt_meta.get("tags", []))
        for dt in default_tags:
            if dt.lower() not in existing_tags:
                yt_meta["tags"].append(dt)

        # Merge default channel hashtags
        if default_hashtags:
            existing_ht = set(h.lower() for h in yt_meta.get("hashtags", []))
            for dh in default_hashtags:
                if dh.lower() not in existing_ht:
                    yt_meta["hashtags"].insert(0, dh)

        return yt_meta
    except Exception as e:
        log.warning(f"YouTube metadata generation failed: {e}")
        return {
            "description": f"{title}\n\n© {year} Emrose Media Studios. All rights reserved.",
            "tags": default_tags,
            "hashtags": [f"#{channel_name}"],
            "category": category,
        }


def compile_videos(channel, video_dirs, title, progress=None):
    """Compile multiple videos into one long-form video with transitions.
    Great for 30-60 minute compilations that drive watch time."""
    import subprocess

    def emit(step, msg):
        if progress:
            progress(step, msg)
        log.info(f"[compile/{step}] {msg}")

    channel_id = channel["channel_id"]
    channel_name = channel["channel_name"]
    vs = channel.get("video_settings", {})
    res = tuple(vs.get("resolution", [1920, 1080]))

    emit("compile", f"Compiling {len(video_dirs)} videos...")

    # Collect video paths
    video_paths = []
    for dir_name in video_dirs:
        vp = OUTPUT_DIR / channel_id / dir_name / "video.mp4"
        if vp.exists():
            video_paths.append(str(vp))
            emit("compile", f"Added: {dir_name}")
        else:
            emit("compile", f"⚠ Skipping {dir_name} — video.mp4 not found")

    if len(video_paths) < 2:
        raise RuntimeError("Need at least 2 videos to compile")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = "".join(c if c.isalnum() or c in "-_ " else "" for c in title)[:50].strip().replace(" ", "_")
    out_dir = OUTPUT_DIR / channel_id / f"{timestamp}_{safe_title}_compilation"
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="compile_"))

    # Build ffmpeg concat list with 2s black transition between videos
    concat_list = work_dir / "concat.txt"

    # Create a 2-second black transition video
    transition_path = work_dir / "transition.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"color=c=black:s={res[0]}x{res[1]}:r=30:d=2",
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
        "-t", "2",
        "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "320k",
        str(transition_path),
    ], capture_output=True)

    entries = []
    for i, vp in enumerate(video_paths):
        if i > 0:
            entries.append(f"file '{transition_path}'")
        entries.append(f"file '{vp}'")

    concat_list.write_text("\n".join(entries))

    # Compile via ffmpeg concat
    output_path = out_dir / "video.mp4"
    emit("compile", "Concatenating videos via ffmpeg...")
    result = subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_list),
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-c:a", "aac", "-b:a", "320k",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ], capture_output=True, text=True)

    if result.returncode != 0:
        log.error(f"Compilation failed: {result.stderr[:300]}")
        raise RuntimeError(f"Compilation failed: {result.stderr[:200]}")

    # Get duration
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(output_path),
    ], capture_output=True, text=True)
    duration = float(probe.stdout.strip()) if probe.stdout.strip() else 0

    # Save metadata
    meta = {
        "title": title,
        "channel_id": channel_id,
        "channel_name": channel_name,
        "timestamp": timestamp,
        "duration": round(duration, 1),
        "is_compilation": True,
        "source_videos": video_dirs,
        "source_count": len(video_dirs),
        "youtube_uploaded": False,
        "created_at": datetime.now().isoformat(),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    # Copy to Desktop
    desktop_dir = _get_desktop_channel_dir(channel_name)
    desktop_path = desktop_dir / f"{safe_title}_{timestamp}_compilation.mp4"
    shutil.copy2(str(output_path), str(desktop_path))

    duration_min = int(duration // 60)
    emit("done", f"Compilation complete! {duration_min}min — saved to Desktop/EmroseMedia/{channel_name}/")

    # Cleanup
    shutil.rmtree(work_dir, ignore_errors=True)

    return {
        "video_path": str(output_path),
        "desktop_path": str(desktop_path),
        "output_dir": str(out_dir),
        "dir_name": out_dir.name,
        "metadata": meta,
    }


def _generate_short(channel, scenes, work_dir, out_dir, api_keys, voice_id, model_id, voice_settings, speed, emit):
    full_script = "\n\n".join([f"Scene {i}: {s['narration']}" for i, s in enumerate(scenes)])
    short_prompt = f"""You are creating a YouTube Short derived from a longer video script.

Select the most engaging, curiosity-driven segment and convert it into a short-form narration.

Rules:
- Length: 20-40 seconds when read at the narrator's pace
- Must start with a strong hook in the first 2 seconds
- Must NOT require prior context
- Must create curiosity, tension, or insight
- Must feel complete but leave the viewer wanting more
- Do NOT summarize — extract a compelling moment

Match the tone of {channel['channel_name']} exactly.

Output ONLY valid JSON:
{{"narration": "short script", "image_prompt": "image prompt matching channel style", "suggested_title": "title", "suggested_caption": "caption", "suggested_hashtags": ["#tag1", "#tag2"]}}

Full script:
{full_script}"""

    result = _call_openai_sync([{"role": "user", "content": short_prompt}], api_keys["openai"])
    result = result.strip()
    if result.startswith("```"):
        result = result.split("\n", 1)[1]
    if result.endswith("```"):
        result = result.rsplit("```", 1)[0]
    short_data = json.loads(result.strip())

    emit("short", "Generating short narration...")
    short_audio = work_dir / "short_narration.mp3"
    dur = _generate_narration_sync(short_data["narration"], voice_id, model_id, voice_settings, speed, api_keys["elevenlabs"], str(short_audio))

    emit("short", "Generating short image...")
    short_img = work_dir / "short_image.png"
    ok = _generate_image(short_data["image_prompt"], str(short_img), api_keys.get("hf_token", ""), width=1080, height=1920)
    if not ok:
        _generate_fallback_image(str(short_img), 99, width=1080, height=1920)

    emit("short", "Animating short...")
    short_kb = work_dir / "short_kb.mp4"
    apply_ken_burns(str(short_img), dur + 1.5, str(short_kb), target_res=(1080, 1920))

    video = VideoFileClip(str(short_kb))
    video = video.subclipped(0, min(dur + 1.5, video.duration))
    audio = AudioFileClip(str(short_audio))
    video = video.with_audio(audio)
    video = video.with_effects([vfx.FadeIn(1.0), vfx.FadeOut(1.5)])

    short_output = out_dir / "short.mp4"
    video.write_videofile(str(short_output), fps=30, codec="libx264", audio_codec="aac", audio_bitrate="320k", preset="slow", threads=4, logger=None)
    video.close()
    audio.close()

    short_data["duration"] = round(dur + 1.5, 1)
    emit("short", "YouTube Short complete")
    return short_data
