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
IMAGE_GEN_WIDTH = 2048
IMAGE_GEN_HEIGHT = 1365
KB_ZOOM_RANGE = (1.05, 1.18)
KB_PAN_RANGE = 0.08
CROSSFADE_DURATION = 1.5
FLUX_SPACE = os.environ.get("FLUX_SPACE", "multimodalart/FLUX.1-merged")
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

    return f"""You are the creative director for {c['channel_name']}.
{c.get('description', '')}

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

TARGET LENGTH: The final video MUST be between {vs.get('target_duration_min', 5)} and {vs.get('target_duration_max', 7)} minutes long. Err on the side of going longer rather than shorter.
- Aim for {vs.get('scene_count_min', 10)}-{vs.get('scene_count_max', 14)} scenes.
- Each scene's narration MUST be 4-6 sentences and 50-80 words. This is critical.
- Total narration across all scenes MUST be {vs.get('target_word_count_min', 700)}-{vs.get('target_word_count_max', 900)} words. This is NON-NEGOTIABLE.
- Do NOT write short 1-2 sentence narration.
- duration_hint should be 10."""


def generate_script(channel, topic, api_key):
    system = _build_director_prompt(channel)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Create an entry about:\n\n{topic}"},
    ]
    result = _call_openai_sync(messages, api_key)
    result = result.strip()
    if result.startswith("```"):
        result = result.split("\n", 1)[1]
    if result.endswith("```"):
        result = result.rsplit("```", 1)[0]
    return json.loads(result.strip())


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
            time.sleep(wait)
            continue
        resp.raise_for_status()
        Path(out_path).write_bytes(resp.content)
        clip = AudioFileClip(str(out_path))
        dur = clip.duration
        clip.close()
        return dur
    raise RuntimeError("ElevenLabs rate limit exceeded")


def generate_ambient_drone(duration, out_path):
    sr = 44100
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    detune = np.sin(2 * np.pi * 0.03 * t) * 0.5
    drone1 = np.sin(2 * np.pi * (38.0 + detune) * t) * 0.3
    drone2 = np.sin(2 * np.pi * (55.0 + detune * 0.7) * t) * 0.2
    harm_lfo = np.sin(2 * np.pi * 0.05 * t) * 0.5 + 0.5
    harm1 = np.sin(2 * np.pi * 82.4 * t) * 0.08 * harm_lfo
    harm2 = np.sin(2 * np.pi * 87.3 * t) * 0.06 * (1.0 - harm_lfo) * 0.8
    noise = np.random.randn(len(t)) * 0.015
    ks = int(sr / 200)
    if ks > 1:
        noise = np.convolve(noise, np.ones(ks) / ks, mode="same")
    noise *= (np.sin(2 * np.pi * 0.02 * t) * 0.5 + 0.5) ** 2
    sweep_freq = np.linspace(25, 30, len(t))
    sweep = np.sin(2 * np.pi * sweep_freq * t / 2) * 0.1
    sweep *= np.sin(2 * np.pi * 0.01 * t) * 0.5 + 0.5
    ambient = drone1 + drone2 + harm1 + harm2 + noise + sweep
    fi = int(sr * 4.0)
    fo = int(sr * 5.0)
    ambient[:fi] *= np.linspace(0, 1, fi) ** 2
    ambient[-fo:] *= np.linspace(1, 0, fo) ** 2
    peak = np.max(np.abs(ambient))
    if peak > 0:
        ambient = ambient / peak * 0.7
    scipy_wav.write(str(out_path), sr, (ambient * 32767).astype(np.int16))


# --- Image generation ---

def _generate_image_flux(prompt, out_path, hf_token, width=2048, height=1365):
    from gradio_client import Client
    for attempt in range(3):
        try:
            # gradio_client versions vary: try token=, then hf_token=, then no auth
            client = None
            if hf_token:
                try:
                    client = Client(FLUX_SPACE, hf_token=hf_token)
                except TypeError:
                    try:
                        client = Client(FLUX_SPACE, token=hf_token)
                    except TypeError:
                        client = Client(FLUX_SPACE)
            else:
                client = Client(FLUX_SPACE)
            result = client.predict(
                prompt=prompt, seed=0, randomize_seed=True,
                width=width, height=height,
                guidance_scale=FLUX_GUIDANCE, num_inference_steps=FLUX_STEPS,
                api_name="/infer",
            )
            img_result = result[0] if isinstance(result, tuple) else result
            src = img_result.get("path", img_result.get("url", "")) if isinstance(img_result, dict) else str(img_result)
            img = Image.open(src)
            img.save(str(out_path), "PNG")
            return True
        except Exception as e:
            log.warning(f"FLUX attempt {attempt+1}/3 failed: {e}")
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
    return False


def _generate_fallback_image(out_path, scene_index, width=2048, height=1365):
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

    total_frames = int(duration * TARGET_FPS)
    img_array = np.array(img)
    writer = imageio.get_writer(
        str(out_path), fps=TARGET_FPS, codec="libx264", quality=8,
        macro_block_size=1,
        output_params=["-preset", "medium", "-pix_fmt", "yuv420p"],
    )
    for frame_idx in range(total_frames):
        t = frame_idx / max(total_frames - 1, 1)
        t_smooth = 0.5 - 0.5 * math.cos(math.pi * t)
        zoom = zoom_start + (zoom_end - zoom_start) * t_smooth
        pan_x = start_x + (end_x - start_x) * t_smooth
        pan_y = start_y + (end_y - start_y) * t_smooth
        crop_w = int(target_w / zoom)
        crop_h = int(target_h / zoom)
        cx = img_w // 2 + int(pan_x)
        cy = img_h // 2 + int(pan_y)
        x1 = max(0, min(cx - crop_w // 2, img_w - crop_w))
        y1 = max(0, min(cy - crop_h // 2, img_h - crop_h))
        frame = img_array[y1:y1+crop_h, x1:x1+crop_w]
        frame_img = Image.fromarray(frame).resize((target_w, target_h), Image.LANCZOS)
        writer.append_data(np.array(frame_img))
    writer.close()
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
    vs = channel.get("video_settings", {})
    res = tuple(vs.get("resolution", [1920, 1080]))
    fps = vs.get("fps", 30)
    crossfade = vs.get("crossfade_seconds", 1.5)
    fade_in = vs.get("fade_in_seconds", 2.0)
    fade_out = vs.get("fade_out_seconds", 3.0)
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
        time.sleep(0.5)  # Brief pause between TTS calls

    total_narration = sum(audio_durations)
    emit("narration", f"All narration complete ({total_narration:.0f}s total)")

    # Step 2: Generate images
    emit("images", "Generating scene images...")
    for i, scene in enumerate(scenes):
        emit("images", f"Generating image for scene {i+1}/{len(scenes)}...")
        img_path = images_dir / f"scene_{i:03d}.png"
        ok = _generate_image_flux(
            scene["image_prompt"], str(img_path),
            api_keys.get("hf_token", ""),
            width=2048, height=1365,
        )
        if not ok:
            emit("images", f"Scene {i+1}: Using fallback image")
            _generate_fallback_image(str(img_path), i, width=2048, height=1365)
        scene["image_path"] = str(img_path)

    emit("images", "All images complete")

    # Step 3: Ken Burns animation
    emit("kenburns", "Applying Ken Burns animation...")
    for i, scene in enumerate(scenes):
        emit("kenburns", f"Animating scene {i+1}/{len(scenes)}...")
        duration = scene.get("audio_duration", 10) + 1.5
        kb_path = work_dir / f"kb_{i:03d}.mp4"
        apply_ken_burns(scene["image_path"], duration, str(kb_path), target_res=res)
        scene["video_path"] = str(kb_path)

    emit("kenburns", "All animations complete")

    # Step 4: Assemble
    emit("assembly", "Assembling final video...")
    clips = []
    for scene in scenes:
        clip = VideoFileClip(scene["video_path"])
        if scene.get("audio_path") and scene.get("audio_duration", 0) > 0:
            audio = AudioFileClip(scene["audio_path"])
            target_dur = scene["audio_duration"] + 1.5
            if clip.duration < target_dur:
                sf = clip.duration / target_dur
                if sf > 0.4:
                    clip = clip.with_effects([vfx.MultiplySpeed(sf)])
                else:
                    clip = clip.with_effects([vfx.Loop(duration=target_dur)])
            clip = clip.subclipped(0, min(target_dur, clip.duration))
            clip = clip.with_audio(audio)
        clip = clip.with_effects([vfx.Resize(res)])
        clips.append(clip)

    for i in range(1, len(clips)):
        clips[i] = clips[i].with_effects([vfx.CrossFadeIn(crossfade)])
    clips[0] = clips[0].with_effects([vfx.FadeIn(fade_in)])
    clips[-1] = clips[-1].with_effects([vfx.FadeOut(fade_out)])

    if len(clips) > 1:
        final = concatenate_videoclips(clips, method="compose", padding=-crossfade)
    else:
        final = clips[0]

    # Ambient drone
    if ambient_cfg.get("enabled", True):
        emit("assembly", "Mixing ambient audio...")
        drone_path = work_dir / "ambient.wav"
        generate_ambient_drone(final.duration, str(drone_path))
        ambient = AudioFileClip(str(drone_path))
        vol = ambient_cfg.get("volume", 0.20)
        ambient = ambient.with_volume_scaled(vol)
        if final.audio is not None:
            final = final.with_audio(CompositeAudioClip([final.audio, ambient]))
        else:
            final = final.with_audio(ambient)

    video_path = out_dir / "video.mp4"
    emit("assembly", "Rendering final video...")
    final.write_videofile(
        str(video_path), fps=fps, codec="libx264",
        audio_codec="aac", audio_bitrate="320k", preset="slow",
        threads=4, logger=None,
    )

    video_duration = final.duration
    for c in clips:
        c.close()
    final.close()

    emit("assembly", f"Video complete: {video_duration:.0f}s")

    # Step 5: Optional Short
    short_meta = None
    if generate_short:
        emit("short", "Generating YouTube Short...")
        try:
            short_meta = _generate_short(channel, scenes, work_dir, out_dir, api_keys, voice_id, model_id, voice_settings, speed, emit)
        except Exception as e:
            emit("short", f"Short generation failed: {e}")

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
        "short": short_meta,
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

    emit("done", f"Complete! Video saved to Desktop/EmroseMedia/{channel_name}/")
    return {
        "video_path": str(video_path),
        "desktop_video_path": str(desktop_vid_path),
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
    ok = _generate_image_flux(short_data["image_prompt"], str(short_img), api_keys.get("hf_token", ""), width=1080, height=1920)
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
