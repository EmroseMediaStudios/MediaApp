#!/usr/bin/env python3
"""
DeadlightCodex — Fully Automated Video Generator

Takes a text prompt, breaks it into scenes via LLM, generates images
via FLUX (free), applies Ken Burns animation, adds narration via ElevenLabs,
and stitches into a final .mp4.

Usage:
    export ELEVENLABS_API_KEY="your-key"
    export OPENAI_API_KEY="your-key"  # or ANTHROPIC_API_KEY

    python deadlight_codex.py "The Observer Paradox Entity"

Requirements:
    pip install -r requirements.txt
"""

import os
import sys
import json
import time
import math
import random
import asyncio
import argparse
import tempfile
import logging
import shutil
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.io import wavfile as scipy_wav
from PIL import Image

import httpx
from moviepy import (
    VideoFileClip,
    AudioFileClip,
    ImageClip,
    concatenate_videoclips,
    ColorClip,
    CompositeVideoClip,
    CompositeAudioClip,
    vfx,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# ElevenLabs voice settings — tuned for "The Keeper"
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")
ELEVENLABS_MODEL_ID = os.environ.get("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")
VOICE_SETTINGS = {
    "stability": 0.85,
    "similarity_boost": 0.80,
    "style": 0.15,
    "use_speaker_boost": False,
}
VOICE_SPEED = float(os.environ.get("ELEVENLABS_SPEED", "0.85"))

# HuggingFace token (free account — increases GPU quotas dramatically)
HF_TOKEN = os.environ.get('HF_TOKEN', '')

# Image generation
FLUX_SPACE = os.environ.get("FLUX_SPACE", "multimodalart/FLUX.1-merged")
FLUX_STEPS = int(os.environ.get("FLUX_STEPS", "8"))
FLUX_GUIDANCE = float(os.environ.get("FLUX_GUIDANCE", "3.5"))
# Generate images larger than target for Ken Burns headroom
IMAGE_GEN_WIDTH = 2048
IMAGE_GEN_HEIGHT = 1365

# Ken Burns animation
KB_ZOOM_RANGE = (1.05, 1.18)  # Min/max zoom factor
KB_PAN_RANGE = 0.08  # Max pan as fraction of image size

# Video assembly
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./output")
CROSSFADE_DURATION = 1.5
TARGET_FPS = 30
TARGET_RESOLUTION = (1920, 1080)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("deadlight")

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Scene:
    index: int
    narration: str
    image_prompt: str
    duration_hint: float
    audio_path: Optional[str] = None
    image_path: Optional[str] = None
    video_path: Optional[str] = None
    audio_duration: float = 0.0

@dataclass
class Entry:
    title: str
    subject: str
    scenes: list[Scene] = field(default_factory=list)
    output_path: Optional[str] = None

# ---------------------------------------------------------------------------
# LLM Director
# ---------------------------------------------------------------------------

DIRECTOR_SYSTEM_PROMPT = """You are the creative director for the DeadlightCodex, a cosmic-horror archival video series.

You will receive a subject prompt. Break it into a sequence of scenes for a 2-5 minute video entry.

NARRATION RULES (The Keeper):
- Deep, controlled, deliberate tone. Ancient, restrained, authoritative.
- Slow, measured pacing. Minimal emotional variation. No urgency.
- Each sentence should feel intentional and weighted.
- The Keeper does not speculate. The Keeper does not question. The Keeper presents.
- No dramatic exaggeration, no flowery language, no sinister tone, no modern phrasing.
- Feels like a record being read, not a story being told.

OPENING (always Scene 1 narration — must start exactly like this):
"You are listening to the DeadlightCodex."
[then a beat]
"This entry concerns [brief subject description]."
Then continue into the narrative without breaking tone.

CLOSING (always last scene narration):
End with a restrained closing like:
"This entry remains incomplete."
or "Further records are either missing or intentionally withheld."
End without resolution. Do not summarize. Do not explain. Leave the implication open.

IMAGE PROMPT RULES (for each scene):
Write detailed prompts for generating a STILL IMAGE that will have slow Ken Burns animation applied. Each prompt must describe:
- A cinematic, atmospheric, dark cosmic-horror composition
- Ancient, abstract environment not tied to any real-world location
- Low, directional, moody lighting with deep shadows
- Muted palette: black, deep red, desaturated gold, cold gray
- Subtle film grain, soft depth of field
- Symbolic rather than literal environments
- Vast, empty, or structurally unnatural spaces
- Elements like: abstract structures, monoliths, void-like spaces, dust/particles, faint light, fog
- Avoid: text, subtitles, readable symbols, copyrighted characters, bright colors, modern environments, human faces as focal point
- The image should look like a frame from a cinematic film
- End the prompt with: cinematic still, 4k, film grain, atmospheric, dark cosmic horror photography

OUTPUT FORMAT — respond with ONLY valid JSON, no markdown fences:
{
  "title": "short title for the entry",
  "subject": "brief subject line used in the opening",
  "scenes": [
    {
      "narration": "The Keeper's narration text for this scene",
      "image_prompt": "Detailed image generation prompt for this scene",
      "duration_hint": 10
    }
  ]
}

TARGET LENGTH: The final video MUST be between 8 and 10 minutes long. Videos MUST be at least 8 minutes for mid-roll ad eligibility.
- Aim for 12-16 scenes.
- Each scene's narration MUST be 5-8 sentences and 70-100 words. This is critical.
- Total narration across all scenes MUST be 1000-1300 words. This is NON-NEGOTIABLE.
- Do NOT write short 1-2 sentence narration. The Keeper speaks in measured, deliberate detail.
- duration_hint should be 10."""

async def direct_scenes(prompt: str) -> Entry:
    log.info("Directing scenes from prompt...")
    messages = [
        {"role": "system", "content": DIRECTOR_SYSTEM_PROMPT},
        {"role": "user", "content": f"Create a DeadlightCodex entry about:\n\n{prompt}"},
    ]
    if ANTHROPIC_API_KEY:
        result = await _call_anthropic(messages)
    elif OPENAI_API_KEY:
        result = await _call_openai(messages)
    else:
        raise RuntimeError("No LLM API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.")

    result = result.strip()
    if result.startswith("```"):
        result = result.split("\n", 1)[1]
    if result.endswith("```"):
        result = result.rsplit("```", 1)[0]
    result = result.strip()

    data = json.loads(result)
    entry = Entry(
        title=data["title"],
        subject=data["subject"],
        scenes=[
            Scene(
                index=i,
                narration=s["narration"],
                image_prompt=s["image_prompt"],
                duration_hint=s.get("duration_hint", 10),
            )
            for i, s in enumerate(data["scenes"])
        ],
    )
    log.info(f"Directed {len(entry.scenes)} scenes: \"{entry.title}\"")
    for s in entry.scenes:
        log.info(f"  Scene {s.index}: {s.narration[:80]}...")
    return entry

async def _call_anthropic(messages):
    system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
    user_msgs = [m for m in messages if m["role"] != "system"]
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": "claude-sonnet-4-20250514", "max_tokens": 4096, "system": system_msg, "messages": user_msgs},
        )
        resp.raise_for_status()
        return resp.json()["content"][0]["text"]

async def _call_openai(messages):
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json={"model": "gpt-4o", "messages": messages, "temperature": 0.7},
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

# ---------------------------------------------------------------------------
# ElevenLabs TTS
# ---------------------------------------------------------------------------

async def generate_narration(scene: Scene, work_dir: Path, semaphore: asyncio.Semaphore | None = None) -> str:
    log.info(f"  Scene {scene.index}: Generating narration...")
    out_path = work_dir / f"narration_{scene.index:03d}.mp3"

    async def _do_request(client):
        payload = {"text": scene.narration, "model_id": ELEVENLABS_MODEL_ID, "voice_settings": VOICE_SETTINGS}
        if VOICE_SPEED != 1.0:
            payload["speed"] = VOICE_SPEED
        for attempt in range(5):
            resp = await client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}",
                headers={"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json", "Accept": "audio/mpeg"},
                json=payload,
            )
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", str(5 * (attempt + 1))))
                log.warning(f"  Scene {scene.index}: Rate limited, retrying in {wait}s ({attempt+1}/5)")
                await asyncio.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.content
        raise RuntimeError(f"ElevenLabs rate limit exceeded for scene {scene.index}")

    if semaphore:
        async with semaphore:
            async with httpx.AsyncClient(timeout=180) as client:
                data = await _do_request(client)
    else:
        async with httpx.AsyncClient(timeout=180) as client:
            data = await _do_request(client)

    out_path.write_bytes(data)
    clip = AudioFileClip(str(out_path))
    scene.audio_duration = clip.duration
    clip.close()
    scene.audio_path = str(out_path)
    log.info(f"  Scene {scene.index}: Narration ready ({scene.audio_duration:.1f}s)")
    return str(out_path)

# ---------------------------------------------------------------------------
# FLUX Image Generation (free via HuggingFace Spaces)
# ---------------------------------------------------------------------------

def generate_fallback_image(scene: Scene, work_dir: Path) -> str:
    """Generate a procedural dark atmospheric image as fallback."""
    log.info(f"  Scene {scene.index}: Generating procedural fallback image...")
    out_path = work_dir / f"image_{scene.index:03d}.png"
    w, h = IMAGE_GEN_WIDTH, IMAGE_GEN_HEIGHT

    # Dark base with subtle gradient
    img = np.zeros((h, w, 3), dtype=np.float32)
    # Vertical gradient (slightly lighter at center)
    for y in range(h):
        v = 0.02 + 0.03 * math.exp(-((y - h * 0.4) ** 2) / (2 * (h * 0.3) ** 2))
        img[y, :] = v
    # Radial vignette
    cy, cx = h * 0.45, w * 0.5
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 / (w * 0.4) ** 2 + (Y - cy) ** 2 / (h * 0.4) ** 2)
    vignette = np.clip(1.0 - dist * 0.5, 0, 1)
    img *= vignette[:, :, None]
    # Add subtle color tint (deep red / desaturated gold)
    tints = [(0.08, 0.02, 0.02), (0.04, 0.03, 0.01), (0.02, 0.02, 0.04)]
    tint = tints[scene.index % len(tints)]
    img[:, :, 0] += tint[0]
    img[:, :, 1] += tint[1]
    img[:, :, 2] += tint[2]
    # Noise/grain
    noise = np.random.randn(h, w, 3) * 0.015
    img += noise
    # Floating particles
    for _ in range(random.randint(20, 50)):
        px, py = random.randint(0, w - 1), random.randint(0, h - 1)
        brightness = random.uniform(0.08, 0.2)
        size = random.randint(1, 3)
        y1, y2 = max(0, py - size), min(h, py + size)
        x1, x2 = max(0, px - size), min(w, px + size)
        img[y1:y2, x1:x2] += brightness

    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(str(out_path), "PNG")
    scene.image_path = str(out_path)
    log.info(f"  Scene {scene.index}: Fallback image ready")
    return str(out_path)


def generate_scene_image(scene: Scene, work_dir: Path) -> str:
    """Generate a still image for a scene via FLUX on HuggingFace Spaces."""
    from gradio_client import Client
    log.info(f"  Scene {scene.index}: Generating image via FLUX...")
    out_path = work_dir / f"image_{scene.index:03d}.png"

    for attempt in range(3):
        try:
            client = Client(FLUX_SPACE, token=HF_TOKEN if HF_TOKEN else None)
            result = client.predict(
                prompt=scene.image_prompt,
                seed=0,
                randomize_seed=True,
                width=IMAGE_GEN_WIDTH,
                height=IMAGE_GEN_HEIGHT,
                guidance_scale=FLUX_GUIDANCE,
                num_inference_steps=FLUX_STEPS,
                api_name="/infer",
            )
            # result is (image_dict, seed) or (filepath, seed)
            if isinstance(result, tuple):
                img_result = result[0]
            else:
                img_result = result

            if isinstance(img_result, dict):
                src_path = img_result.get("path", img_result.get("url", ""))
            else:
                src_path = str(img_result)

            # Copy/convert to PNG
            img = Image.open(src_path)
            img.save(str(out_path), "PNG")
            scene.image_path = str(out_path)
            log.info(f"  Scene {scene.index}: Image ready ({img.size[0]}x{img.size[1]})")
            return str(out_path)

        except Exception as e:
            log.warning(f"  Scene {scene.index}: FLUX attempt {attempt+1}/3 failed: {e}")
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
            else:
                log.warning(f"  Scene {scene.index}: FLUX exhausted, using fallback image")
                return generate_fallback_image(scene, work_dir)

def generate_images_threaded(scenes: list[Scene], work_dir: Path):
    """Generate images with limited concurrency (HF Spaces can be slow)."""
    log.info("Generating scene images via FLUX...")
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(generate_scene_image, s, work_dir): s for s in scenes}
        for future in futures:
            future.result()  # Raises on failure

# ---------------------------------------------------------------------------
# Ken Burns Animation
# ---------------------------------------------------------------------------

def apply_ken_burns(image_path: str, duration: float, work_dir: Path, scene_index: int) -> str:
    """
    Apply Ken Burns (slow zoom + pan) effect to a still image.
    Returns path to the generated .mp4 clip.
    """
    log.info(f"  Scene {scene_index}: Applying Ken Burns animation ({duration:.1f}s)...")
    out_path = work_dir / f"kb_{scene_index:03d}.mp4"

    img = Image.open(image_path)
    img_w, img_h = img.size
    target_w, target_h = TARGET_RESOLUTION
    # Auto-detect from image aspect ratio for shorts (vertical)
    if img_w < img_h:
        target_w, target_h = 1080, 1920

    # Randomly choose a Ken Burns motion type
    motion = random.choice(["zoom_in", "zoom_out", "pan_left", "pan_right", "drift"])

    # Calculate zoom parameters
    zoom_start = 1.0
    zoom_end = random.uniform(*KB_ZOOM_RANGE)
    if motion == "zoom_out":
        zoom_start, zoom_end = zoom_end, zoom_start

    # Calculate pan parameters
    max_pan_x = int(img_w * KB_PAN_RANGE)
    max_pan_y = int(img_h * KB_PAN_RANGE)

    if motion == "pan_left":
        start_x, end_x = max_pan_x, -max_pan_x
        start_y, end_y = 0, 0
    elif motion == "pan_right":
        start_x, end_x = -max_pan_x, max_pan_x
        start_y, end_y = 0, 0
    elif motion == "drift":
        start_x = random.randint(-max_pan_x, max_pan_x)
        end_x = random.randint(-max_pan_x, max_pan_x)
        start_y = random.randint(-max_pan_y, max_pan_y)
        end_y = random.randint(-max_pan_y, max_pan_y)
    else:  # zoom_in or zoom_out
        start_x, end_x = 0, 0
        start_y, end_y = 0, 0
        # Add slight drift during zoom
        end_x = random.randint(-max_pan_x // 2, max_pan_x // 2)
        end_y = random.randint(-max_pan_y // 2, max_pan_y // 2)

    total_frames = int(duration * TARGET_FPS)
    img_array = np.array(img)

    # Pre-allocate frame writer via imageio
    import imageio
    writer = imageio.get_writer(
        str(out_path), fps=TARGET_FPS, codec="libx264",
        quality=8, macro_block_size=1,
        output_params=["-preset", "medium", "-pix_fmt", "yuv420p"],
    )

    for frame_idx in range(total_frames):
        t = frame_idx / max(total_frames - 1, 1)
        # Smooth easing (ease-in-out)
        t_smooth = 0.5 - 0.5 * math.cos(math.pi * t)

        # Interpolate zoom and pan
        zoom = zoom_start + (zoom_end - zoom_start) * t_smooth
        pan_x = start_x + (end_x - start_x) * t_smooth
        pan_y = start_y + (end_y - start_y) * t_smooth

        # Calculate crop region
        crop_w = int(target_w / zoom)
        crop_h = int(target_h / zoom)

        # Center point with pan offset
        cx = img_w // 2 + int(pan_x)
        cy = img_h // 2 + int(pan_y)

        # Clamp to image bounds
        x1 = max(0, min(cx - crop_w // 2, img_w - crop_w))
        y1 = max(0, min(cy - crop_h // 2, img_h - crop_h))
        x2 = x1 + crop_w
        y2 = y1 + crop_h

        # Crop and resize
        frame = img_array[y1:y2, x1:x2]
        frame_img = Image.fromarray(frame).resize(TARGET_RESOLUTION, Image.LANCZOS)
        writer.append_data(np.array(frame_img))

    writer.close()
    log.info(f"  Scene {scene_index}: Ken Burns clip ready ({motion}, {total_frames} frames)")
    return str(out_path)

# ---------------------------------------------------------------------------
# Ambient Audio Generation
# ---------------------------------------------------------------------------

def generate_ambient_drone(duration: float, work_dir: Path) -> str:
    log.info(f"Generating ambient drone ({duration:.1f}s)...")
    out_path = work_dir / "ambient_drone.wav"
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
    log.info(f"Ambient drone ready: {out_path}")
    return str(out_path)

# ---------------------------------------------------------------------------
# Video Assembly (moviepy v2)
# ---------------------------------------------------------------------------

def assemble_video(entry: Entry, work_dir: Path) -> str:
    log.info("Assembling final video...")
    final_clips = []

    for scene in entry.scenes:
        if not scene.video_path:
            log.warning(f"  Scene {scene.index}: No video, creating black clip")
            duration = scene.audio_duration if scene.audio_duration > 0 else scene.duration_hint
            clip = ColorClip(size=TARGET_RESOLUTION, color=(5, 5, 5), duration=duration)
        else:
            clip = VideoFileClip(scene.video_path)

        if scene.audio_path and scene.audio_duration > 0:
            audio = AudioFileClip(scene.audio_path)
            target_duration = scene.audio_duration + 1.5
            if clip.duration < target_duration:
                speed_factor = clip.duration / target_duration
                if speed_factor > 0.4:
                    clip = clip.with_effects([vfx.MultiplySpeed(speed_factor)])
                else:
                    clip = clip.with_effects([vfx.Loop(duration=target_duration)])
            actual_end = min(target_duration, clip.duration)
            clip = clip.subclipped(0, actual_end)
            clip = clip.with_audio(audio)
        else:
            actual_end = min(scene.duration_hint, clip.duration)
            clip = clip.subclipped(0, actual_end)

        clip = clip.with_effects([vfx.Resize(TARGET_RESOLUTION)])
        final_clips.append(clip)

    if not final_clips:
        raise RuntimeError("No clips to assemble")

    for i in range(1, len(final_clips)):
        final_clips[i] = final_clips[i].with_effects([vfx.CrossFadeIn(CROSSFADE_DURATION)])

    final_clips[0] = final_clips[0].with_effects([vfx.FadeIn(2.0)])
    final_clips[-1] = final_clips[-1].with_effects([vfx.FadeOut(3.0)])

    if len(final_clips) > 1:
        final = concatenate_videoclips(final_clips, method="compose", padding=-CROSSFADE_DURATION)
    else:
        final = final_clips[0]

    # Mix ambient drone
    total_duration = final.duration
    ambient_path = generate_ambient_drone(total_duration, work_dir)
    ambient_audio = AudioFileClip(ambient_path)
    vol = float(os.environ.get("AMBIENT_VOLUME", "0.20"))
    ambient_audio = ambient_audio.with_volume_scaled(vol)

    if final.audio is not None:
        final = final.with_audio(CompositeAudioClip([final.audio, ambient_audio]))
    else:
        final = final.with_audio(ambient_audio)

    log.info(f"Ambient drone mixed at {vol:.0%} volume")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = "".join(c if c.isalnum() or c in "-_ " else "" for c in entry.title)
    safe_title = safe_title.strip().replace(" ", "_")[:50]
    output_path = Path(OUTPUT_DIR) / f"deadlight_{safe_title}_{timestamp}.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"Rendering to {output_path}...")
    final.write_videofile(
        str(output_path), fps=TARGET_FPS, codec="libx264",
        audio_codec="aac", audio_bitrate="320k", preset="slow",
        threads=4, logger=None,
    )

    for clip in final_clips:
        clip.close()
    ambient_audio.close()
    final.close()

    entry.output_path = str(output_path)
    log.info(f"✓ Final video: {output_path}")
    return str(output_path)

# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

async def generate_entry(prompt: str) -> str:
    log.info("=" * 60)
    log.info("DeadlightCodex — Entry Generation")
    log.info("=" * 60)

    if not ELEVENLABS_API_KEY:
        raise RuntimeError("ELEVENLABS_API_KEY not set")
    if not OPENAI_API_KEY and not ANTHROPIC_API_KEY:
        raise RuntimeError("Set OPENAI_API_KEY or ANTHROPIC_API_KEY")

    work_dir = Path(tempfile.mkdtemp(prefix="deadlight_"))
    log.info(f"Working directory: {work_dir}")

    # Step 1: Direct scenes
    entry = await direct_scenes(prompt)

    plan_path = work_dir / "scene_plan.json"
    plan_path.write_text(json.dumps({
        "title": entry.title, "subject": entry.subject,
        "scenes": [{"index": s.index, "narration": s.narration, "image_prompt": s.image_prompt, "duration_hint": s.duration_hint} for s in entry.scenes],
    }, indent=2))
    log.info(f"Scene plan saved: {plan_path}")

    # Step 2: Generate narration (async, throttled)
    log.info("Generating narration audio...")
    sem = asyncio.Semaphore(2)
    await asyncio.gather(*[generate_narration(s, work_dir, sem) for s in entry.scenes])

    # Step 3: Generate images (threaded, free FLUX)
    generate_images_threaded(entry.scenes, work_dir)

    # Step 4: Apply Ken Burns animation to each image
    log.info("Applying Ken Burns animation...")
    for scene in entry.scenes:
        if scene.image_path:
            duration = scene.audio_duration + 1.5 if scene.audio_duration > 0 else scene.duration_hint
            scene.video_path = apply_ken_burns(scene.image_path, duration, work_dir, scene.index)

    # Step 5: Assemble final video
    output = assemble_video(entry, work_dir)

    plan_out = Path(OUTPUT_DIR) / f"{Path(output).stem}_plan.json"
    shutil.copy2(plan_path, plan_out)
    log.info(f"Scene plan copied: {plan_out}")

    log.info("=" * 60)
    log.info(f"DONE — {output}")
    log.info("=" * 60)
    return output

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DeadlightCodex — Automated cosmic horror video generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("prompt", nargs="?", help="The entry subject/prompt")
    parser.add_argument("--from-file", "-f", help="Read prompt from a text file")
    parser.add_argument("--output", "-o", default="./output", help="Output directory")
    parser.add_argument("--voice-id", help="Override ElevenLabs voice ID")
    parser.add_argument("--speed", type=float, help="Override voice speed (0.5-1.0)")
    parser.add_argument("--dry-run", action="store_true", help="Scene plan only")

    args = parser.parse_args()

    if args.from_file:
        prompt = Path(args.from_file).read_text().strip()
    elif args.prompt:
        prompt = args.prompt
    else:
        parser.error("Provide a prompt or use --from-file")

    global OUTPUT_DIR, ELEVENLABS_VOICE_ID, VOICE_SPEED
    OUTPUT_DIR = args.output
    if args.voice_id:
        ELEVENLABS_VOICE_ID = args.voice_id
    if args.speed:
        VOICE_SPEED = args.speed

    if args.dry_run:
        log.info("DRY RUN — generating scene plan only")
        entry = asyncio.run(direct_scenes(prompt))
        print(json.dumps({
            "title": entry.title, "subject": entry.subject,
            "scenes": [{"index": s.index, "narration": s.narration, "image_prompt": s.image_prompt} for s in entry.scenes],
        }, indent=2))
        return

    output = asyncio.run(generate_entry(prompt))
    print(f"\n✓ Output: {output}")

if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# YouTube Shorts Generator
# ---------------------------------------------------------------------------

SHORTS_SYSTEM_PROMPT = """You are creating a YouTube Short derived from a longer video script.

Your task:
Select the most engaging, curiosity-driven, or thought-provoking segment from the script and convert it into a short-form video narration.

Rules:

- Length: 20-40 seconds (VERY IMPORTANT)
- Must start with a strong hook in the first 2 seconds
- Must NOT require prior context
- Must create curiosity, tension, or insight
- Must feel complete but leave the viewer wanting more
- Do NOT summarize — extract a compelling moment

Structure:

1. Hook (first line must grab attention immediately)
2. Core moment or idea
3. Subtle unresolved or impactful ending

Tone:
Match the original channel tone exactly.

Output ONLY valid JSON, no markdown fences:
{
  "narration": "The short narration script (20-40 seconds when read slowly)",
  "image_prompt": "A single image prompt for the short's visual, matching the channel's visual style",
  "suggested_title": "YouTube Short title",
  "suggested_caption": "YouTube Short description/caption",
  "suggested_hashtags": ["#tag1", "#tag2", "#tag3", "#tag4", "#tag5"]
}"""

async def generate_short(entry: Entry, work_dir: Path) -> dict:
    """Generate a YouTube Short from the main video's script."""
    log.info("Generating YouTube Short...")

    # Compile full script for context
    full_script = "\n\n".join([f"Scene {s.index}: {s.narration}" for s in entry.scenes])

    messages = [
        {"role": "system", "content": SHORTS_SYSTEM_PROMPT},
        {"role": "user", "content": f"Channel: {entry.title}\n\nFull script:\n{full_script}\n\nCreate a YouTube Short from the most compelling moment."},
    ]

    if ANTHROPIC_API_KEY:
        result = await _call_anthropic(messages)
    elif OPENAI_API_KEY:
        result = await _call_openai(messages)
    else:
        raise RuntimeError("No LLM API key for shorts generation")

    result = result.strip()
    if result.startswith("```"):
        result = result.split("\n", 1)[1]
    if result.endswith("```"):
        result = result.rsplit("```", 1)[0]

    short_data = json.loads(result.strip())
    log.info(f"  Short title: {short_data['suggested_title']}")

    # Generate narration
    short_audio_path = work_dir / "short_narration.mp3"
    async with httpx.AsyncClient(timeout=180) as client:
        payload = {"text": short_data["narration"], "model_id": ELEVENLABS_MODEL_ID, "voice_settings": VOICE_SETTINGS}
        if VOICE_SPEED != 1.0:
            payload["speed"] = VOICE_SPEED
        resp = await client.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}",
            headers={"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json", "Accept": "audio/mpeg"},
            json=payload,
        )
        resp.raise_for_status()
        short_audio_path.write_bytes(resp.content)

    audio_clip = AudioFileClip(str(short_audio_path))
    short_duration = audio_clip.duration + 1.5
    log.info(f"  Short duration: {short_duration:.1f}s")

    # Generate image
    try:
        from gradio_client import Client as GradioClient
        gc = GradioClient(FLUX_SPACE, token=HF_TOKEN if HF_TOKEN else None)
        img_result = gc.predict(
            prompt=short_data["image_prompt"], seed=0, randomize_seed=True,
            width=1080, height=1920,  # 9:16 vertical for Shorts
            guidance_scale=FLUX_GUIDANCE, num_inference_steps=FLUX_STEPS, api_name="/infer",
        )
        src = img_result[0]["path"] if isinstance(img_result[0], dict) else str(img_result[0])
        short_img_path = work_dir / "short_image.png"
        Image.open(src).save(str(short_img_path), "PNG")
    except Exception as e:
        log.warning(f"  Short image generation failed, using fallback: {e}")
        short_img_path = work_dir / "short_image.png"
        # Fallback: dark procedural image in 9:16
        img = np.zeros((1920, 1080, 3), dtype=np.uint8)
        img[:] = (8, 4, 4)
        noise = np.random.randint(0, 10, (1920, 1080, 3), dtype=np.uint8)
        img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
        Image.fromarray(img).save(str(short_img_path), "PNG")

    # Apply Ken Burns (9:16 vertical)
    short_video_path = apply_ken_burns(
        str(short_img_path), short_duration, work_dir, scene_index=99,
    )

    # Assemble short with audio + ambient
    video = VideoFileClip(short_video_path)
    video = video.subclipped(0, min(short_duration, video.duration))
    video = video.with_audio(audio_clip)

    # Add ambient drone
    ambient_path = generate_ambient_drone(short_duration, work_dir)
    ambient = AudioFileClip(ambient_path)
    vol = float(os.environ.get("AMBIENT_VOLUME", "0.20"))
    ambient = ambient.with_volume_scaled(vol)
    video = video.with_audio(CompositeAudioClip([video.audio, ambient]))

    # Fade in/out
    video = video.with_effects([vfx.FadeIn(1.0), vfx.FadeOut(1.5)])

    # Render
    safe_title = "".join(c if c.isalnum() or c in "-_ " else "" for c in short_data["suggested_title"])[:40]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_output = Path(OUTPUT_DIR) / f"short_{safe_title}_{timestamp}.mp4"
    video.write_videofile(
        str(short_output), fps=TARGET_FPS, codec="libx264",
        audio_codec="aac", audio_bitrate="320k", preset="slow",
        threads=4, logger=None,
    )
    video.close()
    audio_clip.close()
    ambient.close()

    short_data["output_path"] = str(short_output)
    short_data["duration"] = short_duration
    log.info(f"  ✓ Short ready: {short_output}")

    # Save metadata
    meta_path = Path(OUTPUT_DIR) / f"short_{safe_title}_{timestamp}_meta.json"
    meta_path.write_text(json.dumps(short_data, indent=2))

    return short_data
