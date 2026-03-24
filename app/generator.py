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
            "Real-world locations, objects, or events that defy explanation",
            "Archaeological anomalies — things found where they shouldn't be, in layers too old",
            "Engineering impossibilities — ancient structures that modern tools can't replicate",
            "Scientific measurements that don't add up — readings, signals, data that breaks models",
            "Forgotten or suppressed discoveries that were documented then buried",
            "Natural phenomena that science acknowledges but cannot fully explain",
            "Historical records that contradict the official timeline",
            "Objects or materials with properties that shouldn't exist",
            "Places where instruments behave erratically — compasses, clocks, radios",
            "Expeditions or research projects that were abruptly abandoned without explanation",
            "Medical and biological anomalies — documented cases of impossible survival, people who don't feel pain, identical strangers, foreign accent syndrome",
            "Audio and sensory anomalies — sounds with no source (The Hum, Bloop), frequencies that affect the body, places where everyone reports the same phantom smell",
            "Mathematical and pattern anomalies — sequences that repeat where they shouldn't, data that forms impossible patterns, numerical coincidences too precise to be chance",
            "Behavioral anomalies at scale — events where large groups of people or animals did the same unexplained thing simultaneously",
            "Cartographic weirdness — borders drawn around nothing, features on old maps that shouldn't exist, geographic formations with no geological explanation",
            "Documented temporal anomalies — anachronistic objects found in sealed sites, events recorded out of sequence, cases investigated by credible sources",
        ],
        "avoid": [
            "Supernatural claims without grounded realism",
            "Conspiracy theories (that's EchelonVeil's territory)",
            "Clickbait speculation — present facts and let the mystery speak",
            "Generic 'scientists can't explain this' framing without specific detail",
        ],
        "examples": [
            "The Tunnel That Appeared Overnight Beneath a School",
            "The Case of the Missing Floor in a Government Building",
            "The Stone Block That Weighs More Every Time It's Measured",
            "The Map Drawn 300 Years Before the Continent Was Discovered",
            "The Frequency That Only Plays on the Third Thursday of Each Month",
            "The Lake That Drains Completely Every 7 Years — and Nobody Knows Where the Water Goes",
            "The Drill Core Sample That Contained Something That Shouldn't Be There",
            "The Building That Appears in Photographs From Before It Was Built",
            "The Woman Who Woke Up Speaking a Language She'd Never Learned",
            "The Hum: Thousands of People Hear It. No One Can Find the Source.",
            "The Identical Strangers Who Met by Accident — and Shared the Same Life",
            "The Flock That Fell from the Sky: 5,000 Birds, One Moment, No Explanation",
            "The Number Station That's Been Broadcasting Since 1982. Nobody Claims It.",
            "The Border Drawn Around Nothing: Why This Line Exists on Every Map",
            "The Patient Who Survived What Doctors Say Is Impossible — Twice",
            "The Data Set That Contains a Pattern No One Programmed",
        ],
    },
    "gray_meridian": {
        "focus": [
            "Human behavior patterns and why people do what they do",
            "Cognitive biases and mental shortcuts",
            "Subtle psychological truths most people never notice",
            "Social dynamics, persuasion, and influence",
            "The gap between what people say and what they actually do",
            "Everyday decisions that reveal deeper psychology",
        ],
        "avoid": [
            "Generic self-help or motivational content",
            "Clinical/academic tone — keep it conversational",
            "Pop psychology clichés",
        ],
        "examples": [
            "Why People Defend Ideas They Know Are Wrong",
            "The Reason You Can't Remember What You Walked Into the Room For",
            "Why Strangers Trust You Less When You're Too Honest",
            "The Psychology of Why We Root for the Underdog",
            "Why You Always Pick the Slowest Line at the Store",
        ],
    },
    "autonomous_stack": {
        "focus": [
            "Automation workflows that save real hours — not theory, actual implementations",
            "AI tools and how to chain them together for practical outcomes",
            "The hidden inefficiencies in common workflows that most people don't notice",
            "Building systems that monitor, repair, or improve themselves",
            "Integrating AI into existing business processes without rebuilding everything",
            "The art of elimination — finding what to stop doing, not just what to automate",
            "Real cost-benefit analysis of automation — when it's worth it and when it's not",
            "Infrastructure as code — managing servers, deployments, environments automatically",
            "Scraping, parsing, and transforming data from messy real-world sources",
            "Building personal dashboards and monitoring systems that actually get used",
            "The 80/20 of developer tooling — the few tools that handle most of the work",
            "Automation failures and what they teach — when systems break in interesting ways",
        ],
        "avoid": [
            "Basic beginner coding tutorials",
            "Generic 'top 10 AI tools' listicle content",
            "Hype-driven AI coverage without practical application",
            "Content that's outdated within a month — focus on principles over specific tool versions",
        ],
        "examples": [
            "The One Automation Layer Everyone Forgets",
            "I Replaced 4 Hours of Daily Work with a 50-Line Script",
            "Why Your CI/CD Pipeline Is Slower Than It Needs to Be",
            "Building a Self-Healing Server Stack for Under $20/Month",
            "The Automation That Broke Everything — and What I Learned",
            "How to Make AI Actually Useful in Your Workflow (Not Just Cool)",
            "The Monitoring Dashboard That Paid for Itself in a Week",
            "Stop Automating the Wrong Things",
        ],
    },
    "the_unwritten_wing": {
        "focus": [
            "Stories told from unusual perspectives — objects, places, emotions as narrators",
            "Characters who discover they're not who they thought they were",
            "Relationships that exist across time, memory, or altered reality",
            "The weight of choices not taken — parallel lives, alternate timelines experienced emotionally",
            "Letters, recordings, or messages found that reframe everything",
            "Small, quiet moments that contain entire lifetimes of meaning",
            "Stories where the real twist is emotional, not plot-based",
            "The spaces between people — things unsaid, distances that grow or shrink",
            "Characters meeting versions of themselves — past, future, imagined",
            "Stories rooted in a single powerful image or metaphor that unfolds slowly",
            "Grief and loss told indirectly — through objects, habits, or spaces left behind rather than stating it outright",
            "Quiet existential moments — the instant someone realizes something fundamental about their life has shifted",
            "Stories told through unusual structures — voicemails, to-do lists, margin notes, receipts, missed connections posts",
            "Intergenerational stories — things passed between people across decades: recipes, lies, houses, names, debts",
            "Place as character — stories narrated by or centered on a specific location and everything it has witnessed",
            "Warmth and gentle absurdity — not everything is melancholy; tenderness, humor, and the comedy of being human",
            "Era-specific stories that transport — a specific time and place rendered with sensory precision",
        ],
        "avoid": [
            "High-action or plot-heavy ideas",
            "Standard romance or love story structure",
            "Stories that rely on a twist ending — the journey is the point",
            "Abstract concepts without a human anchor — always ground it in someone's experience",
            "Repeating the same 'woman remembers a different life' premise",
        ],
        "examples": [
            "The Day She Remembered a Life That Wasn't Hers",
            "A Letter Written to Someone Who Doesn't Exist Yet",
            "The Room That Remembers Everyone Who Ever Slept In It",
            "He Spent 30 Years Building Something He'd Never Finish",
            "The Photograph That Shows a Moment That Never Happened",
            "She Left a Voicemail Every Day for a Year After He Died",
            "The Story Told by the Chair in the Waiting Room",
            "Two Strangers Who Keep Almost Meeting",
            "The Last Voicemail He Never Deleted",
            "A To-Do List Found in a Coat Pocket at Goodwill",
            "The Diner Booth That Witnessed Forty Years of First Dates",
            "She Taught Her Daughter to Make the Soup. She Never Told Her Why.",
            "The Man Who Wrote Letters to His Future Self — and One Day Got a Reply",
            "A Lighthouse Keeper's Log, Written to No One",
            "The Payphone Call That Lasted Three Hours in 1994",
            "He Kept Watering Her Garden for a Year After She Left",
        ],
    },
    "remnants_project": {
        "focus": [
            "Systems and machines that continue operating without human oversight",
            "The last signals, broadcasts, or transmissions from abandoned places",
            "Animals adapting to and repurposing human-built environments",
            "Automated processes that outlive their creators — satellites, servers, dams",
            "What specific objects experience after humans leave — a clock, a traffic light, a vending machine",
            "The sociology of abandoned places — why some decay and others are preserved",
            "Time capsules, buried archives, and messages left for no one",
            "How different materials and technologies age at different rates",
            "The eerie beauty of liminal spaces — empty malls, silent highways, dark stadiums",
            "What the last day looks like in a specific place before it's abandoned forever",
            "Digital remnants — websites that outlive their creators, social media profiles of the dead, abandoned servers still running, automated emails sent to no one",
            "Cultural extinction — languages with one speaker left, traditions nobody practices, recipes that die with a single person",
            "Personal objects left behind — suitcases never claimed, letters never opened, wedding dresses in attics, shoes on power lines",
            "Space remnants — Voyager's golden record, flags on the Moon, rovers on Mars, orbital debris circling a planet that may forget them",
            "Submerged and underwater remains — flooded cities, shipwrecks with preserved cargo, forests swallowed by reservoirs",
            "Sound and music after the end — the last record pressed, instruments in abandoned buildings, silence replacing noise",
        ],
        "avoid": [
            "Action or survival narratives",
            "Generic 'nature reclaims civilization' without a specific angle",
            "Just describing buildings crumbling and plants growing — that's the backdrop, not the story",
            "Repeating the same infrastructure-decay premise with different locations",
        ],
        "examples": [
            "The Satellite That Will Outlast Everything We've Ever Built",
            "What a Traffic Light Does After the Last Car Drives Away",
            "The Vending Machine at the Edge of the Exclusion Zone",
            "Why Elevators Are the First Thing to Die in an Abandoned Building",
            "The Last Radio Station Still Broadcasting to No One",
            "What Happens Inside a Server Room When the Power Finally Goes Out",
            "The Animals That Moved Into Chernobyl's Swimming Pool",
            "A Hotel Room That Hasn't Been Opened in 40 Years",
            "The Geocities Page That's Been Online Since 1998 — and Nobody Knows Who Made It",
            "The Language That Dies When She Does",
            "The Wedding Dress in the Attic of a House Nobody Lives In",
            "Voyager Is Still Transmitting. Nobody Is Listening.",
            "The Town They Flooded to Build the Dam — You Can Still See the Steeple",
            "An Automated Birthday Email Sent Every Year to Someone Who Died in 2014",
            "The Piano in the Flood-Damaged House That Still Holds Its Tune",
            "The Last Blockbuster Closed. The Rewound Tapes Are Still Inside.",
        ],
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
            "Grounded creepypasta-style horror storytelling",
            "Real-feeling locations, people, and events that turn sinister",
            "Slow-burn dread — things that feel wrong before you know why",
            "Urban legends, cursed places, found footage style narratives",
            "Horror rooted in the familiar — houses, roads, small towns, jobs",
            "Objects with wrong histories — things that shouldn't exist, or exist differently than recorded",
            "Personal horror — journals, voicemails, security footage that reveals something about the narrator",
            "Temporal wrongness — events that repeat, time that doesn't pass correctly, days that don't end",
            "Horror of documentation — official records, forms, databases that contain impossible entries",
            "The uncanny mundane — routines, neighbors, coworkers that are subtly and inexplicably off",
            "Places that are wrong on the inside — dimensions don't match, rooms that shouldn't fit, spaces that change",
            "Horror from the perspective of someone who doesn't realize they're in danger yet",
            "Real-life horror accounts — fictional stories told as first-person experiences that feel genuinely lived-in",
            "Encounters in remote or isolated places — forests, highways, abandoned buildings, night shifts",
            "Stories from everyday jobs that turned sinister — night security, long-haul trucking, rural EMT, park ranger",
            "Home intrusion and stalker horror — the terror of realizing someone has been watching, entering, or living nearby",
            "Camping, hiking, and wilderness horror — things seen, heard, or found in the deep woods",
            "Small town secrets — communities where everyone knows something they won't talk about",
            "Childhood memories that make sense only in hindsight — things you witnessed as a kid that were actually horrifying",
            "Online encounters that crossed into real life — messages, videos, or strangers from the internet that became real threats",
        ],
        "avoid": [
            "Abstract cosmic entities without grounding",
            "Standard horror tropes or jumpscares",
            "Gore or shock value",
            "Overly supernatural without a real-world anchor",
            "Repeating the same '[location] that [spooky adjective]' formula",
            "Generic haunted house or ghost stories without a unique angle",
        ],
        "examples": [
            "The House That Grew a Room Nobody Built",
            "I Found My Neighbor's Journal. He Was Writing About Me.",
            "The Road That Doesn't Show Up on Any Map",
            "There's a Reason Nobody Fishes at Mirror Lake Anymore",
            "The Voicemail Was from My Own Number. I Never Made the Call.",
            "Every Photo from That Night Has an Extra Person in It",
            "The Employee Database Lists Someone Who Doesn't Work Here",
            "I Checked My Security Camera. I Never Came Home Last Night.",
            "The Town's Records Show a Street That No One Can Find",
            "My Dentist Said My X-Rays Show Teeth That Aren't Mine",
            "I Worked the Night Shift at a Gas Station in the Middle of Nowhere. I Quit After the Third Night.",
            "The Campsite Was Perfect. Then We Found the Other Tent.",
            "My Uber Driver Took a Route That Doesn't Exist",
            "I Found Scratches on the Inside of My Closet Door. I Live Alone.",
            "The Babysitter Called to Check on the Kids. We Don't Have a Babysitter.",
            "A Park Ranger's Account of the Trail That Hikers Don't Come Back From",
            "The Town I Grew Up In Has a Rule: Never Go Outside After the Sirens",
            "Someone Has Been Leaving Notes Under My Windshield Wiper. They Know My Schedule.",
        ],
    },
    "softlight_kingdom": {
        "focus": [
            "Classic children's fantasy: princesses, kings, queens, castles",
            "Talking animal friends on gentle adventures",
            "Magical creatures: friendly dragons, unicorns, fairies, wise owls",
            "Enchanted forests, hidden kingdoms, cozy villages",
            "Simple lessons: kindness, sharing, bravery, being yourself",
            "Bedtime-appropriate stories that wind down toward sleep",
        ],
        "avoid": [
            "Anything scary, dark, or threatening",
            "Complex plots or twists",
            "Villains or real danger",
            "Adult themes or abstract concepts",
            "Modern technology or real-world settings",
        ],
        "examples": [
            "The Little Bear Who Learned to Share the Stars",
            "Princess Rosepetal and the Sleepy Dragon",
            "The Bunny Who Found a Secret Garden",
            "The Owl King's Moonlight Festival",
            "The Kitten and the Friendly Cloud",
        ],
    },
    "echelon_veil": {
        "focus": [
            "Real-world cryptid sightings and monster encounters — told as investigation files, not folklore",
            "UFO/UAP encounters that were officially documented, investigated, then quietly shelved",
            "Government programs investigating the paranormal — remote viewing, psychic warfare, dimensional research",
            "Missing persons cases with inexplicable details — time gaps, impossible locations, witnesses who saw something",
            "Places where multiple unrelated people report the same impossible thing",
            "Men in Black-style encounters — strangers who show up after sightings and tell people to stop talking",
            "Military and pilot encounters with things that defy known physics",
            "Small town monster sightings that were taken seriously by law enforcement",
            "Cattle mutilations, crop formations, and physical evidence cases that were never explained",
            "People who came back from disappearances with memories that don't match reality",
            "Shadow government and black budget programs that fund research into things that 'don't exist'",
            "The thin line between conspiracy and confirmed — things once called crazy that turned out to be real",
        ],
        "avoid": [
            "Ancient aliens / History Channel speculation without investigation framing",
            "Pure science fiction without a 'this actually happened' anchor",
            "QAnon or politically partisan conspiracy theories",
            "Flat earth, anti-vax, or debunked fringe theories",
            "Debunking tone — maintain mystery, present it like an open case file",
            "Pure ZeroTrace overlap — EchelonVeil always has a conspiracy/cover-up angle, not just 'unexplained'",
        ],
        "examples": [
            "The Pilot Who Chased Something Over Lake Michigan — Then Was Told It Never Happened",
            "Skinwalker Ranch: The Government Bought It. Then Classified Everything.",
            "The Town That Reported the Same Creature for 30 Years — and Has the Photos to Prove It",
            "Three Hikers Went Missing in the Same Forest. All Three Came Back — With the Same Story.",
            "The Military Base That Monitors Something Underground — and Won't Say What",
            "The Man Who Walked Into an FBI Office with Proof of Something. He Was Never Seen Again.",
            "They Told the Fishermen It Was a Submarine. The Fishermen Disagree.",
            "The Classified File on Mothman That Nobody Was Supposed to Find",
            "Point Pleasant, 1967: The Bridge, the Moth, and the Men Who Came After",
            "The Rancher Found His Cattle Surgically Mutilated. The Sheriff's Report Was Seized the Same Day.",
        ],
    },
    "loreletics": {
        "focus": [
            "Legendary sports moments told as mythology — ALL sports are fair game",
            "Underdog stories and impossible comebacks",
            "The human drama behind iconic sports events",
            "Cinematic, emotional sports storytelling",
            "Athletes who defied fate, physics, or expectation",
            "Cover the full spectrum: football, basketball, baseball, soccer, hockey, boxing, MMA, tennis, golf, Olympic sports, motorsport, cricket, rugby, swimming, track & field, and any other sport with a great story",
            "Obscure and lesser-known sports stories that most people have never heard — the best stories aren't always the famous ones",
            "Tragedy, darkness, and downfall in sports — careers destroyed, corruption, doping scandals, athletes who lost everything",
            "The full human story — where athletes came from, what they sacrificed, and what happened after the spotlight faded",
            "Great rivalries told as mythology — not just one moment, but the entire arc of two forces colliding",
            "Cursed franchises, blown leads, and 'what if' moments that haunt cities and fanbases for generations",
            "Women's sports stories — massively underrepresented and full of incredible untold narratives",
            "Obscure and unusual sports — ultra-marathons, kabaddi, sumo, chess boxing, competitive events most viewers don't know exist",
            "The psychology of pressure — what happens inside an athlete's mind at the moment everything depends on them",
        ],
        "avoid": [
            "Stats-heavy analysis or play-by-play breakdowns",
            "Current sports news or commentary",
            "Generic highlight reels without narrative depth",
            "Repeating the same sport too often — rotate across different sports",
        ],
        "examples": [
            "The Night the Ice Cracked: Miracle on Ice Retold",
            "The Barefoot God of Rome: Abebe Bikila's Marathon",
            "The Ball That Bent Physics: Roberto Carlos Free Kick",
            "The Rumble in the Jungle: When Ali Rewrote Boxing",
            "The Catch That Haunted a City: Steve Bartman and the Cubs",
            "The Impossible 28-3: Super Bowl LI's Greatest Comeback",
            "The High School Team That Won State with Four Players",
            "She Wasn't Allowed to Run. She Ran Anyway. Boston, 1967.",
            "The Boxer Who Fought Six Rounds With a Broken Hand — and Won",
            "The Curse of Cleveland: 52 Years of Almost",
            "The Goalkeeper Who Scored an Own Goal and Never Played Again",
            "Andrés Escobar Scored an Own Goal at the World Cup. Ten Days Later, He Was Dead.",
            "The Gymnast Who Landed on a Broken Ankle — and Smiled",
            "The Ultra-Marathon Nobody Was Supposed to Survive",
            "They Banned Her From Competing. She Broke the Record Anyway.",
            "The Chess Match That Lasted 20 Hours and Nearly Killed Both Players",
        ],
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
KB_ZOOM_RANGE = (1.06, 1.18)
KB_PAN_RANGE = 0.10

# Per-channel Ken Burns overrides.  Channels not listed use the globals above.
# upscale_size controls how large the source image is scaled before cropping —
# smaller values mean more of the original image is visible in each frame.
CHANNEL_KB_OVERRIDES = {
    "softlight_kingdom": {
        "zoom_range": (1.03, 1.08),
        "pan_range": 0.05,
        "upscale_size": (2304, 1312),  # ~83% visible (vs default ~71%)
    },
    "somnus_protocol": {
        "zoom_range": (1.02, 1.06),
        "pan_range": 0.03,
        "upscale_size": (2304, 1312),
    },
}
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
                meta["has_thumbnail"] = (d / "thumbnail.png").exists()
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
Use ellipses (...) liberally to indicate pauses WITHIN sentences. Add "..." between phrases to signal the narrator to slow down and drift.
Each sentence should stand alone, separated by natural silence.
The narrator is drifting off too. Each sentence should feel like it takes effort to say.
Example style: "The sky... is very still tonight... Nothing moves... nothing needs to... Just the quiet... and the dark... settling in around you..."
"""

    min_words = vs.get('target_word_count_min', 1100)
    max_words = vs.get('target_word_count_max', 1300)
    min_scenes = vs.get('scene_count_min', 12)
    max_scenes = vs.get('scene_count_max', 16)
    target_per_scene = max(min_words // min_scenes, 80)

    # Anti-repetition rule (skip for sleep/meditation channels where repetition is intentional)
    repetition_rule = ""
    if c.get("channel_id") != "somnus_protocol":
        repetition_rule = """
CRITICAL QUALITY RULE — NO REPETITION:
- Do NOT repeat the same idea, sentence, or phrase across scenes to pad word count
- Each scene must introduce NEW information, a new angle, a new detail, or advance the narrative
- If two scenes say essentially the same thing in different words, you have FAILED
- Reach the word count through DEPTH (more detail, more atmosphere, more world-building) not through REDUNDANCY
- Every sentence should earn its place — if removing it loses nothing, it shouldn't be there"""

    return f"""You are the creative director for {c['channel_name']}.
{c.get('description', '')}
{pacing_note}

*** CRITICAL LENGTH REQUIREMENT — READ THIS FIRST ***
This video MUST be 8+ minutes for YouTube mid-roll ad eligibility. This is a hard business requirement.
- Write EXACTLY {min_scenes} to {max_scenes} scenes (no fewer than {min_scenes})
- Each scene MUST be 80-100 words of narration (5-8 sentences). Count your words.
- Total word count MUST be {min_words}-{max_words} words across all scenes
- Target: {min_scenes} scenes × {target_per_scene} words = {min_scenes * target_per_scene} words minimum
- If you write fewer than {min_words} total words, the output will be rejected and regenerated at additional cost
- duration_hint = 15 for every scene
- WRITE LONG. Err heavily toward {max_words} words, not {min_words}
{repetition_rule}

EXAMPLE of a properly-sized scene narration (85 words):
"The structure was first documented in the spring of 1987, though local accounts suggest it had been present for far longer. Its surface was smooth, almost polished, and yet no tool marks could be identified under magnification. Researchers noted that photographs of the object consistently failed to capture its true dimensions. Measurements taken on different days produced different results, sometimes by as much as several centimeters. The surrounding soil showed no signs of excavation or placement. It was simply there, as though it had always been."

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
      "narration": "Narration text for this scene (80-100 words, 5-8 sentences)",
      "image_prompt": "Detailed image generation prompt",
      "duration_hint": 15
    }}
  ]
}}

FINAL REMINDER: You MUST write {min_scenes}+ scenes with {min_words}+ total words. Count them before responding."""


def generate_script(channel, topic, api_key):
    system = _build_director_prompt(channel)
    vs = channel.get("video_settings", {})
    min_words = vs.get("target_word_count_min", 1100)
    max_words = vs.get("target_word_count_max", 1400)
    min_scenes = vs.get("scene_count_min", 12)

    # Use higher temperature on first call to encourage longer output
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Create an entry about:\n\n{topic}\n\nRemember: {min_scenes}+ scenes, {min_words}+ total words, 80-100 words per scene. Count before responding."},
    ]

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
        log.info(f"Script accepted: {total_words} words, {scene_count} scenes")
        return script

    # If short, try to expand in-place rather than full regeneration
    log.warning(f"Script short: {total_words} words, {scene_count} scenes. Expanding...")

    # Calculate how many more words/scenes we need
    words_needed = min_words - total_words
    scenes_needed = max(0, min_scenes - scene_count)

    expand_prompt = f"""The script below has {total_words} words across {scene_count} scenes.
I need AT LEAST {min_words} words across at least {min_scenes} scenes.

{"Add " + str(scenes_needed) + " more scenes to reach " + str(min_scenes) + " total." if scenes_needed > 0 else ""}
{"Expand existing scenes — each needs 80-100 words (5-8 sentences). Add atmospheric detail, sensory description, and deliberate pacing." if words_needed > 200 else "Slightly expand the shorter scenes to reach the word count target."}

IMPORTANT: Return the COMPLETE updated script as a full JSON object with ALL scenes (existing + new), not just the additions.
Same JSON format as before. No markdown fences."""

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Create an entry about:\n\n{topic}"},
        {"role": "assistant", "content": result},
        {"role": "user", "content": expand_prompt},
    ]

    result2 = _call_openai_sync(messages, api_key)
    result2 = result2.strip()
    if result2.startswith("```"):
        result2 = result2.split("\n", 1)[1]
    if result2.endswith("```"):
        result2 = result2.rsplit("```", 1)[0]

    try:
        script2 = json.loads(result2.strip())
        total_words2 = sum(len(s.get("narration", "").split()) for s in script2.get("scenes", []))
        scene_count2 = len(script2.get("scenes", []))
        log.info(f"Expanded script: {total_words2} words, {scene_count2} scenes (was {total_words}w/{scene_count}s)")

        # Use expanded version if it's actually better
        if total_words2 > total_words:
            return script2
    except (json.JSONDecodeError, Exception) as e:
        log.warning(f"Expansion parse failed: {e}")

    # Fall back to original
    log.warning(f"Using original script: {total_words} words, {scene_count} scenes")
    return script


# --- Audio generation ---

def _generate_narration_sync(text, voice_id, model_id, voice_settings, speed, api_key, out_path,
                              sentence_pause=0):
    """Generate narration audio via ElevenLabs TTS.
    
    If sentence_pause > 0, splits text into individual sentences,
    generates each separately, and concatenates with silence between them.
    This creates deliberate pauses between sentences for sleep/meditation content.
    """
    if sentence_pause > 0:
        return _generate_narration_with_pauses(
            text, voice_id, model_id, voice_settings, speed, api_key, out_path, sentence_pause
        )
    
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
            log.error(f"ElevenLabs 401: {resp.text[:300]}")
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


def _split_into_sentences(text):
    """Split narration text into individual sentences, preserving ellipses as pause markers."""
    import re
    # Split on sentence-ending punctuation followed by space or end
    # Keep ellipses as part of the sentence they're in
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z"])', text)
    # Filter out empty strings
    return [s.strip() for s in parts if s.strip()]


def _generate_narration_with_pauses(text, voice_id, model_id, voice_settings, speed, api_key, out_path, pause_seconds):
    """Generate narration sentence-by-sentence with silence gaps between them."""
    import subprocess
    import httpx as hx
    
    sentences = _split_into_sentences(text)
    if len(sentences) <= 1:
        # Single sentence or couldn't split — fall back to normal generation
        return _generate_narration_sync(text, voice_id, model_id, voice_settings, speed, api_key, out_path, sentence_pause=0)
    
    log.info(f"Generating narration with {pause_seconds}s pauses between {len(sentences)} sentences")
    
    work_dir = Path(out_path).parent
    sentence_files = []
    
    for idx, sentence in enumerate(sentences):
        sentence_path = work_dir / f"_sentence_{Path(out_path).stem}_{idx:03d}.mp3"
        payload = {"text": sentence, "model_id": model_id, "voice_settings": voice_settings}
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
                log.warning(f"ElevenLabs 429 (sentence {idx+1}), retrying in {wait}s")
                time.sleep(wait)
                continue
            if resp.status_code == 401:
                if attempt < 1:
                    time.sleep(3)
                    continue
                raise RuntimeError(f"ElevenLabs auth failed: {resp.text[:200]}")
            resp.raise_for_status()
            sentence_path.write_bytes(resp.content)
            break
        else:
            raise RuntimeError(f"ElevenLabs rate limit exceeded for sentence {idx+1}")
        
        sentence_files.append(sentence_path)
        time.sleep(1.0)  # Rate limit courtesy between sentences
    
    # Convert all sentence MP3s to WAV and build concat list with silence gaps
    concat_list = work_dir / f"_concat_{Path(out_path).stem}.txt"
    silence_path = work_dir / f"_pause_{Path(out_path).stem}.wav"
    
    # Generate the silence gap file
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=mono",
        "-t", str(pause_seconds), str(silence_path),
    ], capture_output=True)
    
    entries = []
    for idx, mp3_path in enumerate(sentence_files):
        wav_path = work_dir / f"_sentence_{Path(out_path).stem}_{idx:03d}.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", str(mp3_path),
            "-ar", "44100", "-ac", "1", str(wav_path),
        ], capture_output=True)
        
        if idx > 0:
            entries.append(f"file '{silence_path}'")
        entries.append(f"file '{wav_path}'")
    
    concat_list.write_text("\n".join(entries))
    
    # Concat into final output as WAV first, then convert to MP3
    combined_wav = work_dir / f"_combined_{Path(out_path).stem}.wav"
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_list),
        "-c:a", "pcm_s16le", str(combined_wav),
    ], capture_output=True)
    
    # Convert to MP3 to match expected output format
    subprocess.run([
        "ffmpeg", "-y", "-i", str(combined_wav),
        "-c:a", "libmp3lame", "-b:a", "192k", str(out_path),
    ], capture_output=True)
    
    clip = AudioFileClip(str(out_path))
    dur = clip.duration
    clip.close()
    log.info(f"Narration with pauses: {len(sentences)} sentences, {dur:.1f}s total")
    return dur


def generate_ambient_audio(duration, out_path, channel_id=None, title=None, topic=None, api_keys=None):
    """Generate thematic ambient audio using ElevenLabs Sound Generation API.
    
    Creates unique ambient audio for each video based on its content and channel theme.
    Falls back to procedural synthesis if the API is unavailable.
    """
    
    # Build rich, specific ambient descriptions — the more detail, the better the output
    channel_ambient_base = {
        "deadlight_codex": "deep rumbling cosmic drone, vast underground cavern reverb, distant metallic resonance, low ominous sub-bass, wind through impossible architecture, dark ambient soundscape",
        "zero_trace_archive": "quiet empty room tone with distant ventilation hum, fluorescent light buzz, occasional muffled footstep echo, tense investigative atmosphere, low frequency tension",
        "the_unwritten_wing": "warm vinyl crackle, soft rain on windows, gentle piano room reverb, nostalgic and intimate, library ambiance with distant clock ticking",
        "remnants_project": "wind through abandoned buildings, distant creaking metal, birdsong echoing in empty spaces, dripping water, nature slowly reclaiming silence",
        "somnus_protocol": "soft warm ambient drone, deeply calming and sleep-inducing, slow evolving texture, no sudden sounds, gentle and meditative",
        "autonomous_stack": "clean server room hum, soft digital processing sounds, minimal electronic textures, cool and precise data center ambient",
        "gray_meridian": "quiet contemplative room tone, soft warm analog hum, gentle breathing space, psychological stillness, minimal and introspective",
        "softlight_kingdom": "gentle music box melody with soft wind chimes, quiet nighttime crickets, warm cozy fireplace crackle, magical fairy dust shimmer sounds, calming and safe",
        "echelon_veil": "low electronic hum with distant radio static, quiet server room drone, subtle signal interference and data transmission sounds, slightly unsettling modern ambient",
        "loreletics": "distant stadium crowd murmur building slowly, dramatic orchestral undertone, heartbeat tension pulse, cinematic sports atmosphere, epic and emotional",
    }
    
    base_desc = channel_ambient_base.get(channel_id, "cinematic atmospheric ambient soundscape, slow evolving drone, moody and immersive")
    
    # Make ambient content-aware — match the video's actual topic
    # ElevenLabs Sound Generation has a 450 char limit
    if topic:
        base_desc = f"{base_desc}, evoking {topic[:80]}"
    elif title:
        base_desc = f"{base_desc}, evoking {title[:80]}"
    
    # Enforce 450 char limit
    if len(base_desc) > 440:
        base_desc = base_desc[:440]
    
    # ElevenLabs Sound Generation
    elevenlabs_key = None
    if api_keys:
        elevenlabs_key = api_keys.get("elevenlabs", "")
    if not elevenlabs_key:
        elevenlabs_key = os.environ.get("ELEVENLABS_API_KEY", "")
    
    if elevenlabs_key:
        try:
            import httpx as hx
            import subprocess
            
            # Generate two different clips and crossfade them together
            # This avoids the obvious loop seam from repeating one 22s clip
            log.info(f"Generating ambient audio via ElevenLabs Sound Generation...")
            log.info(f"  Prompt: {base_desc[:120]}...")
            
            clips = []
            for clip_idx in range(2):
                # Vary the prompt slightly for the second clip to get variation
                clip_prompt = base_desc if clip_idx == 0 else f"{base_desc}, slowly evolving and shifting"
                
                resp = hx.post(
                    "https://api.elevenlabs.io/v1/sound-generation",
                    headers={"xi-api-key": elevenlabs_key, "Content-Type": "application/json"},
                    json={"text": clip_prompt, "duration_seconds": 22},
                    timeout=60,
                )
                
                if resp.status_code == 200 and len(resp.content) > 1000:
                    clip_path = str(out_path) + f".clip{clip_idx}.mp3"
                    Path(clip_path).write_bytes(resp.content)
                    clips.append(clip_path)
                    log.info(f"ElevenLabs SFX clip {clip_idx+1}: {len(resp.content)} bytes")
                    time.sleep(1)  # Rate limit courtesy
                else:
                    log.warning(f"ElevenLabs Sound Generation clip {clip_idx+1} failed ({resp.status_code}): {resp.text[:200]}")
                    break
            
            if clips:
                # Convert clips to WAV
                wav_clips = []
                for cp in clips:
                    wav_path = cp.replace(".mp3", ".wav")
                    subprocess.run([
                        "ffmpeg", "-y", "-i", cp,
                        "-ar", "44100", "-ac", "1", wav_path,
                    ], capture_output=True)
                    wav_clips.append(wav_path)
                
                if len(wav_clips) == 2:
                    # Crossfade the two clips into one longer seamless base (~40s)
                    merged_path = str(out_path) + ".merged.wav"
                    subprocess.run([
                        "ffmpeg", "-y",
                        "-i", wav_clips[0], "-i", wav_clips[1],
                        "-filter_complex",
                        f"[0:a][1:a]acrossfade=d=4:c1=tri:c2=tri[out]",
                        "-map", "[out]",
                        "-ar", "44100", "-ac", "1",
                        merged_path,
                    ], capture_output=True)
                    base_clip = merged_path
                else:
                    base_clip = wav_clips[0]
                
                # Loop the base clip with crossfade to fill full duration
                loops_needed = int(duration / 35) + 2
                loop_result = subprocess.run([
                    "ffmpeg", "-y",
                    "-stream_loop", str(loops_needed),
                    "-i", base_clip,
                    "-af", (
                        f"afade=t=in:d=4,"
                        f"afade=t=out:st={max(duration - 4, 1)}:d=4,"
                        f"atrim=0:{duration}"
                    ),
                    "-ar", "44100", "-ac", "1",
                    str(out_path),
                ], capture_output=True, text=True)
                
                # Cleanup temp files
                for cp in clips:
                    Path(cp).unlink(missing_ok=True)
                for wp in wav_clips:
                    Path(wp).unlink(missing_ok=True)
                Path(str(out_path) + ".merged.wav").unlink(missing_ok=True)
                
                if loop_result.returncode == 0 and Path(out_path).exists() and Path(out_path).stat().st_size > 1000:
                    log.info(f"Ambient audio generated via ElevenLabs SFX ({duration:.0f}s, {len(clips)} clips merged)")
                    return
                else:
                    log.warning(f"SFX loop failed: {loop_result.stderr[:200] if loop_result.returncode != 0 else 'output too small'}")
            
        except Exception as e:
            log.warning(f"ElevenLabs Sound Generation failed: {str(e)[:150]}")
    
    # FALLBACK: Procedural synthesis
    log.info("Falling back to procedural ambient generation...")
    _generate_procedural_ambient(duration, out_path, channel_id)


def _generate_procedural_ambient(duration, out_path, channel_id=None):
    """Fallback procedural ambient — simple and inoffensive."""
    sr = 44100
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Keep it very simple — just a warm low pad with gentle movement
    # No harsh frequencies, no "helicopter" artifacts
    pad1 = np.sin(2 * np.pi * 55.0 * t) * 0.1
    pad2 = np.sin(2 * np.pi * 82.5 * t) * 0.06
    
    # Very slow LFO modulation
    lfo = np.sin(2 * np.pi * 0.02 * t) * 0.3 + 0.7
    ambient = (pad1 + pad2) * lfo
    
    # Gentle filtered noise (very quiet)
    noise = np.random.randn(len(t)) * 0.008
    ks = int(sr / 150)
    if ks > 1:
        noise = np.convolve(noise, np.ones(ks) / ks, mode="same")
    ambient += noise
    
    # Fade in/out
    fi = int(sr * 4.0)
    fo = int(sr * 5.0)
    if len(ambient) > fi:
        ambient[:fi] *= np.linspace(0, 1, fi) ** 2
    if len(ambient) > fo:
        ambient[-fo:] *= np.linspace(1, 0, fo) ** 2

    peak = np.max(np.abs(ambient))
    if peak > 0:
        ambient = ambient / peak * 0.7
    scipy_wav.write(str(out_path), sr, (ambient * 32767).astype(np.int16))


# --- Image generation ---

def _generate_image(prompt, out_path, hf_token, width=1792, height=1024):
    """Generate image. Tries DALL-E 3 first (best quality), falls back to FLUX via HuggingFace."""

    # Soften prompts for DALL-E to avoid safety filter rejections.
    # DALL-E's content filter is aggressive — it blocks many words that are
    # perfectly fine in context. This replacement map handles known triggers
    # across ALL channel types (horror, kids, sports, documentary, etc.)
    dalle_prompt = prompt
    _dalle_replacements = {
        # Horror / dark themes
        "tentacles": "organic flowing forms",
        "tentacle": "organic flowing form",
        "eldritch": "ancient mysterious",
        "blood": "crimson liquid",
        "bloody": "crimson-stained",
        "bleeding": "dripping crimson",
        "corpse": "still figure",
        "dead body": "motionless figure",
        "dead": "lifeless",
        "death": "passing",
        "dying": "fading",
        "gore": "dark texture",
        "gory": "visceral",
        "skull": "bone structure",
        "skulls": "bone structures",
        "skeleton": "skeletal frame",
        "demon": "dark entity",
        "demonic": "otherworldly",
        "devil": "dark figure",
        "hell": "abyss",
        "hellish": "abyssal",
        "occult": "arcane",
        "satanic": "ritualistic",
        "pentagram": "ancient symbol",
        "sacrifice": "ritual offering",
        "sacrificial": "ceremonial",
        "murder": "dark event",
        "murdered": "vanished",
        "killer": "shadowy figure",
        "kill": "consume",
        "killing": "consuming",
        "weapon": "artifact",
        "weapons": "artifacts",
        "gun": "metal object",
        "knife": "sharp implement",
        "sword": "bladed artifact",
        "torture": "confinement",
        "torment": "anguish",
        "scream": "silent cry",
        "screaming": "anguished",
        "terror": "dread",
        "terrifying": "unsettling",
        "horrifying": "deeply unsettling",
        "horrific": "disturbing",
        "gruesome": "haunting",
        "mutilated": "damaged",
        "mutilation": "deterioration",
        "severed": "separated",
        "decapitated": "broken",
        "dismembered": "fragmented",
        "wound": "mark",
        "wounds": "marks",
        "scar": "weathered mark",
        "zombie": "hollow figure",
        "zombies": "hollow figures",
        "undead": "restless presence",
        "vampire": "pale figure",
        "ghost": "translucent presence",
        "haunted": "unsettled",
        "possessed": "overtaken",
        "possession": "influence",
        "exorcism": "ritual",
        "curse": "dark influence",
        "cursed": "marked",
        "evil": "malevolent",
        "sinister": "foreboding",
        "malicious": "threatening",
        "victim": "figure",
        "victims": "figures",
        "prey": "target",
        "predator": "pursuer",
        "stalker": "watcher",
        "stalking": "following",
        "abduction": "disappearance",
        "kidnap": "vanishing",
        "hostage": "captive figure",
        "captive": "confined figure",
        "prison": "confined space",
        "prisoner": "confined figure",
        "hanging": "suspended",
        "noose": "rope",
        "suicide": "departure",
        "drown": "submerge",
        "drowning": "submerging",
        "suffocate": "smother",
        "strangle": "constrict",
        "poison": "toxic substance",
        "toxic": "hazardous",
        "plague": "spreading condition",
        "disease": "affliction",
        "infection": "contamination",
        "infected": "contaminated",
        "flesh": "surface",
        "skin": "surface layer",
        "naked": "bare",
        "nude": "unclothed figure",
        "exposed": "revealed",

        # Children / minors — DALL-E is very strict about these
        "child": "young character",
        "children": "young characters",
        "kid": "young character",
        "kids": "young characters",
        "little girl": "small fairy-tale character",
        "little boy": "small fairy-tale character",
        "young girl": "small fairy-tale character",
        "young boy": "small fairy-tale character",
        "baby": "tiny creature",
        "babies": "tiny creatures",
        "infant": "tiny creature",
        "toddler": "tiny character",
        "toddlers": "tiny characters",
        "boy": "young adventurer",
        "girl": "young adventurer",
        "teenage": "youthful",
        "teenager": "youthful figure",
        "minor": "small figure",
        "underage": "youthful",
        "schoolgirl": "young student character",
        "schoolboy": "young student character",
        "orphan": "lone young character",
        "newborn": "tiny sleeping creature",

        # Violence / conflict
        "fight": "confrontation",
        "fighting": "clashing",
        "battle": "epic clash",
        "war": "great conflict",
        "warfare": "conflict",
        "combat": "clash",
        "attack": "surge",
        "assault": "advance",
        "explosion": "burst of energy",
        "exploding": "erupting",
        "bomb": "device",
        "bombing": "devastation",
        "destroy": "shatter",
        "destruction": "devastation",
        "destroyed": "shattered",
        "wreckage": "debris",
        "crash": "impact",
        "crashed": "impacted",
        "bullet": "projectile",
        "bullets": "projectiles",
        "shooting": "dramatic action",
        "shot": "struck",
        "punching": "striking",
        "punch": "strike",
        "knockout": "decisive moment",
        "brutal": "intense",
        "brutality": "intensity",
        "violent": "intense",
        "violence": "intensity",
        "aggressive": "forceful",

        # Surveillance / conspiracy — can trigger political content filters
        "surveillance": "monitoring",
        "spy": "observer",
        "spying": "observing",
        "espionage": "covert observation",
        "classified": "restricted",
        "top secret": "highly restricted",
        "government cover-up": "institutional concealment",
        "cover-up": "concealment",
        "propaganda": "messaging",
        "terrorist": "extremist figure",
        "terrorism": "extremism",

        # Sports — injury terms that might trigger
        "injured": "fallen",
        "injury": "setback",
        "broken bone": "physical setback",
        "concussion": "impact",
        "bleeding athlete": "exhausted competitor",

        # Medical / body horror
        "surgery": "medical procedure",
        "surgical": "clinical",
        "autopsy": "examination",
        "dissection": "analysis",
        "organ": "internal structure",
        "organs": "internal structures",
        "intestines": "internal forms",
        "brain": "neural structure",

        # Religious imagery that can trigger
        "crucifixion": "ancient execution method",
        "crucified": "suspended figure",
        "cross": "wooden structure",
        "church burning": "structure in flames",

        # Drugs / substance
        "drugs": "substances",
        "drug": "substance",
        "cocaine": "white powder",
        "heroin": "dark substance",
        "needle": "thin implement",
        "syringe": "medical instrument",
        "overdose": "collapse",
        "intoxicated": "disoriented",
        "drunk": "unsteady",
    }

    # Apply replacements — check longest phrases first to avoid partial matches
    # (e.g., "little girl" before "girl")
    import re
    sorted_replacements = sorted(_dalle_replacements.items(), key=lambda x: -len(x[0]))
    for old_word, new_word in sorted_replacements:
        # Case-insensitive replacement
        pattern = re.compile(re.escape(old_word), re.IGNORECASE)
        dalle_prompt = pattern.sub(new_word, dalle_prompt)

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
            
            # Try DALL-E up to 3 times — on content filter blocks, progressively
            # strip the prompt down to pass the filter while staying on DALL-E
            dalle_attempts = [
                dalle_prompt,  # Attempt 1: already-softened prompt
                None,          # Attempt 2: further stripped (built on failure)
                None,          # Attempt 3: maximally generic (built on failure)
            ]
            
            for attempt_idx, attempt_prompt in enumerate(dalle_attempts):
                if attempt_prompt is None:
                    break
                    
                r = hx.post(
                    "https://api.openai.com/v1/images/generations",
                    headers={"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"},
                    json={
                        "model": "dall-e-3",
                        "prompt": attempt_prompt,
                        "n": 1,
                        "size": dalle_size,
                        "quality": "hd",
                        "response_format": "url",
                    },
                    timeout=120,
                )
                if r.status_code == 200:
                    img_url = r.json()["data"][0]["url"]
                    img_resp = hx.get(img_url, timeout=60)
                    if img_resp.status_code == 200:
                        with open(str(out_path), "wb") as f:
                            f.write(img_resp.content)
                        img = Image.open(str(out_path))
                        img.save(str(out_path), "PNG")
                        if attempt_idx > 0:
                            log.info(f"Image generated via DALL-E 3 (attempt {attempt_idx+1}, softened): {img.size[0]}x{img.size[1]}")
                        else:
                            log.info(f"Image generated via DALL-E 3: {img.size[0]}x{img.size[1]}")
                        return True
                    else:
                        log.warning(f"DALL-E 3 image download failed: {img_resp.status_code}")
                        break  # download issue, not content filter — move to fallback
                else:
                    error_msg = r.text[:300]
                    
                    # Content filter block — retry with progressively softer prompt
                    if r.status_code == 400 and "content" in error_msg.lower() and "filter" in error_msg.lower():
                        log.warning(f"DALL-E 3 content filter block (attempt {attempt_idx+1}/3)")
                        
                        if attempt_idx == 0:
                            # Attempt 2: Strip to just visual descriptors, remove narrative elements
                            stripped = re.sub(r'(?i)(mysterious|eerie|sinister|dark|creepy|haunted|ominous|threatening|menacing|disturbing|unsettling|foreboding|dread|horror|scary|frightening|terrifying|nightmarish)', 'atmospheric', attempt_prompt)
                            stripped = re.sub(r'(?i)(abandoned|decaying|rotting|crumbling|ruined|decrepit|desolate|forsaken)', 'weathered', stripped)
                            stripped = re.sub(r'(?i)(figure|silhouette|shadow figure|shadowy|lurking|watching|stalking)', 'form', stripped)
                            dalle_attempts[1] = stripped
                            log.info("Retrying DALL-E with further softened prompt...")
                        elif attempt_idx == 1:
                            # Attempt 3: Maximally generic — just the visual style and setting
                            # Extract the image_prompt_suffix (always at end) and build minimal prompt
                            generic = f"Atmospheric cinematic scene, moody lighting, {dalle_size.replace('x', ' by ')} composition, "
                            # Try to keep the last part (usually the style suffix)
                            parts = attempt_prompt.rsplit(',', 5)
                            if len(parts) > 3:
                                generic += ', '.join(parts[-3:])
                            else:
                                generic += "4k, film grain, cinematic photography"
                            dalle_attempts[2] = generic
                            log.info("Retrying DALL-E with generic fallback prompt...")
                        continue
                    
                    # Non-content-filter error
                    log.warning(f"DALL-E 3 failed ({r.status_code}): {error_msg[:200]}")
                    if r.status_code == 401:
                        log.error("OpenAI API key invalid for DALL-E 3")
                    elif "billing" in error_msg.lower() or "quota" in error_msg.lower():
                        log.warning("OpenAI billing/quota issue — falling back to FLUX")
                    break  # Don't retry on auth/billing errors
        except Exception as e:
            log.warning(f"DALL-E 3 failed: {str(e)[:150]}")

    # FALLBACK 1: HuggingFace Inference API (free with HF Pro)
    # NOTE: FLUX.1-dev deprecated on HF Inference as of March 2026
    inference_configs = [
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


def _generate_fallback_image(out_path, scene_index, width=1792, height=1024):
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

def apply_ken_burns(image_path, duration, out_path, target_res=(1920, 1080), channel_id=None):
    # Resolve per-channel overrides (fall back to globals)
    overrides = CHANNEL_KB_OVERRIDES.get(channel_id, {})
    kb_zoom = overrides.get("zoom_range", KB_ZOOM_RANGE)
    kb_pan = overrides.get("pan_range", KB_PAN_RANGE)

    img = Image.open(image_path)
    img_w, img_h = img.size
    target_w, target_h = target_res
    if img_w < img_h:
        target_w, target_h = 1080, 1920

    # Ensure source image has enough headroom for Ken Burns zoom/pan.
    # Target ~1.4x the output res so the initial view shows most of the image
    # while leaving room for the motion range.
    min_dim = max(target_w, target_h) * 1.4
    if img_w < min_dim or img_h < min_dim:
        scale = min_dim / min(img_w, img_h)
        img = img.resize((int(img_w * scale), int(img_h * scale)), Image.LANCZOS)
        img_w, img_h = img.size

    motion = random.choice(["zoom_in", "zoom_out", "pan_left", "pan_right", "drift"])
    zoom_start = 1.0
    zoom_end = random.uniform(*kb_zoom)
    if motion == "zoom_out":
        zoom_start, zoom_end = zoom_end, zoom_start

    max_pan_x = int(img_w * kb_pan)
    max_pan_y = int(img_h * kb_pan)
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

    # Pre-convert to float32 for subpixel affine transforms (eliminates jitter)
    img_float = np.array(img).astype(np.float32)
    import cv2

    def make_frame(t_val):
        t_norm = t_val / max(duration - 0.001, 0.001)
        t_norm = max(0.0, min(1.0, t_norm))
        # Smoothstep easing
        t_smooth = t_norm * t_norm * (3.0 - 2.0 * t_norm)
        zoom = zoom_start + (zoom_end - zoom_start) * t_smooth
        pan_x = start_x + (end_x - start_x) * t_smooth
        pan_y = start_y + (end_y - start_y) * t_smooth

        # Subpixel crop via affine transform — no integer snapping
        crop_w = target_w / zoom
        crop_h = target_h / zoom
        cx = img_w / 2.0 + pan_x
        cy = img_h / 2.0 + pan_y

        # Clamp center so crop stays within image bounds
        x1 = max(0.0, min(cx - crop_w / 2.0, img_w - crop_w))
        y1 = max(0.0, min(cy - crop_h / 2.0, img_h - crop_h))

        # Build affine matrix: maps output pixels to source pixels (subpixel precision)
        scale_x = crop_w / target_w
        scale_y = crop_h / target_h
        M = np.array([
            [scale_x, 0,       x1],
            [0,       scale_y, y1],
        ], dtype=np.float32)

        frame = cv2.warpAffine(
            img_float, M, (target_w, target_h),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        return np.clip(frame, 0, 255).astype(np.uint8)

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

def _get_channel_font(channel_id, size=62):
    """Get the channel-specific font at the requested size. Returns (font, path_used)."""
    from PIL import ImageFont
    
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
        "softlight_kingdom": [
            "/System/Library/Fonts/Supplemental/Cochin.ttc",
            "/System/Library/Fonts/Supplemental/Savoye LET.ttc",
            "/System/Library/Fonts/Supplemental/Georgia.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        ],
        "echelon_veil": [
            "/System/Library/Fonts/Supplemental/Helvetica Neue.ttc",
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/SFNSMono.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ],
        "loreletics": [
            "/System/Library/Fonts/Supplemental/Rockwell.ttc",
            "/System/Library/Fonts/Supplemental/Impact.ttf",
            "/System/Library/Fonts/Supplemental/Arial Black.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ],
    }

    generic_fonts = [
        "/System/Library/Fonts/Supplemental/Georgia.ttf",
        "/System/Library/Fonts/Georgia.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    ]

    font_prefs = channel_font_prefs.get(channel_id, [])
    for fp in font_prefs + generic_fonts:
        if Path(fp).exists():
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
    return ImageFont.load_default()


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
    ok = _generate_image(title_bg_prompt, bg_path, hf_token, width=1792, height=1024)

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

        title_font = _get_channel_font(channel_id, 62)

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

        main_font = _get_channel_font(channel_id, 52)
        sub_font = _get_channel_font(channel_id, 28)
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


def _get_thumbnail_font(channel_id, size=90):
    """Get a bold, thematic font for thumbnails — must be heavy enough for small-size
    readability while matching the channel's visual identity."""
    from PIL import ImageFont

    # Each channel gets bold/heavy variants of their thematic font family.
    # Thumbnails need more weight than title cards — prefer Bold/Black weights
    # of the same typeface families used in _get_channel_font.
    channel_thumb_fonts = {
        "deadlight_codex": [
            # Copperplate is already heavy — matches title cards
            "/System/Library/Fonts/Supplemental/Copperplate.ttc",
            "/System/Library/Fonts/Supplemental/Impact.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
        ],
        "zero_trace_archive": [
            # Bold monospace — investigative/classified document feel
            "/System/Library/Fonts/Supplemental/Courier New Bold.ttf",
            "/System/Library/Fonts/Supplemental/Andale Mono.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
        ],
        "the_unwritten_wing": [
            # Bold serif — literary, elegant but heavy
            "/System/Library/Fonts/Supplemental/Baskerville.ttc",
            "/System/Library/Fonts/Supplemental/Georgia Bold.ttf",
            "/System/Library/Fonts/Supplemental/Palatino.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
        ],
        "remnants_project": [
            # Bold clean sans — futuristic, post-human
            "/System/Library/Fonts/Supplemental/Futura.ttc",
            "/System/Library/Fonts/Supplemental/Helvetica Neue.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ],
        "somnus_protocol": [
            # Bold elegant serif — soft but readable, dreamy
            "/System/Library/Fonts/Supplemental/Didot.ttc",
            "/System/Library/Fonts/Supplemental/Georgia Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
        ],
        "softlight_kingdom": [
            # Rounded, warm, storybook — friendly but bold
            "/System/Library/Fonts/Supplemental/Arial Rounded Bold.ttf",
            "/System/Library/Fonts/Supplemental/Cochin.ttc",
            "/System/Library/Fonts/Supplemental/Georgia Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
        ],
        "autonomous_stack": [
            # Bold monospace — techy, code-like
            "/System/Library/Fonts/SFMono-Bold.otf",
            "/System/Library/Fonts/Supplemental/Menlo.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
        ],
        "gray_meridian": [
            # Bold clean sans — intellectual, modern
            "/System/Library/Fonts/Supplemental/Avenir Next.ttc",
            "/System/Library/Fonts/Supplemental/Gill Sans.ttc",
            "/System/Library/Fonts/Supplemental/Helvetica Neue.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ],
        "echelon_veil": [
            # Bold condensed sans — governmental, classified feel
            "/System/Library/Fonts/Supplemental/Helvetica Neue.ttc",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ],
        "loreletics": [
            # Heavy slab/impact — sports broadcast, ESPN-style
            "/System/Library/Fonts/Supplemental/Rockwell.ttc",
            "/System/Library/Fonts/Supplemental/Impact.ttf",
            "/System/Library/Fonts/Supplemental/Arial Black.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ],
    }

    # Generic bold fallbacks if nothing channel-specific is found
    bold_fallbacks = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]

    font_prefs = channel_thumb_fonts.get(channel_id, [])
    for fp in font_prefs + bold_fallbacks:
        if Path(fp).exists():
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _generate_thumbnail(channel, title, scene_image_path, out_path, res=(1280, 720)):
    """Generate a YouTube thumbnail from a scene image + title text overlay.
    YouTube recommends 1280x720 (16:9), minimum 640x360.

    Design principles:
    - Bold, high-contrast text that's readable at tiny sizes (mobile feed)
    - Strong bottom gradient so text pops against any background
    - Semi-transparent dark bar behind text for guaranteed readability
    - Channel-specific accent colors with outer glow for depth
    - Max 2 lines of text — fewer words = more impact
    """
    w, h = res
    channel_id = channel.get("channel_id", "")

    # Load the best scene image as background
    try:
        bg = Image.open(scene_image_path)
        bg = bg.resize((w, h), Image.LANCZOS)
    except Exception:
        bg = Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))

    # Boost contrast and saturation for thumbnail pop
    from PIL import ImageEnhance, ImageDraw, ImageFont, ImageFilter
    bg = ImageEnhance.Contrast(bg).enhance(1.4)
    bg = ImageEnhance.Color(bg).enhance(1.3)
    bg = ImageEnhance.Brightness(bg).enhance(0.8)

    img_array = np.array(bg).astype(np.float32)

    # Strong bottom gradient — heavier than before for text readability
    for y in range(h):
        fade = max(0, (y - h * 0.35) / (h * 0.65))
        img_array[y] *= (1.0 - fade * 0.85)

    # Vignette — stronger on edges to draw eye to center
    cy, cx = h * 0.4, w // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt(((X - cx) / (w * 0.6)) ** 2 + ((Y - cy) / (h * 0.6)) ** 2)
    vignette = np.clip(1.0 - dist * 0.4, 0.25, 1.0)
    img_array *= vignette[:, :, None]

    pil_img = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

    try:
        draw = ImageDraw.Draw(pil_img)

        # Use a bold/heavy font at larger size for thumbnail impact
        title_font = _get_thumbnail_font(channel_id, 90)

        # UPPERCASE title — wrap to max 2 lines for readability at small sizes
        title_upper = title.upper()
        words = title_upper.split()
        lines = []
        current = ""
        max_text_width = w - 120  # generous padding
        for word in words:
            test = f"{current} {word}".strip()
            bbox = draw.textbbox((0, 0), test, font=title_font)
            if bbox[2] - bbox[0] > max_text_width:
                if current:
                    lines.append(current)
                current = word
            else:
                current = test
        if current:
            lines.append(current)

        # Hard cap at 2 lines — truncate with ellipsis if needed
        if len(lines) > 2:
            # Try to fit on 2 lines by trimming
            lines = lines[:2]
            # Don't add ellipsis — just cut cleanly

        # Measure text block
        line_height = 95
        total_text_h = len(lines) * line_height
        text_block_top = h - total_text_h - 55
        text_block_bottom = h - 25

        # Semi-transparent dark bar behind text
        bar_img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        bar_draw = ImageDraw.Draw(bar_img)
        bar_top = text_block_top - 20
        bar_draw.rectangle(
            [(0, bar_top), (w, text_block_bottom + 10)],
            fill=(0, 0, 0, 140),
        )
        # Feather the top edge of the bar with gradient transparency
        for y_off in range(30):
            alpha = int(140 * (y_off / 30))
            bar_draw.line([(0, bar_top - 30 + y_off), (w, bar_top - 30 + y_off)], fill=(0, 0, 0, alpha))

        pil_img = pil_img.convert("RGBA")
        pil_img = Image.alpha_composite(pil_img, bar_img)
        draw = ImageDraw.Draw(pil_img)

        # Channel-specific accent colors (brighter/more saturated for thumbnails)
        accent_colors = {
            "deadlight_codex": (220, 40, 40),
            "zero_trace_archive": (220, 215, 190),
            "the_unwritten_wing": (255, 215, 120),
            "remnants_project": (100, 200, 80),
            "somnus_protocol": (140, 160, 230),
            "autonomous_stack": (80, 210, 255),
            "gray_meridian": (210, 190, 220),
            "softlight_kingdom": (255, 200, 230),
            "echelon_veil": (180, 220, 180),
            "loreletics": (255, 180, 60),
        }
        text_color = accent_colors.get(channel_id, (255, 255, 255))

        # Glow color — tinted version of accent, used for outer glow
        glow_color = tuple(max(0, c - 60) for c in text_color) + (80,)

        for i, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=title_font)
            lw = bbox[2] - bbox[0]
            x = (w - lw) // 2
            y = text_block_top + i * line_height

            # Layer 1: Outer glow (blurred large outline for depth)
            glow_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            glow_draw = ImageDraw.Draw(glow_layer)
            for ox in range(-6, 7):
                for oy in range(-6, 7):
                    if abs(ox) + abs(oy) > 3:  # only outer ring
                        glow_draw.text((x + ox, y + oy), line, fill=glow_color, font=title_font)
            glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=3))
            pil_img = Image.alpha_composite(pil_img, glow_layer)
            draw = ImageDraw.Draw(pil_img)

            # Layer 2: Thick black outline for crisp separation
            outline_color = (0, 0, 0, 255)
            for ox in range(-4, 5):
                for oy in range(-4, 5):
                    if ox * ox + oy * oy <= 20:  # circular outline
                        draw.text((x + ox, y + oy), line, fill=outline_color, font=title_font)

            # Layer 3: Main text in accent color
            draw.text((x, y), line, fill=text_color + (255,), font=title_font)

        # Convert back to RGB for PNG save
        pil_img = pil_img.convert("RGB")

    except Exception as e:
        log.warning(f"Thumbnail text rendering failed: {e}")
        pil_img = pil_img.convert("RGB") if pil_img.mode != "RGB" else pil_img

    pil_img.save(str(out_path), "PNG")
    log.info(f"Thumbnail generated: {out_path}")
    return str(out_path)


# --- Main video generation pipeline ---

def _apply_narration_volume(input_path, output_path, volume):
    """Reduce narration volume via ffmpeg. Used for sleep/meditation channels."""
    import subprocess
    result = subprocess.run([
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-af", f"volume={volume}",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "320k",
        str(output_path),
    ], capture_output=True, text=True)
    if result.returncode == 0:
        log.info(f"Narration volume reduced to {volume}")
    else:
        log.warning(f"Narration volume reduction failed: {result.stderr[:200]}")
        import shutil as sh
        sh.move(str(input_path), str(output_path))


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
    narration_volume = voice.get("narration_volume", 1.0)  # Optional per-channel narration volume reduction
    sentence_pause = voice.get("sentence_pause_seconds", 0)  # Silence between sentences (sleep/meditation channels)

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
            sentence_pause=sentence_pause,
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
        # Generate at 1792x1024 to match DALL-E 3 output size across all sources
        ok = _generate_image(
            scene["image_prompt"], str(img_path),
            api_keys.get("hf_token", ""),
            width=1792, height=1024,
        )
        if not ok:
            fallback_count += 1
            emit("images", f"⚠ Scene {i+1}: Using fallback image")
            _generate_fallback_image(str(img_path), i, width=1792, height=1024)

        # Upscale for Ken Burns headroom — use per-channel size if configured,
        # otherwise default 2688x1536 (~71% visible at 1080p).
        ch_overrides = CHANNEL_KB_OVERRIDES.get(channel_id, {})
        upscale_w, upscale_h = ch_overrides.get("upscale_size", (2688, 1536))
        try:
            raw_img = Image.open(str(img_path))
            upscaled = raw_img.resize((upscale_w, upscale_h), Image.LANCZOS)
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
        apply_ken_burns(scene["image_path"], duration, str(kb_path), target_res=res, channel_id=channel_id)
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
            # Keep tail shorter — the gap_silence between scenes handles the main pause
            extra_buffer = max(3.0, scene_pause * 0.5) if scene_pause > 0 else 2.0
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
        generate_ambient_audio(video_duration + 2, str(drone_path), channel_id=channel_id, title=title, topic=topic, api_keys=api_keys)
        
        if not Path(drone_path).exists() or Path(drone_path).stat().st_size < 1000:
            emit("assembly", "⚠ Ambient audio generation failed — video will have narration only")
            if narration_volume < 1.0:
                _apply_narration_volume(video_path_with_narration, video_path, narration_volume)
            else:
                import shutil as sh
                sh.move(str(video_path_with_narration), str(video_path))
        else:
            vol = ambient_cfg.get("volume", 0.25)
            emit("assembly", f"Mixing ambient at volume {vol}...")

            # Mix using filter_complex with explicit volume control
            # Avoid amix which normalizes/reduces volume of both streams
            # narration_volume allows per-channel voice softening (e.g., SomnusProtocol)
            narr_vol_filter = f"volume={narration_volume}," if narration_volume < 1.0 else ""
            mix_cmd = [
                "ffmpeg", "-y",
                "-i", str(video_path_with_narration),
                "-i", str(drone_path),
                "-filter_complex",
                f"[1:a]atrim=0:{video_duration:.2f},apad=whole_dur={video_duration:.2f},aformat=sample_rates=44100:channel_layouts=stereo,volume={vol}[drone];"
                f"[0:a]apad=whole_dur={video_duration:.2f},aformat=sample_rates=44100:channel_layouts=stereo,{narr_vol_filter}asetpts=PTS-STARTPTS[narr];"
                f"[narr][drone]amix=inputs=2:duration=longest:normalize=0[out]",
                "-map", "0:v",
                "-map", "[out]",
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "320k",
                "-shortest",
                str(video_path),
            ]
            result = subprocess.run(mix_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                log.info(f"Ambient mixed successfully at volume {vol}")
            else:
                log.warning(f"Ambient mix failed: {result.stderr[:300]}")
                emit("assembly", f"⚠ Ambient mix failed — using narration only")
                if narration_volume < 1.0:
                    _apply_narration_volume(video_path_with_narration, video_path, narration_volume)
                else:
                    import shutil as sh
                    sh.move(str(video_path_with_narration), str(video_path))
    else:
        if narration_volume < 1.0:
            _apply_narration_volume(video_path_with_narration, video_path, narration_volume)
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

    ai_disclosure = "This video includes AI-assisted narration and visual generation.\nAll content is original and created for entertainment purposes."

    prompt = f"""You are writing YouTube upload metadata for a video.

Channel: {channel_name}
Channel Description: {channel_desc}
Video Title: {title}
Topic: {topic}
Video length: ~{word_count} words of narration, approximately {len(scenes)} scenes

Write the following in JSON format:

1. "description" — A YouTube description (150-300 words) that:
   - Opens with a compelling 1-2 sentence hook about the video (this is what appears in search results)
   - Briefly describes what the viewer will experience
   - Weaves searchable keywords NATURALLY into the prose — NEVER list keywords explicitly, never write "Keywords:", never break the fourth wall about SEO. The description should read like a human wrote it, not a marketing bot.
   - Includes a line: "If you enjoyed this, please like, comment, and subscribe for more."
   - Includes a call to turn on notifications
   - Matches the channel's tone perfectly
   - After the main video description and CTA, add a blank line separator, then "———" on its own line, then another blank line
   - BELOW the separator, include this exact AI disclosure block: "{ai_disclosure}"
   - Then a blank line, then exactly this copyright line: "© {year} Emrose Media Studios. All rights reserved."
   - The structure MUST be: [video description + CTA] → blank line → ——— → blank line → [AI disclosure] → blank line → [copyright]

2. "tags" — An array of 15-25 YouTube tags optimized for search discovery. Include:
   - The exact video title as a tag
   - 3-5 broad category terms (e.g., "horror", "psychology", "documentary")
   - 5-10 specific long-tail phrases people actually search for (e.g., "scary stories at night", "what happens when humans disappear")
   - 3-5 related channel/creator terms for discoverability
   - The channel name as a tag

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
            "description": f"{title}\n\n———\n\nThis video includes AI-assisted narration and visual generation.\nAll content is original and created for entertainment purposes.\n\n© {year} Emrose Media Studios. All rights reserved.",
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

    # Probe each source video's duration for chapter timestamps
    source_durations = []
    for vp in video_paths:
        probe = subprocess.run([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", vp,
        ], capture_output=True, text=True)
        dur = float(probe.stdout.strip()) if probe.stdout.strip() else 0
        source_durations.append(dur)

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

    # Build chapter timestamps for YouTube description
    # YouTube requires first chapter at 0:00, each chapter must be >= 10 seconds
    chapters = []
    current_time = 0.0
    transition_dur = 2.0
    for i, dir_name in enumerate(video_dirs):
        # Get the source video's title from its metadata
        source_meta_path = OUTPUT_DIR / channel_id / dir_name / "metadata.json"
        chapter_title = dir_name  # fallback
        if source_meta_path.exists():
            try:
                source_meta = json.loads(source_meta_path.read_text())
                chapter_title = source_meta.get("title", dir_name)
            except Exception:
                pass

        # Format timestamp as H:MM:SS or M:SS
        total_secs = int(current_time)
        hours = total_secs // 3600
        mins = (total_secs % 3600) // 60
        secs = total_secs % 60
        if hours > 0:
            ts = f"{hours}:{mins:02d}:{secs:02d}"
        else:
            ts = f"{mins}:{secs:02d}"

        chapters.append({"timestamp": ts, "title": chapter_title, "start_seconds": current_time})

        # Advance by this video's duration + transition (except after last)
        if i < len(source_durations):
            current_time += source_durations[i]
        if i < len(video_dirs) - 1:
            current_time += transition_dur

    chapters_text = "\n".join(f"{ch['timestamp']} {ch['title']}" for ch in chapters)
    log.info(f"Generated {len(chapters)} chapter markers for compilation")

    # Save chapters as a readable text file for review/editing before upload
    (out_dir / "chapters.txt").write_text(chapters_text)
    emit("compile", f"Generated {len(chapters)} YouTube chapter markers")

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
        "chapters": chapters,
        "chapters_text": chapters_text,
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

CRITICAL RULES FOR SHORTS SUCCESS:

HOOK (FIRST 2 SECONDS — THIS IS EVERYTHING):
- The VERY FIRST SENTENCE must be a jarring, provocative, or deeply curious statement
- It must make someone STOP scrolling immediately
- Examples of strong hooks:
  - "This building hasn't been touched in 200 years."
  - "Your brain is lying to you right now."
  - "No one has ever explained what happened here."
  - "You've been making this decision wrong your entire life."
- Do NOT start with atmosphere, setting, or slow buildup
- Do NOT start with "Imagine..." or "What if..." — be declarative, not hypothetical
- The hook must work WITHOUT any visual context — audio alone must grab attention

NARRATION RULES:
- Length: 20-40 seconds when read at the narrator's pace
- Must NOT require prior context from the full video
- Must create curiosity, tension, or insight
- Must feel complete but leave the viewer wanting more
- Do NOT summarize — extract a compelling moment
- Every sentence after the hook should escalate interest

IMAGE PROMPT:
- The image must be VISUALLY STRIKING and immediately attention-grabbing
- High contrast, bold composition, dramatic lighting
- Must be compelling even as a tiny thumbnail in a scroll feed
- No subtle or minimal imagery — this needs to stop a thumb mid-scroll
- The image should match or amplify the hook's impact

Match the tone of {channel['channel_name']} exactly.

Output ONLY valid JSON:
{{"narration": "short script", "image_prompt": "detailed image prompt — MUST be visually striking and scroll-stopping", "suggested_title": "title", "suggested_caption": "caption", "suggested_hashtags": ["#tag1", "#tag2"]}}

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
    # Pad Ken Burns duration to ensure video is always >= narration length
    kb_duration = dur + 2.0
    apply_ken_burns(str(short_img), kb_duration, str(short_kb), target_res=(1080, 1920), channel_id=channel.get("channel_id"))

    # Use the narration duration as the master timeline — never truncate audio
    target_duration = dur + 1.5
    video = VideoFileClip(str(short_kb))
    if video.duration < target_duration:
        # KB clip came out short — extend by freezing on last frame
        log.warning(f"Short KB clip ({video.duration:.1f}s) shorter than target ({target_duration:.1f}s), extending with freeze frame")
        from moviepy import ImageClip as _IC
        # Grab last frame and freeze it for the remaining time
        last_frame = video.get_frame(video.duration - 0.1)
        freeze = _IC(last_frame).with_duration(target_duration - video.duration + 0.5)
        from moviepy import concatenate_videoclips
        video = concatenate_videoclips([video, freeze])
    video = video.subclipped(0, target_duration)
    audio = AudioFileClip(str(short_audio))
    # Trim audio to match video duration so nothing gets cut off mid-sentence
    if audio.duration > target_duration:
        log.warning(f"Short narration ({audio.duration:.1f}s) longer than video ({target_duration:.1f}s) — this shouldn't happen")
    video = video.with_audio(audio)
    # Apply fade only to the visual — do NOT fade the audio (causes narration to drop)
    video = video.with_effects([vfx.FadeIn(1.0)])
    # For fade out: apply to a copy without audio, then re-attach audio
    faded_video = video.without_audio().with_effects([vfx.FadeOut(1.5)])
    video = faded_video.with_audio(video.audio)

    # Mix ambient audio into the short (same as full video)
    short_with_narration = work_dir / "short_with_narration.mp4"
    video.write_videofile(str(short_with_narration), fps=30, codec="libx264", audio_codec="aac", audio_bitrate="320k", preset="slow", threads=4, logger=None)
    video.close()
    audio.close()

    short_output = out_dir / "short.mp4"

    # Generate and mix ambient audio for the short
    channel_id = channel.get("channel_id", "")
    ambient_cfg = channel.get("ambient_audio", {})
    ambient_vol = ambient_cfg.get("volume", 0.25)
    short_drone = work_dir / "short_ambient.wav"

    try:
        emit("short", "Generating ambient audio for short...")
        generate_ambient_audio(target_duration + 1, str(short_drone), channel_id=channel_id,
                               title=short_data.get("suggested_title", ""), topic="", api_keys=api_keys)
    except Exception as e:
        log.warning(f"Short ambient generation failed: {e}")

    if short_drone.exists() and short_drone.stat().st_size > 1000:
        emit("short", f"Mixing short ambient at volume {ambient_vol}...")
        import subprocess
        mix_cmd = [
            "ffmpeg", "-y",
            "-i", str(short_with_narration),
            "-i", str(short_drone),
            "-filter_complex",
            f"[1:a]atrim=0:{target_duration:.2f},apad=whole_dur={target_duration:.2f},aformat=sample_rates=44100:channel_layouts=stereo,volume={ambient_vol}[drone];"
            f"[0:a]apad=whole_dur={target_duration:.2f},aformat=sample_rates=44100:channel_layouts=stereo,asetpts=PTS-STARTPTS[narr];"
            f"[narr][drone]amix=inputs=2:duration=longest:normalize=0[out]",
            "-map", "0:v", "-map", "[out]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "320k",
            "-shortest",
            str(short_output),
        ]
        subprocess.run(mix_cmd, check=True, capture_output=True)
    else:
        import shutil as sh
        sh.move(str(short_with_narration), str(short_output))

    # Validate final short — make sure audio isn't truncated
    try:
        probe = VideoFileClip(str(short_output))
        final_dur = probe.duration
        probe.close()
        if final_dur < dur - 1.0:
            log.warning(f"⚠ Short may be truncated: final={final_dur:.1f}s, narration={dur:.1f}s")
            emit("short", f"⚠ Warning: Short duration ({final_dur:.1f}s) is shorter than narration ({dur:.1f}s)")
    except Exception as e:
        log.warning(f"Could not validate short duration: {e}")

    short_data["duration"] = round(target_duration, 1)
    emit("short", "YouTube Short complete")
    return short_data
