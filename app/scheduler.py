"""
Background scheduler for scheduled video uploads to YouTube.
Monitors metadata.json files and automatically uploads videos at their scheduled times.
"""
import os
import json
import logging
import threading
import time
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from . import generator
from . import youtube_upload

log = logging.getLogger("scheduler")

# Posting schedule configuration (days: 0=Mon, 6=Sun)
# ---- BOOST MODE ----
# Persisted to boost_mode.json so it survives restarts.
# Toggle via dashboard UI or set BOOST_MODE_DEFAULT for fresh installs.
BOOST_MODE_DEFAULT = True
BOOST_GAP_DAYS = 3   # Min days between uploads in boost mode
NORMAL_GAP_DAYS = 7  # Min days between uploads in normal mode (true 1x/week)

_BOOST_FILE = Path(__file__).parent.parent / "boost_mode.json"

def get_boost_mode():
    """Read boost mode from persistent file."""
    if _BOOST_FILE.exists():
        try:
            with open(_BOOST_FILE) as f:
                return json.load(f).get("boost", BOOST_MODE_DEFAULT)
        except (json.JSONDecodeError, IOError):
            pass
    return BOOST_MODE_DEFAULT

def set_boost_mode(enabled: bool):
    """Write boost mode to persistent file."""
    with open(_BOOST_FILE, "w") as f:
        json.dump({"boost": enabled}, f)
    print(f"[scheduler] Boost mode {'ON' if enabled else 'OFF'}")

POSTING_SCHEDULE = {
    "deadlight_codex": {"days": [3, 4, 5], "hour": 18},  # Thu-Sat, 6PM ET
    "zero_trace_archive": {"days": [2, 3, 4], "hour": 15},  # Wed-Fri, 3PM ET
    "the_unwritten_wing": {"days": [2, 3, 4], "hour": 15},
    "remnants_project": {"days": [2, 3, 4], "hour": 15},
    "somnus_protocol": {"days": [0, 1, 2, 3, 4, 5, 6], "hour": 20},  # Any day, 8PM ET
    "softlight_kingdom": {"days": [0, 1, 2, 3, 4, 5, 6], "hour": 18},
    "gray_meridian": {"days": [1, 2, 3], "hour": 15},  # Tue-Thu, 3PM ET
    "echelon_veil": {"days": [3, 4, 5], "hour": 18},
    "loreletics": {"days": [4, 5, 6], "hour": 13},  # Fri-Sun, 1PM ET
}

# In boost mode, all channels can post any day (keeps preferred hour)
BOOST_DAYS = [0, 1, 2, 3, 4, 5, 6]

ET = ZoneInfo("America/New_York")


def _load_metadata(channel_id, dir_name):
    """Load metadata.json for a video. Returns dict or None."""
    meta_path = generator.OUTPUT_DIR / channel_id / dir_name / "metadata.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text())
    except Exception as e:
        log.warning(f"Failed to load metadata for {channel_id}/{dir_name}: {e}")
        return None


def _save_metadata(channel_id, dir_name, meta):
    """Save metadata.json for a video."""
    meta_path = generator.OUTPUT_DIR / channel_id / dir_name / "metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        meta_path.write_text(json.dumps(meta, indent=2))
    except Exception as e:
        log.error(f"Failed to save metadata for {channel_id}/{dir_name}: {e}")


def _format_display_time(iso_string):
    """Convert ISO 8601 string to display format like 'Thursday, Mar 27 at 6:00 PM ET'."""
    try:
        dt = datetime.fromisoformat(iso_string).astimezone(ET)
        day_name = dt.strftime("%A")
        month_abbr = dt.strftime("%b")
        day = dt.day
        hour_12 = dt.strftime("%I:%M %p")
        return f"{day_name}, {month_abbr} {day} at {hour_12} ET"
    except Exception as e:
        log.warning(f"Failed to format time {iso_string}: {e}")
        return iso_string


def recommend_upload_time(channel_id):
    """
    Recommend the next available upload time for a channel.
    
    Returns:
        dict: {"datetime": "ISO string", "display": "Human readable", "reasoning": "Why this time"}
    """
    if channel_id not in POSTING_SCHEDULE:
        return None
    
    schedule = POSTING_SCHEDULE[channel_id]
    preferred_days = BOOST_DAYS if get_boost_mode() else schedule["days"]
    preferred_hour = schedule["hour"]
    
    # Scan all videos for this channel to find the last scheduled/uploaded date
    output_channel_dir = generator.OUTPUT_DIR / channel_id
    last_scheduled = None
    
    if output_channel_dir.exists():
        for video_dir in output_channel_dir.iterdir():
            if not video_dir.is_dir():
                continue
            meta = _load_metadata(channel_id, video_dir.name)
            if not meta:
                continue
            
            # Check if this video has been scheduled or uploaded
            if meta.get("scheduled_upload"):
                try:
                    dt = datetime.fromisoformat(meta["scheduled_upload"])
                    if last_scheduled is None or dt > last_scheduled:
                        last_scheduled = dt
                except Exception:
                    pass
            
            if meta.get("youtube_uploaded") and meta.get("youtube_url"):
                # Try to infer upload date from youtube_url or metadata
                # For now, we'll use a heuristic: if uploaded, assume it was recently
                pass
    
    # Find next available slot
    now_et = datetime.now(ET)
    candidate = now_et.replace(hour=preferred_hour, minute=0, second=0, microsecond=0)
    
    # If we're past the preferred hour today, start tomorrow
    if candidate <= now_et:
        candidate += timedelta(days=1)
    
    # Find the next day that matches the preferred days with appropriate gap
    min_gap_days = BOOST_GAP_DAYS if get_boost_mode() else NORMAL_GAP_DAYS
    attempts = 0
    max_attempts = 365
    
    while attempts < max_attempts:
        # Check if this candidate is on a preferred day
        if candidate.weekday() in preferred_days:
            # Check if it's at least min_gap_days after the last upload
            if last_scheduled is None or (candidate - last_scheduled).days >= min_gap_days:
                break
        
        candidate += timedelta(days=1)
        candidate = candidate.replace(hour=preferred_hour, minute=0, second=0, microsecond=0)
        attempts += 1
    
    if attempts >= max_attempts:
        log.error(f"Could not find available upload slot for {channel_id}")
        return None
    
    # Convert to ISO format
    iso_string = candidate.isoformat()
    display = _format_display_time(iso_string)
    
    # Build reasoning
    mode_label = "BOOST 2x/week" if get_boost_mode() else "Normal 1x/week"
    if last_scheduled:
        days_after = (candidate - last_scheduled).days
        reasoning = f"[{mode_label}] Next available {candidate.strftime('%A')} slot, {days_after} days after last upload"
    else:
        reasoning = f"[{mode_label}] First upload: {candidate.strftime('%A')} at {preferred_hour}:00 ET"
    
    # datetime_local is formatted for HTML datetime-local inputs (no timezone suffix)
    datetime_local = candidate.strftime("%Y-%m-%dT%H:%M")

    return {
        "datetime": iso_string,
        "datetime_local": datetime_local,
        "display": display,
        "reasoning": reasoning,
    }


def get_all_scheduled():
    """
    Scan all channels and return all videos with upload_status="scheduled".
    
    Returns:
        list: [{"channel_id", "channel_name", "dir_name", "title", "scheduled_upload", "display_time"}, ...]
    """
    scheduled = []
    
    # Iterate all channels
    for channel_dir in generator.OUTPUT_DIR.iterdir():
        if not channel_dir.is_dir():
            continue
        
        channel_id = channel_dir.name
        
        # Load channel info
        try:
            ch = generator.load_channel(channel_id)
            channel_name = ch.get("channel_name", channel_id) if ch else channel_id
        except Exception:
            channel_name = channel_id
        
        # Iterate all videos for this channel
        for video_dir in channel_dir.iterdir():
            if not video_dir.is_dir():
                continue
            
            dir_name = video_dir.name
            meta = _load_metadata(channel_id, dir_name)
            if not meta:
                continue
            
            # Check if scheduled
            if meta.get("upload_status") == "scheduled" and meta.get("scheduled_upload"):
                scheduled.append({
                    "channel_id": channel_id,
                    "channel_name": channel_name,
                    "dir_name": dir_name,
                    "title": meta.get("title", "Untitled"),
                    "scheduled_upload": meta["scheduled_upload"],
                    "display_time": _format_display_time(meta["scheduled_upload"]),
                })
    
    # Sort by scheduled_upload datetime
    scheduled.sort(key=lambda x: x["scheduled_upload"])
    return scheduled


def _do_scheduled_upload(channel_id, dir_name, api_keys):
    """
    Execute a scheduled upload for a video.
    This is the shared logic used by both the scheduler and the /api/youtube_upload route.
    
    Returns:
        dict: {"success": bool, "video_id": str, "url": str, "error": str}
    """
    try:
        # Load video metadata
        video_dir = generator.OUTPUT_DIR / channel_id / dir_name
        video_path = video_dir / "video.mp4"
        meta_path = video_dir / "metadata.json"
        yt_meta_path = video_dir / "youtube.json"
        thumb_path = video_dir / "thumbnail.png"
        
        if not video_path.exists():
            raise RuntimeError("Video file not found")
        
        meta = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
        
        yt_meta = {}
        if yt_meta_path.exists():
            yt_meta = json.loads(yt_meta_path.read_text())
        elif meta.get("youtube_meta"):
            yt_meta = meta["youtube_meta"]
        
        title = meta.get("title", "Untitled")
        description = yt_meta.get("description", "")
        
        # For compilations, auto-generate description
        if meta.get("is_compilation") and meta.get("chapters_text"):
            chapters_block = meta["chapters_text"]
            if not description:
                year = datetime.now().year
                channel_name = meta.get("channel_name", "")
                description = (
                    f"{title}\n\n"
                    f"A compilation of {meta.get('source_count', '?')} episodes from {channel_name}.\n\n"
                    f"📑 Chapters:\n{chapters_block}\n\n"
                    f"If you enjoyed this, please like, comment, and subscribe for more.\n"
                    f"🔔 Turn on notifications so you never miss an upload.\n\n"
                    f"———\n\n"
                    f"This video includes AI-assisted narration and visual generation.\n"
                    f"All content is original and created for entertainment purposes.\n\n"
                    f"© {year} Emrose Media Studios. All rights reserved."
                )
            elif "0:00" not in description:
                description += f"\n\n📑 Chapters:\n{chapters_block}"
        
        # Add hashtags
        hashtags = yt_meta.get("hashtags", [])
        if hashtags:
            description = " ".join(hashtags) + "\n\n" + description
        
        tags = yt_meta.get("tags", [])
        
        # Also inject top tags as hashtags at the bottom of description (fallback if keyword tags rejected)
        if tags:
            tag_hashtags = [f"#{t.replace(' ', '')}" for t in tags[:8] if t.strip()]
            description = description.rstrip() + "\n\n" + " ".join(tag_hashtags)
        category = youtube_upload.CATEGORIES.get(
            yt_meta.get("category", meta.get("youtube", {}).get("category", "Entertainment")),
            "24"
        )
        
        # Get made_for_kids flag
        ch = generator.load_channel(channel_id)
        made_for_kids = False
        if ch:
            made_for_kids = ch.get("youtube", {}).get("made_for_kids", False)
        
        # YouTube rejects ALL tags on made_for_kids content (COPPA restriction)
        if made_for_kids and tags:
            log.info(f"[SCHEDULER] Dropping {len(tags)} tags for made_for_kids channel {channel_id}")
            tags = []
        
        # TEMP: Disable tags on ALL uploads — YouTube API rejecting every tag set.
        # Hashtags in description still work as fallback. Remove this once root cause found.
        if tags:
            log.info(f"[SCHEDULER] Tags disabled (API bug): dropping {len(tags)} tags for {channel_id}")
            tags = []
        
        # Upload main video
        log.info(f"[SCHEDULER] Uploading {channel_id}/{dir_name}: {title}")
        result = youtube_upload.upload_video(
            video_path=str(video_path),
            title=title,
            description=description,
            tags=tags,
            category_id=category,
            privacy="public",
            thumbnail_path=str(thumb_path) if thumb_path.exists() else None,
            app_channel_id=channel_id,
            made_for_kids=made_for_kids,
        )
        
        # Update metadata with upload info
        meta["youtube_uploaded"] = True
        meta["youtube_video_id"] = result["video_id"]
        meta["youtube_url"] = result["url"]
        meta["youtube_privacy"] = "public"
        
        # Also upload Short if it exists
        short_result = None
        short_path = video_dir / "short.mp4"
        if short_path.exists():
            try:
                short_meta = meta.get("short") or {}
                short_title = short_meta.get("suggested_title", title + " #Shorts")
                if not short_title.endswith("#Shorts"):
                    short_title += " #Shorts"
                
                short_caption = short_meta.get("suggested_caption", "")
                short_hashtags = short_meta.get("suggested_hashtags", ["#Shorts"])
                if "#Shorts" not in short_hashtags:
                    short_hashtags.insert(0, "#Shorts")
                
                # Merge channel default hashtags
                if ch:
                    default_ht = ch.get("youtube", {}).get("default_hashtags", [])
                    existing = set(h.lower() for h in short_hashtags)
                    for dh in default_ht:
                        if dh.lower() not in existing:
                            short_hashtags.append(dh)
                
                short_desc = " ".join(short_hashtags) + "\n\n"
                if short_caption:
                    short_desc += short_caption + "\n\n"
                short_desc += f"Full video: {result['url']}\n\n"
                short_desc += f"Subscribe for more from {meta.get('channel_name', '')}!\n\n"
                short_desc += "———\n\n"
                short_desc += "This video includes AI-assisted narration and visual generation.\n"
                short_desc += "All content is original and created for entertainment purposes.\n\n"
                year = datetime.now().year
                short_desc += f"© {year} Emrose Media Studios. All rights reserved."
                
                short_tags = []  # Tags disabled (same as main video)
                
                log.info(f"[SCHEDULER] Uploading Short for {channel_id}/{dir_name}")
                short_result = youtube_upload.upload_video(
                    video_path=str(short_path),
                    title=short_title,
                    description=short_desc,
                    tags=short_tags,
                    category_id=category,
                    privacy="public",
                    app_channel_id=channel_id,
                    made_for_kids=made_for_kids,
                )
                meta["youtube_short_id"] = short_result["video_id"]
                meta["youtube_short_url"] = short_result["url"]
            except Exception as e:
                log.warning(f"[SCHEDULER] Short upload failed for {channel_id}/{dir_name}: {e}")
        
        # Save updated metadata
        _save_metadata(channel_id, dir_name, meta)
        
        log.info(f"[SCHEDULER] Upload complete for {channel_id}/{dir_name}")
        return {
            "success": True,
            "video_id": result["video_id"],
            "url": result["url"],
        }
    
    except Exception as e:
        error_msg = str(e)
        log.error(f"[SCHEDULER] Upload failed for {channel_id}/{dir_name}: {error_msg}")
        return {
            "success": False,
            "error": error_msg,
        }


def _scheduler_loop(app, api_keys):
    """
    Main scheduler loop. Runs as a daemon thread.
    Checks every 60 seconds for videos to upload.
    """
    log.info("[SCHEDULER] Background upload scheduler started")
    
    while True:
        try:
            # Check all scheduled videos
            now_et = datetime.now(ET)
            log.debug(f"[SCHEDULER] Checking for scheduled uploads at {now_et.isoformat()}")
            
            for channel_dir in generator.OUTPUT_DIR.iterdir():
                if not channel_dir.is_dir():
                    continue
                
                channel_id = channel_dir.name
                
                for video_dir in channel_dir.iterdir():
                    if not video_dir.is_dir():
                        continue
                    
                    dir_name = video_dir.name
                    meta = _load_metadata(channel_id, dir_name)
                    if not meta:
                        continue
                    
                    # Check if this video is scheduled and ready
                    if meta.get("upload_status") == "scheduled" and meta.get("scheduled_upload"):
                        # Skip if already uploaded (prevents double uploads)
                        if meta.get("youtube_uploaded"):
                            meta["upload_status"] = "uploaded"
                            _save_metadata(channel_id, dir_name, meta)
                            continue
                        
                        try:
                            scheduled_dt = datetime.fromisoformat(meta["scheduled_upload"])
                            
                            # Is it time to upload?
                            if scheduled_dt <= now_et:
                                log.info(f"[SCHEDULER] Time to upload {channel_id}/{dir_name}")
                                
                                # Mark as uploading
                                meta["upload_status"] = "uploading"
                                _save_metadata(channel_id, dir_name, meta)
                                
                                # Do the upload
                                result = _do_scheduled_upload(channel_id, dir_name, api_keys)
                                
                                # Reload metadata — _do_scheduled_upload saves youtube_uploaded, video_id, url etc.
                                meta = _load_metadata(channel_id, dir_name) or meta
                                
                                if result["success"]:
                                    meta["upload_status"] = "uploaded"
                                    meta["upload_error"] = None
                                    log.info(f"[SCHEDULER] Successfully uploaded {channel_id}/{dir_name}")
                                else:
                                    meta["upload_status"] = "failed"
                                    meta["upload_error"] = result.get("error", "Unknown error")
                                    log.error(f"[SCHEDULER] Failed to upload {channel_id}/{dir_name}: {result.get('error')}")
                                
                                _save_metadata(channel_id, dir_name, meta)
                        
                        except Exception as e:
                            log.error(f"[SCHEDULER] Error processing {channel_id}/{dir_name}: {e}")
            
        except Exception as e:
            log.error(f"[SCHEDULER] Loop error: {e}")
        
        # Sleep for 60 seconds before checking again
        time.sleep(60)


def start_scheduler(app, api_keys):
    """
    Start the background upload scheduler thread.
    Call this once at app startup.
    """
    # Only start if authentication is available
    if not youtube_upload.is_authenticated():
        log.warning("[SCHEDULER] YouTube not authenticated — scheduler will not process uploads")
        return
    
    # Start daemon thread
    t = threading.Thread(
        target=_scheduler_loop,
        args=(app, api_keys),
        daemon=True,
        name="UploadScheduler"
    )
    t.start()
    log.info("[SCHEDULER] Background scheduler thread started")
