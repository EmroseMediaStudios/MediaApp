"""
YouTube metrics module for Emrose Media Studios.
Fetches channel and video statistics via YouTube Data API v3.
Caches results locally to minimize quota usage.

Quota costs:
- channels.list (statistics) = 1 point per call
- videos.list (statistics) = 1 point per call (up to 50 video IDs per request)
- Daily quota: 10,000 points
- Refresh every 6h across 7 channels ≈ ~42 points/refresh × 4/day = ~168/day
- Leaves ~9,800 for uploads (1,600 per upload = ~6 uploads/day)
"""
import json
import logging
import time
from pathlib import Path
from datetime import datetime

log = logging.getLogger("youtube_metrics")

CACHE_PATH = Path(__file__).parent.parent / "metrics_cache.json"
CACHE_MAX_AGE = 21600  # 6 hours in seconds (saves ~800 quota units/day vs 1hr)


def _load_cache():
    """Load cached metrics from disk."""
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text())
        except Exception:
            pass
    return {"channels": {}, "videos": {}, "last_refresh": None}


def _save_cache(cache):
    """Save metrics cache to disk."""
    CACHE_PATH.write_text(json.dumps(cache, indent=2))


def _get_youtube_service(youtube_channel_id=None):
    """Build a YouTube API service object."""
    from googleapiclient.discovery import build
    from . import youtube_upload
    
    creds = youtube_upload._get_credentials(youtube_channel_id)
    if not creds:
        creds = youtube_upload._get_credentials()
    if not creds:
        return None
    
    return build("youtube", "v3", credentials=creds)


def fetch_channel_stats(youtube, yt_channel_id):
    """Fetch subscriber count, total views, and video count for a YouTube channel."""
    try:
        resp = youtube.channels().list(
            part="statistics,snippet",
            id=yt_channel_id
        ).execute()
        
        items = resp.get("items", [])
        if not items:
            return None
        
        stats = items[0]["statistics"]
        snippet = items[0]["snippet"]
        
        return {
            "channel_id": yt_channel_id,
            "channel_title": snippet.get("title", ""),
            "subscribers": int(stats.get("subscriberCount", 0)),
            "total_views": int(stats.get("viewCount", 0)),
            "total_videos": int(stats.get("videoCount", 0)),
            "hidden_subscribers": stats.get("hiddenSubscriberCount", False),
            "fetched_at": datetime.now().isoformat(),
        }
    except Exception as e:
        log.warning(f"Failed to fetch channel stats for {yt_channel_id}: {e}")
        return None


def fetch_video_list(youtube, yt_channel_id, max_results=50):
    """Fetch the list of uploaded videos for a channel."""
    try:
        # First get the uploads playlist ID
        resp = youtube.channels().list(
            part="contentDetails",
            id=yt_channel_id
        ).execute()
        
        items = resp.get("items", [])
        if not items:
            return []
        
        uploads_playlist = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]
        
        # Fetch videos from the uploads playlist
        videos = []
        next_page = None
        
        while len(videos) < max_results:
            pl_resp = youtube.playlistItems().list(
                part="snippet,contentDetails",
                playlistId=uploads_playlist,
                maxResults=min(50, max_results - len(videos)),
                pageToken=next_page,
            ).execute()
            
            for item in pl_resp.get("items", []):
                videos.append({
                    "video_id": item["contentDetails"]["videoId"],
                    "title": item["snippet"]["title"],
                    "published_at": item["snippet"]["publishedAt"],
                    "thumbnail": item["snippet"].get("thumbnails", {}).get("medium", {}).get("url", ""),
                })
            
            next_page = pl_resp.get("nextPageToken")
            if not next_page:
                break
        
        return videos
    except Exception as e:
        log.warning(f"Failed to fetch video list for {yt_channel_id}: {e}")
        return []


def fetch_video_stats(youtube, video_ids):
    """Fetch statistics for a batch of videos (up to 50 per call)."""
    if not video_ids:
        return {}
    
    stats = {}
    # Process in batches of 50
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i+50]
        try:
            resp = youtube.videos().list(
                part="statistics,contentDetails,snippet",
                id=",".join(batch)
            ).execute()
            
            for item in resp.get("items", []):
                vid = item["id"]
                s = item["statistics"]
                cd = item["contentDetails"]
                
                # Parse duration to determine if Short (<=60s)
                duration_str = cd.get("duration", "PT0S")
                duration_secs = _parse_duration(duration_str)
                is_short = duration_secs <= 60
                
                stats[vid] = {
                    "video_id": vid,
                    "title": item["snippet"]["title"],
                    "published_at": item["snippet"]["publishedAt"],
                    "views": int(s.get("viewCount", 0)),
                    "likes": int(s.get("likeCount", 0)),
                    "comments": int(s.get("commentCount", 0)),
                    "duration_seconds": duration_secs,
                    "is_short": is_short,
                    "fetched_at": datetime.now().isoformat(),
                }
        except Exception as e:
            log.warning(f"Failed to fetch video stats batch: {e}")
    
    return stats


def _parse_duration(iso_duration):
    """Parse ISO 8601 duration (PT1H2M3S) to seconds."""
    import re
    match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', iso_duration)
    if not match:
        return 0
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    return hours * 3600 + minutes * 60 + seconds


def refresh_all_metrics(channel_map):
    """
    Refresh metrics for all channels in the map.
    channel_map: dict of app_channel_id -> youtube_channel_id
    Returns the updated cache.
    """
    youtube = _get_youtube_service()
    if not youtube:
        log.warning("YouTube not authenticated — cannot fetch metrics")
        return _load_cache()
    
    cache = _load_cache()
    
    for app_id, yt_id in channel_map.items():
        log.info(f"Fetching metrics for {app_id} ({yt_id})...")
        
        # Channel-level stats
        ch_stats = fetch_channel_stats(youtube, yt_id)
        if ch_stats:
            cache["channels"][app_id] = ch_stats
        
        # Video list + stats
        videos = fetch_video_list(youtube, yt_id, max_results=100)
        if videos:
            video_ids = [v["video_id"] for v in videos]
            vid_stats = fetch_video_stats(youtube, video_ids)
            
            # Calculate breakdowns
            full_views = 0
            short_views = 0
            video_list = []
            
            for v in videos:
                vid = v["video_id"]
                if vid in vid_stats:
                    vs = vid_stats[vid]
                    if vs["is_short"]:
                        short_views += vs["views"]
                    else:
                        full_views += vs["views"]
                    video_list.append(vs)
            
            # Sort by views descending for compilation candidates
            video_list.sort(key=lambda x: x["views"], reverse=True)
            
            cache["videos"][app_id] = {
                "full_length_views": full_views,
                "shorts_views": short_views,
                "total_views": full_views + short_views,
                "video_count": len(video_list),
                "videos": video_list,
                "fetched_at": datetime.now().isoformat(),
            }
        
        time.sleep(0.5)  # Courtesy pause between channels
    
    cache["last_refresh"] = datetime.now().isoformat()
    _save_cache(cache)
    log.info(f"Metrics refresh complete for {len(channel_map)} channels")
    return cache


def get_metrics(force_refresh=False, channel_map=None):
    """Get metrics from cache. Only refreshes when explicitly requested (force_refresh=True)."""
    cache = _load_cache()
    
    if force_refresh and channel_map:
        return refresh_all_metrics(channel_map)
    
    return cache


def get_dashboard_summary(cache):
    """Build a summary dict for the dashboard template."""
    summary = {}
    for app_id, ch_data in cache.get("channels", {}).items():
        vid_data = cache.get("videos", {}).get(app_id, {})
        summary[app_id] = {
            "subscribers": ch_data.get("subscribers", 0),
            "total_views": ch_data.get("total_views", 0),
            "full_length_views": vid_data.get("full_length_views", 0),
            "shorts_views": vid_data.get("shorts_views", 0),
            "video_count": vid_data.get("video_count", 0),
            "top_videos": vid_data.get("videos", [])[:5],
        }
    return summary


def get_channel_videos_with_stats(app_channel_id, cache):
    """Get video list with stats for a specific channel, sorted by views."""
    vid_data = cache.get("videos", {}).get(app_channel_id, {})
    return vid_data.get("videos", [])
