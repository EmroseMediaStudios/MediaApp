"""
YouTube upload module for Emrose Media Studios.
Handles OAuth2 authentication and video uploads via YouTube Data API v3.
"""
import os
import json
import logging
import time
from pathlib import Path

log = logging.getLogger("youtube_upload")

CLIENT_SECRET_PATH = Path(__file__).parent.parent / "client_secret.json"
TOKEN_DIR = Path(__file__).parent.parent / "youtube_tokens"
TOKEN_PATH = Path(__file__).parent.parent / "youtube_token.json"  # Legacy default

# YouTube API scopes
SCOPES = ["https://www.googleapis.com/auth/youtube.upload",
          "https://www.googleapis.com/auth/youtube"]


def _token_path_for_channel(channel_id=None):
    """Get the token file path for a specific YouTube channel."""
    if channel_id:
        TOKEN_DIR.mkdir(exist_ok=True)
        return TOKEN_DIR / f"{channel_id}.json"
    return TOKEN_PATH


def _get_credentials(youtube_channel_id=None):
    """Get or refresh OAuth2 credentials. Returns None if not authenticated."""
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request

    token_path = _token_path_for_channel(youtube_channel_id)
    
    creds = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    elif youtube_channel_id and TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            token_path.write_text(creds.to_json())
        except Exception as e:
            log.warning(f"Token refresh failed: {e}")
            creds = None

    return creds


def is_authenticated(youtube_channel_id=None):
    """Check if we have valid YouTube credentials."""
    creds = _get_credentials(youtube_channel_id)
    return creds is not None and creds.valid


def get_authenticated_channels():
    """List which YouTube channels have saved auth tokens."""
    authenticated = {}
    if TOKEN_PATH.exists():
        authenticated["default"] = True
    if TOKEN_DIR.exists():
        for f in TOKEN_DIR.glob("*.json"):
            channel_id = f.stem
            authenticated[channel_id] = True
    return authenticated


def get_auth_url():
    """Generate OAuth2 authorization URL for first-time setup."""
    from google_auth_oauthlib.flow import InstalledAppFlow

    if not CLIENT_SECRET_PATH.exists():
        return None

    flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRET_PATH), SCOPES)
    flow.redirect_uri = "http://localhost:8090"
    auth_url, _ = flow.authorization_url(prompt="consent", access_type="offline")
    return auth_url


def complete_auth(auth_code):
    """Complete OAuth2 flow with the authorization code."""
    from google_auth_oauthlib.flow import InstalledAppFlow

    flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRET_PATH), SCOPES)
    flow.redirect_uri = "http://localhost:8090"
    flow.fetch_token(code=auth_code)
    creds = flow.credentials
    TOKEN_PATH.write_text(creds.to_json())
    log.info("YouTube OAuth2 credentials saved")
    return True


def run_local_auth(youtube_channel_id=None):
    """Run the full OAuth2 flow using a local server (opens browser).
    
    For Brand Accounts: switch to the correct channel in YouTube BEFORE authorizing.
    Each channel needs its own separate auth with the correct account selected.
    """
    from google_auth_oauthlib.flow import InstalledAppFlow

    if not CLIENT_SECRET_PATH.exists():
        raise RuntimeError("client_secret.json not found")

    flow = InstalledAppFlow.from_client_secrets_file(
        str(CLIENT_SECRET_PATH),
        SCOPES,
        # Force account selection every time so user picks the right Brand Account
    )
    creds = flow.run_local_server(
        port=8090,
        open_browser=True,
        authorization_prompt_message="Opening browser for YouTube auth...\nSwitch to the correct Brand Account channel BEFORE clicking authorize.",
        prompt="consent",  # Force re-consent to ensure correct account
    )
    
    token_path = _token_path_for_channel(youtube_channel_id)
    token_path.write_text(creds.to_json())
    log.info(f"YouTube OAuth2 credentials saved to {token_path}")
    return True


def list_channels():
    """List YouTube channels accessible with current credentials."""
    from googleapiclient.discovery import build

    creds = _get_credentials()
    if not creds:
        return []

    youtube = build("youtube", "v3", credentials=creds)
    
    channels = []
    resp = youtube.channels().list(part="snippet,contentDetails", mine=True).execute()
    for ch in resp.get("items", []):
        channels.append({
            "id": ch["id"],
            "title": ch["snippet"]["title"],
            "description": ch["snippet"].get("description", ""),
        })
    
    try:
        resp2 = youtube.channels().list(part="snippet,contentDetails", managedByMe=True, maxResults=50).execute()
        for ch in resp2.get("items", []):
            if ch["id"] not in [c["id"] for c in channels]:
                channels.append({
                    "id": ch["id"],
                    "title": ch["snippet"]["title"],
                    "description": ch["snippet"].get("description", ""),
                })
    except Exception as e:
        log.warning(f"Could not list managed channels: {e}")
    
    return channels


def _sanitize_tags(tags):
    """Sanitize YouTube tags — strip everything except alphanumeric, spaces, hyphens, apostrophes."""
    import re
    if not tags:
        return []
    
    sanitized = []
    total_chars = 0
    seen = set()
    for tag in tags:
        if not isinstance(tag, str):
            continue
        tag = re.sub(r"[^a-zA-Z0-9\s\-']", '', tag)
        tag = re.sub(r'\s+', ' ', tag).strip()
        if not tag or len(tag) < 2:
            continue
        if tag.startswith("http") or ".com" in tag or ".org" in tag:
            continue
        tag_lower = tag.lower()
        if tag_lower in seen:
            continue
        seen.add(tag_lower)
        tag = tag[:30].rstrip()
        tag_cost = len(tag) + (1 if sanitized else 0)
        if total_chars + tag_cost > 480:
            break
        sanitized.append(tag)
        total_chars += tag_cost
    
    log.info(f"Sanitized {len(tags)} tags -> {len(sanitized)} tags ({total_chars} chars): {sanitized}")
    return sanitized


def _is_invalid_tags_error(error):
    """Check if an HttpError is specifically about invalid tags."""
    err_str = str(error)
    return "invalidTags" in err_str or "invalid video keywords" in err_str


def _do_upload(youtube, video_path, title, description, tags, category_id, privacy, progress, made_for_kids=False):
    """Execute the actual YouTube upload. Returns the API response."""
    from googleapiclient.http import MediaFileUpload

    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags,
            "categoryId": category_id,
        },
        "status": {
            "privacyStatus": privacy,
            "selfDeclaredMadeForKids": made_for_kids,
            "containsSyntheticMedia": True,
        },
    }

    media = MediaFileUpload(
        video_path,
        mimetype="video/mp4",
        resumable=True,
        chunksize=10 * 1024 * 1024,
    )

    log.info(f"Uploading '{title}' ({privacy}) with {len(tags)} tags: {tags}")
    if progress:
        progress(0, "Starting upload...")

    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            pct = int(status.progress() * 100)
            log.info(f"Upload progress: {pct}%")
            if progress:
                progress(pct, f"Uploading... {pct}%")

    return response


def _find_bad_tags_by_elimination(youtube, video_path, title, description, tags, category_id, privacy):
    """When tags are rejected, just drop them all rather than testing each one.
    
    Testing each tag individually costs 1,600 quota units per tag (videos.insert),
    which would obliterate the 10,000 daily quota. Instead, log the rejected tags
    for manual review and retry with no tags.
    """
    log.warning(f"Tags rejected by YouTube — dropping all {len(tags)} tags to preserve quota")
    log.warning(f"Rejected tags were: {tags}")
    log.warning("Review these tags manually and update your tag lists")
    return [], tags  # All tags treated as bad


def upload_video(video_path, title, description="", tags=None, category_id="22",
                 privacy="private", thumbnail_path=None, progress=None, app_channel_id=None,
                 made_for_kids=False):
    """
    Upload a video to YouTube.
    On invalid tags: tests each tag individually, removes bad ones, retries.
    """
    import re
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError

    # Resolve YouTube channel ID and get appropriate credentials
    yt_channel_id = YOUTUBE_CHANNEL_MAP.get(app_channel_id) if app_channel_id else None
    creds = _get_credentials(yt_channel_id)
    if not creds:
        creds = _get_credentials()
    if not creds:
        raise RuntimeError("Not authenticated with YouTube. Run /youtube/auth first.")

    youtube = build("youtube", "v3", credentials=creds)

    # Sanitize tags and title
    clean_tags = _sanitize_tags(tags)
    clean_title = re.sub(r'[<>]', '', title or "Untitled")[:100].strip()

    # First attempt with all sanitized tags
    try:
        response = _do_upload(youtube, video_path, clean_title, description, clean_tags, category_id, privacy, progress, made_for_kids)
    except HttpError as e:
        if not _is_invalid_tags_error(e):
            raise

        log.warning(f"Upload failed with invalidTags. Testing each tag individually...")
        log.warning(f"Failed tags were: {clean_tags}")

        # Test each tag to find the bad ones
        good_tags, bad_tags = _find_bad_tags_by_elimination(
            youtube, video_path, clean_title, description, clean_tags, category_id, privacy
        )

        if bad_tags:
            log.warning(f"Removed {len(bad_tags)} bad tags: {bad_tags}")
            log.info(f"Retrying upload with {len(good_tags)} good tags: {good_tags}")
        else:
            # Couldn't isolate any — might be a combination issue. Try with no tags.
            log.warning("No individual bad tags found — might be a combination issue. Uploading with no tags.")
            good_tags = []

        response = _do_upload(youtube, video_path, clean_title, description, good_tags, category_id, privacy, progress, made_for_kids)

    video_id = response["id"]
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    log.info(f"Upload complete: {video_url}")

    if progress:
        progress(100, "Upload complete!")

    # Set thumbnail if provided
    if thumbnail_path and Path(thumbnail_path).exists():
        try:
            from googleapiclient.http import MediaFileUpload as MFU
            thumb_media = MFU(thumbnail_path, mimetype="image/png")
            youtube.thumbnails().set(videoId=video_id, media_body=thumb_media).execute()
            log.info(f"Thumbnail set for {video_id}")
        except Exception as e:
            log.warning(f"Thumbnail upload failed (may need channel verification): {e}")

    return {
        "video_id": video_id,
        "url": video_url,
        "title": clean_title,
        "privacy": privacy,
    }


# YouTube category IDs
CATEGORIES = {
    "Entertainment": "24",
    "Education": "27",
    "Science & Technology": "28",
    "Film & Animation": "1",
    "Music": "10",
    "People & Blogs": "22",
    "Gaming": "20",
    "Howto & Style": "26",
}

# Map app channel IDs to YouTube channel IDs
YOUTUBE_CHANNEL_MAP = {
    "gray_meridian": "UC9bKvKkjAtoA9ysy2RnRPaQ",
    "zero_trace_archive": "UC8Hy4GyP9T_qTp72c0kAYpg",
    "remnants_project": "UC7tUZwk9l23qVwryihjBZSg",
    "the_unwritten_wing": "UCyjZXmPy4gG1xX2RJVJpfGA",
    "autonomous_stack": "UCzZU7Gn_5eQAfUz6a9rXPAA",
    "somnus_protocol": "UCukqYDdDPGG8G_jNvkb_2VQ",
    "deadlight_codex": "UCeR5uvuGWIQgxHsCg6KOYlg",
    "softlight_kingdom": "UC7AI57ebEXWAXw6Uhku0hJQ",
    "echelon_veil": "UChiZ0p3OOyBfk8gWC9DJ0NQ",
    "loreletics": "UCkabQW3XUx83Cnj25LJcf3A",
}
