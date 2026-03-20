"""
YouTube upload module for Emrose Media Studios.
Handles OAuth2 authentication and video uploads via YouTube Data API v3.
"""
import os
import json
import logging
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
        # Fall back to default token
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
    # Check default token
    if TOKEN_PATH.exists():
        authenticated["default"] = True
    # Check per-channel tokens
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
    For Brand Accounts, switch to the target channel in YouTube before authorizing."""
    from google_auth_oauthlib.flow import InstalledAppFlow

    if not CLIENT_SECRET_PATH.exists():
        raise RuntimeError("client_secret.json not found")

    flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRET_PATH), SCOPES)
    creds = flow.run_local_server(port=8090, open_browser=True)
    
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
    
    # Get own channels
    channels = []
    resp = youtube.channels().list(part="snippet,contentDetails", mine=True).execute()
    for ch in resp.get("items", []):
        channels.append({
            "id": ch["id"],
            "title": ch["snippet"]["title"],
            "description": ch["snippet"].get("description", ""),
        })
    
    # Also list channels managed via Brand Accounts
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
    """Sanitize YouTube tags to avoid 'invalid video keywords' API errors.
    
    YouTube rejects tags that:
    - Contain < or > characters
    - Exceed 500 characters combined (including comma separators)
    - Are excessively long individually (>30 chars is unusual)
    - Contain special characters like = { } [ ] or multiple spaces
    - Are empty or whitespace-only
    """
    import re
    if not tags:
        return []
    
    sanitized = []
    total_chars = 0
    seen = set()
    for tag in tags:
        if not isinstance(tag, str):
            continue
        # Strip HTML-like characters and other problematic chars
        tag = re.sub(r'[<>{}\[\]=|\\~`]', '', tag)
        # Collapse multiple spaces
        tag = re.sub(r'\s+', ' ', tag)
        # Remove leading/trailing whitespace and quotes
        tag = tag.strip().strip('"').strip("'").strip()
        # Skip empty tags or tags that are just punctuation/symbols
        if not tag or len(tag) < 2:
            continue
        # Skip if it looks like a URL
        if tag.startswith("http") or ".com" in tag or ".org" in tag:
            continue
        # Skip duplicate tags (case-insensitive)
        tag_lower = tag.lower()
        if tag_lower in seen:
            continue
        seen.add(tag_lower)
        # Truncate individual tags at 30 chars (YouTube best practice)
        tag = tag[:30].rstrip()
        # Check combined limit (500 chars total, YouTube counts commas as separators)
        tag_cost = len(tag) + (1 if sanitized else 0)  # comma separator
        if total_chars + tag_cost > 480:  # Leave some margin
            break
        sanitized.append(tag)
        total_chars += tag_cost
    
    # Log what we're sending for debugging
    log.info(f"Sanitized {len(tags)} tags → {len(sanitized)} tags ({total_chars} chars)")
    
    return sanitized


def upload_video(video_path, title, description="", tags=None, category_id="22",
                 privacy="private", thumbnail_path=None, progress=None, app_channel_id=None):
    """
    Upload a video to YouTube.
    app_channel_id maps to the YouTube channel via YOUTUBE_CHANNEL_MAP.
    """
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    # Resolve YouTube channel ID and get appropriate credentials
    yt_channel_id = YOUTUBE_CHANNEL_MAP.get(app_channel_id) if app_channel_id else None
    creds = _get_credentials(yt_channel_id)
    if not creds:
        creds = _get_credentials()  # Fall back to default
    if not creds:
        raise RuntimeError("Not authenticated with YouTube. Run /youtube/auth first.")

    youtube = build("youtube", "v3", credentials=creds)

    # Sanitize tags to avoid YouTube API "invalid video keywords" errors
    clean_tags = _sanitize_tags(tags)
    log.info(f"Upload tags for '{title}': {clean_tags}")

    # Also sanitize title (max 100 chars, no < >)
    import re
    clean_title = re.sub(r'[<>]', '', title or "Untitled")[:100].strip()

    body = {
        "snippet": {
            "title": clean_title,
            "description": description,
            "tags": clean_tags,
            "categoryId": category_id,
        },
        "status": {
            "privacyStatus": privacy,
            "selfDeclaredMadeForKids": False,
        },
    }

    media = MediaFileUpload(
        video_path,
        mimetype="video/mp4",
        resumable=True,
        chunksize=10 * 1024 * 1024,  # 10MB chunks
    )

    log.info(f"Uploading '{title}' to YouTube ({privacy})...")
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

    video_id = response["id"]
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    log.info(f"Upload complete: {video_url}")

    if progress:
        progress(100, "Upload complete!")

    # Set thumbnail if provided
    if thumbnail_path and Path(thumbnail_path).exists():
        try:
            thumb_media = MediaFileUpload(thumbnail_path, mimetype="image/png")
            youtube.thumbnails().set(videoId=video_id, media_body=thumb_media).execute()
            log.info(f"Thumbnail set for {video_id}")
        except Exception as e:
            log.warning(f"Thumbnail upload failed (may need channel verification): {e}")

    return {
        "video_id": video_id,
        "url": video_url,
        "title": title,
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
}
