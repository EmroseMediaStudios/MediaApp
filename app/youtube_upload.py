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
TOKEN_PATH = Path(__file__).parent.parent / "youtube_token.json"

# YouTube API scopes
SCOPES = ["https://www.googleapis.com/auth/youtube.upload",
          "https://www.googleapis.com/auth/youtube"]


def _get_credentials():
    """Get or refresh OAuth2 credentials. Returns None if not authenticated."""
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request

    creds = None
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            TOKEN_PATH.write_text(creds.to_json())
        except Exception as e:
            log.warning(f"Token refresh failed: {e}")
            creds = None

    return creds


def is_authenticated():
    """Check if we have valid YouTube credentials."""
    creds = _get_credentials()
    return creds is not None and creds.valid


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


def run_local_auth():
    """Run the full OAuth2 flow using a local server (opens browser)."""
    from google_auth_oauthlib.flow import InstalledAppFlow

    if not CLIENT_SECRET_PATH.exists():
        raise RuntimeError("client_secret.json not found")

    flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRET_PATH), SCOPES)
    creds = flow.run_local_server(port=8090, open_browser=True)
    TOKEN_PATH.write_text(creds.to_json())
    log.info("YouTube OAuth2 credentials saved via local server")
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


def upload_video(video_path, title, description="", tags=None, category_id="22",
                 privacy="private", thumbnail_path=None, progress=None):
    """
    Upload a video to YouTube.

    Args:
        video_path: Path to the .mp4 file
        title: Video title
        description: Video description
        tags: List of tags
        category_id: YouTube category (22=Entertainment, 28=Science, etc.)
        privacy: "private", "unlisted", or "public"
        thumbnail_path: Optional path to thumbnail image
        progress: Optional callback(percent, message)

    Returns:
        dict with video_id, url, etc.
    """
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    creds = _get_credentials()
    if not creds:
        raise RuntimeError("Not authenticated with YouTube. Run /youtube/auth first.")

    youtube = build("youtube", "v3", credentials=creds)

    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags or [],
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
