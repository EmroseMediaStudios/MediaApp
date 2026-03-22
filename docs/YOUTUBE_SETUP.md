# YouTube Channel Setup Guide

## 1. Create Brand Account Channels

For each new channel:
1. Go to [youtube.com](https://youtube.com), click your profile icon
2. **Settings** → **Add or manage your channels** → **Create a channel**
3. Name it exactly as the channel name (e.g., "SoftlightKingdom")
4. Set profile picture and banner

## 2. Get Channel IDs

For each channel:
1. Go to [studio.youtube.com](https://studio.youtube.com)
2. Switch to the correct Brand Account (profile icon → Switch account)
3. **Settings** → **Channel** → **Advanced settings**
4. Copy the **Channel ID** (starts with `UC`, 24 characters)

Alternative: Go to your channel page — the URL `youtube.com/channel/UCxxxxxxxx` contains the ID.

## 3. Update Channel Map

In `app/youtube_upload.py`, add/update the channel ID in `YOUTUBE_CHANNEL_MAP`:

```python
YOUTUBE_CHANNEL_MAP = {
    "softlight_kingdom": "UCxxxxxxxxxxxxxxxxxxxxxx",
    "echelon_veil": "UCxxxxxxxxxxxxxxxxxxxxxx",
    "loreletics": "UCxxxxxxxxxxxxxxxxxxxxxx",
    # ... existing channels ...
}
```

## 4. Authenticate Each Channel

**Important:** You must authenticate each YouTube channel separately.

1. Start the app: `./run.sh`
2. Open `http://localhost:7749`
3. For each channel, visit the auth URL:
   - `/youtube/auth/softlight_kingdom`
   - `/youtube/auth/echelon_veil`
   - `/youtube/auth/loreletics`
4. **Before clicking "Authorize"** in the Google popup:
   - Make sure you're signed into the correct Google account
   - Switch to the correct Brand Account channel in YouTube
   - The auth tokens are channel-specific — wrong account = uploads go to wrong channel
5. After authorizing, the token is saved to `youtube_tokens/<channel_id>.json`

## 5. Verify

After auth, check status at:
- `http://localhost:7749/youtube/status`

This should show `authenticated: true` and list all accessible channels.

## Existing Channel IDs

| App Channel ID | YouTube Channel ID | Channel Name |
|---|---|---|
| `deadlight_codex` | `UCeR5uvuGWIQgxHsCg6KOYlg` | DeadlightCodex |
| `gray_meridian` | `UC9bKvKkjAtoA9ysy2RnRPaQ` | GrayMeridian |
| `zero_trace_archive` | `UC8Hy4GyP9T_qTp72c0kAYpg` | ZeroTraceArchive |
| `remnants_project` | `UC7tUZwk9l23qVwryihjBZSg` | RemnantsProject |
| `the_unwritten_wing` | `UCyjZXmPy4gG1xX2RJVJpfGA` | TheUnwrittenWing |
| `somnus_protocol` | `UCukqYDdDPGG8G_jNvkb_2VQ` | SomnusProtocol |
| `autonomous_stack` | `UCzZU7Gn_5eQAfUz6a9rXPAA` | AutonomousStack |
| `softlight_kingdom` | *(pending)* | SoftlightKingdom |
| `echelon_veil` | *(pending)* | EchelonVeil |
| `loreletics` | *(pending)* | Loreletics |

## Re-authentication

Tokens auto-refresh, but if one expires or breaks:
1. Delete the token file: `youtube_tokens/<channel_id>.json`
2. Re-visit the auth URL for that channel
3. Re-authorize

## Troubleshooting

- **Uploads going to wrong channel**: You were signed into the wrong Brand Account during auth. Delete the token and re-auth with the correct account.
- **"Not authenticated" error**: Token expired and couldn't refresh. Re-auth.
- **Videos stuck as private**: Your API project may need [audit verification](https://support.google.com/youtube/contact/yt_api_form).
