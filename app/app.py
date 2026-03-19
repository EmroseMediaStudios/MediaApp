"""
Flask web application for multi-channel video generation.
"""
import os
import json
import queue
import threading
from pathlib import Path

from flask import (
    Flask, render_template, request, jsonify,
    redirect, url_for, Response, send_from_directory,
)

from . import generator
from . import youtube_upload

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Global progress queues for SSE
_progress_queues = {}
_generation_results = {}

API_KEYS = {
    "openai": os.environ.get("OPENAI_API_KEY", ""),
    "elevenlabs": os.environ.get("ELEVENLABS_API_KEY", ""),
    "hf_token": os.environ.get("HF_TOKEN", ""),
}


@app.route("/")
def dashboard():
    channels = generator.list_channels()
    return render_template("dashboard.html", channels=channels)


@app.route("/channel/<channel_id>")
def channel_workspace(channel_id):
    ch = generator.load_channel(channel_id)
    if not ch:
        return "Channel not found", 404
    videos = generator.list_videos(channel_id)
    return render_template("channel.html", channel=ch, videos=videos)


@app.route("/channel/<channel_id>/idea", methods=["GET", "POST"])
def topic_idea(channel_id):
    ch = generator.load_channel(channel_id)
    if not ch:
        return "Channel not found", 404
    idea = None
    if request.method == "POST":
        idea = generator.generate_topic_idea(ch, API_KEYS["openai"])
    return render_template("idea.html", channel=ch, idea=idea)


@app.route("/channel/<channel_id>/script", methods=["GET", "POST"])
def script_editor(channel_id):
    ch = generator.load_channel(channel_id)
    if not ch:
        return "Channel not found", 404

    script = None
    topic = request.args.get("topic", "") or request.form.get("topic", "")

    if request.method == "POST" and "generate" in request.form:
        script = generator.generate_script(ch, topic, API_KEYS["openai"])

    return render_template("script.html", channel=ch, script=script, topic=topic)


@app.route("/channel/<channel_id>/generate", methods=["GET", "POST"])
def generate_page(channel_id):
    ch = generator.load_channel(channel_id)
    if not ch:
        return "Channel not found", 404

    if request.method == "POST":
        scenes_json = request.form.get("scenes_json", "[]")
        scenes = json.loads(scenes_json)
        title = request.form.get("title", "Untitled")
        topic = request.form.get("topic", "")
        gen_short = request.form.get("generate_short") == "on"

        # Create progress queue
        job_id = f"{channel_id}_{id(threading.current_thread())}"
        q = queue.Queue()
        _progress_queues[job_id] = q

        def progress_cb(step, msg):
            q.put({"step": step, "message": msg})

        def run_generation():
            try:
                result = generator.generate_video(
                    ch, scenes, title, topic, API_KEYS,
                    generate_short=gen_short, progress=progress_cb,
                )
                _generation_results[job_id] = result
                q.put({"step": "complete", "message": "done", "result": result})
            except Exception as e:
                q.put({"step": "error", "message": str(e)})

        t = threading.Thread(target=run_generation, daemon=True)
        t.start()

        return render_template("generate.html", channel=ch, job_id=job_id, title=title)

    return redirect(url_for("channel_workspace", channel_id=channel_id))


@app.route("/progress/<job_id>")
def progress_stream(job_id):
    def event_stream():
        q = _progress_queues.get(job_id)
        if not q:
            yield f"data: {json.dumps({'step': 'error', 'message': 'Job not found'})}\n\n"
            return
        while True:
            try:
                msg = q.get(timeout=30)
                yield f"data: {json.dumps(msg)}\n\n"
                if msg.get("step") in ("complete", "error"):
                    break
            except queue.Empty:
                yield f"data: {json.dumps({'step': 'heartbeat', 'message': 'waiting...'})}\n\n"

    return Response(event_stream(), mimetype="text/event-stream")


@app.route("/video/<channel_id>/<dir_name>/<filename>")
def serve_video(channel_id, dir_name, filename):
    video_dir = generator.OUTPUT_DIR / channel_id / dir_name
    return send_from_directory(str(video_dir), filename)


@app.route("/api/youtube_toggle", methods=["POST"])
def youtube_toggle():
    data = request.json
    meta = generator.update_video_meta(
        data["channel_id"], data["dir_name"],
        {"youtube_uploaded": data["uploaded"]}
    )
    return jsonify({"ok": True})


@app.route("/api/delete_video", methods=["POST"])
def delete_video():
    """Delete a video and all its files from the output directory."""
    data = request.json
    channel_id = data.get("channel_id")
    dir_name = data.get("dir_name")
    if not channel_id or not dir_name:
        return jsonify({"ok": False, "error": "Missing channel_id or dir_name"})

    # Safety: ensure dir_name doesn't contain path traversal
    if ".." in dir_name or "/" in dir_name:
        return jsonify({"ok": False, "error": "Invalid directory name"})

    target = generator.OUTPUT_DIR / channel_id / dir_name
    if target.exists() and target.is_dir():
        import shutil
        shutil.rmtree(str(target))
        log.info(f"Deleted video: {channel_id}/{dir_name}")
        return jsonify({"ok": True})
    else:
        return jsonify({"ok": False, "error": "Directory not found"})


@app.route("/api/elevenlabs_quota")
def elevenlabs_quota():
    """Return ElevenLabs character usage for quota display in UI."""
    try:
        import httpx
        resp = httpx.get(
            "https://api.elevenlabs.io/v1/user",
            headers={"xi-api-key": API_KEYS["elevenlabs"]},
            timeout=10,
        )
        if resp.status_code == 200:
            sub = resp.json().get("subscription", {})
            char_used = sub.get("character_count", 0)
            char_limit = sub.get("character_limit", 0)
            return jsonify({
                "ok": True,
                "used": char_used,
                "limit": char_limit,
                "remaining": char_limit - char_used,
            })
        return jsonify({"ok": False, "error": f"HTTP {resp.status_code}"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/channel/<channel_id>/compile", methods=["GET", "POST"])
def compile_videos(channel_id):
    ch = generator.load_channel(channel_id)
    if not ch:
        return "Channel not found", 404

    videos = generator.list_videos(channel_id)
    # Only show videos that have actual video files
    available = [v for v in videos if v.get("has_video")]

    if request.method == "POST":
        selected = request.form.getlist("video_dirs")
        comp_title = request.form.get("title", f"{ch['channel_name']} Compilation")
        if len(selected) < 2:
            return render_template("compile.html", channel=ch, videos=available,
                                   error="Select at least 2 videos to compile.")

        # Create progress queue
        job_id = f"compile_{channel_id}_{id(threading.current_thread())}"
        q = queue.Queue()
        _progress_queues[job_id] = q

        def progress_cb(step, msg):
            q.put({"step": step, "message": msg})

        def run_compile():
            try:
                result = generator.compile_videos(ch, selected, comp_title, progress=progress_cb)
                _generation_results[job_id] = result
                q.put({"step": "complete", "message": "done", "result": result})
            except Exception as e:
                q.put({"step": "error", "message": str(e)})

        t = threading.Thread(target=run_compile, daemon=True)
        t.start()

        return render_template("generate.html", channel=ch, job_id=job_id, title=comp_title)

    return render_template("compile.html", channel=ch, videos=available)


@app.route("/channel/<channel_id>/regenerate/<dir_name>")
def regenerate_page(channel_id, dir_name):
    ch = generator.load_channel(channel_id)
    if not ch:
        return "Channel not found", 404

    # Load the original plan
    plan_path = generator.OUTPUT_DIR / channel_id / dir_name / "plan.json"
    if not plan_path.exists():
        return "Original plan not found — cannot regenerate", 404

    plan = json.loads(plan_path.read_text())
    script = {
        "title": plan["title"],
        "subject": plan.get("subject", ""),
        "scenes": plan["scenes"],
    }

    return render_template(
        "script.html", channel=ch, script=script,
        topic=plan.get("subject", ""), regenerating=True,
    )


@app.route("/channel/<channel_id>/youtube/<dir_name>")
def youtube_meta_page(channel_id, dir_name):
    ch = generator.load_channel(channel_id)
    if not ch:
        return "Channel not found", 404

    yt_path = generator.OUTPUT_DIR / channel_id / dir_name / "youtube.json"
    meta_path = generator.OUTPUT_DIR / channel_id / dir_name / "metadata.json"

    yt_meta = {}
    title = "Video"
    if yt_path.exists():
        yt_meta = json.loads(yt_path.read_text())
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        title = meta.get("title", "Video")
        if not yt_meta and meta.get("youtube_meta"):
            yt_meta = meta["youtube_meta"]

    return render_template("youtube.html", channel=ch, yt_meta=yt_meta, title=title, dir_name=dir_name)


# --- YouTube Upload ---

@app.route("/youtube/auth")
def youtube_auth():
    """Start YouTube OAuth2 flow."""
    if youtube_upload.is_authenticated():
        return redirect(url_for("dashboard"))
    try:
        youtube_upload.run_local_auth()
        return redirect(url_for("dashboard"))
    except Exception as e:
        return f"YouTube auth failed: {e}", 500


@app.route("/youtube/status")
def youtube_status():
    """Check YouTube auth status and list available channels."""
    authenticated = youtube_upload.is_authenticated()
    channels = []
    if authenticated:
        try:
            channels = youtube_upload.list_channels()
        except Exception as e:
            channels = [{"error": str(e)}]
    return jsonify({
        "authenticated": authenticated,
        "has_client_secret": youtube_upload.CLIENT_SECRET_PATH.exists(),
        "channels": channels,
    })


@app.route("/api/youtube_upload", methods=["POST"])
def youtube_upload_video():
    """Upload a video to YouTube. Called from the channel video list."""
    data = request.json
    channel_id = data.get("channel_id")
    dir_name = data.get("dir_name")
    privacy = data.get("privacy", "private")

    if not channel_id or not dir_name:
        return jsonify({"ok": False, "error": "Missing channel_id or dir_name"})

    if not youtube_upload.is_authenticated():
        return jsonify({"ok": False, "error": "Not authenticated with YouTube. Visit /youtube/auth first."})

    # Load video metadata
    video_dir = generator.OUTPUT_DIR / channel_id / dir_name
    video_path = video_dir / "video.mp4"
    meta_path = video_dir / "metadata.json"
    yt_meta_path = video_dir / "youtube.json"
    thumb_path = video_dir / "thumbnail.png"

    if not video_path.exists():
        return jsonify({"ok": False, "error": "Video file not found"})

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

    # Add hashtags to top of description (YouTube shows these above the title)
    hashtags = yt_meta.get("hashtags", [])
    if hashtags:
        description = " ".join(hashtags) + "\n\n" + description

    tags = yt_meta.get("tags", [])
    category = youtube_upload.CATEGORIES.get(
        yt_meta.get("category", meta.get("youtube", {}).get("category", "Entertainment")),
        "24"
    )

    try:
        result = youtube_upload.upload_video(
            video_path=str(video_path),
            title=title,
            description=description,
            tags=tags,
            category_id=category,
            privacy=privacy,
            thumbnail_path=str(thumb_path) if thumb_path.exists() else None,
        )

        # Update metadata
        meta["youtube_uploaded"] = True
        meta["youtube_video_id"] = result["video_id"]
        meta["youtube_url"] = result["url"]
        meta["youtube_privacy"] = privacy
        meta_path.write_text(json.dumps(meta, indent=2))

        # Also upload Short if it exists and was requested
        short_result = None
        short_path = video_dir / "short.mp4"
        if data.get("include_short", True) and short_path.exists():
            short_meta = meta.get("short", {})
            short_title = short_meta.get("suggested_title", title + " #Shorts")
            if not short_title.endswith("#Shorts"):
                short_title += " #Shorts"

            short_caption = short_meta.get("suggested_caption", "")
            short_hashtags = short_meta.get("suggested_hashtags", ["#Shorts"])
            if "#Shorts" not in short_hashtags:
                short_hashtags.insert(0, "#Shorts")

            short_desc = " ".join(short_hashtags) + "\n\n"
            if short_caption:
                short_desc += short_caption + "\n\n"
            short_desc += f"Full video: {result['url']}\n\n"
            short_desc += f"Subscribe for more from {meta.get('channel_name', '')}!"

            short_tags = tags + ["Shorts", "YouTube Shorts"]

            try:
                short_result = youtube_upload.upload_video(
                    video_path=str(short_path),
                    title=short_title,
                    description=short_desc,
                    tags=short_tags,
                    category_id=category,
                    privacy=privacy,
                )
                meta["youtube_short_id"] = short_result["video_id"]
                meta["youtube_short_url"] = short_result["url"]
                meta_path.write_text(json.dumps(meta, indent=2))
            except Exception as e:
                log.warning(f"Short upload failed: {e}")

        response = {"ok": True, "video_id": result["video_id"], "url": result["url"]}
        if short_result:
            response["short_id"] = short_result["video_id"]
            response["short_url"] = short_result["url"]
        return jsonify(response)

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


if __name__ == "__main__":
    app.run(debug=True, port=7749, host="0.0.0.0")
