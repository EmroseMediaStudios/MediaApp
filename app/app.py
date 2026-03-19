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


if __name__ == "__main__":
    app.run(debug=True, port=7749, host="0.0.0.0")
