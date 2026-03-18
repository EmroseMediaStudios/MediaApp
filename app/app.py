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


if __name__ == "__main__":
    app.run(debug=True, port=7749, host="0.0.0.0")
