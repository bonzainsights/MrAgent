"""
MRAgent — Web UI
Flask-based browser interface with SSE streaming, chat history,
model/mode settings, code highlighting, and responsive design.

Created: 2026-02-15
Updated: 2026-02-16 — Full redesign with sidebar, settings, code copy

Usage:
    python main.py                         # CLI + Web on port 16226
    python main.py --mode web --port 8080  # Web-only, custom port
"""

import os
import json
import queue
import threading
from functools import wraps

from flask import Flask, render_template_string, request, Response, jsonify, send_from_directory

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        expected_token = os.getenv("MRAGENT_ACCESS_TOKEN")
        if not expected_token:
            return f(*args, **kwargs)
        
        auth_header = request.headers.get("Authorization")
        if not auth_header or auth_header != f"Bearer {expected_token}":
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated

from agents.core import AgentCore
from agents.model_selector import ModelSelector
from config.settings import MODEL_REGISTRY, IMAGES_DIR, DATA_DIR
from memory.chat_store import ChatStore
from utils.logger import get_logger

logger = get_logger("ui.web")

# The agent instance (created per-app)
_agent = None
_chat_store = None
_event_queues = {}
_approval_events = {}

def _on_event(event_type: str, data: str):
    """Callback to push events to the SSE queue."""
    if _agent and _agent.chat_id in _event_queues:
        _event_queues[_agent.chat_id].put({"type": event_type, "data": data})

def web_approval_callback(prompt: str) -> bool:
    """Blocks the execution thread until the UI sends an approval or rejection."""
    chat_id = _agent.chat_id
    if chat_id not in _approval_events:
        _approval_events[chat_id] = queue.Queue()
        
    # Wait for the user to approve/reject via the API endpoint
    try:
        response = _approval_events[chat_id].get(timeout=300) # 5 min timeout
        return response == "approve"
    except queue.Empty:
        return False

def create_app() -> Flask:
    """Create and configure the Flask app."""
    global _agent, _chat_store

    app = Flask(__name__)
    _agent = AgentCore()
    _agent.on_response(_on_event)
    _agent.approval_callback = web_approval_callback
    _chat_store = ChatStore()

    @app.route("/")
    def index():
        return render_template_string(HTML_TEMPLATE)

    @app.route("/api/login", methods=["POST"])
    def login():
        expected_token = os.getenv("MRAGENT_ACCESS_TOKEN")
        if not expected_token:
            return jsonify({"status": "ok"})
        data = request.json or {}
        if data.get("token") == expected_token or (request.headers.get("Authorization") == f"Bearer {expected_token}"):
            return jsonify({"status": "ok"})
        return jsonify({"error": "Unauthorized"}), 401

    @app.route("/api/approve", methods=["POST"])
    @require_auth
    def approve_action():
        """Handle human-in-the-loop approval requests."""
        data = request.json
        action = data.get("action") # "approve" or "reject"
        chat_id = _agent.chat_id
        if chat_id not in _approval_events:
            _approval_events[chat_id] = queue.Queue()
        _approval_events[chat_id].put(action)
        return jsonify({"status": "ok"})

    @app.route("/api/chat", methods=["POST"])
    @require_auth
    def chat():
        """Handle a chat message (non-streaming response)."""
        data = request.json
        message = data.get("message", "")
        if not message:
            return jsonify({"error": "No message provided"}), 400

        try:
            response = _agent.chat(message, stream=False)
            _chat_store.save_message(_agent.chat_id, "user", message)
            _chat_store.save_message(_agent.chat_id, "assistant", response)
            return jsonify({
                "response": response,
                "stats": _agent.get_stats(),
            })
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/chat/stream", methods=["POST"])
    @require_auth
    def chat_stream():
        """Handle a chat message with SSE streaming."""
        data = request.json
        message = data.get("message", "")
        if not message:
            return jsonify({"error": "No message provided"}), 400

        chat_id = _agent.chat_id
        if chat_id not in _event_queues:
            _event_queues[chat_id] = queue.Queue()
            
        q = _event_queues[chat_id]
        
        # Clear queue
        while not q.empty():
            q.get()

        def generate():
            def _run():
                try:
                    response = _agent.chat(message, stream=True)
                    _chat_store.save_message(_agent.chat_id, "user", message)
                    _chat_store.save_message(_agent.chat_id, "assistant", response)
                    # Auto-title chat from first user message
                    chat_info = _chat_store.get_chat(_agent.chat_id)
                    if chat_info and chat_info.get("title") == "New Chat":
                        title = message[:50] + ("..." if len(message) > 50 else "")
                        _chat_store.update_chat_title(_agent.chat_id, title)
                    q.put({"type": "done", "data": response})
                except Exception as e:
                    q.put({"type": "error", "data": str(e)})

            thread = threading.Thread(target=_run, daemon=True)
            thread.start()

            while True:
                try:
                    event = q.get(timeout=60)
                    yield f"data: {json.dumps(event)}\n\n"
                    if event["type"] in ("done", "error"):
                        break
                except queue.Empty:
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"

        return Response(generate(), mimetype="text/event-stream")

    @app.route("/api/stats")
    @require_auth
    def stats():
        return jsonify(_agent.get_stats())

    @app.route("/api/upload", methods=["POST"])
    @require_auth
    def upload_files():
        """Handle multiple file uploads from Web UI."""
        if 'files' not in request.files:
            return jsonify({"error": "No files provided"}), 400
        
        uploaded_files = request.files.getlist('files')
        results = []
        
        # Ensure uploads directory exists
        uploads_dir = DATA_DIR / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        
        for file in uploaded_files:
            if file.filename == '':
                continue
                
            file_ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
            
            # Text based files
            if file_ext in ['txt', 'json', 'xml', 'csv', 'md', 'py', 'js', 'html', 'css', 'jsonl', 'yaml', 'yml']:
                content = file.read().decode('utf-8', errors='replace')
                results.append(f"[Attached File: {file.filename}]\n```\n{content}\n```")
            
            # PDF files
            elif file_ext == 'pdf':
                try:
                    import PyPDF2
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                    results.append(f"[Attached Document: {file.filename}]\n```\n{text}\n```")
                except Exception as e:
                    results.append(f"[Failed to parse PDF {file.filename}: {e}]")
                    logger.error(f"PDF Parsing error on {file.filename}: {e}")
            
            # Image files
            elif file_ext in ['png', 'jpg', 'jpeg', 'webp']:
                # Save locally for the agent to access via absolute path
                file_path = uploads_dir / file.filename
                # Handle filename collisions
                counter = 1
                while file_path.exists():
                    name, ext = file.filename.rsplit('.', 1) if '.' in file.filename else (file.filename, '')
                    file_path = uploads_dir / f"{name}_{counter}.{ext}"
                    counter += 1
                    
                file.save(str(file_path))
                results.append(f"[Attached Image: {file_path.absolute()}]")
                
            else:
                results.append(f"[Unsupported file attached: {file.filename}]")
                
        return jsonify({"status": "ok", "results": results})

    @app.route("/api/newchat", methods=["POST"])
    @require_auth
    def newchat():
        _agent.new_chat()
        return jsonify({"status": "ok", "chat_id": _agent.chat_id})

    @app.route("/api/images/<path:filename>")
    def serve_image(filename):
        """Serve generated images from the images directory."""
        return send_from_directory(str(IMAGES_DIR), filename)

    @app.route("/api/history")
    @require_auth
    def history():
        """List past chat sessions with preview."""
        chats = _chat_store.list_chats()
        result = []
        for c in chats:
            # Get first user message as preview if title is default
            preview = c.get("title", "New Chat")
            if preview == "New Chat":
                msgs = _chat_store.get_messages(c["id"], limit=1)
                if msgs:
                    preview = msgs[0]["content"][:50] + "..."
            result.append({
                "chat_id": c["id"],
                "title": preview,
                "updated_at": c.get("updated_at", ""),
                "message_count": c.get("token_count", 0),
            })
        return jsonify(result)

    @app.route("/api/history/<chat_id>")
    @require_auth
    def history_detail(chat_id):
        """Load messages for a specific chat and switch to it."""
        messages = _chat_store.get_messages(chat_id)
        # Switch agent to this chat so new messages go to the right place
        _agent.chat_id = chat_id
        return jsonify(messages)

    @app.route("/api/models")
    @require_auth
    def models():
        """List available LLM models grouped by category."""
        llm_models = []
        for name, info in MODEL_REGISTRY.items():
            if info.get("type") in ("llm", "vlm"):
                llm_models.append({
                    "name": name,
                    "categories": info.get("categories", []),
                    "description": info.get("description", ""),
                })
        current_mode = _agent.model_selector.mode
        current_model = _agent.model_override or ModelSelector.get_default_for_mode(
            current_mode if current_mode != "auto" else "thinking"
        )
        return jsonify({
            "models": llm_models,
            "current_model": current_model,
            "current_mode": current_mode,
        })

    @app.route("/api/model", methods=["POST"])
    @require_auth
    def set_model():
        """Switch to a specific model."""
        data = request.json
        model_name = data.get("model", "")
        if model_name not in MODEL_REGISTRY:
            return jsonify({"error": f"Unknown model: {model_name}"}), 400
        _agent.set_model(model_name)
        return jsonify({"status": "ok", "model": model_name})

    @app.route("/api/mode", methods=["POST"])
    @require_auth
    def set_mode():
        """Switch model selection mode."""
        data = request.json
        mode = data.get("mode", "")
        if mode not in ("auto", "thinking", "fast", "code"):
            return jsonify({"error": f"Invalid mode: {mode}"}), 400
        _agent.set_model_mode(mode)
        _agent.model_override = None
        return jsonify({
            "status": "ok",
            "mode": mode,
            "default_model": ModelSelector.get_default_for_mode(mode),
        })

    @app.route("/api/voices", methods=["GET"])
    @require_auth
    def get_voices():
        """Get available TTS voices."""
        from providers.tts import VOICES, DEFAULT_VOICE
        return jsonify({"voices": VOICES, "default": DEFAULT_VOICE})

    @app.route("/api/voice", methods=["POST"])
    @require_auth
    def voice():
        """Handle voice audio upload and transcription."""
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files["file"]
        voice_id = request.form.get("voice", None) # Get selected voice
        
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        try:
            # Read audio bytes
            audio_bytes = file.read()
            
            # Transcribe
            from providers.nvidia_stt import NvidiaSTTProvider
            stt = NvidiaSTTProvider()
            if not stt.available:
                return jsonify({"error": "STT not available (check GROQ_API_KEY)"}), 503
            
            transcript = stt.speech_to_text(audio_bytes)
            
            # Chat with agent (blocking for now)
            response = _agent.chat(transcript, stream=False)
            _chat_store.save_message(_agent.chat_id, "user", transcript)
            _chat_store.save_message(_agent.chat_id, "assistant", response)

            # Generate TTS
            audio_base64 = None
            try:
                import os
                import tempfile
                import base64
                from providers.tts import text_to_speech, DEFAULT_VOICE
                
                target_voice = voice_id if voice_id else DEFAULT_VOICE
                
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                    tts_path = tmp.name
                
                # We need to run async function in sync Flask route. 
                import asyncio
                asyncio.run(text_to_speech(response[:1000], tts_path, voice=target_voice))
                
                if os.path.exists(tts_path):
                    with open(tts_path, "rb") as audio_f:
                        audio_base64 = base64.b64encode(audio_f.read()).decode('utf-8')
                    os.remove(tts_path)
            except Exception as e:
                logger.error(f"Web TTS failed: {e}")

            return jsonify({
                "transcript": transcript,
                "response": response,
                "audio": audio_base64,
                "chat_id": _agent.chat_id,
                "voice_used": target_voice
            })

        except Exception as e:
            logger.error(f"Voice error: {e}")
            return jsonify({"error": str(e)}), 500

    logger.info("Flask web app created")
    return app


# ──────────────────────────────────────────────
# HTML Template (single file, no build step)
# ──────────────────────────────────────────────
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MRAgent — AI Assistant</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Fira+Code:wght@400;500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
  :root {
    --bg: #0a0a0f;
    --surface: #12121a;
    --surface2: #1a1a2e;
    --surface3: #222240;
    --border: #2a2a3e;
    --border-hover: #3a3a5e;
    --text: #e4e4ef;
    --text-dim: #8888aa;
    --text-muted: #555570;
    --accent: #6c63ff;
    --accent-light: #8b83ff;
    --accent-glow: rgba(108, 99, 255, 0.25);
    --success: #4ade80;
    --error: #f87171;
    --warning: #fbbf24;
    --gradient: linear-gradient(135deg, #6c63ff, #a855f7, #ec4899);
    --radius: 12px;
    --sidebar-w: 260px;
    --settings-w: 240px;
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    font-family: 'Inter', -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    height: 100vh;
    overflow: hidden;
  }

  /* ── LAYOUT ── */
  .app {
    display: grid;
    grid-template-columns: var(--sidebar-w) 1fr var(--settings-w);
    grid-template-rows: auto 1fr;
    height: 100vh;
  }

  /* ── HEADER ── */
  .header {
    grid-column: 1 / -1;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 20px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    z-index: 100;
  }

  .header-left { display: flex; align-items: center; gap: 12px; }

  .hamburger {
    display: none;
    background: none; border: none; color: var(--text);
    font-size: 1.4rem; cursor: pointer; padding: 4px;
  }

  .header h1 {
    font-size: 1.15rem; font-weight: 700;
    background: var(--gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }

  .header-right {
    display: flex; align-items: center; gap: 12px;
  }

  .model-badge {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.72rem;
    color: var(--accent-light);
    font-weight: 500;
    display: flex; align-items: center; gap: 6px;
  }

  .model-badge .dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--success);
    animation: pulse 2s infinite;
  }

  @keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.3;} }

  .status-text { font-size: 0.72rem; color: var(--text-dim); }

  /* ── SIDEBAR (Chat History) ── */
  .sidebar {
    background: var(--surface);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  .sidebar-header {
    padding: 14px 16px;
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; justify-content: space-between;
  }

  .sidebar-header h3 { font-size: 0.8rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 1px; }

  .new-chat-btn {
    background: var(--accent);
    border: none; border-radius: 6px;
    color: white; font-size: 0.75rem;
    padding: 5px 10px; cursor: pointer;
    transition: all 0.15s;
    font-weight: 500;
  }
  .new-chat-btn:hover { background: var(--accent-light); transform: scale(1.03); }

  .chat-list {
    flex: 1;
    overflow-y: auto;
    padding: 8px;
  }

  .chat-item {
    padding: 10px 12px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 0.82rem;
    color: var(--text-dim);
    transition: all 0.15s;
    margin-bottom: 2px;
    display: flex;
    align-items: center;
    gap: 8px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .chat-item:hover { background: var(--surface2); color: var(--text); }
  .chat-item.active { background: var(--surface3); color: var(--text); border-left: 2px solid var(--accent); }
  .chat-item .icon { flex-shrink: 0; }

  /* ── MAIN CHAT AREA ── */
  .main {
    display: flex;
    flex-direction: column;
    overflow: hidden;
    background: var(--bg);
  }

  .chat-area {
    flex: 1;
    overflow-y: auto;
    padding: 20px 24px;
    scroll-behavior: smooth;
  }

  .message {
    margin-bottom: 20px;
    display: flex;
    gap: 10px;
    animation: fadeIn 0.3s ease;
    max-width: 860px;
    margin-left: auto;
    margin-right: auto;
  }

  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .message .avatar {
    width: 30px; height: 30px;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.85rem; flex-shrink: 0;
    margin-top: 2px;
  }

  .message.user .avatar { background: var(--accent); }
  .message.assistant .avatar { background: var(--gradient); }
  .message.tool .avatar { background: var(--surface2); font-size: 0.75rem; }

  .message .content {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 12px 16px;
    max-width: calc(100% - 44px);
    line-height: 1.65;
    font-size: 0.88rem;
    overflow-wrap: break-word;
    word-break: break-word;
  }

  .message.user .content {
    background: var(--surface2);
    border-color: rgba(108, 99, 255, 0.3);
  }

  .message.tool .content {
    background: rgba(26, 26, 46, 0.5);
    border-color: var(--border);
    font-size: 0.8rem;
    color: var(--text-dim);
    font-style: italic;
  }

  /* ── MARKDOWN RENDERED CONTENT ── */
  .content h1,.content h2,.content h3,.content h4 {
    margin: 14px 0 8px 0; font-weight: 600;
  }
  .content h1 { font-size: 1.2rem; }
  .content h2 { font-size: 1.05rem; }
  .content h3 { font-size: 0.95rem; }
  .content p { margin: 6px 0; }
  .content ul,.content ol { margin: 6px 0 6px 20px; }
  .content li { margin: 3px 0; }
  .content strong { color: var(--accent-light); }
  .content a { color: var(--accent-light); text-decoration: none; }
  .content a:hover { text-decoration: underline; }
  .content blockquote {
    border-left: 3px solid var(--accent);
    padding: 4px 12px;
    margin: 8px 0;
    color: var(--text-dim);
    background: rgba(108,99,255,0.05);
    border-radius: 0 6px 6px 0;
  }
  .content hr { border: none; border-top: 1px solid var(--border); margin: 12px 0; }
  .content table {
    border-collapse: collapse; margin: 8px 0; width: 100%;
    font-size: 0.82rem;
  }
  .content th, .content td {
    border: 1px solid var(--border); padding: 6px 10px; text-align: left;
  }
  .content th { background: var(--surface2); font-weight: 600; }

  /* ── INLINE IMAGES ── */
  .content .chat-image {
    max-width: 100%;
    max-height: 420px;
    border-radius: var(--radius);
    border: 1px solid var(--border);
    margin: 10px 0;
    cursor: pointer;
    transition: transform 0.2s, box-shadow 0.2s;
    display: block;
  }
  .content .chat-image:hover {
    transform: scale(1.02);
    box-shadow: 0 4px 20px rgba(108, 99, 255, 0.3);
  }
  .image-caption {
    font-size: 0.72rem;
    color: var(--text-dim);
    margin-top: 4px;
    font-style: italic;
  }

  /* ── CODE BLOCKS WITH COPY BUTTON ── */
  .code-block-wrapper {
    position: relative;
    margin: 10px 0;
  }

  .code-block-header {
    display: flex; justify-content: space-between; align-items: center;
    background: #1e1e30;
    border: 1px solid var(--border);
    border-bottom: none;
    border-radius: var(--radius) var(--radius) 0 0;
    padding: 6px 12px;
    font-size: 0.72rem;
    color: var(--text-dim);
  }

  .copy-btn {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--text-dim);
    padding: 3px 10px;
    border-radius: 4px;
    font-size: 0.7rem;
    cursor: pointer;
    transition: all 0.15s;
    font-family: inherit;
  }
  .copy-btn:hover { border-color: var(--accent); color: var(--text); }
  .copy-btn.copied { border-color: var(--success); color: var(--success); }

  .content pre {
    background: #0d0d18;
    border: 1px solid var(--border);
    border-radius: 0 0 var(--radius) var(--radius);
    padding: 14px;
    overflow-x: auto;
    margin: 0;
    font-size: 0.82rem;
    font-family: 'Fira Code', monospace;
    line-height: 1.5;
  }

  .content pre code {
    background: none; padding: 0; border-radius: 0;
    font-size: inherit;
  }

  .content code {
    background: #1e1e30;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.82rem;
    font-family: 'Fira Code', monospace;
  }

  /* ── INPUT AREA ── */
  .input-area {
    padding: 12px 20px 16px;
    background: var(--surface);
    border-top: 1px solid var(--border);
  }

  .input-container {
    max-width: 860px;
    margin: 0 auto;
  }

  .input-wrapper {
    display: flex;
    gap: 8px;
    align-items: flex-end;
  }

  .input-wrapper textarea {
    flex: 1;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 11px 14px;
    color: var(--text);
    font-size: 0.88rem;
    font-family: inherit;
    resize: none;
    min-height: 44px;
    max-height: 180px;
    outline: none;
    transition: border-color 0.2s, box-shadow 0.2s;
  }

  .input-wrapper textarea:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px var(--accent-glow);
  }

  .input-wrapper textarea::placeholder { color: var(--text-muted); }

  .send-btn {
    background: var(--gradient);
    border: none;
    border-radius: 10px;
    width: 44px; height: 44px;
    cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem;
    transition: transform 0.15s, opacity 0.15s;
    flex-shrink: 0;
    color: white;
  }
  .send-btn:hover { transform: scale(1.06); }
  .send-btn:active { transform: scale(0.94); }
  .send-btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .send-btn.stop { background: var(--error); border-radius: 10px; }
  .send-btn.stop:hover { transform: scale(1.06); }

  .input-hint {
    font-size: 0.68rem; color: var(--text-muted);
    margin-top: 6px; text-align: center;
    display: flex; align-items: center; justify-content: center; gap: 8px;
  }

  /* ── MIC BUTTON ── */
  .mic-btn {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 50%;
    width: 44px; height: 44px;
    display: flex; align-items: center; justify-content: center;
    cursor: pointer;
    color: var(--text-dim);
    transition: all 0.2s;
    font-size: 1.2rem;
    flex-shrink: 0;
  }
  .mic-btn:hover { color: var(--text); border-color: var(--accent); background: var(--surface3); }
  .mic-btn.recording {
    background: rgba(248, 113, 113, 0.2);
    border-color: var(--error);
    color: var(--error);
    animation: pulseMic 1.5s infinite;
  }
  @keyframes pulseMic { 0%{box-shadow:0 0 0 0 rgba(248,113,113,0.4);} 70%{box-shadow:0 0 0 10px rgba(248,113,113,0);} 100%{box-shadow:0 0 0 0 rgba(248,113,113,0);} }

  /* ── QUEUE BAR ── */
  .queue-bar {
    max-width: 860px;
    margin: 0 auto;
    padding: 0 0 8px 0;
    display: none;
  }
  .queue-bar.visible { display: block; }

  .queue-label {
    font-size: 0.7rem;
    color: var(--text-dim);
    margin-bottom: 5px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .queue-clear {
    background: none; border: none;
    color: var(--error); font-size: 0.7rem;
    cursor: pointer; padding: 0;
    text-decoration: underline;
  }

  .queue-items {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }

  .queue-pill {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 4px 10px;
    font-size: 0.75rem;
    color: var(--text-dim);
    display: flex;
    align-items: center;
    gap: 6px;
    max-width: 200px;
    animation: fadeIn 0.2s ease;
  }

  .queue-pill .pill-text {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .queue-pill .pill-remove {
    background: none; border: none;
    color: var(--text-muted); cursor: pointer;
    font-size: 0.8rem; padding: 0;
    line-height: 1;
  }
  .queue-pill .pill-remove:hover { color: var(--error); }

  /* ── SETTINGS PANEL ── */
  .settings {
    background: var(--surface);
    border-left: 1px solid var(--border);
    display: flex; flex-direction: column;
    overflow-y: auto;
    padding: 0;
  }

  .settings-section {
    padding: 14px 16px;
    border-bottom: 1px solid var(--border);
  }

  .settings-section h3 {
    font-size: 0.72rem;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 10px;
  }

  .setting-group { margin-bottom: 12px; }

  .setting-label {
    font-size: 0.78rem;
    color: var(--text-dim);
    margin-bottom: 5px;
    display: block;
  }

  .setting-select {
    width: 100%;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 10px;
    color: var(--text);
    font-size: 0.82rem;
    font-family: inherit;
    outline: none;
    cursor: pointer;
    appearance: none;
    -webkit-appearance: none;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%238888aa' stroke-width='2'%3E%3Cpath d='M6 9l6 6 6-6'/%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: right 10px center;
  }
  .setting-select:focus { border-color: var(--accent); }

  .mode-cards { display: flex; flex-direction: column; gap: 4px; }

  .mode-card {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 10px;
    border-radius: 8px;
    cursor: pointer;
    border: 1px solid transparent;
    transition: all 0.15s;
    font-size: 0.78rem;
  }
  .mode-card:hover { background: var(--surface2); }
  .mode-card.active {
    background: var(--surface3);
    border-color: var(--accent);
  }
  .mode-card .mode-icon { font-size: 1rem; }
  .mode-card .mode-name { font-weight: 500; color: var(--text); }
  .mode-card .mode-desc { font-size: 0.7rem; color: var(--text-dim); }

  /* ── TYPING INDICATOR ── */
  .typing-indicator {
    display: flex; gap: 4px; padding: 4px 0;
  }
  .typing-indicator span {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--accent);
    animation: typingBounce 1.4s infinite;
  }
  .typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
  .typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
  @keyframes typingBounce { 0%,80%,100%{transform:scale(0.6);opacity:0.3;} 40%{transform:scale(1);opacity:1;} }

  /* ── SCROLLBAR ── */
  ::-webkit-scrollbar { width: 5px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: var(--border-hover); }

  /* ── MOBILE RESPONSIVE ── */
  @media (max-width: 900px) {
    .app { grid-template-columns: 1fr; }
    .sidebar, .settings { display: none; }
    .sidebar.open, .settings.open {
      display: flex;
      position: fixed;
      top: 0; bottom: 0;
      width: 280px;
      z-index: 200;
      box-shadow: 4px 0 24px rgba(0,0,0,0.5);
    }
    .sidebar.open { left: 0; }
    .settings.open { right: 0; }
    .hamburger { display: block; }
    .overlay {
      position: fixed; inset: 0;
      background: rgba(0,0,0,0.5);
      z-index: 199;
    }
  }

  @media (max-width: 600px) {
    .message .content { max-width: 100%; font-size: 0.84rem; }
    .chat-area { padding: 12px; }
    .header h1 { font-size: 1rem; }
  }
</style>
</head>
<body>

<!-- Login Overlay -->
<div id="login-overlay" style="display:none; position:fixed; inset:0; background:var(--bg); z-index:9999; align-items:center; justify-content:center; flex-direction:column;">
  <div style="background:var(--surface); padding:30px; border-radius:12px; border:1px solid var(--border); text-align:center; width:300px;">
    <h2>🔒 Login Required</h2>
    <p style="font-size:0.8rem; color:var(--text-dim); margin-bottom:20px;">Please enter your access token</p>
    <input type="password" id="login-token" placeholder="MRAGENT_ACCESS_TOKEN" style="width:100%; padding:10px; margin-bottom:15px; background:var(--surface2); border:1px solid var(--border); border-radius:8px; color:var(--text); outline:none;" onkeydown="if(event.key==='Enter') submitLogin()">
    <button onclick="submitLogin()" style="width:100%; padding:10px; background:var(--accent); color:white; border:none; border-radius:8px; cursor:pointer; font-weight:600;">Authenticate</button>
    <div id="login-error" style="color:var(--error); font-size:0.8rem; margin-top:10px; display:none;">Invalid token</div>
  </div>
</div>

<div class="app">
  <!-- Header -->
  <div class="header">
    <div class="header-left">
      <button class="hamburger" onclick="toggleSidebar()">☰</button>
      <h1>🤖 MRAgent</h1>
    </div>
    <div class="header-right">
      <div class="model-badge">
        <span class="dot"></span>
        <span id="model-badge-text">auto</span>
      </div>
      <span class="status-text" id="status-text">Ready</span>
      <button class="hamburger" onclick="toggleSettings()" style="font-size:1.1rem;">⚙️</button>
    </div>
  </div>

  <!-- Sidebar: Chat History -->
  <div class="sidebar" id="sidebar">
    <div class="sidebar-header">
      <h3>💬 Chats</h3>
      <button class="new-chat-btn" onclick="newChat()">+ New</button>
    </div>
    <div class="chat-list" id="chatList">
      <div class="chat-item active"><span class="icon">💬</span> New conversation</div>
    </div>
  </div>

  <!-- Main Chat Area -->
  <div class="main">
    <div class="chat-area" id="chat">
      <div class="message assistant">
        <div class="avatar">🤖</div>
        <div class="content">
          <strong>Welcome to MRAgent!</strong><br>
          I'm your AI assistant powered by NVIDIA NIM. I can help with coding,
          answer questions, generate images, search the web, and more.
          <br><br>Type a message below to get started. Use the ⚙️ panel to switch models and modes.
        </div>
      </div>
    </div>

    <div class="input-area">
      <div class="input-container">
        <div class="queue-bar" id="queueBar">
          <div class="queue-label">
            <span>⏳ Queued messages</span>
            <button class="queue-clear" onclick="clearQueue()">Clear all</button>
          </div>
          <div class="queue-items" id="queueItems"></div>
        </div>
        <div class="input-wrapper">
          <textarea id="input" placeholder="Type your message... (Shift+Enter for new line)"
                    rows="1" onkeydown="handleKey(event)" oninput="autoResize(this)"></textarea>
          <label class="mic-btn" for="file-upload" title="Attach file" style="cursor: pointer; display: flex; align-items: center; justify-content: center;">📎</label>
          <input type="file" id="file-upload" style="display: none;" onchange="handleFileUpload(event)" accept=".txt,.json,.xml,.csv,.md,.py,.js,.html,.css,.jsonl,.yaml,.yml,.pdf,.png,.jpg,.jpeg,.webp" multiple>
          <button class="mic-btn" id="mic-btn" onclick="toggleRecording()" title="Hold to record">🎤</button>
          <button class="send-btn" id="sendBtn" onclick="handleSendClick()">➤</button>
        </div>
        <div class="input-hint">Enter to send · Shift+Enter for new line</div>
      </div>
    </div>
  </div>

  <!-- Settings Panel -->
  <div class="settings" id="settings">
    <div class="settings-section">
      <h3>Mode</h3>
      <div class="mode-cards" id="modeCards">
        <div class="mode-card active" data-mode="auto" onclick="setMode('auto')">
          <span class="mode-icon">🔄</span>
          <div><div class="mode-name">Auto</div><div class="mode-desc">Best model per message</div></div>
        </div>
        <div class="mode-card" data-mode="thinking" onclick="setMode('thinking')">
          <span class="mode-icon">🧠</span>
          <div><div class="mode-name">Thinking</div><div class="mode-desc">Deep reasoning</div></div>
        </div>
        <div class="mode-card" data-mode="fast" onclick="setMode('fast')">
          <span class="mode-icon">⚡</span>
          <div><div class="mode-name">Fast</div><div class="mode-desc">Quick replies</div></div>
        </div>
        <div class="mode-card" data-mode="code" onclick="setMode('code')">
          <span class="mode-icon">💻</span>
          <div><div class="mode-name">Code</div><div class="mode-desc">Programming tasks</div></div>
        </div>
      </div>
    </div>

    <div class="settings-section">
      <h3>Model</h3>
      <div class="setting-group">
        <label class="setting-label">Active model</label>
        <select class="setting-select" id="modelSelect" onchange="setModel(this.value)">
        </select>
      </div>

      <h3 style="margin-top:20px">Voice</h3>
      <div class="setting-group">
        <label class="setting-label">TTS Voice</label>
        <select class="setting-select" id="voiceSelect">
             <option value="">Loading...</option>
        </select>
      </div>
    </div>

    <div class="settings-section">
      <h3>Info</h3>
      <div style="font-size:0.75rem; color:var(--text-dim); line-height:1.6;">
        <div>Chat ID: <span id="infoChat" style="color:var(--text);">—</span></div>
        <div>Tools: <span id="infoTools" style="color:var(--text);">—</span></div>
        <div>Messages: <span id="infoMsgs" style="color:var(--text);">—</span></div>
      </div>
    </div>
  </div>
</div>

<!-- Mobile overlay -->
<div class="overlay" id="overlay" style="display:none;" onclick="closePanels()"></div>

<script>

// ── Fetch Wrapper for Auth ──
const originalFetch = window.fetch;
window.fetch = async function() {
    let [resource, config] = arguments;
    if (typeof resource === 'string' && resource.startsWith('/api/')) {
        config = config || {};
        config.headers = config.headers || {};
        const token = localStorage.getItem('mragent_token');
        if (token && !config.headers['Authorization']) {
            config.headers['Authorization'] = `Bearer ${token}`;
        }
    }
    const response = await originalFetch(resource, config);
    if (response.status === 401 && resource !== '/api/login') {
        document.getElementById('login-overlay').style.display = 'flex';
    }
    return response;
};

async function submitLogin() {
    const token = document.getElementById('login-token').value.trim();
    localStorage.setItem('mragent_token', token);
    
    try {
        const resp = await fetch('/api/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ token })
        });
        
        if (resp.status === 200) {
            document.getElementById('login-overlay').style.display = 'none';
            document.getElementById('login-error').style.display = 'none';
            // Reload initial data
            loadModels();
            loadHistory();
            updateStats();
            loadVoices();
        } else {
            document.getElementById('login-error').style.display = 'block';
            localStorage.removeItem('mragent_token');
        }
    } catch(e) {
        document.getElementById('login-error').style.display = 'block';
    }
}

// Initial auth check
fetch('/api/login', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({}) }).then(resp => {
  if (resp.status === 401) {
    document.getElementById('login-overlay').style.display = 'flex';
  } else {
     // ── Init if no auth required ──
     loadModels();
     loadHistory();
     updateStats();
     loadVoices();
  }
});

// ── Elements ──
const chat = document.getElementById('chat');
const input = document.getElementById('input');
const sendBtn = document.getElementById('sendBtn');
const statusText = document.getElementById('status-text');
const modelBadge = document.getElementById('model-badge-text');
const modelSelect = document.getElementById('modelSelect');

// ── Configure marked.js ──
marked.setOptions({
  breaks: true,
  gfm: true,
  highlight: function(code, lang) {
    if (lang && hljs.getLanguage(lang)) {
      return hljs.highlight(code, {language: lang}).value;
    }
    return hljs.highlightAuto(code).value;
  }
});

// ── Auto-resize textarea ──
function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 180) + 'px';
}

function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
}

// ── Format content with markdown + code highlighting ──
function formatContent(text) {
  // Fix broken image markdown: "! [text](url)" → "![text](url)"
  text = text.replace(/!\s+\[/g, '![');

  // Detect raw /api/images/filename references not already in markdown image syntax
  text = text.replace(
    /(?<!!)\[?(?:\/api\/images\/)([\w._-]+\.(?:png|jpg|jpeg|webp))/gi,
    function(match, filename) {
      return `\n![Generated image](/api/images/${filename})\n`;
    }
  );

  // Detect absolute file paths to data/images/ and convert to /api/images/
  text = text.replace(
    /(?:(?:\/[^\s]+\/)?data\/images\/)([\w._-]+\.(?:png|jpg|jpeg|webp))/gi,
    function(match, filename) {
      return `\n![Generated image](/api/images/${filename})\n`;
    }
  );

  // Use marked.js for markdown rendering
  let html = marked.parse(text);

  // Convert markdown images to styled chat images
  html = html.replace(
    /<img\s+src="([^"]+)"\s*alt="([^"]*)"[^>]*>/gi,
    function(match, src, alt) {
      return `<img class="chat-image" src="${src}" alt="${alt}" onclick="window.open('${src}','_blank')" loading="lazy">
        <div class="image-caption">${alt || 'Generated image'}</div>`;
    }
  );

  // Wrap code blocks with copy button
  html = html.replace(/<pre><code class="language-(\w+)">([\s\S]*?)<\/code><\/pre>/g,
    function(match, lang, code) {
      const id = 'code-' + Math.random().toString(36).substr(2, 8);
      return `<div class="code-block-wrapper">
        <div class="code-block-header">
          <span>${lang}</span>
          <button class="copy-btn" onclick="copyCode('${id}', this)">📋 Copy</button>
        </div>
        <pre><code id="${id}" class="language-${lang}">${code}</code></pre>
      </div>`;
    }
  );

  // Also handle code blocks without language
  html = html.replace(/<pre><code>([\s\S]*?)<\/code><\/pre>/g,
    function(match, code) {
      const id = 'code-' + Math.random().toString(36).substr(2, 8);
      return `<div class="code-block-wrapper">
        <div class="code-block-header">
          <span>code</span>
          <button class="copy-btn" onclick="copyCode('${id}', this)">📋 Copy</button>
        </div>
        <pre><code id="${id}">${code}</code></pre>
      </div>`;
    }
  );

  return html;
}

// ── Copy code to clipboard ──
function copyCode(id, btn) {
  const el = document.getElementById(id);
  if (!el) return;
  const text = el.textContent;
  navigator.clipboard.writeText(text).then(() => {
    btn.textContent = '✅ Copied!';
    btn.classList.add('copied');
    setTimeout(() => {
      btn.textContent = '📋 Copy';
      btn.classList.remove('copied');
    }, 2000);
  });
}

// ── Add message to chat ──
function addMessage(role, content) {
  const div = document.createElement('div');
  div.className = 'message ' + role;
  const avatar = role === 'user' ? '👤' : role === 'assistant' ? '🤖' : '🔧';
  const formatted = role === 'tool' ? `<em>${escapeHtml(content)}</em>` : formatContent(content);
  div.innerHTML = `<div class="avatar">${avatar}</div><div class="content">${formatted}</div>`;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

function escapeHtml(text) {
  const d = document.createElement('div');
  d.textContent = text;
  return d.innerHTML;
}

// ── Approval UI ──
function addApprovalUI(promptText) {
  const div = document.createElement('div');
  div.className = 'message tool';
  div.innerHTML = `
    <div class="avatar">⚠️</div>
    <div class="content" style="border:1px solid var(--accent); padding:10px; border-radius:8px;">
      <strong>Action Required</strong><br>
      <pre style="white-space:pre-wrap; font-size:0.8rem; margin:10px 0; font-family:monospace;">${escapeHtml(promptText)}</pre>
      <div style="display:flex; gap:10px; margin-top:10px;">
        <button onclick="respondApproval(this, 'approve')" style="padding:6px 12px; background:var(--accent); color:white; border:none; border-radius:6px; cursor:pointer;">Approve</button>
        <button onclick="respondApproval(this, 'reject')" style="padding:6px 12px; background:var(--surface3); color:white; border:none; border-radius:6px; cursor:pointer;">Reject</button>
      </div>
    </div>`;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

async function respondApproval(btn, action) {
  const container = btn.parentElement;
  container.innerHTML = `<span style="font-weight:bold; color: ${action === 'approve' ? 'var(--accent)' : 'var(--error)'}">${action === 'approve' ? '✅ Approved' : '❌ Rejected'}</span>`;
  try {
    await fetch('/api/approve', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action })
    });
  } catch(e) {}
}

// ── Message Queue & Abort ──
const messageQueue = [];
let isSending = false;
let abortController = null;
let stopRequested = false;

const queueBar = document.getElementById('queueBar');
const queueItems = document.getElementById('queueItems');

// ── File Upload ──
async function handleFileUpload(event) {
  const files = event.target.files;
  if (!files || files.length === 0) return;

  const formData = new FormData();
  for (let i = 0; i < files.length; i++) {
    formData.append("files", files[i]);
  }

  // Show a temporary uploading message
  const uploadingPill = document.createElement('div');
  uploadingPill.className = 'queue-pill';
  uploadingPill.innerHTML = `<span class="pill-text">⏳ Uploading ${files.length} file(s)...</span>`;
  queueBar.classList.add('visible');
  queueItems.appendChild(uploadingPill);

  try {
    const resp = await fetch('/api/upload', {
      method: 'POST',
      body: formData
    });
    
    if (resp.ok) {
      const data = await resp.json();
      if (data.results && data.results.length > 0) {
        let currentVal = input.value;
        data.results.forEach(res => {
            currentVal = currentVal + (currentVal ? '\n\n' : '') + res;
        });
        input.value = currentVal;
        autoResize(input);
      }
      if (data.error) {
        alert("Upload error: " + data.error);
      }
    } else {
      alert("Upload failed: " + resp.statusText);
    }
  } catch (err) {
    alert("Upload error: " + err.message);
  } finally {
    uploadingPill.remove();
    renderQueue();
    event.target.value = ''; // clear input
  }
}

// ── Handle send/stop button click ──
function handleSendClick() {
  if (isSending) { stopGeneration(); } else { send(); }
}

// ── Stop current generation ──
function stopGeneration() {
  stopRequested = true;
  if (abortController) abortController.abort();
  statusText.textContent = 'Stopped';
}

// ── Queue UI ──
function renderQueue() {
  if (messageQueue.length === 0) {
    queueBar.classList.remove('visible');
    return;
  }
  queueBar.classList.add('visible');
  queueItems.innerHTML = '';
  messageQueue.forEach((msg, i) => {
    const pill = document.createElement('div');
    pill.className = 'queue-pill';
    pill.innerHTML = `<span class="pill-text">${escapeHtml(msg)}</span>
      <button class="pill-remove" onclick="removeFromQueue(${i})" title="Remove">✕</button>`;
    queueItems.appendChild(pill);
  });
}

function removeFromQueue(index) {
  messageQueue.splice(index, 1);
  renderQueue();
}

function clearQueue() {
  messageQueue.length = 0;
  renderQueue();
}

// ── Send message (with queue) ──
async function send() {
  const msg = input.value.trim();
  if (!msg) return;
  input.value = '';
  input.style.height = 'auto';

  if (isSending) {
    // Queue the message, show in queue bar
    messageQueue.push(msg);
    renderQueue();
    return;
  }

  await processMessage(msg);

  // Process queued messages one by one
  while (messageQueue.length > 0 && !stopRequested) {
    const next = messageQueue.shift();
    renderQueue();
    await processMessage(next);
  }

  stopRequested = false;
  setButtonSend();
}

function setButtonSend() {
  sendBtn.textContent = '➤';
  sendBtn.title = 'Send';
  sendBtn.classList.remove('stop');
  sendBtn.disabled = false;
}

function setButtonStop() {
  sendBtn.textContent = '■';
  sendBtn.title = 'Stop generating';
  sendBtn.classList.add('stop');
  sendBtn.disabled = false;
}

async function processMessage(msg) {
  isSending = true;
  stopRequested = false;
  setButtonStop();
  statusText.textContent = 'Thinking...';

  addMessage('user', msg);

  // Typing indicator
  const assistantDiv = addMessage('assistant', '');
  const contentEl = assistantDiv.querySelector('.content');
  contentEl.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';

  let fullContent = '';
  abortController = new AbortController();

  try {
    const resp = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: msg}),
      signal: abortController.signal,
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream: true});

      const lines = buffer.split('\n');
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        try {
          const event = JSON.parse(line.slice(6));
          if (event.type === 'delta') {
            fullContent += event.data;
            contentEl.innerHTML = formatContent(fullContent);
            chat.scrollTop = chat.scrollHeight;
          } else if (event.type === 'model') {
            modelBadge.textContent = event.data;
          } else if (event.type === 'tool_start' || event.type === 'tool_result') {
            addMessage('tool', event.data);
          } else if (event.type === 'approval_required') {
            addApprovalUI(event.data);
          } else if (event.type === 'done') {
            if (!fullContent) {
              contentEl.innerHTML = formatContent(event.data);
            }
          } else if (event.type === 'error') {
            contentEl.innerHTML = '<span style="color:var(--error)">⚠️ Error: ' + escapeHtml(event.data) + '</span>';
          }
        } catch(e) {}
      }
    }
  } catch(e) {
    if (e.name === 'AbortError') {
      // User pressed stop
      if (fullContent) {
        contentEl.innerHTML = formatContent(fullContent + '\n\n*— generation stopped —*');
      } else {
        contentEl.innerHTML = '<span style="color:var(--warning)">⏹ Generation stopped</span>';
      }
    } else {
      contentEl.innerHTML = '<span style="color:var(--error)">⚠️ Connection error</span>';
    }
  }

  abortController = null;
  isSending = false;
  setButtonSend();
  statusText.textContent = 'Ready';
  input.focus();
  updateStats();
  loadHistory();
}

// ── New Chat ──
async function newChat() {
  await fetch('/api/newchat', {method: 'POST'});
  chat.innerHTML = '';
  addMessage('assistant', '🔄 **New conversation started.** How can I help?');
  loadHistory();
  closePanels();
}

// ── Load Chat History ──
let activeChatId = null;

async function loadHistory() {
  try {
    const resp = await fetch('/api/history');
    const chats = await resp.json();
    const list = document.getElementById('chatList');
    list.innerHTML = '';
    if (chats.length === 0) {
      list.innerHTML = '<div class="chat-item active"><span class="icon">💬</span> New conversation</div>';
      return;
    }
    chats.forEach((c, i) => {
      const div = document.createElement('div');
      const isActive = activeChatId ? c.chat_id === activeChatId : i === 0;
      div.className = 'chat-item' + (isActive ? ' active' : '');
      div.innerHTML = `<span class="icon">💬</span> ${escapeHtml(c.title || 'New Chat')}`;
      div.onclick = () => loadChat(c.chat_id, div);
      list.appendChild(div);
    });
  } catch(e) {}
}

async function loadChat(chatId, el) {
  try {
    activeChatId = chatId;
    const resp = await fetch(`/api/history/${chatId}`);
    const messages = await resp.json();
    chat.innerHTML = '';
    messages.forEach(m => addMessage(m.role, m.content));
    // Update active state
    document.querySelectorAll('.chat-item').forEach(item => item.classList.remove('active'));
    if (el) el.classList.add('active');
    closePanels();
  } catch(e) {}
}

// ── Model & Mode Switching ──
async function loadModels() {
  try {
    const resp = await fetch('/api/models');
    const data = await resp.json();

    modelSelect.innerHTML = '';
    data.models.forEach(m => {
      const opt = document.createElement('option');
      opt.value = m.name;
      opt.textContent = `${m.name} — ${m.description}`;
      if (m.name === data.current_model) opt.selected = true;
      modelSelect.appendChild(opt);
    });

    modelBadge.textContent = data.current_model + ' · ' + data.current_mode;

    // Highlight active mode
    document.querySelectorAll('.mode-card').forEach(card => {
      card.classList.toggle('active', card.dataset.mode === data.current_mode);
    });
  } catch(e) {}
}

async function setMode(mode) {
  try {
    const resp = await fetch('/api/mode', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({mode}),
    });
    const data = await resp.json();
    if (data.status === 'ok') {
      modelBadge.textContent = data.default_model + ' · ' + mode;
      document.querySelectorAll('.mode-card').forEach(card => {
        card.classList.toggle('active', card.dataset.mode === mode);
      });
      loadModels();
    }
  } catch(e) {}
}

async function setModel(name) {
  try {
    await fetch('/api/model', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({model: name}),
    });
    loadModels();
  } catch(e) {}
}

// ── Stats ──
async function updateStats() {
  try {
    const resp = await fetch('/api/stats');
    const s = await resp.json();
    document.getElementById('infoChat').textContent = (s.chat_id || '—').substring(0, 12) + '...';
    document.getElementById('infoTools').textContent = s.tools || '—';
    document.getElementById('infoMsgs').textContent = s.context?.message_count || '—';
  } catch(e) {}
}

// ── Mobile Panels ──
function toggleSidebar() {
  const sb = document.getElementById('sidebar');
  const st = document.getElementById('settings');
  st.classList.remove('open');
  sb.classList.toggle('open');
  document.getElementById('overlay').style.display = sb.classList.contains('open') ? 'block' : 'none';
}

function toggleSettings() {
  const sb = document.getElementById('sidebar');
  const st = document.getElementById('settings');
  sb.classList.remove('open');
  st.classList.toggle('open');
  document.getElementById('overlay').style.display = st.classList.contains('open') ? 'block' : 'none';
}

function closePanels() {
  document.getElementById('sidebar').classList.remove('open');
  document.getElementById('settings').classList.remove('open');
  document.getElementById('overlay').style.display = 'none';
}

// ── Init ──
input.focus();

// ── Voice Options ──
async function loadVoices() {
    try {
        const resp = await fetch('/api/voices');
        const data = await resp.json();
        const select = document.getElementById('voiceSelect');
        select.innerHTML = '';
        
        data.voices.forEach(v => {
            const opt = document.createElement('option');
            opt.value = v;
            // Make friendly names
            let name = v.split('-')[2].replace('Neural', '');
            opt.textContent = name;
            if (v === data.default) opt.selected = true;
            select.appendChild(opt);
        });
    } catch(e) { console.error("Failed to load voices", e); }
}

// ── VOICE RECORDING ──
let mediaRecorder;
let audioChunks = [];
let isRecording = false;

async function toggleRecording() {
  const micBtn = document.getElementById('mic-btn');
  
  if (!isRecording) {
    // Start Recording
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = event => {
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            await sendVoiceMessage(audioBlob);
            
            // Stop tracks to release mic
            stream.getTracks().forEach(track => track.stop());
        };

        mediaRecorder.start();
        isRecording = true;
        micBtn.classList.add('recording');
        micBtn.innerHTML = '⏹'; // Stop icon
    } catch (err) {
        console.error("Mic access denied:", err);
        alert("Could not access microphone.");
    }
  } else {
    // Stop Recording
    mediaRecorder.stop();
    isRecording = false;
    micBtn.classList.remove('recording');
    micBtn.innerHTML = '🎤';
  }
}

async function sendVoiceMessage(audioBlob) {
    const formData = new FormData();
    formData.append("file", audioBlob, "voice.wav");
    
    // Attach selected voice
    const voiceSelect = document.getElementById('voiceSelect');
    if (voiceSelect && voiceSelect.value) {
        formData.append("voice", voiceSelect.value);
    }

    // UI Feedback
    addMessage('user', "🎤 Voice message sent...");
    
    // Show typing
    const assistantDiv = addMessage('assistant', '');
    const contentEl = assistantDiv.querySelector('.content');
    contentEl.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';

    try {
        const response = await fetch('/api/voice', { method: 'POST', body: formData });
        const data = await response.json();
        
        if (data.error) {
           contentEl.innerHTML = `<span style="color:var(--error)">⚠️ Error: ${escapeHtml(data.error)}</span>`;
           return;
        }
        
        const userMsgs = document.querySelectorAll('.message.user .content');
        if (userMsgs.length > 0) {
            userMsgs[userMsgs.length - 1].innerText = `🎤 ${data.transcript}`;
        }

        // Play Audio if available
        if (data.audio) {
            const audio = new Audio("data:audio/mp3;base64," + data.audio);
            audio.play().catch(e => console.error("Audio play failed:", e));
        }

        // Update assistant message
        contentEl.innerHTML = formatContent(data.response);
        hljs.highlightAll();
        chat.scrollTop = chat.scrollHeight;
        
    } catch (err) {
        contentEl.innerHTML = `<span style="color:var(--error)">⚠️ Network error: ${escapeHtml(err)}</span>`;
    }
}
</script>
</body>
</html>"""
