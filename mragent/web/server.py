"""MRAgent Web UI — aiohttp server with chat + voice endpoints.

Routes:
  GET  /              → index.html
  GET  /static/*      → static assets
  POST /api/chat      → chat with agent (JSON)
  WS   /ws            → streaming agent responses
  POST /api/voice     → audio → Groq transcription → text
"""

from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from aiohttp import WSMsgType, web
from loguru import logger

if TYPE_CHECKING:
    from mragent.agent.loop import AgentLoop

_STATIC_DIR = Path(__file__).parent / "static"


class WebServer:
    """Lightweight aiohttp-based web UI for MRAgent."""

    def __init__(
        self,
        agent: "AgentLoop",
        host: str = "127.0.0.1",
        port: int = 6326,
        groq_api_key: str | None = None,
    ):
        self.agent = agent
        self.host = host
        self.port = port
        self.groq_api_key = groq_api_key
        self._runner: web.AppRunner | None = None
        self._app = self._build_app()
        # Config path — used by /api/config endpoint
        from mragent.config.loader import get_config_path
        self._config_path = get_config_path()

    def _build_app(self) -> web.Application:
        app = web.Application(client_max_size=50 * 1024 * 1024)  # 50MB for audio
        app.router.add_get("/", self._handle_index)
        app.router.add_get("/static/{filename:.*}", self._handle_static)
        app.router.add_post("/api/chat", self._handle_chat)
        app.router.add_get("/ws", self._handle_ws)
        app.router.add_post("/api/voice", self._handle_voice)
        app.router.add_post("/api/upload", self._handle_upload)
        app.router.add_get("/api/status", self._handle_status)
        app.router.add_get("/api/models", self._handle_models)
        app.router.add_post("/api/config", self._handle_config)
        app.router.add_get("/api/history", self._handle_get_history)
        app.router.add_delete("/api/history", self._handle_delete_history)
        app.router.add_get("/api/sessions", self._handle_get_sessions)
        return app

    # ------------------------------------------------------------------
    # Static file handlers
    # ------------------------------------------------------------------

    async def _handle_index(self, request: web.Request) -> web.Response:
        index_path = _STATIC_DIR / "index.html"
        if not index_path.exists():
            return web.Response(text="MRAgent Web UI - index.html not found", status=404)
        return web.Response(
            body=index_path.read_bytes(),
            content_type="text/html",
            headers={"Cache-Control": "no-cache"},
        )

    async def _handle_static(self, request: web.Request) -> web.Response:
        filename = request.match_info["filename"]
        file_path = _STATIC_DIR / filename
        if not file_path.exists() or not file_path.is_file():
            return web.Response(text="Not found", status=404)
        # Basic path traversal protection
        try:
            file_path.relative_to(_STATIC_DIR)
        except ValueError:
            return web.Response(text="Forbidden", status=403)
        mime, _ = mimetypes.guess_type(str(file_path))
        return web.Response(
            body=file_path.read_bytes(),
            content_type=mime or "application/octet-stream",
        )

    # ------------------------------------------------------------------
    # Chat API
    # ------------------------------------------------------------------

    async def _handle_chat(self, request: web.Request) -> web.Response:
        """POST /api/chat — {"message": "...", "session": "web:user"}"""
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        message = (body.get("message") or "").strip()
        session_id = body.get("session", "web:user")
        media = body.get("media") or []
        if not message and not media:
            return web.json_response({"error": "Empty message"}, status=400)

        try:
            response = await self.agent.process_direct(
                message,
                session_key=session_id,
                channel="web",
                chat_id="user",
                media=media,
            )
            return web.json_response({"response": response or ""})
        except Exception as e:
            logger.error("Chat error: {}", e)
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_get_history(self, request: web.Request) -> web.Response:
        """GET /api/history?session=web:user — returns chat history."""
        session_id = request.rel_url.query.get("session", "web:user").strip()
        if not session_id:
            return web.json_response({"error": "Empty session ID"}, status=400)

        try:
            # We bypass the LLM and just return raw text messages directly
            session = self.agent.sessions.get_or_create(session_id)
            messages = []
            for m in session.messages:
                # We only want to show user inputs and final agent text responses
                # We hide tool_calls and tool_results from the basic web UI view
                if m.get("role") in ("user", "assistant"):
                    content = (m.get("content") or "").strip()
                    if content and not m.get("tool_calls"):
                        messages.append({
                            "role": "agent" if m["role"] == "assistant" else "user",
                            "content": content
                        })
            return web.json_response({"messages": messages})
        except Exception as e:
            logger.error("History fetch error: {}", e)
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_delete_history(self, request: web.Request) -> web.Response:
        """DELETE /api/history?session=web:user — clears the session."""
        session_id = request.rel_url.query.get("session", "web:user").strip()
        if not session_id:
            return web.json_response({"error": "Empty session ID"}, status=400)

        try:
            self.agent.sessions.delete_session(session_id)
            return web.json_response({"status": "cleared"})
        except Exception as e:
            logger.error("History delete error: {}", e)
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_get_sessions(self, request: web.Request) -> web.Response:
        """GET /api/sessions — returns lightweight metadata of all 'web:' sessions."""
        try:
            all_sessions = self.agent.sessions.list_sessions()
            # Filter to only 'web:' prefixed sessions to keep matrix/cli sessions out
            web_sessions = [s for s in all_sessions if s.get("key", "").startswith("web:")]
            return web.json_response({"sessions": web_sessions})
        except Exception as e:
            logger.error("Sessions list error: {}", e)
            return web.json_response({"error": str(e)}, status=500)

    # ------------------------------------------------------------------
    # WebSocket streaming
    # ------------------------------------------------------------------

    async def _handle_ws(self, request: web.Request) -> web.WebSocketResponse:
        """WS /ws — streams agent messages back to browser."""
        ws = web.WebSocketResponse(heartbeat=30)
        await ws.prepare(request)

        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except json.JSONDecodeError:
                        await ws.send_json({"error": "Invalid JSON"})
                        continue

                    message = (data.get("message") or "").strip()
                    session_id = data.get("session", "web:user")
                    media = data.get("media") or []
                    if not message and not media:
                        continue

                    chunks: list[str] = []

                    async def _progress(content: str, *, tool_hint: bool = False) -> None:
                        if not ws.closed:
                            await ws.send_json({
                                "type": "progress",
                                "content": content,
                                "tool_hint": tool_hint,
                            })

                    try:
                        response = await self.agent.process_direct(
                            message,
                            session_key=session_id,
                            channel="web",
                            chat_id="user",
                            on_progress=_progress,
                            media=media,
                        )
                        if not ws.closed:
                            await ws.send_json({"type": "response", "content": response or ""})
                    except Exception as e:
                        if not ws.closed:
                            await ws.send_json({"type": "error", "content": str(e)})

                elif msg.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
                    break
        except Exception as e:
            logger.error("WebSocket error: {}", e)

        return ws

    # ------------------------------------------------------------------
    # Voice transcription (Groq Whisper)
    # ------------------------------------------------------------------

    async def _handle_voice(self, request: web.Request) -> web.Response:
        """POST /api/voice — multipart audio → transcribed text."""
        if not self.groq_api_key:
            return web.json_response(
                {"error": "Groq API key not configured. Set providers.groq.api_key in config."},
                status=503,
            )

        try:
            reader = await request.multipart()
            field = await reader.next()
            if field is None:
                return web.json_response({"error": "No audio data"}, status=400)

            # Save to temp file
            suffix = ".webm"
            if field.filename:
                suffix = Path(field.filename).suffix or ".webm"

            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp_path = Path(tmp.name)
                while True:
                    chunk = await field.read_chunk(65536)
                    if not chunk:
                        break
                    tmp.write(chunk)

            try:
                from mragent.providers.transcription import GroqTranscriptionProvider
                transcriber = GroqTranscriptionProvider(api_key=self.groq_api_key)
                text = await transcriber.transcribe(tmp_path)
                return web.json_response({"text": text})
            finally:
                tmp_path.unlink(missing_ok=True)

        except Exception as e:
            logger.error("Voice transcription error: {}", e)
            return web.json_response({"error": str(e)}, status=500)

    # ------------------------------------------------------------------
    # File upload
    # ------------------------------------------------------------------

    async def _handle_upload(self, request: web.Request) -> web.Response:
        """POST /api/upload — multipart file upload."""
        try:
            reader = await request.multipart()
            field = await reader.next()
            if field is None:
                return web.json_response({"error": "No file data"}, status=400)

            filename = field.filename or "uploaded_file"
            
            # Save file to workspace
            uploads_dir = self.agent.workspace / "uploads"
            uploads_dir.mkdir(parents=True, exist_ok=True)
            
            # basic sanitization
            safe_name = "".join(c for c in filename if c.isalnum() or c in "._- ")
            file_path = uploads_dir / f"{int(asyncio.get_event_loop().time())}_{safe_name}"
            
            with open(file_path, "wb") as f:
                while True:
                    chunk = await field.read_chunk(65536)
                    if not chunk:
                        break
                    f.write(chunk)
            
            return web.json_response({
                "path": str(file_path.absolute()),
                "filename": safe_name,
            })

        except Exception as e:
            logger.error("File upload error: {}", e)
            return web.json_response({"error": str(e)}, status=500)

    # ------------------------------------------------------------------
    # Status / Config / Model-list endpoints
    # ------------------------------------------------------------------

    async def _handle_status(self, request: web.Request) -> web.Response:
        """GET /api/status — returns current model + provider name."""
        model = getattr(self.agent, "model", None) or ""
        return web.json_response({"model": model, "status": "ok"})

    async def _handle_models(self, request: web.Request) -> web.Response:
        """GET /api/models?key=nvapi-...&family=meta/ — proxy NVIDIA model list.

        Args:
          key    — NVIDIA API key (nvapi-...)
          family — optional prefix filter: 'meta/', 'qwen/', 'nvidia/', etc.
        """
        api_key = request.rel_url.query.get("key", "").strip()
        family = request.rel_url.query.get("family", "").strip()

        if not api_key:
            try:
                from mragent.config.loader import load_config
                cfg = load_config(self._config_path)
                api_key = cfg.providers.nvidia_nim.api_key or ""
            except Exception:
                pass

        if not api_key:
            return web.json_response({"error": "Missing 'key' parameter and no key configured"}, status=400)
        if not api_key.startswith("nvapi-"):
            return web.json_response({"error": "Key must start with 'nvapi-'"}, status=400)

        import httpx
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    "https://integrate.api.nvidia.com/v1/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=20.0,
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as e:
            return web.json_response(
                {"error": f"NVIDIA API error: {e.response.status_code}"},
                status=502,
            )
        except Exception as e:  # noqa: BLE001
            logger.error("Model list fetch error: {}", e)
            return web.json_response({"error": str(e)}, status=502)

        models = data.get("data", [])
        if family:
            models = [m for m in models if m.get("id", "").startswith(family)]

        # Sort by id, return only id + owned_by for compactness
        result = sorted(
            [{"id": m["id"], "owned_by": m.get("owned_by", "")} for m in models],
            key=lambda x: x["id"].lower(),
        )
        return web.json_response({"models": result, "total": len(result)})

    async def _handle_config(self, request: web.Request) -> web.Response:
        """POST /api/config — write nvidia_nim api_key + model to config file.

        Localhost-only: rejects requests from non-loopback IPs.
        Body: {"api_key": "nvapi-...", "model": "meta/llama-..."}
        """
        # Security: only accept from loopback
        peer = request.remote or ""
        if peer not in ("127.0.0.1", "::1", "localhost"):
            return web.json_response({"error": "Forbidden"}, status=403)

        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        api_key = (body.get("api_key") or "").strip()
        model = (body.get("model") or "").strip()

        if not api_key and not model:
            return web.json_response({"error": "Provide at least api_key or model"}, status=400)

        if api_key and not api_key.startswith("nvapi-"):
            return web.json_response({"error": "Key must start with 'nvapi-'"}, status=400)

        try:
            from mragent.config.loader import load_config, save_config
            cfg = load_config(self._config_path)
            if api_key:
                cfg.providers.nvidia_nim.api_key = api_key
                masked = api_key[:8] + "****"
                logger.info("Config updated: nvidia_nim.api_key = {}", masked)
            if model:
                cfg.agents.defaults.model = model
                logger.info("Config updated: agents.defaults.model = {}", model)
            save_config(cfg, self._config_path)
        except Exception as e:  # noqa: BLE001
            logger.error("Config save error: {}", e)
            return web.json_response({"error": str(e)}, status=500)

        return web.json_response({"ok": True, "model": model or cfg.agents.defaults.model})

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the aiohttp web server."""
        self._runner = web.AppRunner(self._app, access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        logger.info("MRAgent Web UI started at http://{}:{}", self.host, self.port)
        # Keep running until cancelled
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            pass

    async def stop(self) -> None:
        """Stop the aiohttp web server."""
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
