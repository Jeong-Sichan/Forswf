"""Run with `python -m webapp` on an allocated GPU compute node."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response

from webapp.engine import ModelEngine, Settings
from webapp.schemas import ChatRequest, ChatResponse


STATIC = Path(__file__).parent / "static"
ASSETS = {
    "index.html": ("text/html; charset=utf-8", (STATIC / "index.html").read_bytes()),
    "styles.css": ("text/css; charset=utf-8", (STATIC / "styles.css").read_bytes()),
    "app.js": ("text/javascript; charset=utf-8", (STATIC / "app.js").read_bytes()),
    "mascot.svg": ("image/svg+xml", (STATIC / "mascot.svg").read_bytes()),
}


class SecurityHeadersMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.extend([
                    (b"x-content-type-options", b"nosniff"),
                    (b"referrer-policy", b"strict-origin-when-cross-origin"),
                    (b"content-security-policy",
                     b"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
                     b"connect-src 'self'; object-src 'none'; frame-ancestors 'none'"),
                ])
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_headers)


def create_app(engine=None, settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = engine if engine is not None else (ModelEngine(settings) if settings.revision else None)
        yield

    app = FastAPI(title="Malmaek", version="0.1.0", lifespan=lifespan,
                  docs_url=None, redoc_url=None)
    app.add_middleware(SecurityHeadersMiddleware)

    @app.get("/", include_in_schema=False)
    async def home():
        media_type, content = ASSETS["index.html"]
        return Response(content, media_type=media_type)

    @app.get("/static/{asset_name}", include_in_schema=False)
    async def static_asset(asset_name: str):
        if asset_name not in ASSETS or asset_name == "index.html":
            raise HTTPException(status_code=404)
        media_type, content = ASSETS[asset_name]
        return Response(content, media_type=media_type, headers={"Cache-Control": "public, max-age=3600"})

    @app.get("/api/health")
    async def health():
        active = app.state.engine
        corpora = {name: len(rows) for name, rows in active.hub.corpora.items()} if active else {}
        return {"ready": active is not None, "corpora": corpora,
                "adapter": bool(active and active.settings.adapter)}

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest, response: Response):
        active = app.state.engine
        if active is None:
            raise HTTPException(status_code=503, detail="Model is not configured on this server")
        try:
            result = await asyncio.get_running_loop().run_in_executor(None, active.reply, request)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        response.headers["Cache-Control"] = "no-store"
        return result

    return app


app = create_app()
