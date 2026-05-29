"""FastAPI application factory for the Rembrandt SPA."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import Response
from starlette.types import Scope

from rembrandt.web.api import router as api_router

_VITE_DEV_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)


class SPAStaticFiles(StaticFiles):
    """Serve static assets with an ``index.html`` fallback for SPA routes."""

    async def get_response(self, path: str, scope: Scope) -> Response:
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code != 404:
                raise
            if Path(path).suffix:
                raise
            return await super().get_response("index.html", scope)


def create_app(*, frontend_dist: Path | None = None) -> FastAPI:
    """Create the bpy-free FastAPI app.

    Args:
        frontend_dist: Optional path to the built frontend assets. If omitted,
            ``frontend/dist`` relative to the repository root is used.

    Returns:
        A configured FastAPI application.
    """
    app = FastAPI(title="Rembrandt")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(_VITE_DEV_ORIGINS),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(_api_router(), prefix="/api")

    static_dir = frontend_dist or _default_frontend_dist()
    if (static_dir / "index.html").is_file():
        app.mount("/", SPAStaticFiles(directory=static_dir, html=True), name="spa")

    return app


def _api_router() -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    router.include_router(api_router)
    return router


def _default_frontend_dist() -> Path:
    return Path(__file__).resolve().parents[3] / "frontend" / "dist"
