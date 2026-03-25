"""FastAPI application factory for Market Data Analyzer."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .api.deps import init_routes
from .bootstrap import AppServices, build_services
from .routes import router

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
NO_CACHE_PATHS = frozenset(
    {
        "/",
        "/market-data-lab",
        "/ml-lab",
        "/strategy-lab",
        "/compare-lab",
        "/leadlag-lab",
        "/static/app.terminal.js",
        "/static/styles.css",
        "/static/index.html",
    }
)


def create_lifespan(hub):
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await hub.start()
        try:
            yield
        finally:
            await hub.stop()

    return lifespan


def register_no_cache_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def disable_monitor_asset_cache(request, call_next):
        response = await call_next(request)
        request_path = request.url.path
        if request_path in NO_CACHE_PATHS or request_path.startswith("/historical/"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response


def create_app(services: AppServices | None = None) -> FastAPI:
    resolved_services = services or build_services()
    app = FastAPI(title="Market Data Analyzer", lifespan=create_lifespan(resolved_services.hub))

    init_routes(
        app,
        hub=resolved_services.hub,
        symbol_catalog_store=resolved_services.symbol_catalog_store,
        paper_portfolio_store=resolved_services.paper_portfolio_store,
        ui_state_store=resolved_services.ui_state_store,
        stock_ml_page_store=resolved_services.stock_ml_page_store,
        ml_job_store=resolved_services.ml_job_store,
    )
    register_no_cache_middleware(app)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.include_router(router)
    return app
