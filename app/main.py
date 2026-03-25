"""Market Data Analyzer application entry point."""

from __future__ import annotations

from .application import create_app
from .bootstrap import build_services

services = build_services()
hub = services.hub
app = create_app(services)
