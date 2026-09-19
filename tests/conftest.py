"""Shared fixtures.

Every test runs against a temporary database and the offline providers, so the
suite needs no API keys, no network and no fixed ordering.
"""

from __future__ import annotations

import pytest

from app.config import Settings
from app.ingest import run_ingest
from app.store import Store


@pytest.fixture(scope="session")
def cfg(tmp_path_factory) -> Settings:
    settings = Settings()
    settings.db_path = str(tmp_path_factory.mktemp("db") / "test.db")
    # Force the keyless path regardless of the developer's .env, so results are
    # identical on a laptop with keys and in CI without them.
    settings.groq_api_key = ""
    settings.gemini_api_key = ""
    settings.today_override = "2026-09-19"
    return settings


@pytest.fixture(scope="session")
def store(cfg: Settings) -> Store:
    s = Store(cfg)
    s.init_schema()
    run_ingest(cfg=cfg, store=s, verbose=False)
    yield s
    s.close()
