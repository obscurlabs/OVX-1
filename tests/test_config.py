"""Project-root resolution.

This exists because of a deployment bug that build-time checks cannot catch.
Locally the package is imported from the source tree and PROJECT_ROOT can be
derived from __file__. In the container `pip install .` moves the package into
site-packages, where that same derivation resolves to /usr/local/lib/python3.12
- a root-owned directory containing neither the index nor web/. The image built
cleanly and would have died on first boot with a PermissionError.

The Dockerfile pins VOICERAG_ROOT=/app. These tests hold both halves of that
contract: the override is honoured, and the local derivation still works when
it is absent.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import voicerag.config


def _reload_with_root(monkeypatch, value: str | None):
    """Re-import config under a given VOICERAG_ROOT.

    PROJECT_ROOT is a module-level constant read at import time, so the env var
    has to be set before the reload rather than after.
    """
    if value is None:
        monkeypatch.delenv("VOICERAG_ROOT", raising=False)
    else:
        monkeypatch.setenv("VOICERAG_ROOT", value)
    return importlib.reload(voicerag.config)


def test_env_root_overrides_file_derivation(monkeypatch, tmp_path):
    cfg = _reload_with_root(monkeypatch, str(tmp_path))
    assert cfg.PROJECT_ROOT == tmp_path.resolve()


def test_all_paths_follow_the_override(monkeypatch, tmp_path):
    """Every path must move with the root.

    A path that kept pointing at the source tree would be the same bug in a
    smaller form - the container would find some artifacts and not others.
    """
    cfg = _reload_with_root(monkeypatch, str(tmp_path))

    for path in (
        cfg.Paths.root,
        cfg.Paths.cache,
        cfg.Paths.data,
        cfg.Paths.indexes,
        cfg.Paths.vector_index,
        cfg.Paths.bm25_index,
        cfg.Paths.chunk_meta,
        cfg.Paths.onnx_encoder,
        cfg.Paths.stt_cache,
    ):
        assert path.is_relative_to(tmp_path.resolve()), path


def test_falls_back_to_source_tree_without_the_env_var(monkeypatch):
    """No VOICERAG_ROOT (i.e. local dev) must still find the real project."""
    cfg = _reload_with_root(monkeypatch, None)

    assert (cfg.PROJECT_ROOT / "pyproject.toml").exists()
    assert (cfg.PROJECT_ROOT / "src" / "voicerag").is_dir()


def test_web_dir_resolves_under_the_override(monkeypatch, tmp_path):
    """The Dockerfile copies web/ to /app/web; the app must look there.

    WEB_DIR is computed at import in api.py, so it is recomputed here from the
    reloaded config rather than imported directly.
    """
    cfg = _reload_with_root(monkeypatch, str(tmp_path))
    assert cfg.Paths.root / "web" == tmp_path.resolve() / "web"


def teardown_module() -> None:
    """Restore the real config for any test module that runs after this one."""
    importlib.reload(voicerag.config)


def test_repo_web_assets_exist() -> None:
    """Guards the other half: web/index.html must actually be in the repo."""
    root = Path(voicerag.config.__file__).resolve().parents[2]
    assert (root / "web" / "index.html").exists()
