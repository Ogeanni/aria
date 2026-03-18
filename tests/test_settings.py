"""
tests/test_settings.py
Tests for config/settings.py

Verifies configuration loads correctly, computed properties work,
and the system handles missing optional values gracefully.
"""
import pytest
from config.settings import get_settings, Settings


class TestSettings:

    def test_settings_load(self):
        """Settings should load without raising errors."""
        s = get_settings()
        assert s is not None

    def test_demo_mode_is_bool(self):
        s = get_settings()
        assert isinstance(s.demo_mode, bool)

    def test_demo_mode_true_in_test_env(self):
        """Test environment sets DEMO_MODE=true via conftest."""
        s = get_settings()
        assert s.demo_mode is True

    def test_database_url_set(self):
        s = get_settings()
        assert s.database_url is not None
        assert len(s.database_url) > 0

    def test_path_properties_return_paths(self):
        from pathlib import Path
        s = get_settings()
        assert isinstance(s.data_dir, Path)
        assert isinstance(s.models_dir, Path)
        assert isinstance(s.features_path, Path)
        assert isinstance(s.xgb_model_path, Path)

    def test_has_serpapi_false_with_empty_key(self):
        s = Settings(serpapi_key="")
        assert s.has_serpapi is False

    def test_has_serpapi_false_with_placeholder(self):
        s = Settings(serpapi_key="your_serpapi_key_here")
        assert s.has_serpapi is False

    def test_has_serpapi_true_with_real_key(self):
        s = Settings(serpapi_key="real-key-abc123")
        assert s.has_serpapi is True

    def test_has_redis_false_without_url(self):
        s = Settings(redis_url=None)
        assert s.has_redis is False

    def test_has_redis_true_with_url(self):
        s = Settings(redis_url="redis://localhost:6379/0")
        assert s.has_redis is True

    def test_agent_thresholds_are_positive(self):
        s = get_settings()
        assert s.agent_auto_approve_max_pct > 0
        assert s.agent_schedule_minutes > 0

    def test_settings_cached(self):
        """get_settings() should return the same instance each call."""
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2