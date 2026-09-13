"""平台 judge 端点接入（环境指标走 tuneplane.report，不在教程仓自建 HTTP）。"""
from __future__ import annotations

import importlib


def _reload_reward(monkeypatch, **env):
    for key in ("TUNEPLANE_JOB_JUDGE_ENDPOINT", "TUNEPLANE_JOB_JUDGE_TOKEN",
                "JUDGE_BASE_URL", "JUDGE_MODEL", "JUDGE_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    import common.rewards.qa_judge_reward as mod

    return importlib.reload(mod)


def test_platform_judge_endpoint_takes_priority(monkeypatch):
    mod = _reload_reward(
        monkeypatch,
        TUNEPLANE_JOB_JUDGE_ENDPOINT="https://console.internal/api/judge",
        TUNEPLANE_JOB_JUDGE_TOKEN="tok-judge",
        JUDGE_BASE_URL="http://should-not-win:8001/v1",
    )
    assert mod.JUDGE_BASE_URL == "https://console.internal/api/judge/v1"
    assert mod.JUDGE_API_KEY == "tok-judge"


def test_local_judge_env_still_works_without_platform(monkeypatch):
    mod = _reload_reward(monkeypatch, JUDGE_BASE_URL="http://127.0.0.1:9001/v1", JUDGE_API_KEY="k")
    assert mod.JUDGE_BASE_URL == "http://127.0.0.1:9001/v1"
    assert mod.JUDGE_API_KEY == "k"
    _reload_reward(monkeypatch)
