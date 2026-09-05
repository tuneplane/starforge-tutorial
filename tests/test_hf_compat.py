from __future__ import annotations

import inspect

from transformers import AutoConfig

from common.hf_compat import _register_deepseek_v4


def test_deepseek_v4_registration_does_not_import_vllm():
    source = inspect.getsource(_register_deepseek_v4)
    assert "__import__" not in source
    assert "vllm.transformers_utils" not in source


def test_deepseek_v4_fallback_config_preserves_model_fields():
    config = AutoConfig.for_model(
        "deepseek_v4",
        hidden_size=4096,
        max_position_embeddings=None,
        rope_scaling={"rope_type": "yarn", "factor": 16.0},
    )

    assert config.hidden_size == 4096
    assert config.max_position_embeddings == 1_048_576
    assert config.rope_parameters["factor"] == 16.0
    assert config.rope_scaling == config.rope_parameters
    assert config.rope_theta == 10_000.0
