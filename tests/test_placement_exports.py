"""Compatibility checks for the placement package's public helpers."""

from wasp.placement import transformers_utils
from wasp.placement.transformers_utils import config, flops, memory


def test_transformer_helpers_remain_available_from_package():
    assert transformers_utils.HFUnifiedConfig is config.HFUnifiedConfig
    for name in ("forward_layer_flops", "backward_layer_flops"):
        assert getattr(transformers_utils, name) is getattr(flops, name)


def test_memory_exports_remain_available_from_package():
    # The package historically exposed all public names from memory.py.
    for name, value in vars(memory).items():
        if not name.startswith("_"):
            assert getattr(transformers_utils, name) is value
