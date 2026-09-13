"""Model backends.

Two backends ship with UniqKache:

``synthetic``
    A small dense-attention transformer with randomly-initialised weights. No
    download, no tokenizer, deterministic. The default for tests and for
    development on constrained hardware. See
    :mod:`uniqkache.models.synthetic` for what its outputs are and are not
    valid evidence for.

``hf``
    Any Hugging Face ``AutoModelForCausalLM``, adapted to UniqKache's cache.
    Requires the ``hf`` extra (``pip install -e ".[hf]"``).

Both satisfy :class:`~uniqkache.models.base.LanguageModel`, so the runtime,
the benchmark runner and the quality evaluators are backend-agnostic.
"""

from __future__ import annotations

from uniqkache.models.base import LanguageModel, Tokenizer
from uniqkache.models.synthetic import (
    PRESETS,
    SyntheticCausalLM,
    SyntheticConfig,
    build_cache_for_model,
    build_model,
    get_preset,
)

__all__ = [
    "PRESETS",
    "LanguageModel",
    "SyntheticCausalLM",
    "SyntheticConfig",
    "Tokenizer",
    "build_cache_for_model",
    "build_model",
    "get_preset",
]
