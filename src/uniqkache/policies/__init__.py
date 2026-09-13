"""Cache policies.

Importing this package registers every built-in policy with
:mod:`uniqkache.policies.registry`, so ``build_policy("sliding_window")`` works
from anywhere — including from an experiment config file.

Built-in policies
-----------------
======================  ==============  ==========================================
Name                    Signals         Status
======================  ==============  ==========================================
``full_cache``          none            Stable — reference baseline
``sliding_window``      position        Stable — StreamingLLM reproduction
``lru``                 recency         Stable
``attention_based``     attention       Stable — H2O reproduction
``token_importance``    multi-signal    Stable — engineering ablation target
``adaptive``            attention,      **Research prototype, unvalidated**
                        recency
======================  ==============  ==========================================

Adding a policy
---------------
Subclass :class:`~uniqkache.policies.base.BaseCachePolicy`, implement ``score``,
decorate with :func:`~uniqkache.policies.registry.register_policy`, and import it
here. Nothing else needs to change: the CLI, the benchmark runner and the docs
table generator all discover policies through the registry.
"""

from __future__ import annotations

from uniqkache.policies.adaptive import AdaptivePolicy
from uniqkache.policies.attention_based import AttentionBasedPolicy
from uniqkache.policies.base import BaseCachePolicy, CachePolicy
from uniqkache.policies.full import FullCachePolicy
from uniqkache.policies.lru import LRUPolicy
from uniqkache.policies.registry import (
    available_aliases,
    available_policies,
    build_policy,
    get_policy_class,
    register_alias,
    register_policy,
    resolve_name,
)
from uniqkache.policies.sliding_window import SlidingWindowPolicy
from uniqkache.policies.token_importance import TokenImportancePolicy

# Aliases accept the names contributors arrive with, so a config file does not
# have to guess our internal naming. Aliases must be registered after the real
# policies, and must not collide with a policy name.
_ALIASES: dict[str, str] = {
    "streamingllm": "sliding_window",
    "streaming_llm": "sliding_window",
    "h2o": "attention_based",
    "heavy_hitter": "attention_based",
    "full": "full_cache",
    "none": "full_cache",
    "sliding": "sliding_window",
    "recency": "lru",
}
for _alias, _target in _ALIASES.items():
    register_alias(_alias, _target)
del _alias, _target

__all__ = [
    "AdaptivePolicy",
    "AttentionBasedPolicy",
    "BaseCachePolicy",
    "CachePolicy",
    "FullCachePolicy",
    "LRUPolicy",
    "SlidingWindowPolicy",
    "TokenImportancePolicy",
    "available_aliases",
    "available_policies",
    "build_policy",
    "get_policy_class",
    "register_alias",
    "register_policy",
    "resolve_name",
]
