"""Pin policy precision, ties, and registry reachability."""

import pytest
import torch

from tests.unit.test_policies import make_state
from uniqkache.policies import LRUPolicy, TokenImportancePolicy, build_policy
from uniqkache.policies.registry import register_policy
from uniqkache.policies.signals import rank_normalize
from uniqkache.utils.errors import PolicyError


def test_lru_preserves_long_context_position_order():
    policy = LRUPolicy()
    state = make_state(3, positions=[131070, 131071, 131072], last_access=[1000] * 3)
    scores = policy.score(state)
    assert scores.dtype == torch.float64
    assert policy.select(scores, 1).tolist() == [2]


def test_rank_ties_preserve_policy_slot_tiebreak():
    assert rank_normalize(torch.ones(4)).tolist() == [0.0] * 4
    scores = rank_normalize(torch.tensor([3.0, 1.0, 3.0, 1.0]))
    assert scores[0] == scores[2]
    assert scores[1] == scores[3]
    assert TokenImportancePolicy().select(scores, 1).tolist() == [0]


def test_rank_does_not_collapse_distinct_large_integer_signals():
    scores = rank_normalize(torch.tensor([2**25, 2**25 + 1]))
    assert scores.tolist() == [0.0, 1.0]


def test_policy_cannot_shadow_existing_alias():
    original = type(build_policy("h2o"))

    class Shadow(LRUPolicy):
        name = "h2o"

    with pytest.raises(PolicyError, match="alias"):
        register_policy(Shadow)
    assert type(build_policy("h2o")) is original
