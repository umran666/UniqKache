"""Unit tests for cache policies and the policy contract.

The policy layer is where an ablation's *decision rule* lives, so these tests
pin down what each rule does: which tokens it keeps, what it scores, and how it
behaves at boundaries.
"""

from __future__ import annotations

import pytest
import torch

from uniqkache.cache.types import PolicyState
from uniqkache.policies import (
    AdaptivePolicy,
    AttentionBasedPolicy,
    BaseCachePolicy,
    CachePolicy,
    FullCachePolicy,
    LRUPolicy,
    SlidingWindowPolicy,
    TokenImportancePolicy,
    available_aliases,
    available_policies,
    build_policy,
)
from uniqkache.policies.registry import register_policy
from uniqkache.policies.signals import minmax, rank_normalize, weighted_sum
from uniqkache.utils.errors import PolicyError


def make_state(
    num_cached: int,
    *,
    capacity: int | None = None,
    positions: list[int] | None = None,
    last_access: list[int] | None = None,
    cum_attention: list[float] | None = None,
    sinks: int = 0,
    step: int = 0,
) -> PolicyState:
    """Build a PolicyState directly, so policies can be tested in isolation."""
    positions = positions if positions is not None else list(range(num_cached))
    return PolicyState(
        layer_idx=0,
        step=step,
        num_cached=num_cached,
        capacity=capacity,
        positions=torch.tensor(positions, dtype=torch.long),
        last_access=torch.tensor(
            last_access if last_access is not None else [0] * num_cached, dtype=torch.long
        ),
        cum_attention=torch.tensor(
            cum_attention if cum_attention is not None else [0.0] * num_cached,
            dtype=torch.float32,
        ),
        hit_count=torch.zeros(num_cached, dtype=torch.long),
        is_sink=torch.tensor([p < sinks for p in positions], dtype=torch.bool),
        memory_pressure=(num_cached / capacity) if capacity else 0.0,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_all_builtin_policies_are_registered(self):
        for name in (
            "full_cache",
            "sliding_window",
            "lru",
            "attention_based",
            "token_importance",
            "adaptive",
        ):
            assert name in available_policies()

    def test_aliases_resolve_to_registered_policies(self):
        for alias, target in available_aliases().items():
            assert target in available_policies(), f"{alias} -> unknown {target}"

    def test_h2o_alias_builds_attention_based(self):
        assert isinstance(build_policy("h2o"), AttentionBasedPolicy)

    def test_streamingllm_alias_builds_sliding_window(self):
        assert isinstance(build_policy("streamingllm"), SlidingWindowPolicy)

    def test_unknown_policy_raises_with_suggestions(self):
        with pytest.raises(PolicyError, match="unknown policy"):
            build_policy("does_not_exist")

    def test_duplicate_name_is_rejected(self):
        with pytest.raises(PolicyError, match="already registered"):

            @register_policy
            class ClashingPolicy(BaseCachePolicy):
                name = "full_cache"

                def score(self, state):  # pragma: no cover - never called
                    return torch.zeros(state.num_cached)

    def test_unnamed_policy_is_rejected(self):
        with pytest.raises(PolicyError, match="non-empty class attribute"):

            @register_policy
            class NamelessPolicy(BaseCachePolicy):
                name = ""

                def score(self, state):  # pragma: no cover - never called
                    return torch.zeros(state.num_cached)

    def test_every_policy_satisfies_the_protocol(self):
        for name in available_policies():
            policy = build_policy(name)
            assert isinstance(policy, CachePolicy), f"{name} violates the protocol"
            assert isinstance(policy, BaseCachePolicy)


# ---------------------------------------------------------------------------
# select() contract
# ---------------------------------------------------------------------------


class TestSelectContract:
    def test_budget_above_length_keeps_everything(self):
        policy = FullCachePolicy()
        scores = torch.rand(10)
        keep = policy.select(scores, budget=10)
        assert keep.tolist() == list(range(10))

    def test_selects_exactly_the_budget(self):
        policy = FullCachePolicy()
        keep = policy.select(torch.rand(10), budget=4)
        assert len(keep) == 4

    def test_returns_sorted_indices(self):
        policy = FullCachePolicy()
        keep = policy.select(torch.rand(20), budget=5)
        assert torch.all(keep[1:] > keep[:-1]), "indices must be ascending"

    def test_higher_score_wins(self):
        policy = FullCachePolicy()
        scores = torch.tensor([0.1, 0.9, 0.5, 0.2, 0.8])
        keep = policy.select(scores, budget=2)
        assert keep.tolist() == [1, 4]

    def test_protected_tokens_always_survive(self):
        policy = FullCachePolicy()
        scores = torch.tensor([0.0, 0.0, 5.0, 5.0, 5.0])
        protect = torch.tensor([True, True, False, False, False])
        keep = policy.select(scores, budget=3, protect=protect)
        assert 0 in keep.tolist() and 1 in keep.tolist()

    def test_protected_tokens_consume_budget(self):
        policy = FullCachePolicy()
        protect = torch.tensor([True, True, False, False])
        keep = policy.select(torch.rand(4), budget=3, protect=protect)
        assert len(keep) == 3

    def test_protection_exceeding_budget_raises(self):
        """A configuration conflict must be reported, not silently resolved."""
        policy = FullCachePolicy()
        protect = torch.tensor([True, True, True, False])
        with pytest.raises(PolicyError, match="are protected"):
            policy.select(torch.rand(4), budget=2, protect=protect)

    def test_negative_budget_raises(self):
        with pytest.raises(PolicyError, match="budget must be >= 0"):
            FullCachePolicy().select(torch.rand(4), budget=-1)

    def test_multidimensional_scores_raise(self):
        with pytest.raises(PolicyError, match="1-D tensor"):
            FullCachePolicy().select(torch.rand(4, 2), budget=2)

    def test_wrong_length_protect_mask_raises(self):
        with pytest.raises(PolicyError, match="protect mask has length"):
            FullCachePolicy().select(torch.rand(4), budget=2, protect=torch.tensor([True]))


# ---------------------------------------------------------------------------
# Individual policies
# ---------------------------------------------------------------------------


class TestSlidingWindow:
    def test_keeps_the_most_recent_tokens(self):
        policy = SlidingWindowPolicy()
        state = make_state(10)
        keep = policy.select(policy.score(state), budget=3)
        assert keep.tolist() == [7, 8, 9]

    def test_sinks_plus_recent_window(self):
        policy = SlidingWindowPolicy()
        state = make_state(20, sinks=2)
        keep = policy.select(policy.score(state), budget=5, protect=state.is_sink)
        # 2 sinks plus the 3 newest tokens.
        assert keep.tolist() == [0, 1, 17, 18, 19]

    def test_scores_follow_absolute_position_not_slot(self):
        """After eviction, slot order and position order must not be confused."""
        policy = SlidingWindowPolicy()
        state = make_state(4, positions=[0, 1, 50, 51])
        keep = policy.select(policy.score(state), budget=2)
        assert keep.tolist() == [2, 3]

    def test_explicit_window_caps_the_budget(self):
        policy = SlidingWindowPolicy(window=2)
        state = make_state(10)
        keep = policy.select(policy.score(state), budget=8)
        assert len(keep) <= 2

    def test_invalid_window_is_rejected(self):
        with pytest.raises(ValueError, match="window must be >= 1"):
            SlidingWindowPolicy(window=0)


class TestLRU:
    def test_evicts_the_least_recently_used(self):
        policy = LRUPolicy()
        # Token 1 was used most recently, token 0 least recently.
        state = make_state(3, last_access=[0, 5, 2])
        keep = policy.select(policy.score(state), budget=1)
        assert keep.tolist() == [1]

    def test_ties_break_toward_the_older_token(self):
        policy = LRUPolicy()
        state = make_state(4, last_access=[3, 3, 3, 3], positions=[0, 1, 2, 3])
        keep = policy.select(policy.score(state), budget=2)
        # All equally recent, so the two newest positions are kept.
        assert keep.tolist() == [2, 3]

    def test_declares_that_it_uses_attention(self):
        assert LRUPolicy().uses_attention is True


class TestAttentionBased:
    def test_keeps_the_highest_accumulated_attention(self):
        policy = AttentionBasedPolicy()
        state = make_state(4, cum_attention=[0.1, 0.9, 0.2, 0.8])
        keep = policy.select(policy.score(state), budget=2)
        assert keep.tolist() == [1, 3]

    def test_normalisation_preserves_the_ranking(self):
        raw = AttentionBasedPolicy(normalize=False)
        norm = AttentionBasedPolicy(normalize=True)
        state = make_state(5, cum_attention=[0.1, 0.9, 0.2, 0.8, 0.5])
        assert (
            raw.select(raw.score(state), 2).tolist() == norm.select(norm.score(state), 2).tolist()
        )

    def test_uses_attention_flag_is_set(self):
        assert AttentionBasedPolicy().uses_attention is True


class TestTokenImportance:
    def test_attention_only_matches_attention_based(self):
        """With other weights zeroed, this must reduce to the attention rule."""
        policy = TokenImportancePolicy(
            weights={"attention": 1.0, "recency": 0.0, "frequency": 0.0, "position": 0.0}
        )
        state = make_state(4, cum_attention=[0.1, 0.9, 0.2, 0.8])
        keep = policy.select(policy.score(state), budget=2)
        assert keep.tolist() == [1, 3]

    def test_recency_only_matches_lru_ranking(self):
        policy = TokenImportancePolicy(
            weights={"attention": 0.0, "recency": 1.0, "frequency": 0.0, "position": 0.0}
        )
        state = make_state(3, last_access=[0, 5, 2])
        keep = policy.select(policy.score(state), budget=1)
        assert keep.tolist() == [1]

    def test_unknown_signal_is_rejected(self):
        with pytest.raises(ValueError, match="unknown signal"):
            TokenImportancePolicy(weights={"nonsense": 1.0})

    def test_all_zero_weights_are_rejected(self):
        with pytest.raises(ValueError, match="at least one signal weight"):
            TokenImportancePolicy(
                weights={"attention": 0.0, "recency": 0.0, "frequency": 0.0, "position": 0.0}
            )


class TestAdaptivePolicy:
    def test_recency_weight_rises_with_pressure(self):
        policy = AdaptivePolicy(
            recency_weight_at_low_pressure=0.0,
            recency_weight_at_high_pressure=1.0,
            pressure_floor=0.0,
            pressure_ceiling=1.0,
            min_free_fraction=0.0,
        )
        low = policy.recency_weight(0.0, free_fraction=1.0)
        mid = policy.recency_weight(0.5, free_fraction=0.5)
        high = policy.recency_weight(1.0, free_fraction=0.0)
        assert low < mid < high

    def test_weight_is_clamped_outside_the_pressure_window(self):
        policy = AdaptivePolicy(
            recency_weight_at_low_pressure=0.1,
            recency_weight_at_high_pressure=0.9,
            pressure_floor=0.4,
            pressure_ceiling=0.8,
            min_free_fraction=0.0,
        )
        assert policy.recency_weight(0.0, 1.0) == pytest.approx(0.1)
        assert policy.recency_weight(1.0, 0.0) == pytest.approx(0.9)

    def test_safety_floor_forces_the_high_weight(self):
        policy = AdaptivePolicy(
            recency_weight_at_low_pressure=0.0,
            recency_weight_at_high_pressure=0.75,
            min_free_fraction=0.2,
        )
        # Almost no free space forces the high-pressure weight regardless.
        assert policy.recency_weight(0.0, free_fraction=0.05) == pytest.approx(0.75)

    def test_realised_weights_are_recorded_for_inspection(self):
        policy = AdaptivePolicy()
        state = make_state(8, capacity=8, cum_attention=[float(i) for i in range(8)])
        policy.score(state)
        assert set(policy.last_realised_weights) == {"attention", "recency", "memory_pressure"}
        total = policy.last_realised_weights["attention"] + policy.last_realised_weights["recency"]
        assert total == pytest.approx(1.0)

    def test_state_dict_declares_that_it_is_unvalidated(self):
        """The prototype must not be able to masquerade as a validated method."""
        assert AdaptivePolicy().state_dict()["validated"] is False

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"pressure_floor": 0.9, "pressure_ceiling": 0.5}, "pressure_floor"),
            ({"recency_weight_at_low_pressure": 2.0}, "recency_weight_at_low_pressure"),
            ({"recency_weight_at_high_pressure": -1.0}, "recency_weight_at_high_pressure"),
            ({"normalize": "bogus"}, "normalize"),
        ],
    )
    def test_invalid_parameters_are_rejected(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            AdaptivePolicy(**kwargs)


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------


class TestSignalHelpers:
    def test_minmax_scales_to_unit_interval(self):
        assert torch.allclose(minmax(torch.tensor([1.0, 3.0, 5.0])), torch.tensor([0.0, 0.5, 1.0]))

    def test_minmax_of_constant_returns_zeros_not_nan(self):
        out = minmax(torch.tensor([2.0, 2.0, 2.0]))
        assert torch.equal(out, torch.zeros(3))
        assert not torch.isnan(out).any()

    def test_rank_normalize_is_uniformly_spaced(self):
        out = rank_normalize(torch.tensor([100.0, 1.0, 50.0]))
        assert torch.allclose(out, torch.tensor([1.0, 0.0, 0.5]))

    def test_rank_normalize_is_robust_to_outliers(self):
        """A single dominant value must not crush every other score to zero."""
        out = rank_normalize(torch.tensor([1000.0, 1.0, 2.0, 3.0]))
        assert out.max() == pytest.approx(1.0)
        assert out.min() == pytest.approx(0.0)
        assert len(set(out.tolist())) == 4

    def test_weighted_sum_normalises_by_total_weight(self):
        signals = {"a": torch.tensor([1.0, 0.0]), "b": torch.tensor([0.0, 1.0])}
        out = weighted_sum(signals, {"a": 1.0, "b": 1.0})
        assert torch.allclose(out, torch.tensor([0.5, 0.5]))

    def test_weighted_sum_rejects_unknown_signal(self):
        with pytest.raises(KeyError, match="not supplied"):
            weighted_sum({"a": torch.zeros(2)}, {"b": 1.0})

    def test_weighted_sum_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="share a length"):
            weighted_sum({"a": torch.zeros(2), "b": torch.zeros(3)}, {"a": 1.0, "b": 1.0})

    def test_weighted_sum_rejects_zero_total_weight(self):
        with pytest.raises(ValueError, match="positive value"):
            weighted_sum({"a": torch.zeros(2)}, {"a": 0.0})
