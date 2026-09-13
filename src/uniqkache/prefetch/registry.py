"""Prefetch policy registry.

Mirrors :mod:`uniqkache.policies.registry` so that a prefetch policy can be
selected by name from a configuration file without the CLI knowing about it.
"""

from __future__ import annotations

from typing import TypeVar

from uniqkache.prefetch.base import BasePrefetchPolicy
from uniqkache.utils.errors import PolicyError

_REGISTRY: dict[str, type[BasePrefetchPolicy]] = {}

PolicyT = TypeVar("PolicyT", bound=type[BasePrefetchPolicy])


def register_prefetch_policy(cls: PolicyT) -> PolicyT:
    """Register a prefetch policy under its ``name`` attribute."""
    name = getattr(cls, "name", None)
    if not isinstance(name, str) or not name:
        raise PolicyError(f"{cls.__name__} must define a non-empty class attribute `name`")

    existing = _REGISTRY.get(name)
    if existing is not None and existing is not cls:
        raise PolicyError(
            f"prefetch policy name {name!r} is already registered by "
            f"{existing.__module__}.{existing.__name__}"
        )
    _REGISTRY[name] = cls
    return cls


def available_prefetch_policies() -> list[str]:
    return sorted(_REGISTRY)


def get_prefetch_policy_class(name: str) -> type[BasePrefetchPolicy]:
    """Look up a prefetch policy class by name."""
    if name not in _REGISTRY:
        known = ", ".join(available_prefetch_policies()) or "<none>"
        raise PolicyError(f"unknown prefetch policy {name!r}. Registered: {known}")
    return _REGISTRY[name]


def build_prefetch_policy(name: str, **kwargs: object) -> BasePrefetchPolicy:
    """Instantiate a prefetch policy by name."""
    cls = get_prefetch_policy_class(name)
    try:
        return cls(**kwargs)  # type: ignore[arg-type]
    except TypeError as exc:
        raise PolicyError(
            f"could not construct prefetch policy {name!r} with {kwargs!r}: {exc}"
        ) from exc


__all__ = [
    "available_prefetch_policies",
    "build_prefetch_policy",
    "get_prefetch_policy_class",
    "register_prefetch_policy",
]
