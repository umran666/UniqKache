"""Allocation strategy registry.

Allows registering and resolving allocation strategies by name.
"""

from __future__ import annotations

from typing import TypeVar

from uniqkache.allocation.base import BaseAllocationStrategy
from uniqkache.utils.errors import ConfigError
from uniqkache.utils.logging import get_logger

_log = get_logger(__name__)

_REGISTRY: dict[str, type[BaseAllocationStrategy]] = {}
_ALIASES: dict[str, str] = {}

StrategyT = TypeVar("StrategyT", bound=type[BaseAllocationStrategy])


def register_allocation_strategy(cls: StrategyT) -> StrategyT:
    """Register an allocation strategy class under its ``name`` attribute."""
    name = getattr(cls, "name", None)
    if not isinstance(name, str) or not name:
        raise ConfigError(f"{cls.__name__} must define a non-empty class attribute `name`")

    existing = _REGISTRY.get(name)
    if existing is not None and existing is not cls:
        raise ConfigError(
            f"allocation strategy name {name!r} is already registered by "
            f"{existing.__module__}.{existing.__name__}; pick a unique name for "
            f"{cls.__module__}.{cls.__name__}"
        )

    _REGISTRY[name] = cls
    return cls


def register_alias(alias: str, target: str) -> None:
    """Map an alternative name onto a registered allocation strategy."""
    _ALIASES[alias] = target


def resolve_allocation_strategy_name(name: str) -> str:
    """Resolve an alias to its canonical allocation strategy name."""
    return _ALIASES.get(name, name)


def available_allocation_strategies() -> list[str]:
    """All registered canonical strategy names, sorted."""
    return sorted(_REGISTRY.keys())


def get_allocation_strategy_class(name: str) -> type[BaseAllocationStrategy]:
    """Look up a strategy class by canonical name or alias."""
    canonical = resolve_allocation_strategy_name(name)
    cls = _REGISTRY.get(canonical)
    if cls is None:
        raise ConfigError(
            f"unknown allocation strategy {name!r}. Available: {', '.join(available_allocation_strategies())}"
        )
    return cls


def build_allocation_strategy(name: str, **kwargs) -> BaseAllocationStrategy:
    """Construct an allocation strategy by name."""
    cls = get_allocation_strategy_class(name)
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise ConfigError(f"could not construct allocation strategy {name!r}: {exc}") from exc
