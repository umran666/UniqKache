"""Policy registry.

Adding a policy should not require editing the CLI, the benchmark runner or the
documentation generator. A contributor registers a class here and it becomes
available everywhere by name — including from an experiment config file.

.. code-block:: python

    from uniqkache.policies.registry import register_policy

    @register_policy
    class MyPolicy(BaseCachePolicy):
        name = "my_policy"
        ...
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from uniqkache.policies.base import BaseCachePolicy
from uniqkache.utils.errors import PolicyError
from uniqkache.utils.logging import get_logger

_log = get_logger(__name__)

_REGISTRY: dict[str, type[BaseCachePolicy]] = {}
_ALIASES: dict[str, str] = {}

PolicyT = TypeVar("PolicyT", bound=type[BaseCachePolicy])


def register_policy(cls: PolicyT) -> PolicyT:
    """Register a policy class under its ``name`` attribute.

    Raises
    ------
    PolicyError
        If the class has no ``name``, or the name is already taken by a
        different class. Re-registering the *same* class is allowed so that
        modules can be imported more than once.
    """
    name = getattr(cls, "name", None)
    if not isinstance(name, str) or not name:
        raise PolicyError(f"{cls.__name__} must define a non-empty class attribute `name`")

    existing = _REGISTRY.get(name)
    if existing is not None and existing is not cls:
        raise PolicyError(
            f"policy name {name!r} is already registered by {existing.__module__}."
            f"{existing.__name__}; pick a unique name for {cls.__module__}.{cls.__name__}"
        )

    _REGISTRY[name] = cls
    return cls


def register_alias(alias: str, target: str) -> None:
    """Map an alternative name onto a registered policy.

    Used to accept the names people arrive with (``"h2o"``, ``"streamingllm"``)
    without duplicating implementations. An alias must not collide with a real
    policy name.
    """
    if alias in _REGISTRY:
        raise PolicyError(f"alias {alias!r} collides with a registered policy name")
    if target not in _REGISTRY:
        raise PolicyError(f"cannot alias {alias!r} to unknown policy {target!r}")
    _ALIASES[alias] = target


def available_policies() -> list[str]:
    """Sorted names of every registered policy, excluding aliases."""
    return sorted(_REGISTRY)


def available_aliases() -> dict[str, str]:
    return dict(_ALIASES)


def resolve_name(name: str) -> str:
    """Resolve an alias to its canonical policy name."""
    return _ALIASES.get(name, name)


def get_policy_class(name: str) -> type[BaseCachePolicy]:
    """Look up a policy class by name or alias.

    Raises
    ------
    PolicyError
        If the name is unknown, with a suggestion list so a typo is obvious.
    """
    canonical = resolve_name(name)
    if canonical not in _REGISTRY:
        known = ", ".join(available_policies()) or "<none>"
        raise PolicyError(f"unknown policy {name!r}. Registered policies: {known}")
    return _REGISTRY[canonical]


def build_policy(name: str, **kwargs: object) -> BaseCachePolicy:
    """Instantiate a policy by name, forwarding keyword arguments.

    Raises
    ------
    PolicyError
        If construction fails, wrapping the underlying ``TypeError`` so the CLI
        can present a clean message about which arguments a policy accepts.
    """
    cls = get_policy_class(name)
    try:
        return cls(**kwargs)  # type: ignore[arg-type]
    except TypeError as exc:
        raise PolicyError(f"could not construct policy {name!r} with {kwargs!r}: {exc}") from exc


def policy_factory(func: Callable[..., BaseCachePolicy]) -> Callable[..., BaseCachePolicy]:
    """Decorator form for building policies from experiment configs.

    An experiment config may declare ``{"name": "sliding_window", "window": 512}``
    and the runner calls ``policy_factory`` to turn that into an instance. Keeping
    this as a function (rather than ``lambda``) means configs stay declarative.
    """
    return func


__all__ = [
    "available_aliases",
    "available_policies",
    "build_policy",
    "get_policy_class",
    "register_alias",
    "register_policy",
    "resolve_name",
]
