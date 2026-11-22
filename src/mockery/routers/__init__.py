"""Domain-scoped FastAPI APIRouters for Mockery.

Each router owns a vertical slice of the HTTP API and must never import sibling
routers or `mockery.server` (enforced by ARCH004 in `scripts/lint_arch.py`).
"""

from .addons import build_addons_router
from .practice import build_practice_router
from .profile import build_profile_router

__all__ = [
    "build_addons_router",
    "build_practice_router",
    "build_profile_router",
]
