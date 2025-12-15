# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""VLA backend implementations for different models."""

from typing import Literal

from crane_x7_vla.core.base import VLABackend


# Type alias for backend names
BackendType = Literal["openvla", "openvla-oft", "minivla", "pi0", "pi0.5"]


def get_backend(backend_type: BackendType) -> type[VLABackend]:
    """
    Get backend class by name with lazy loading.

    This function delays importing backend modules until they are actually
    needed, avoiding dependency issues when not all backends are installed.

    Args:
        backend_type: Name of the backend ("openvla", "minivla", "pi0", "pi0.5", etc.)

    Returns:
        Backend class (not instantiated)

    Raises:
        ValueError: If backend_type is unknown
        ImportError: If required dependencies for the backend are not installed

    Example:
        >>> BackendClass = get_backend("openvla")
        >>> backend = BackendClass(config)
        >>> backend.train()

        >>> # Pi0 and Pi0.5 use the same backend class
        >>> BackendClass = get_backend("pi0")
        >>> backend = BackendClass(config)
    """
    if backend_type == "openvla":
        from crane_x7_vla.backends.openvla import OpenVLABackend

        return OpenVLABackend
    elif backend_type == "openvla-oft":
        from crane_x7_vla.backends.openvla_oft import OpenVLAOFTBackend

        return OpenVLAOFTBackend
    elif backend_type == "minivla":
        from crane_x7_vla.backends.minivla import MiniVLABackend

        return MiniVLABackend
    elif backend_type in ("pi0", "pi0.5"):
        from crane_x7_vla.backends.pi0 import Pi0Backend

        return Pi0Backend
    else:
        raise ValueError(
            f"Unknown backend: {backend_type}. " f"Available backends: openvla, openvla-oft, minivla, pi0, pi0.5"
        )


__all__ = [
    "BackendType",
    "VLABackend",
    "get_backend",
]
