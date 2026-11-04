"""
Finite-horizon return confidence levels for a fixed agent-state policy.

Core pieces
-----------
- ``build_quantized`` — reachable finite model on X=(S,Z)
- ``mgf`` — exact tilted MGF recursion + Chernoff levels
- ``moments`` — moment recursion, exact support, polynomial intervals
- ``polybound`` — Chebyshev TailLP / endpoint bisection
- ``continuous`` — closed-form Gaussian reference (not a certificate for the finite model)
"""

from .build_quantized import FiniteModel, build, simulate
from . import mgf, moments, polybound

__all__ = [
    "FiniteModel",
    "build",
    "simulate",
    "mgf",
    "moments",
    "polybound",
]
