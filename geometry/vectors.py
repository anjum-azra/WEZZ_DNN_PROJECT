"""
Vector mathematics utilities.
"""

from __future__ import annotations
import numpy as np


def vector(x: float, y: float, z: float) -> np.ndarray:
    """Create a 3D vector."""
    return np.array([x, y, z], dtype=np.float64)


def add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a + b


def subtract(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a - b


def scale(v: np.ndarray, s: float) -> np.ndarray:
    return v * s


def magnitude(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def normalize(v: np.ndarray) -> np.ndarray:
    mag = magnitude(v)

    if mag < 1e-10:
        return np.zeros(3, dtype=np.float64)

    return v / mag


def dot(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.cross(a, b)


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return magnitude(a - b)


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """
    Returns angle in radians.
    """

    na = normalize(a)
    nb = normalize(b)

    cosine = np.clip(dot(na, nb), -1.0, 1.0)

    return float(np.arccos(cosine))