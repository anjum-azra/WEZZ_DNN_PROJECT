"""
Coordinate conversion utilities.
"""

from __future__ import annotations
import numpy as np


def cartesian_to_spherical(position: np.ndarray):
    """
    Cartesian → Spherical

    Returns
    -------
    range
    azimuth
    elevation
    """

    x, y, z = position

    r = np.linalg.norm(position)

    azimuth = np.arctan2(y, x)

    horizontal = np.sqrt(x**2 + y**2)

    elevation = np.arctan2(z, horizontal)

    return r, azimuth, elevation


def spherical_to_cartesian(
    radius: float,
    azimuth: float,
    elevation: float,
) -> np.ndarray:
    """
    Spherical → Cartesian
    """

    x = radius * np.cos(elevation) * np.cos(azimuth)

    y = radius * np.cos(elevation) * np.sin(azimuth)

    z = radius * np.sin(elevation)

    return np.array([x, y, z], dtype=np.float64)


def rotation_matrix_x(theta: float):

    c = np.cos(theta)
    s = np.sin(theta)

    return np.array(
        [
            [1, 0, 0],
            [0, c, -s],
            [0, s, c],
        ],
        dtype=np.float64,
    )


def rotation_matrix_y(theta: float):

    c = np.cos(theta)
    s = np.sin(theta)

    return np.array(
        [
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c],
        ],
        dtype=np.float64,
    )


def rotation_matrix_z(theta: float):

    c = np.cos(theta)
    s = np.sin(theta)

    return np.array(
        [
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )