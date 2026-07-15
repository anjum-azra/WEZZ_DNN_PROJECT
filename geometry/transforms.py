"""
geometry/transforms.py

Coordinate transformation utilities for the WEZ simulator.

Author: Anjum Azra
"""

from __future__ import annotations

import numpy as np

from utils.units import (
    feet_to_meters,
    knots_to_mps,
    deg_to_rad,
)

from geometry.vectors import vector


# ==========================================================
# SHOOTER
# ==========================================================

def shooter_position(altitude_ft: float) -> np.ndarray:
    """
    Shooter is placed at the origin.

    Returns
    -------
    np.ndarray
        [0,0,z]
    """

    altitude = feet_to_meters(altitude_ft)

    return vector(0.0, 0.0, altitude)


def shooter_velocity(
    speed_knots: float,
    pitch_deg: float,
) -> np.ndarray:
    """
    Shooter velocity vector.
    """

    speed = knots_to_mps(speed_knots)

    pitch = deg_to_rad(pitch_deg)

    vx = speed * np.cos(pitch)

    vy = 0.0

    vz = speed * np.sin(pitch)

    return vector(vx, vy, vz)


# ==========================================================
# TARGET
# ==========================================================

def target_position(
    launch_distance: float,
    off_boresight_deg: float,
    altitude_ft: float,
) -> np.ndarray:
    """
    Target initial position.
    """

    angle = deg_to_rad(off_boresight_deg)

    altitude = feet_to_meters(altitude_ft)

    x = launch_distance * np.cos(angle)

    y = launch_distance * np.sin(angle)

    z = altitude

    return vector(x, y, z)


def target_velocity(
    speed_knots: float,
    heading_deg: float,
) -> np.ndarray:
    """
    Target velocity vector.
    """

    speed = knots_to_mps(speed_knots)

    heading = deg_to_rad(heading_deg)

    vx = speed * np.cos(heading)

    vy = speed * np.sin(heading)

    vz = 0.0

    return vector(vx, vy, vz)


# ==========================================================
# MISSILE
# ==========================================================

def missile_position(
    shooter_position_vector: np.ndarray,
) -> np.ndarray:
    """
    Missile launches from shooter.
    """

    return shooter_position_vector.copy()


def missile_velocity(
    shooter_velocity_vector: np.ndarray,
    missile_speed: float,
) -> np.ndarray:
    """
    Missile initially flies
    in the shooter's direction.
    """

    norm = np.linalg.norm(shooter_velocity_vector)

    if norm < 1e-9:

        return vector(missile_speed, 0.0, 0.0)

    direction = shooter_velocity_vector / norm

    return direction * missile_speed