"""
geometry/geometry.py

Builds the complete engagement geometry.

Author: Anjum Azra
"""

from __future__ import annotations

import numpy as np

from config.config import MISSILE_INITIAL_SPEED

from geometry.transforms import (
    shooter_position,
    shooter_velocity,
    target_position,
    target_velocity,
    missile_position,
    missile_velocity,
)

from geometry.vectors import (
    subtract,
    magnitude,
    normalize,
    dot,
    cross,
)

from entities.shooter import Shooter
from entities.target import Target
from entities.missile import Missile


class Geometry:
    """
    Creates the complete engagement geometry from the
    seven paper input parameters.
    """

    def __init__(
        self,
        shooter_altitude_ft: float,
        shooter_speed_knots: float,
        shooter_pitch_deg: float,
        target_altitude_ft: float,
        target_speed_knots: float,
        target_heading_deg: float,
        target_off_boresight_deg: float,
        launch_distance: float,
    ):

        # =====================================================
        # SHOOTER
        # =====================================================

        sp = shooter_position(shooter_altitude_ft)

        sv = shooter_velocity(
            shooter_speed_knots,
            shooter_pitch_deg,
        )

        self.shooter = Shooter(
            position=sp,
            velocity=sv,
            altitude=sp[2],
            speed=np.linalg.norm(sv),
            pitch=shooter_pitch_deg,
        )

        # =====================================================
        # TARGET
        # =====================================================

        tp = target_position(
            launch_distance,
            target_off_boresight_deg,
            target_altitude_ft,
        )

        tv = target_velocity(
            target_speed_knots,
            target_heading_deg,
        )

        self.target = Target(
            position=tp,
            velocity=tv,
            acceleration=np.zeros(3),
            altitude=tp[2],
            speed=np.linalg.norm(tv),
            heading=target_heading_deg,
        )

        # =====================================================
        # MISSILE
        # =====================================================

        mp = missile_position(sp)

        mv = missile_velocity(
            sv,
            MISSILE_INITIAL_SPEED,
        )

        self.missile = Missile(
            position=mp,
            velocity=mv,
        )

        # =====================================================
        # Compute initial geometry
        # =====================================================

        self.update()

    # =========================================================

    def update(self):

        # Relative Position

        self.relative_position = subtract(
            self.target.position,
            self.missile.position,
        )

        # Relative Velocity

        self.relative_velocity = subtract(
            self.target.velocity,
            self.missile.velocity,
        )

        # Range

        self.range = magnitude(
            self.relative_position
        )

        # LOS

        self.los = normalize(
            self.relative_position
        )

        # Closing Velocity

        if self.range > 1e-6:

            self.closing_velocity = (

                -dot(
                    self.relative_position,
                    self.relative_velocity,
                )

                / self.range

            )

        else:

            self.closing_velocity = 0.0

        # LOS Rate

        if self.range > 1e-6:

            self.los_rate = cross(

                self.relative_position,

                self.relative_velocity,

            ) / (self.range ** 2)

        else:

            self.los_rate = np.zeros(3)

    # =========================================================

    def print_geometry(self):

        print("\n==============================")

        print("ENGAGEMENT GEOMETRY")

        print("==============================")

        print("\nShooter Position")

        print(self.shooter.position)

        print("\nTarget Position")

        print(self.target.position)

        print("\nMissile Position")

        print(self.missile.position)

        print("\nRelative Position")

        print(self.relative_position)

        print("\nRelative Velocity")

        print(self.relative_velocity)

        print("\nRange")

        print(self.range)

        print("\nLOS")

        print(self.los)

        print("\nLOS Rate")

        print(self.los_rate)

        print("\nClosing Velocity")

        print(self.closing_velocity)