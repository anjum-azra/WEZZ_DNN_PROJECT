"""
Augmented Proportional Navigation
"""

from __future__ import annotations

import numpy as np


class APNGuidance:

    def __init__(self, navigation_constant: float = 4.0):

        self.N = navigation_constant

    def command(self, geometry):

        # ----------------------------------------------------
        # Values already computed by Geometry
        # ----------------------------------------------------

        vc = geometry.closing_velocity

        los = geometry.los

        los_rate = geometry.los_rate

        target_acceleration = geometry.target.acceleration

        # ----------------------------------------------------
        # PN Term
        # ----------------------------------------------------

        pn = self.N * vc * np.cross(
            los_rate,
            los,
        )

        # ----------------------------------------------------
        # APN Target Compensation
        # ----------------------------------------------------

        apn = (self.N / 2.0) * target_acceleration

        # ----------------------------------------------------
        # Total Acceleration Command
        # ----------------------------------------------------

        command = pn + apn

        return command