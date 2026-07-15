"""
Proportional Navigation Guidance
"""

from __future__ import annotations

import numpy as np


class PNGuidance:

    def __init__(self, navigation_constant: float = 4.0):
        self.N = navigation_constant

    def command(self, geometry):

        vc = geometry.closing_velocity

        los_rate = geometry.los_rate

        los = geometry.los

        # Pure PN acceleration
        acceleration = self.N * vc * np.cross(
            los_rate,
            los,
        )

        return acceleration