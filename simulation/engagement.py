"""
simulation/engagement.py

Runs one missile-target engagement.
"""

from __future__ import annotations

import numpy as np

from config.config import TIME_STEP

from physics.integrator import EulerIntegrator

from guidance.apn import APNGuidance


class Engagement:

    def __init__(self, geometry):

        self.geometry = geometry

        self.integrator = EulerIntegrator(TIME_STEP)

        self.guidance = APNGuidance()

        self.time = 0.0

        self.intercept = False

    # -------------------------------------------------

    def hit(self):

        return self.geometry.range <= self.geometry.missile.kill_radius

    # -------------------------------------------------

    def step(self):

        command = self.guidance.command(

            self.geometry

        )

        self.geometry.missile.set_acceleration(

            command

        )

        self.integrator.integrate(

            self.geometry.missile

        )

        self.integrator.integrate(

            self.geometry.target

        )

        self.geometry.update()

        self.time += TIME_STEP

        if self.hit():

            self.intercept = True

    # -------------------------------------------------

    def run(self):

        while (

            self.geometry.missile.alive

            and

            not self.intercept

        ):

            self.step()

        return self.intercept