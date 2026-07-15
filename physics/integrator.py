"""
physics/integrator.py

Simple Euler Integrator
"""

from __future__ import annotations


class EulerIntegrator:

    def __init__(self, dt: float):

        self.dt = dt

    def integrate(self, body):

        body.velocity += body.acceleration * self.dt

        body.position += body.velocity * self.dt

        if hasattr(body, "flight_time"):

            body.flight_time += self.dt