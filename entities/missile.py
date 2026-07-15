"""
Missile Entity
"""

from dataclasses import dataclass, field
import numpy as np

from config.config import (
    MISSILE_MAX_G,
    MISSILE_KILL_RADIUS,
    MISSILE_MAX_FLIGHT_TIME,
    GRAVITY,
)


@dataclass
class Missile:

    position: np.ndarray

    velocity: np.ndarray

    acceleration: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )

    max_g: float = MISSILE_MAX_G

    kill_radius: float = MISSILE_KILL_RADIUS

    max_flight_time: float = MISSILE_MAX_FLIGHT_TIME

    flight_time: float = 0.0

    def __post_init__(self):

        self.position = np.asarray(
            self.position,
            dtype=np.float64,
        )

        self.velocity = np.asarray(
            self.velocity,
            dtype=np.float64,
        )

        self.acceleration = np.asarray(
            self.acceleration,
            dtype=np.float64,
        )

    @property
    def speed(self):

        return np.linalg.norm(self.velocity)

    def set_acceleration(self, command):

        command = np.asarray(command, dtype=np.float64)

        limit = self.max_g * GRAVITY

        mag = np.linalg.norm(command)

        if mag > limit:

            command = command / mag

            command *= limit

        self.acceleration = command

    def update(self, dt):

        self.velocity += self.acceleration * dt

        self.position += self.velocity * dt

        self.flight_time += dt

    @property
    def alive(self):

        return self.flight_time < self.max_flight_time

    def __str__(self):

        return (
            f"Missile\n"
            f"Position : {self.position}\n"
            f"Velocity : {self.velocity}\n"
            f"Acceleration : {self.acceleration}\n"
            f"Speed : {self.speed:.2f} m/s\n"
            f"Flight Time : {self.flight_time:.2f} s"
        )