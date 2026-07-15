"""
Target Aircraft Entity
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class Target:

    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    altitude: float
    speed: float
    heading: float

    def __post_init__(self):

        self.position = np.asarray(self.position, dtype=np.float64)

        self.velocity = np.asarray(self.velocity, dtype=np.float64)

        self.acceleration = np.asarray(
            self.acceleration,
            dtype=np.float64,
        )

    def update(self, dt):

        self.velocity += self.acceleration * dt

        self.position += self.velocity * dt

    @property
    def velocity_magnitude(self):

        return np.linalg.norm(self.velocity)

    def __str__(self):

        return (
            f"Target\n"
            f"Position : {self.position}\n"
            f"Velocity : {self.velocity}\n"
            f"Acceleration : {self.acceleration}\n"
            f"Heading : {self.heading:.2f} deg"
        )