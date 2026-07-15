"""
Shooter Aircraft Entity
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class Shooter:
    """
    Represents the launch aircraft.
    """

    position: np.ndarray
    velocity: np.ndarray
    altitude: float
    speed: float
    pitch: float

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=np.float64)
        self.velocity = np.asarray(self.velocity, dtype=np.float64)

    @property
    def velocity_magnitude(self):
        return np.linalg.norm(self.velocity)

    def __str__(self):
        return (
            f"Shooter\n"
            f"Position : {self.position}\n"
            f"Velocity : {self.velocity}\n"
            f"Altitude : {self.altitude:.2f} m\n"
            f"Speed    : {self.speed:.2f} m/s\n"
            f"Pitch    : {self.pitch:.2f} deg"
        )