"""
simulation/wez.py

Weapon Engagement Zone (WEZ) Estimator
Finds the Maximum Launch Range (Rmax)
using Binary Search.
"""

from geometry.geometry import Geometry
from simulation.engagement import Engagement


class WEZEstimator:

    def __init__(
        self,
        shooter_altitude_ft,
        shooter_speed_knots,
        shooter_pitch_deg,
        target_altitude_ft,
        target_speed_knots,
        target_heading_deg,
        target_off_boresight_deg,
    ):

        self.shooter_altitude_ft = shooter_altitude_ft
        self.shooter_speed_knots = shooter_speed_knots
        self.shooter_pitch_deg = shooter_pitch_deg

        self.target_altitude_ft = target_altitude_ft
        self.target_speed_knots = target_speed_knots
        self.target_heading_deg = target_heading_deg
        self.target_off_boresight_deg = target_off_boresight_deg

    # -----------------------------------------------------

    def simulate(self, launch_distance):

        geometry = Geometry(

            shooter_altitude_ft=self.shooter_altitude_ft,

            shooter_speed_knots=self.shooter_speed_knots,

            shooter_pitch_deg=self.shooter_pitch_deg,

            target_altitude_ft=self.target_altitude_ft,

            target_speed_knots=self.target_speed_knots,

            target_heading_deg=self.target_heading_deg,

            target_off_boresight_deg=self.target_off_boresight_deg,

            launch_distance=launch_distance,

        )

        engagement = Engagement(geometry)

        return engagement.run()

    # -----------------------------------------------------

    def find_rmax(

        self,

        minimum=1000,

        maximum=60000,

        tolerance=5,

    ):

        low = minimum

        high = maximum

        while (high - low) > tolerance:

            mid = (low + high) / 2

            hit = self.simulate(mid)

            if hit:

                low = mid

            else:

                high = mid

        return low