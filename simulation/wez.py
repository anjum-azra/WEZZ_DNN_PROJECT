"""
simulation/wez.py

Weapon Engagement Zone (WEZ) Estimator
"""

from geometry.geometry import Geometry
from simulation.engagement import Engagement

from config.config import (
    WEZ_MIN_RANGE,
    WEZ_MAX_RANGE,
    WEZ_SEARCH_STEP,
    WEZ_SEARCH_TOLERANCE,
)


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

    # ---------------------------------------------------------

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

    # ---------------------------------------------------------

    def find_rmax(
        self,
        minimum=WEZ_MIN_RANGE,
        maximum=WEZ_MAX_RANGE,
        tolerance=WEZ_SEARCH_TOLERANCE,
        step=WEZ_SEARCH_STEP,
    ):
        """
        Estimate the maximum launch range (Rmax).
        """

        # Cannot intercept even at minimum distance
        if not self.simulate(minimum):
            return None

        low = minimum
        high = minimum

        # -------------------------
        # Bracketing
        # -------------------------

        while high <= maximum:

            if self.simulate(high):
                low = high
                high += step
            else:
                break

        # Missile still hits at maximum search range
        if high > maximum:
            return float(maximum)

        # -------------------------
        # Binary Search
        # -------------------------

        while (high - low) > tolerance:

            mid = (low + high) / 2.0

            if self.simulate(mid):
                low = mid
            else:
                high = mid

        return (low + high) / 2.0