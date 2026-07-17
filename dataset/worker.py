"""
dataset/worker.py

Multiprocessing worker for WEZ dataset generation.

Author: Anjum Azra
"""

from simulation.wez import WEZEstimator


def process_sample(sample):

    estimator = WEZEstimator(

        shooter_altitude_ft=float(sample[0]),
        shooter_speed_knots=float(sample[1]),
        shooter_pitch_deg=float(sample[2]),

        target_altitude_ft=float(sample[3]),
        target_speed_knots=float(sample[4]),
        target_heading_deg=float(sample[5]),
        target_off_boresight_deg=float(sample[6]),

    )

    rmax = estimator.find_rmax()

    if rmax is None:
        return None

    return [

        float(sample[0]),
        float(sample[1]),
        float(sample[2]),

        float(sample[3]),
        float(sample[4]),
        float(sample[5]),
        float(sample[6]),

        float(rmax),

    ]