"""
Unit conversion utilities.
"""

import numpy as np
from utils.constants import *


def feet_to_meters(value):
    return value * FEET2METER


def meters_to_feet(value):
    return value * METER2FEET


def knots_to_mps(value):
    return value * KNOT2MPS


def mps_to_knots(value):
    return value * MPS2KNOT


def deg_to_rad(value):
    return value * DEG2RAD


def rad_to_deg(value):
    return value * RAD2DEG


def normalize_angle(angle):
    """
    Normalize angle into [-180,180] degrees.
    """
    return ((angle + 180) % 360) - 180