"""
Global configuration file.

Author: Anjum Azra
Project: WEZ Maximum Launch Range Estimation using Deep Neural Networks
"""

# ======================================================
# SIMULATION PARAMETERS
# ======================================================

TIME_STEP = 0.01                 # seconds
MAX_SIMULATION_TIME = 60.0       # seconds

# ======================================================
# MISSILE PARAMETERS
# ======================================================

MISSILE_INITIAL_SPEED = 900.0    # m/s
MISSILE_MAX_G = 30.0             # g
MISSILE_MASS = 150.0             # kg
MISSILE_KILL_RADIUS = 5.0        # meters
MISSILE_MAX_FLIGHT_TIME = 60.0   # seconds

# ======================================================
# GUIDANCE
# ======================================================

NAVIGATION_CONSTANT = 4

# ======================================================
# ENVIRONMENT
# ======================================================

GRAVITY = 9.81

# ======================================================
# DATASET
# ======================================================

NUMBER_OF_SCENARIOS = 50000

TRAIN_SPLIT = 0.70
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15

# ======================================================
# PARAMETER LIMITS (Paper)
# ======================================================

SHOOTER_ALTITUDE = (1000, 45000)      # ft
SHOOTER_SPEED = (400, 600)            # knots
SHOOTER_PITCH = (-45, 45)             # deg

TARGET_ALTITUDE = (1000, 45000)       # ft
TARGET_SPEED = (400, 600)             # knots
TARGET_HEADING = (-180, 180)          # deg
TARGET_OFFBORESIGHT = (-60, 60)       # deg