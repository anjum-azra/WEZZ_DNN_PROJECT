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

MISSILE_INITIAL_SPEED = 1200.0      # was 900
MISSILE_MAX_G = 40.0                # was 30
MISSILE_MASS = 150.0
MISSILE_KILL_RADIUS = 10.0          # was 5
MISSILE_MAX_FLIGHT_TIME = 120.0     # was 60

# ======================================================
# WEZ SEARCH PARAMETERS
# ======================================================

WEZ_MIN_RANGE = 5000.0
WEZ_MAX_RANGE = 80000.0
WEZ_SEARCH_STEP = 5000.0
WEZ_SEARCH_TOLERANCE = 5.0

# ======================================================
# DATASET
# ======================================================

DEFAULT_DATASET_SIZE = 1000

# ======================================================
# RANDOM SEED
# ======================================================

RANDOM_SEED = 42

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




# ======================================================
# CPU SETTINGS
# ======================================================

MAX_WORKERS = None