



"""
dataset/lhs.py

Latin Hypercube Sampling

Author: Anjum Azra
"""

import numpy as np

from scipy.stats import qmc

from config.config import RANDOM_SEED


def generate_lhs_samples(bounds, number_of_samples):

    dimensions = len(bounds)

    sampler = qmc.LatinHypercube(

        d=dimensions,

        seed=RANDOM_SEED,

    )

    samples = sampler.random(number_of_samples)

    lower_bounds = np.array([b[0] for b in bounds])

    upper_bounds = np.array([b[1] for b in bounds])

    scaled = qmc.scale(

        samples,

        lower_bounds,

        upper_bounds,

    )

    return scaled