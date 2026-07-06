import numpy as np
import pandas as pd


def linear_combo(parent_values, coefs, intercept, noise_std, rng):
    """y = intercept + sum(coef_i * parent_i) + Gaussian noise"""
    n = len(next(iter(parent_values.values())))
    y = np.full(n, intercept, dtype=float)
    for name, coef in coefs.items():
        y += coef * parent_values[name]
    y += rng.normal(0, noise_std, size=n)
    return y

def concave_transform(x, scale = 1.0, shift= 0.0):
    """Concave, monotonically-increasing transform: scale * log1p(relu(x - shift))"""
    return scale * np.log1p(np.clip(x - shift, a_min=0, a_max=None))

def threshold_multiplicative(base, trigger, threshold, multiplier):
    """base value gets multiplied by `multiplier` whenever trigger exceeds threshold."""
    out = base.copy()
    mask = trigger > threshold
    out[mask] = out[mask] * multiplier
    return out