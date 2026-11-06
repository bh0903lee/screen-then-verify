"""Tolerance-band utilities."""
import sys
import logging
from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_REPO / "stv"), str(_REPO / "experiments")]
import numpy as np
import graph_env as base
import common as d0
log = logging.getLogger("bands")
KEYS = base.KEYS; TOL = base.TOL; DEV = d0.DEV
_HARD = np.array([0.0, 0.0, 1.0, 1.0, 0.0])

def _uniform(f):  return np.full(5, f, np.float32)
def _hard(f):
    v = np.ones(5, np.float32); v[_HARD > 0] = f; return v

BAND_DEFS = {
    "u1.0":    (_uniform(1.0),  "worst"),
    "u0.8":    (_uniform(0.8),  "worst"),
    "u0.6":    (_uniform(0.6),  "worst"),
    "u0.5":    (_uniform(0.5),  "worst"),
    "u0.4":    (_uniform(0.4),  "worst"),
    "u0.3":    (_uniform(0.3),  "worst"),
    "u0.2":    (_uniform(0.2),  "worst"),
    "u0.15":   (_uniform(0.15), "worst"),
    "u0.1":    (_uniform(0.1),  "worst"),
    "u0.05":   (_uniform(0.05), "worst"),
    "hard0.4": (_hard(0.4),     "worst"),
    "mean0.5": (_uniform(0.5),  "mean"),
}

FINE_BANDS = ["u1.0", "u0.8", "u0.6", "u0.4", "u0.3", "u0.2", "u0.15", "u0.1", "u0.05"]


def sat_band(res, band):
    factors, crit = band
    scaled = np.abs(np.asarray(res, np.float32)) / factors
    return bool((scaled.max() if crit == "worst" else scaled.mean()) <= 1.0)


def sat_traj(traj, band):
    return any(sat_band(r, band) for r in traj)


