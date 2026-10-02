"""Weighted-network layer for comparing M1 (unweighted) and M2 (weighted) SEIR+D epidemics."""

import random
from unittest import mock

import matplotlib
import numpy as np

# main.py selects TkAgg (which needs tkinter and a display) and seeds the global RNGs at import.
_numpy_state, _python_state = np.random.get_state(), random.getstate()
with mock.patch.object(matplotlib, "use"):
    import main
np.random.set_state(_numpy_state)
random.setstate(_python_state)

__all__ = ["main"]
