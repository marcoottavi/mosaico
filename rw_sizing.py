# Sizing of the Reaction Wheels for GG Torque Control

"""
gg_multi_beta.py
================
GG torque components and angular momentum accumulation for multiple
beta values, held on the same axes.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# ── Configuration ─────────────────────────────────────────────────────────────
INC = np.radians(97.79)
