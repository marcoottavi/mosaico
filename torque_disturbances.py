"""
torque_disturbances.py
=======================

Attitude disturbance torque budget for the MOSAICO hexagonal assembly
spacecraft, 200–1000 km LEO Sun-synchronous orbit.

New in this version: drag-dependent torques (aerodynamic drag) are evaluated
for three solar activity levels (low / mean / high, F10.7 = 75 / 150 / 230 SFU)
so that the solar-cycle sensitivity of the torque environment is visible
alongside the gravity-gradient and SRP torques.

Figures produced (in ./figures/):
    fig_torque_gravity_gradient.svg
    fig_torque_gravity_gradient_phi.svg
    fig_torque_gravity_gradient_theta.svg
    fig_torque_drag.svg                    <- three solar activity bands
    fig_torque_drag_solar_activity.svg     <- NEW: per-activity detail
    fig_torque_srp.svg
    fig_torque_budget.svg                  <- three drag curves + other torques
    fig_torque_budget_solar_activity.svg   <- NEW: full budget per activity
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from orbital_perturbations import R_EARTH, MU_EARTH
import orbital_perturbations as op
from assembler import compute_mass_properties

# ============================================================
# Output directory
# ============================================================
OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# Altitude grid
# ============================================================
H_MIN, H_MAX, N_PTS = 200.0, 1000.0, 80
ALT = np.linspace(H_MIN, H_MAX, N_PTS)

# ============================================================
# Spacecraft geometry / mass assumptions
# ============================================================
N_ATOMS        = 1
NOMINAL_EDGE   = 1.65    # m
NOMINAL_WIDTH  = 0.4     # m
NOMINAL_MASS   = 1250    # kg, total
COM_COP        = 0.2     # m

cr_min = 1.0
cr_max = 2.0
cr_nom = 1.7
Cd     = 2.2
n_exp  = 6

# Drag Areas
am_nom = 0.00945
am_min = 0.00274
am_max = 0.0229

# Solar Areas
a_srp = 49.5 # m^2

# ============================================================
# Build min / nominal / max spacecraft properties
# ============================================================

dict_nominal = compute_mass_properties(NOMINAL_EDGE, NOMINAL_WIDTH, NOMINAL_MASS/7, N_ATOMS) # mass is that of one hexagon
mass_nom    = dict_nominal["total_mass"]
surface_nom = dict_nominal["projected_area_xy"]
inertia_nom = dict_nominal["inertia_about_origin"]


# ============================================================
# Matplotlib styling
# ============================================================
plt.rcParams.update({
    "figure.figsize":    (10, 6),
    "figure.dpi":        300,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "font.family":       "serif",
    "font.size":         14,
    "axes.titleweight":  "bold",
    "axes.grid":         True,
    "grid.linestyle":    "--",
    "grid.linewidth":    0.6,
    "grid.alpha":        0.3,
    "legend.frameon":    True,
    "legend.framealpha": 0.92,
    "lines.linewidth":   2.0,
})

# Activity-level display metadata (mirrors ACT_STYLE in plot_orb_perturbations)
ACT_STYLE = {
    'low':  dict(color="#2ca02c", ls=':', lw=1.8,
                 label='Low activity (F10.7=75)',   fill='#98df8a'),
    'mean': dict(color="#1f77b4", ls='-', lw=2.2,
                 label='Mean activity (F10.7=150)', fill='#aec7e8'),
    'high': dict(color="#d62728", ls='--', lw=1.8,
                 label='High activity (F10.7=230)', fill='#ffbb78'),
}

# ============================================================
# Axis helper
# ============================================================
def style_axes(ax, ylabel="Disturbance torque [N·m]",
               xlabel="Altitude [km]", logy=True):
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    if logy:
        ax.set_yscale("log")
    ax.set_xlim(H_MIN, H_MAX)
    ax.grid(True, which="major")
    ax.grid(True, which="minor", alpha=0.25)
    ax.tick_params(axis="both", labelsize=11)

# ============================================================
# Gravity-gradient torque
# ============================================================
def gravity_gradient_torque(altitude_km, inertia_matrix):

    r = (R_EARTH + altitude_km)*1e3
    phi   = np.linspace(0, 2*np.pi, 100, endpoint=False)
    theta = np.linspace(0, np.pi, 50)
    phi, theta = np.meshgrid(phi, theta)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    r_vec  = np.stack((x, y, z), axis=-1)
    r_norm = np.linalg.norm(r_vec, axis=-1)
    Ir     = np.einsum('ij,...j->...i', inertia_matrix, r_vec)
    torque = (3 * MU_EARTH / r_norm[..., None]**5) * np.cross(r_vec, Ir)*1e9
    torque_mag = np.linalg.norm(torque, axis=-1)
    max_idx = np.unravel_index(np.argmax(torque_mag), torque_mag.shape)

    return torque_mag[max_idx], phi[max_idx], theta[max_idx]

# ============================================================
# Drag torque
# ============================================================
def drag_torque(h_km, area_m2, mass_kg, psi_deg,
                Cd=2.2, n_exp=6, com_cop_m=0.1, activity='mean'):
    a_m    = area_m2 / mass_kg
    a_drag = op.drag_accel(h_km, a_m, psi_deg,
                           Cd=Cd, n_exp=n_exp, activity=activity)
    return mass_kg * a_drag * com_cop_m   # [N·m]

# ============================================================
# SRP torque (no solar activity dependence in the SRP model)
# ============================================================
def srp_torque(area_m2, mass_kg, cr, com_cop_m=COM_COP):
    a_m   = area_m2 / mass_kg
    a_srp = op.srp_accel(a_m, cr, eclipse_fraction=0.0)
    return mass_kg * a_srp * com_cop_m   # [N·m]

# ============================================================
# Figure 1: Gravity-gradient band + worst-case angles
# ============================================================
def fig_gravity_gradient_torque():
    gg = []
    for alt in ALT:
        gg += [gravity_gradient_torque(alt, inertia_nom)[0]]

    plt.figure(figsize=(10,6))
    plt.plot(ALT, gg, color="#e0590b", linewidth=2.3, label="Gravity-gradient torque")
    return gg

# ============================================================
# Figure 2: Drag torque
# ============================================================

def fig_drag_torque_solar_activity():
    """
    Fig — NEW: three-panel figure (one row per solar activity level)
    showing drag torque low/nom/high inertia bands separately.
    Makes it easy to read off the torque range at a specific activity level.
    """

    td_high = drag_torque(ALT, am_max*NOMINAL_MASS, NOMINAL_MASS, psi_deg = 0.0,
                Cd=2.2, n_exp=6, com_cop_m=COM_COP, activity='high')
    td_nom = drag_torque(ALT, am_nom*NOMINAL_MASS, NOMINAL_MASS, psi_deg = 90.0,
                Cd=2.2, n_exp=6, com_cop_m=COM_COP, activity='mean')
    td_low = drag_torque(ALT, am_min*NOMINAL_MASS, NOMINAL_MASS, psi_deg = 180.0,
                Cd=2.2, n_exp=6, com_cop_m=COM_COP, activity='low')
    
    fig, ax = plt.subplots()
    ax.fill_between(ALT, td_low, td_high,
                    color="#d4a017", alpha=0.25,
                    label="SRP torque envelope")
    ax.plot(ALT, td_nom, color="#9c7a00", linewidth=2.3, label="Nominal")
    style_axes(ax)
    ax.set_title("Drag Torque vs. Altitude")
    ax.legend()
    fig.savefig(os.path.join(OUT_DIR, "fig_torque_drag.svg"))
    plt.close(fig)

    return {"low": td_low, "nom": td_nom, "high": td_high}
    


# ============================================================
# Figure 3: SRP torque band (unchanged — no activity dependence)
# ============================================================
def fig_srp_torque():
    srp_low_val  = srp_torque(a_srp, mass_nom, cr_min, COM_COP)
    srp_nom_val  = srp_torque(a_srp, mass_nom, cr_nom, COM_COP)
    srp_high_val = srp_torque(a_srp, mass_nom, cr_max, COM_COP)

    srp_low  = np.full_like(ALT, srp_low_val)
    srp_nom  = np.full_like(ALT, srp_nom_val)
    srp_high = np.full_like(ALT, srp_high_val)

    fig, ax = plt.subplots()
    ax.fill_between(ALT, srp_low, srp_high,
                    color="#d4a017", alpha=0.25,
                    label="SRP torque envelope")
    ax.plot(ALT, srp_nom, color="#9c7a00", linewidth=2.3, label="Nominal")
    style_axes(ax)
    ax.set_title("Solar Radiation Pressure Torque vs. Altitude")
    ax.legend()
    fig.savefig(os.path.join(OUT_DIR, "fig_torque_srp.svg"))
    plt.close(fig)
    return {"low": srp_low, "nom": srp_nom, "high": srp_high}


# ============================================================
# Figure 5: Combined budget — all three solar activity levels  (NEW)
# ============================================================
def fig_torque_budget_solar_activity(gg, drag, srp):
    """
    Full torque budget overlaying drag nominal curves for all three
    solar activity levels, revealing where solar cycle modulates drag
    relative to gravity-gradient and SRP.
    """
    fig, ax = plt.subplots(figsize=(10, 6.5))

    # Gravity-gradient (no activity dependence)
    ax.plot(ALT, gg, color="#e0590b", linewidth=2.4,
            label="Gravity-gradient Torque", zorder=5)

    # SRP (no activity dependence)
    ax.fill_between(ALT, srp["low"], srp["high"],
                    color="#d4a017", alpha=0.18)
    ax.plot(ALT, srp["nom"], color="#9c7a00", linestyle="--",
            linewidth=2.0, label="SRP Torque nominal", zorder=5)

    # Drag — three activity levels
    ax.fill_between(ALT, drag["low"], drag["high"],
                    color="#f5524f", alpha=0.18)
    ax.plot(ALT, drag["nom"], color="#f60400", linestyle="--",
            linewidth=2.0, label="Drag Torque nominal", zorder=5)

    style_axes(ax)
    ax.set_title("Disturbance Torque Budget")
    ax.legend(ncol=2, loc="upper right", fontsize=9)
    fig.savefig(os.path.join(OUT_DIR,
                             "fig_torque_budget_solar_activity.svg"))
    plt.close(fig)

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("Computing gravity-gradient torque band...")
    gg = fig_gravity_gradient_torque()

    print("Computing drag torque bands (slow, nominal, fast, see report)...")
    drag = fig_drag_torque_solar_activity()

    print("Computing SRP torque band...")
    srp = fig_srp_torque()

    print("Computing full torque budget (gravity-gradient + drag + SRP)...")
    fig_torque_budget_solar_activity(gg, drag, srp)

    print("Done. Figures saved in:", os.path.abspath(OUT_DIR))