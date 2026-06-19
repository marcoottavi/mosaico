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
NOMINAL_MASS   = 1250    # kg
COM_COP        = 0.5     # m

SCALING_MIN = 0.8
SCALING_MAX = 1.2

cr_min = 1.0
cr_max = 2.0
cr_nom = 1.3
Cd     = 2.2
n_exp  = 6

# ============================================================
# Build min / nominal / max spacecraft properties
# ============================================================
edge_min = NOMINAL_EDGE * SCALING_MIN
edge_max = NOMINAL_EDGE * SCALING_MAX
thickness_min = NOMINAL_WIDTH * SCALING_MIN
thickness_max = NOMINAL_WIDTH * SCALING_MAX
mass_min = NOMINAL_MASS * SCALING_MIN
mass_max = NOMINAL_MASS * SCALING_MAX
com_cop_min = COM_COP * SCALING_MIN
com_cop_max = COM_COP * SCALING_MAX
com_cop_nom = COM_COP

dict_nominal = compute_mass_properties(NOMINAL_EDGE, NOMINAL_WIDTH, NOMINAL_MASS, N_ATOMS)
dict_min     = compute_mass_properties(edge_min,     thickness_min, mass_min,     N_ATOMS)
dict_max     = compute_mass_properties(edge_max,     thickness_max, mass_max,     N_ATOMS)

mass_nom    = dict_nominal["total_mass"]
mass_min_   = dict_min["total_mass"]
mass_max_   = dict_max["total_mass"]
surface_nom = dict_nominal["projected_area_xy"]
surface_min = dict_min["projected_area_xy"]
surface_max = dict_max["projected_area_xy"]
inertia_nom = dict_nominal["inertia_about_origin"]
inertia_min = dict_min["inertia_about_origin"]
inertia_max = dict_max["inertia_about_origin"]

am_nom = surface_nom / mass_nom
am_min = surface_min / mass_max_
am_max = surface_max / mass_min_

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
    inertia_km = inertia_matrix
    r = (R_EARTH + altitude_km)*1e3
    phi   = np.linspace(0, 2*np.pi, 100, endpoint=False)
    theta = np.linspace(0, np.pi, 50)
    phi, theta = np.meshgrid(phi, theta)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    r_vec  = np.stack((x, y, z), axis=-1)
    r_norm = np.linalg.norm(r_vec, axis=-1)
    Ir     = np.einsum('ij,...j->...i', inertia_km, r_vec)
    torque = (3 * MU_EARTH / r_norm[..., None]**5) * np.cross(r_vec, Ir)
    torque_mag = np.linalg.norm(torque, axis=-1)*1e9
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
def srp_torque(area_m2, mass_kg, cr, com_cop_m=0.1):
    a_m   = area_m2 / mass_kg
    a_srp = op.srp_accel(a_m, cr, eclipse_fraction=0.0)
    return mass_kg * a_srp * com_cop_m   # [N·m]

# ============================================================
# Figure 1: Gravity-gradient band + worst-case angles
# ============================================================
def fig_gravity_gradient_torque():
    gg_low, gg_nom, gg_high   = [], [], []
    phi_low, phi_nom, phi_high = [], [], []
    theta_low, theta_nom, theta_high = [], [], []

    for h in ALT:
        t_l, p_l, th_l = gravity_gradient_torque(h, inertia_min)
        t_n, p_n, th_n = gravity_gradient_torque(h, inertia_nom)
        t_h, p_h, th_h = gravity_gradient_torque(h, inertia_max)
        gg_low.append(t_l);  phi_low.append(p_l);  theta_low.append(th_l)
        gg_nom.append(t_n);  phi_nom.append(p_n);  theta_nom.append(th_n)
        gg_high.append(t_h); phi_high.append(p_h); theta_high.append(th_h)

    gg_low   = np.array(gg_low);   gg_nom  = np.array(gg_nom)
    gg_high  = np.array(gg_high)
    phi_low  = np.array(phi_low);  phi_nom = np.array(phi_nom)
    phi_high = np.array(phi_high)
    theta_low  = np.array(theta_low)
    theta_nom  = np.array(theta_nom)
    theta_high = np.array(theta_high)

    # Torque magnitude
    fig, ax = plt.subplots()
    ax.fill_between(ALT, gg_low, gg_high, color="#4c72b0", alpha=0.20,
                    label="Inertia envelope (±20%)")
    ax.plot(ALT, gg_nom, color="#1f4e79", linewidth=2.3, label="Nominal")
    style_axes(ax)
    ax.set_title("Gravity-Gradient Torque vs. Altitude")
    ax.legend()
    fig.savefig(os.path.join(OUT_DIR, "fig_torque_gravity_gradient.svg"))
    plt.close(fig)

    # Worst-case phi
    phi_low_deg  = np.rad2deg(np.unwrap(phi_low))
    phi_nom_deg  = np.rad2deg(np.unwrap(phi_nom))
    phi_high_deg = np.rad2deg(np.unwrap(phi_high))
    fig, ax = plt.subplots()
    for arr, lbl, col in [
        (phi_low_deg,  "low inertia",     "#4c72b0"),
        (phi_nom_deg,  "nominal inertia", "#1f4e79"),
        (phi_high_deg, "high inertia",    "#08306b"),
    ]:
        ax.plot(ALT, arr, color=col, linewidth=2.0,
                label=rf"Worst-case $\phi$ ({lbl})")
    style_axes(ax, ylabel=r"Worst-case $\phi$ [deg]", logy=False)
    ax.set_title(r"Worst-case Gravity-Gradient Angle $\phi$ vs. Altitude")
    ax.legend()
    fig.savefig(os.path.join(OUT_DIR,
                             "fig_torque_gravity_gradient_phi.svg"))
    plt.close(fig)

    # Worst-case theta
    theta_low_deg  = np.rad2deg(theta_low)
    theta_nom_deg  = np.rad2deg(theta_nom)
    theta_high_deg = np.rad2deg(theta_high)
    fig, ax = plt.subplots()
    for arr, lbl, col in [
        (theta_low_deg,  "low inertia",     "#4c72b0"),
        (theta_nom_deg,  "nominal inertia", "#1f4e79"),
        (theta_high_deg, "high inertia",    "#08306b"),
    ]:
        ax.plot(ALT, arr, color=col, linewidth=2.0,
                label=rf"Worst-case $\theta$ ({lbl})")
    style_axes(ax, ylabel=r"Worst-case $\theta$ [deg]", logy=False)
    ax.set_title(r"Worst-case Gravity-Gradient Angle $\theta$ vs. Altitude")
    ax.legend()
    fig.savefig(os.path.join(OUT_DIR,
                             "fig_torque_gravity_gradient_theta.svg"))
    plt.close(fig)

    return (
        {"low": gg_low, "nom": gg_nom, "high": gg_high},
        {"low": phi_low,   "nom": phi_nom,   "high": phi_high},
        {"low": theta_low, "nom": theta_nom, "high": theta_high},
    )

# ============================================================
# Figure 2: Drag torque — all three solar activity levels
# ============================================================
def _drag_torque_envelopes():
    """
    Compute drag torque (low/nom/high inertia × low/mean/high activity)
    and return nested dict:
        { activity: {'low': arr, 'nom': arr, 'high': arr} }
    """
    psi_grid = np.linspace(0.0, 180.0, 37)
    result = {}
    for act in op.SOLAR_ACTIVITIES:
        lo_arr, nom_arr, hi_arr = [], [], []
        for h in ALT:
            # Lower bound: smallest area, largest mass, anti-bulge, min lever arm
            t_low = drag_torque(h, surface_min, mass_max_, 180.0,
                                Cd=Cd, n_exp=n_exp,
                                com_cop_m=com_cop_min, activity=act)
            # Upper bound: largest area, smallest mass, bulge, max lever arm
            t_high = drag_torque(h, surface_max, mass_min_, 0.0,
                                 Cd=Cd, n_exp=n_exp,
                                 com_cop_m=com_cop_max, activity=act)
            # Nominal: average over psi, nominal geometry
            nom_vals = [drag_torque(h, surface_nom, mass_nom, psi,
                                    Cd=Cd, n_exp=n_exp,
                                    com_cop_m=com_cop_nom, activity=act)
                        for psi in psi_grid]
            lo_arr.append(t_low)
            nom_arr.append(np.mean(nom_vals))
            hi_arr.append(t_high)
        result[act] = {
            'low':  np.array(lo_arr),
            'nom':  np.array(nom_arr),
            'high': np.array(hi_arr),
        }
    return result


def fig_drag_torque(drag_env):
    """
    Fig 2 — drag torque bands for all three solar activity levels
    on one axes, replacing the original single-activity version.
    """
    fig, ax = plt.subplots()
    for act in op.SOLAR_ACTIVITIES:
        st = ACT_STYLE[act]
        d  = drag_env[act]
        ax.fill_between(ALT, d['low'], d['high'],
                        color=st['fill'], alpha=0.22)
        ax.plot(ALT, d['nom'], color=st['color'],
                linestyle=st['ls'], linewidth=st['lw'],
                label=f"Nominal – {st['label']}")

    style_axes(ax)
    ax.set_title("Aerodynamic Drag Torque vs. Altitude\n"
                 "(NRLMSISE-00, shaded = geometry + ψ envelope at each activity level)")
    ax.legend(fontsize=10)
    fig.savefig(os.path.join(OUT_DIR, "fig_torque_drag.svg"))
    plt.close(fig)


def fig_drag_torque_solar_activity(drag_env):
    """
    Fig — NEW: three-panel figure (one row per solar activity level)
    showing drag torque low/nom/high inertia bands separately.
    Makes it easy to read off the torque range at a specific activity level.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
    for ax, act in zip(axes, op.SOLAR_ACTIVITIES):
        st = ACT_STYLE[act]
        d  = drag_env[act]
        ax.fill_between(ALT, d['low'], d['high'],
                        color=st['fill'], alpha=0.35,
                        label="Geometry + ψ envelope")
        ax.plot(ALT, d['nom'], color=st['color'],
                linestyle='-', linewidth=2.2, label="Nominal")
        ax.plot(ALT, d['low'], color=st['color'],
                linestyle=':', linewidth=1.2, alpha=0.8)
        ax.plot(ALT, d['high'], color=st['color'],
                linestyle='--', linewidth=1.2, alpha=0.8)
        ax.set_yscale("log")
        ax.set_xlim(H_MIN, H_MAX)
        ax.set_xlabel("Altitude [km]", fontsize=12)
        ax.set_title(st['label'], fontsize=12, color=st['color'])
        ax.grid(True, which="major")
        ax.grid(True, which="minor", alpha=0.25)
        ax.tick_params(axis="both", labelsize=10)
        ax.legend(fontsize=9)
    axes[0].set_ylabel("Drag torque [N·m]", fontsize=12)
    fig.suptitle("Aerodynamic Drag Torque by Solar Activity Level",
                 fontweight="bold", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT_DIR,
                             "fig_torque_drag_solar_activity.svg"))
    plt.close(fig)

# ============================================================
# Figure 3: SRP torque band (unchanged — no activity dependence)
# ============================================================
def fig_srp_torque():
    srp_low_val  = srp_torque(surface_min, mass_max_, cr_min, com_cop_min)
    srp_nom_val  = srp_torque(surface_nom, mass_nom,  cr_nom, com_cop_nom)
    srp_high_val = srp_torque(surface_max, mass_min_, cr_max, com_cop_max)

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
# Figure 4: Combined budget — mean activity (original)
# ============================================================
def fig_torque_budget(gg, drag_env, srp):
    drag = drag_env['mean']
    fig, ax = plt.subplots()
    ax.fill_between(ALT, gg["low"], gg["high"],
                    color="#e78d51", alpha=0.12)
    ax.plot(ALT, gg["nom"], color="#e0590b", label="Gravity-gradient")
    ax.fill_between(ALT, drag["low"], drag["high"],
                    color="#3182bd", alpha=0.12)
    ax.plot(ALT, drag["nom"], color="#08306b",
            label="Drag nominal (mean activity)")
    ax.fill_between(ALT, srp["low"], srp["high"],
                    color="#d4a017", alpha=0.16)
    ax.plot(ALT, srp["nom"], color="#9c7a00",
            linestyle="--", label="SRP")
    style_axes(ax)
    ax.set_title("Combined Disturbance Torque Budget vs. Altitude\n"
                 "(drag at mean solar activity)")
    ax.legend()
    fig.savefig(os.path.join(OUT_DIR, "fig_torque_budget.svg"))
    plt.close(fig)

# ============================================================
# Figure 5: Combined budget — all three solar activity levels  (NEW)
# ============================================================
def fig_torque_budget_solar_activity(gg, drag_env, srp):
    """
    Full torque budget overlaying drag nominal curves for all three
    solar activity levels, revealing where solar cycle modulates drag
    relative to gravity-gradient and SRP.
    """
    fig, ax = plt.subplots(figsize=(10, 6.5))

    # Gravity-gradient (no activity dependence)
    ax.fill_between(ALT, gg["low"], gg["high"],
                    color="#e78d51", alpha=0.12)
    ax.plot(ALT, gg["nom"], color="#e0590b", linewidth=2.4,
            label="Gravity-gradient (nominal inertia)", zorder=5)

    # SRP (no activity dependence)
    ax.fill_between(ALT, srp["low"], srp["high"],
                    color="#d4a017", alpha=0.18)
    ax.plot(ALT, srp["nom"], color="#9c7a00", linestyle="--",
            linewidth=2.0, label="SRP nominal", zorder=5)

    # Drag — three activity levels
    for act in op.SOLAR_ACTIVITIES:
        st = ACT_STYLE[act]
        d  = drag_env[act]
        ax.fill_between(ALT, d['low'], d['high'],
                        color=st['fill'], alpha=0.15)
        ax.plot(ALT, d['nom'], color=st['color'],
                linestyle=st['ls'], linewidth=st['lw'],
                label=f"Drag nominal – {st['label']}", zorder=4)

    style_axes(ax)
    ax.set_title("Disturbance Torque Budget vs. Solar Activity — LEO SSO\n"
                 "(NRLMSISE-00)")
    ax.legend(ncol=2, loc="upper right", fontsize=9)
    fig.savefig(os.path.join(OUT_DIR,
                             "fig_torque_budget_solar_activity.svg"))
    plt.close(fig)

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("Computing gravity-gradient torque band...")
    gg, phi, theta = fig_gravity_gradient_torque()

    print("Computing drag torque bands (3 solar activity levels)...")
    drag_env = _drag_torque_envelopes()
    fig_drag_torque(drag_env)
    fig_drag_torque_solar_activity(drag_env)

    print("Computing SRP torque band...")
    srp = fig_srp_torque()

    print("Combined disturbance torque budget (mean activity)...")
    fig_torque_budget(gg, drag_env, srp)

    print("Combined disturbance torque budget (all 3 activity levels)...")
    fig_torque_budget_solar_activity(gg, drag_env, srp)

    print("Done. Figures saved in:", os.path.abspath(OUT_DIR))