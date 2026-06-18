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

N_ATOMS = 1
NOMINAL_EDGE = 1.65      # m
NOMINAL_WIDTH = 0.4      # m
NOMINAL_MASS = 1250      # kg
COM_COP = 0.1            # m

SCALING_MIN = 0.8
SCALING_MAX = 1.2

cr_min = 1.0
cr_max = 2.0
cr_nom = 1.3

Cd = 2.2
n_exp = 6

# ============================================================
# Build min / nominal / max spacecraft properties
# ============================================================

edge_min = NOMINAL_EDGE * SCALING_MIN
edge_max = NOMINAL_EDGE * SCALING_MAX

thickness_min = NOMINAL_WIDTH * SCALING_MIN
thickness_max = NOMINAL_WIDTH * SCALING_MAX

mass_guess_min = NOMINAL_MASS * SCALING_MIN
mass_guess_max = NOMINAL_MASS * SCALING_MAX

com_cop_min = COM_COP * SCALING_MIN
com_cop_max = COM_COP * SCALING_MAX
com_cop_nom = COM_COP

dict_nominal = compute_mass_properties(
    NOMINAL_EDGE,
    NOMINAL_WIDTH,
    NOMINAL_MASS,
    N_ATOMS
)

dict_min = compute_mass_properties(
    edge_min,
    thickness_min,
    mass_guess_min,
    N_ATOMS
)

dict_max = compute_mass_properties(
    edge_max,
    thickness_max,
    mass_guess_max,
    N_ATOMS
)

mass_nom = dict_nominal["total_mass"]
mass_min = dict_min["total_mass"]
mass_max = dict_max["total_mass"]

surface_nom = dict_nominal["projected_area_xy"]
surface_min = dict_min["projected_area_xy"]
surface_max = dict_max["projected_area_xy"]

inertia_nom = dict_nominal["inertia_about_origin"]
inertia_min = dict_min["inertia_about_origin"]
inertia_max = dict_max["inertia_about_origin"]

# Area-to-mass ratios
am_nom = surface_nom / mass_nom
am_min = surface_min / mass_max
am_max = surface_max / mass_min

# ============================================================
# Matplotlib styling
# ============================================================

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "font.size": 14,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.3,
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "lines.linewidth": 2.0,
})

# ============================================================
# Helper formatting
# ============================================================

def style_axes(ax, ylabel="Disturbance torque [N m]", xlabel="Altitude [km]", logy=True):
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
    """
    Worst-case gravity-gradient torque magnitude [N m]
    over a spherical grid of body-frame radius directions.

    Returns:
        max_torque [N m]
        worst_phi   [rad]
        worst_theta [rad]
    """
    # Convert Inertia matrix to kg * km^2 for consistency with r in km
    inertia_matrix = inertia_matrix * 1e-6  # kg * m^2 -> kg * km^2

    r = (R_EARTH + altitude_km) 

    # Avoid duplicating phi = 0 and phi = 2*pi
    phi = np.linspace(0, 2*np.pi, 100, endpoint=False)
    theta = np.linspace(0, np.pi, 50)

    phi, theta = np.meshgrid(phi, theta)

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    r_vec = np.stack((x, y, z), axis=-1)      # (50, 100, 3)
    r_norm = np.linalg.norm(r_vec, axis=-1)   # (50, 100)

    Ir = np.einsum('ij,...j->...i', inertia_matrix, r_vec)

    torque = (3 * MU_EARTH / r_norm[..., None]**5) * np.cross(r_vec, Ir)
    torque_mag = np.linalg.norm(torque, axis=-1)

    max_idx = np.unravel_index(np.argmax(torque_mag), torque_mag.shape)
    max_torque = torque_mag[max_idx]
    worst_phi = phi[max_idx]
    worst_theta = theta[max_idx]

    return max_torque, worst_phi, worst_theta

# ============================================================
# Drag torque
# ============================================================

def drag_torque(h_km, area_m2, mass_kg, psi_deg, Cd=2.2, n_exp=6, com_cop_m=0.1):
    """
    Aerodynamic drag torque [N m].

    Uses op.drag_accel(), which returns drag acceleration [m/s^2].
    Then:
        F_drag = m * a_drag
        T_drag = F_drag * lever_arm
    """
    a_m = area_m2 / mass_kg
    a_drag = op.drag_accel(h_km, a_m, psi_deg, Cd=Cd, n_exp=n_exp)   # [m/s^2]
    F_drag = mass_kg * a_drag                                        # [N]
    return F_drag * com_cop_m                                        # [N m]

# ============================================================
# SRP torque
# ============================================================

def srp_torque(area_m2, mass_kg, cr, com_cop_m=0.1):
    """
    Solar radiation pressure torque [N m].

    Uses op.srp_accel(), which returns SRP acceleration [m/s^2].
    Then:
        F_srp = m * a_srp
        T_srp = F_srp * lever_arm
    """
    a_m = area_m2 / mass_kg
    a_srp = op.srp_accel(a_m, cr, eclipse_fraction=0.0)  # [m/s^2]
    F_srp = mass_kg * a_srp                              # [N]
    return F_srp * com_cop_m                             # [N m]

# ============================================================
# Figure 1: Gravity-gradient band + worst-case angles
# ============================================================

def fig_gravity_gradient_torque():
    gg_low = []
    gg_nom = []
    gg_high = []

    phi_low = []
    phi_nom = []
    phi_high = []

    theta_low = []
    theta_nom = []
    theta_high = []

    for h in ALT:
        t_low, p_low, th_low = gravity_gradient_torque(h, inertia_min)
        t_nom, p_nom, th_nom = gravity_gradient_torque(h, inertia_nom)
        t_high, p_high, th_high = gravity_gradient_torque(h, inertia_max)

        gg_low.append(t_low)
        gg_nom.append(t_nom)
        gg_high.append(t_high)

        phi_low.append(p_low)
        phi_nom.append(p_nom)
        phi_high.append(p_high)

        theta_low.append(th_low)
        theta_nom.append(th_nom)
        theta_high.append(th_high)

    gg_low = np.array(gg_low)
    gg_nom = np.array(gg_nom)
    gg_high = np.array(gg_high)

    phi_low = np.array(phi_low)
    phi_nom = np.array(phi_nom)
    phi_high = np.array(phi_high)

    theta_low = np.array(theta_low)
    theta_nom = np.array(theta_nom)
    theta_high = np.array(theta_high)

    # --------------------------------------------------------
    # Torque magnitude plot
    # --------------------------------------------------------
    fig, ax = plt.subplots()

    ax.fill_between(
        ALT, gg_low, gg_high,
        color="#4c72b0", alpha=0.20,
        label="Inertia envelope"
    )
    ax.plot(
        ALT, gg_nom,
        color="#1f4e79", linewidth=2.3,
        label="Nominal"
    )

    style_axes(ax)
    ax.set_title("Gravity-Gradient Torque vs. Altitude")
    ax.legend()

    fig.savefig(os.path.join(OUT_DIR, "fig_torque_gravity_gradient.svg"))
    plt.close(fig)

    # --------------------------------------------------------
    # Worst-case phi plot
    # --------------------------------------------------------
    # Unwrap to avoid artificial 0/360 jumps
    phi_low_deg = np.rad2deg(np.unwrap(phi_low))
    phi_nom_deg = np.rad2deg(np.unwrap(phi_nom))
    phi_high_deg = np.rad2deg(np.unwrap(phi_high))

    fig, ax = plt.subplots()

    ax.plot(
        ALT, phi_low_deg,
        color="#4c72b0",
        linewidth=2.0,
        label=r"Worst-case $\phi$ (low inertia)"
    )
    ax.plot(
        ALT, phi_nom_deg,
        color="#1f4e79",
        linewidth=2.0,
        label=r"Worst-case $\phi$ (nominal inertia)"
    )
    ax.plot(
        ALT, phi_high_deg,
        color="#08306b",
        linewidth=2.0,
        label=r"Worst-case $\phi$ (high inertia)"
    )

    style_axes(ax, ylabel=r"Worst-case $\phi$ [deg]", logy=False)
    ax.set_title(r"Worst-case Gravity-Gradient Angle $\phi$ vs. Altitude")
    ax.legend()

    fig.savefig(os.path.join(OUT_DIR, "fig_torque_gravity_gradient_phi.svg"))
    plt.close(fig)

    # --------------------------------------------------------
    # Worst-case theta plot
    # --------------------------------------------------------
    theta_low_deg = np.rad2deg(theta_low)
    theta_nom_deg = np.rad2deg(theta_nom)
    theta_high_deg = np.rad2deg(theta_high)

    fig, ax = plt.subplots()

    ax.plot(
        ALT, theta_low_deg,
        color="#4c72b0",
        linewidth=2.0,
        label=r"Worst-case $\theta$ (low inertia)"
    )
    ax.plot(
        ALT, theta_nom_deg,
        color="#1f4e79",
        linewidth=2.0,
        label=r"Worst-case $\theta$ (nominal inertia)"
    )
    ax.plot(
        ALT, theta_high_deg,
        color="#08306b",
        linewidth=2.0,
        label=r"Worst-case $\theta$ (high inertia)"
    )

    style_axes(ax, ylabel=r"Worst-case $\theta$ [deg]", logy=False)
    ax.set_title(r"Worst-case Gravity-Gradient Angle $\theta$ vs. Altitude")
    ax.legend()

    fig.savefig(os.path.join(OUT_DIR, "fig_torque_gravity_gradient_theta.svg"))
    plt.close(fig)

    return (
        {"low": gg_low, "nom": gg_nom, "high": gg_high},
        {"low": phi_low, "nom": phi_nom, "high": phi_high},
        {"low": theta_low, "nom": theta_nom, "high": theta_high},
    )

# ============================================================
# Figure 2: Drag torque band
# ============================================================

def fig_drag_torque():
    drag_low = []
    drag_nom = []
    drag_high = []

    psi_grid = np.linspace(0.0, 180.0, 37)

    for h in ALT:
        # Lower bound: low area, high mass, anti-bulge, min lever arm
        t_low = drag_torque(
            h,
            surface_min,
            mass_max,
            psi_deg=180.0,
            Cd=Cd,
            n_exp=n_exp,
            com_cop_m=com_cop_min
        )

        # Upper bound: high area, low mass, bulge apex, max lever arm
        t_high = drag_torque(
            h,
            surface_max,
            mass_min,
            psi_deg=0.0,
            Cd=Cd,
            n_exp=n_exp,
            com_cop_m=com_cop_max
        )

        # Nominal: average over psi at nominal geometry
        nominal_vals = [
            drag_torque(
                h,
                surface_nom,
                mass_nom,
                psi_deg=psi,
                Cd=Cd,
                n_exp=n_exp,
                com_cop_m=com_cop_nom
            )
            for psi in psi_grid
        ]

        drag_low.append(t_low)
        drag_nom.append(np.mean(nominal_vals))
        drag_high.append(t_high)

    drag_low = np.array(drag_low)
    drag_nom = np.array(drag_nom)
    drag_high = np.array(drag_high)

    fig, ax = plt.subplots()

    ax.fill_between(
        ALT, drag_low, drag_high,
        color="#3182bd", alpha=0.20,
        label="Drag torque envelope"
    )
    ax.plot(
        ALT, drag_nom,
        color="#08306b", linewidth=2.3,
        label="Nominal"
    )

    style_axes(ax)
    ax.set_title("Aerodynamic Drag Torque vs. Altitude")
    ax.legend()

    fig.savefig(os.path.join(OUT_DIR, "fig_torque_drag.svg"))
    plt.close(fig)

    return {"low": drag_low, "nom": drag_nom, "high": drag_high}

# ============================================================
# Figure 3: SRP torque band
# ============================================================

def fig_srp_torque():
    srp_low_val = srp_torque(
        surface_min,
        mass_max,
        cr=cr_min,
        com_cop_m=com_cop_min
    )

    srp_nom_val = srp_torque(
        surface_nom,
        mass_nom,
        cr=cr_nom,
        com_cop_m=com_cop_nom
    )

    srp_high_val = srp_torque(
        surface_max,
        mass_min,
        cr=cr_max,
        com_cop_m=com_cop_max
    )

    srp_low = np.full_like(ALT, srp_low_val)
    srp_nom = np.full_like(ALT, srp_nom_val)
    srp_high = np.full_like(ALT, srp_high_val)

    fig, ax = plt.subplots()

    ax.fill_between(
        ALT, srp_low, srp_high,
        color="#d4a017", alpha=0.25,
        label="SRP torque envelope"
    )
    ax.plot(
        ALT, srp_nom,
        color="#9c7a00", linewidth=2.3,
        label="Nominal"
    )

    style_axes(ax)
    ax.set_title("Solar Radiation Pressure Torque vs. Altitude")
    ax.legend()

    fig.savefig(os.path.join(OUT_DIR, "fig_torque_srp.svg"))
    plt.close(fig)

    return {"low": srp_low, "nom": srp_nom, "high": srp_high}

# ============================================================
# Figure 4: Combined budget
# ============================================================

def fig_torque_budget(gg, drag, srp):
    fig, ax = plt.subplots()

    ax.fill_between(ALT, gg["low"], gg["high"], color="#e78d51", alpha=0.12)
    ax.plot(ALT, gg["nom"], color="#e0590b", label="Gravity-gradient")

    ax.fill_between(ALT, drag["low"], drag["high"], color="#3182bd", alpha=0.12)
    ax.plot(ALT, drag["nom"], color="#08306b", label="Drag")

    ax.fill_between(ALT, srp["low"], srp["high"], color="#d4a017", alpha=0.16)
    ax.plot(ALT, srp["nom"], color="#9c7a00", linestyle="--", label="SRP")

    style_axes(ax)
    ax.set_title("Combined Spacecraft Disturbance Torque Budget vs. Altitude")
    ax.legend()

    fig.savefig(os.path.join(OUT_DIR, "fig_torque_budget.svg"))
    plt.close(fig)

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("Computing gravity-gradient torque band...")
    gg, phi, theta = fig_gravity_gradient_torque()

    print("Computing drag torque band...")
    drag = fig_drag_torque()

    print("Computing SRP torque band...")
    srp = fig_srp_torque()

    print("Computing combined disturbance torque budget...")
    fig_torque_budget(gg, drag, srp)

    print("Done. Figures saved in:", os.path.abspath(OUT_DIR))