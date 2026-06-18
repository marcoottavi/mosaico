"""
plot_perturbations.py
======================

Generates a set of publication-quality plots (300 dpi PNG) showing the
order-of-magnitude evolution of the principal orbital perturbations with
altitude, for a LEO Sun-Synchronous mission scenario (200-2000 km).

Atmospheric drag uses the Harris-Priester density model (Montenbruck &
Gill 2000, Table 3.3 / Orekit `HarrisPriester` reference values).

Run:
    python3 plot_perturbations.py

Outputs (in ./figures/):
    fig1_zonal_harmonics.png
    fig2_atmospheric_drag.png
    fig2b_harris_priester_density_profile.png
    fig2c_harris_priester_diurnal_bulge.png
    fig3_srp.png
    fig4_third_body.png
    fig5_disturbance_budget.png
    disturbance_budget_table.csv
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib
from scipy.integrate import solve_ivp

import orbital_perturbations as op

# ---------------------------------------------------------------------------
# Global, "report-ready" matplotlib styling
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.figsize": (10,6),
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "font.size": 15,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.3,
    "grid.color": "gray",
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "lines.linewidth": 2.0,
})

OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

PALETTE = {
    "J2": "#1f4e79", "J3": "#2e8b57", "J4": "#b8860b", "J5": "#8b4a8b",
    "J6": "#c0504d", "total_zonal": "#000000",
    "drag_low": "#9ecae1", "drag_mean": "#3182bd", "drag_high": "#08306b",
    "rho_min": "#6baed6", "rho_max": "#08306b",
    "n2": "#74c476", "n6": "#cb181d",
    "srp_band": "#e6a817", "srp_nom": "#b8860b",
    "moon": "#7b7b7b", "sun": "#d9534f",
}

H_MIN, H_MAX, N_PTS = 200.0, 1000.0, 80
ALT = np.linspace(H_MIN, H_MAX, N_PTS)
REF_ALTS = [200, 300, 400, 500, 600, 700, 800, 900, 1000]

# Representative spacecraft/operational assumptions (clearly stated)
AM_GRID = np.linspace(0.005, 0.04, 12)   # area-to-mass ratio grid [m^2/kg]
CR_GRID = np.linspace(1.0, 2.0, 6)       # SRP reflectivity coefficient grid

# De-orbit decay study settings
DECAY_DAYS = 10.0          # propagation horizon [days]
DECAY_CD = 2.2              # drag coefficient, consistent with the rest of the study
REENTRY_ALT_KM = 120.0      # illustrative "reentry interface" altitude [km]


def style_axes(ax, ylabel="Perturbing acceleration [m/s$^2$]",
               xlabel="Altitude [km]", logy=True):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
    ax.set_xlim(H_MIN, H_MAX)
    ax.grid(True, which="major")
    ax.grid(True, which="minor", alpha=0.3)


# ---------------------------------------------------------------------------
# 1) Zonal harmonics J2..J6
# ---------------------------------------------------------------------------
def fig_zonal_harmonics():
    n_list = (2, 3, 4, 5, 6)
    series = {n: [] for n in n_list}
    total = []
    incl = []
    for h in ALT:
        res = op.zonal_max_accel_profile(h, n_list=n_list)
        for n in n_list:
            series[n].append(res[n])
        total.append(res['total'])
        incl.append(res['inclination_deg'])

    fig, ax = plt.subplots()
    for n in n_list:
        ax.plot(ALT, series[n], label=f"$J_{n}$", color=PALETTE[f"J{n}"])
    ax.plot(ALT, total, label="Total zonal (J2-J6)", color=PALETTE["total_zonal"],
            linestyle="-.", linewidth=2.3)
    style_axes(ax)
    ax.set_title("Earth Gravity Field Perturbations vs. Altitude\n")
    ax.legend(ncol=2, loc="upper right")
    fig.savefig(os.path.join(OUT_DIR, "fig1_zonal_harmonics.png"))
    plt.close(fig)
    return {"J2": series[2], "J3-J6": [t - j2 for t, j2 in zip(total, series[2])],
            "total_zonal": total}


# ---------------------------------------------------------------------------
# 2) Atmospheric drag (Harris-Priester)
# ---------------------------------------------------------------------------
def fig_drag():
    lo, mean, hi = [], [], []
    for h in ALT:
        l, m, hgh = op.drag_accel_envelope(h, AM_GRID)
        lo.append(l); mean.append(m); hi.append(hgh)
    lo, mean, hi = np.array(lo), np.array(mean), np.array(hi)

    fig, ax = plt.subplots()
    ax.fill_between(ALT, lo, hi, color=PALETTE["drag_mean"], alpha=0.18,
                     label="Envelope (A/m: 0.005-0.04 m$^2$/kg,\n"
                           r"diurnal-bulge angle $\psi$: 0-180$^\circ$)")
    ax.plot(ALT, mean, color=PALETTE["drag_mean"], label="Nominal "
            r"(A/m median, $\psi$-averaged)")
    ax.plot(ALT, lo, color=PALETTE["drag_low"], linestyle="--", linewidth=1.3,
            label=r"Lower bound (low A/m, $\psi$=180$^\circ$, $\rho_{min}$)")
    ax.plot(ALT, hi, color=PALETTE["drag_high"], linestyle="--", linewidth=1.3,
            label=r"Upper bound (high A/m, $\psi$=0$^\circ$, $\rho_{max}$)")
    style_axes(ax)
    ax.set_title("Atmospheric Drag Acceleration vs. Altitude\n")
    ax.legend(loc="upper right", fontsize=8.8)
    fig.savefig(os.path.join(OUT_DIR, "fig2_atmospheric_drag.png"))
    plt.close(fig)
    return {"drag_low": lo, "drag_mean": mean, "drag_high": hi}


def fig_harris_priester_density_profile():
    """Companion plot: the Harris-Priester rho_min/rho_max table itself vs. altitude."""
    rho_min, rho_max = op.harris_priester_minmax_density(ALT)

    fig, ax = plt.subplots()
    ax.fill_between(ALT, rho_min, rho_max, color=PALETTE["rho_max"], alpha=0.12)
    ax.plot(ALT, rho_min, color=PALETTE["rho_min"], label=r"$\rho_{min}$ (night side / anti-bulge)")
    ax.plot(ALT, rho_max, color=PALETTE["rho_max"], label=r"$\rho_{max}$ (day side / bulge apex)")
    ax.axvline(op._HP_H_MAX_VALID_KM, color="gray",
               linestyle=":", linewidth=1.2)
    ax.text(op._HP_H_MAX_VALID_KM + 40, 3e-13,
            "Official table\nvalidity limit (1000 km)\nextrapolated beyond", fontsize=8.3,
            color="#555555", va="top")
    style_axes(ax, ylabel="Atmospheric density [kg/m$^3$]")
    ax.set_title("Harris-Priester Atmospheric Density vs. Altitude\n"
                 "(Montenbruck & Gill 2000, Table 3.3 / Orekit reference values)")
    ax.legend(loc="upper right")
    fig.savefig(os.path.join(OUT_DIR, "fig2b_harris_priester_density_profile.png"))
    plt.close(fig)


def fig_harris_priester_diurnal_bulge():
    """Companion plot: density vs. diurnal-bulge angle psi, at representative altitudes,
    comparing the cos^2 (low-inclination) and cos^6 (near-polar/SSO) exponents."""
    psi = np.linspace(0, 180, 181)
    ref_alts = [400, 700, 1000]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=False)
    for ax, h in zip(axes, ref_alts):
        rho_n2 = op.harris_priester_density(h, psi, n_exp=2)
        rho_n6 = op.harris_priester_density(h, psi, n_exp=6)
        ax.plot(psi, rho_n2, color=PALETTE["n2"], label="n=2 (low inclination)")
        ax.plot(psi, rho_n6, color=PALETTE["n6"], label="n=6 (near-polar / SSO)")
        ax.set_title(f"h = {h} km", fontsize=12)
        ax.set_xlabel(r"Diurnal-bulge angle $\psi$ [deg]")
        ax.set_xlim(0, 180)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
    axes[0].set_ylabel("Density [kg/m$^3$]")
    axes[0].legend(loc="upper right", fontsize=9)
    fig.suptitle("Harris-Priester Diurnal Bulge Shape: Inclination-Dependent Exponent",
                 fontweight="bold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(OUT_DIR, "fig2c_harris_priester_diurnal_bulge.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3) Solar radiation pressure
# ---------------------------------------------------------------------------
def fig_srp():
    lo, nom, hi = op.srp_accel_envelope(AM_GRID, CR_GRID)
    lo_arr = np.full_like(ALT, lo)
    nom_arr = np.full_like(ALT, nom)
    hi_arr = np.full_like(ALT, hi)

    fig, ax = plt.subplots()
    ax.fill_between(ALT, lo_arr, hi_arr, color=PALETTE["srp_band"], alpha=0.25,
                     label="Envelope (A/m: 0.005-0.04 m$^2$/kg, $C_r$: 1.0-2.0)")
    ax.plot(ALT, nom_arr, color=PALETTE["srp_nom"], linewidth=2.3,
            label="Nominal ($C_r$=1.3, median A/m)")
    style_axes(ax)
    ax.set_ylim(lo * 0.5, hi * 2.0)
    ax.set_title("Solar Radiation Pressure Acceleration vs. Altitude\n"
                 "(no eclipse, fully sunlit case)")
    ax.legend(loc="center right")
    ax.text(0.02, 0.04,
            "Note: SRP is essentially altitude-independent in LEO\n"
            "($\\Delta R_{LEO}/R_{Sun-Earth} \\sim 10^{-5}$); shown flat by design.",
            transform=ax.transAxes, fontsize=8.5, color="#444444",
            va="bottom", ha="left")
    fig.savefig(os.path.join(OUT_DIR, "fig3_srp.png"))
    plt.close(fig)
    return {"srp_low": lo_arr, "srp_nom": nom_arr, "srp_high": hi_arr}


# ---------------------------------------------------------------------------
# 4) Third-body (Sun, Moon)
# ---------------------------------------------------------------------------
def fig_third_body():
    moon, sun = [], []
    for h in ALT:
        moon.append(op.third_body_max_accel(h, op.MU_MOON, op.D_MOON))
        sun.append(op.third_body_max_accel(h, op.MU_SUN, op.D_SUN))
    moon, sun = np.array(moon), np.array(sun)

    fig, ax = plt.subplots()
    ax.plot(ALT, moon, color=PALETTE["moon"], label="Moon (worst-case coplanar geometry)")
    ax.plot(ALT, sun, color=PALETTE["sun"], label="Sun (worst-case coplanar geometry)")
    style_axes(ax)
    ax.set_title("Third-Body (Lunisolar) Perturbing Acceleration vs. Altitude\n"
                 "(max over orbital phase grid, 0-360$^\\circ$)")
    ax.legend(loc="upper left")
    fig.savefig(os.path.join(OUT_DIR, "fig4_third_body.png"))
    plt.close(fig)
    return {"moon": moon, "sun": sun}


# ---------------------------------------------------------------------------
# 5) Combined disturbance budget
# ---------------------------------------------------------------------------
def fig_disturbance_budget(zonal, drag, srp, tb):
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    ax.plot(ALT, zonal["J2"], color=PALETTE["J2"], label="$J_2$ (gravity)", linewidth=2.4)
    ax.plot(ALT, zonal["J3-J6"], color=PALETTE["J4"], linestyle="--",
            label="$J_3$-$J_6$ combined (gravity)")
    ax.fill_between(ALT, drag["drag_low"], drag["drag_high"],
                     color=PALETTE["drag_mean"], alpha=0.15)
    ax.plot(ALT, drag["drag_mean"], color=PALETTE["drag_mean"], label="Drag (nominal)")
    ax.plot(ALT, srp["srp_nom"], color=PALETTE["srp_nom"], label="SRP (nominal)",
            linestyle=":")
    ax.plot(ALT, tb["moon"], color=PALETTE["moon"], label="Moon (3rd body)")
    ax.plot(ALT, tb["sun"], color=PALETTE["sun"], label="Sun (3rd body)")
    style_axes(ax)
    ax.set_title("Preliminary Disturbance Budget - LEO Sun-Synchronous Mission\n"
                 "Maximum perturbing acceleration vs. altitude (drag: Harris-Priester)")
    ax.legend(ncol=2, loc="upper right", fontsize=9)
    ax.axvspan(600, 800, color="gray", alpha=0.07)
    ax.text(700, ax.get_ylim()[1] * 0.4, "Typical\nSSO band", ha="center",
            fontsize=8.5, color="#555555")
    fig.savefig(os.path.join(OUT_DIR, "fig5_disturbance_budget.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6) De-orbit decay over a 10-day span (Gauss da/dt equation)
# ---------------------------------------------------------------------------
# For a near-circular orbit, the Gauss planetary equation for the secular
# rate of change of the semi-major axis under a perturbing acceleration
# reduces (e=0, so the radial term vanishes) to:
#
#     da/dt = (2/n) * a_s
#
# where n is the mean motion and a_s is the along-track (transverse)
# component of the perturbing acceleration. Drag acts purely along-track,
# opposing the velocity vector, with magnitude a_drag = 0.5*Cd*(A/m)*rho*v^2
# (the same drag_accel() used throughout this study), so a_s = -a_drag and:
#
#     da/dt = -2*a_drag/n = -Cd*(A/m)*rho*sqrt(mu*a)
#
# (the two forms are algebraically identical; the second is the classical
# "King-Hele" circular-decay approximation). This is integrated numerically
# (RK45) over DECAY_DAYS, for every starting altitude in REF_ALTS, under
# three drag regimes that bound and bracket the Harris-Priester envelope
# already used in fig_drag(): "fast" (max A/m, rho_max i.e. psi=0),
# "slow" (min A/m, rho_min i.e. psi=180), and "nominal" (the same
# psi-averaged, median-A/m value already computed by drag_accel_envelope).

def _da_dt_fixed(t, y, area_to_mass, psi_deg):
    a_km = y[0]
    h_km = a_km - op.R_EARTH
    a_drag = op.drag_accel(h_km, area_to_mass, psi_deg, Cd=DECAY_CD)  # m/s^2
    n = np.sqrt(op.MU_EARTH / a_km ** 3)                              # rad/s
    return [-2.0 * (a_drag / 1000.0) / n]                              # km/s


def _da_dt_nominal(t, y):
    a_km = y[0]
    h_km = a_km - op.R_EARTH
    _, a_drag_nom, _ = op.drag_accel_envelope(h_km, AM_GRID)          # m/s^2
    n = np.sqrt(op.MU_EARTH / a_km ** 3)
    return [-2.0 * (a_drag_nom / 1000.0) / n]


def _reentry_event(t, y, *args):
    return (y[0] - op.R_EARTH) - REENTRY_ALT_KM
_reentry_event.terminal = True
_reentry_event.direction = -1


def _propagate_decay(h0_km, rhs, args=()):
    a0 = op.R_EARTH + h0_km
    t_span = (0.0, DECAY_DAYS * 86400.0)
    t_eval = np.linspace(*t_span, 300)
    sol = solve_ivp(rhs, t_span, [a0], args=args, t_eval=t_eval,
                     events=_reentry_event, rtol=1e-9, atol=1e-6, method="RK45")

    t_days = sol.t / 86400.0
    alt_km = sol.y[0] - op.R_EARTH

    reentered = len(sol.t_events[0]) > 0

    if reentered:
        t_event = sol.t_events[0][0]
        a_event = sol.y_events[0][0][0]   # semi-major axis at event
        alt_event = a_event - op.R_EARTH

        # Force consistent final point
        t_days = np.append(t_days, t_event / 86400.0)
        alt_km = np.append(alt_km, alt_event)

        t_reentry = t_event / 86400.0
    else:
        t_reentry = None

    return t_days, alt_km, reentered, t_reentry


from matplotlib.ticker import ScalarFormatter

def fig_deorbit_decay():

    am_min, am_max = AM_GRID.min(), AM_GRID.max()

    fig, axes = plt.subplots(
        5,
        2,
        figsize=(12, 16),
        sharex=True
    )

    axes = axes.ravel()
    summary_rows = []

    for idx, h0 in enumerate(REF_ALTS):

        ax = axes[idx]

        # -------------------------------------------------
        # Propagations
        # -------------------------------------------------
        t_fast, alt_fast, re_fast, tre_fast = _propagate_decay(
            h0,
            _da_dt_fixed,
            args=(am_max, 0.0)
        )

        t_slow, alt_slow, re_slow, tre_slow = _propagate_decay(
            h0,
            _da_dt_fixed,
            args=(am_min, 180.0)
        )

        t_nom, alt_nom, re_nom, tre_nom = _propagate_decay(
            h0,
            _da_dt_nominal
        )

        # -------------------------------------------------
        # Curves
        # -------------------------------------------------
        ax.plot(
            t_nom,
            alt_nom,
            color="black",
            linewidth=2.2,
            label="Nominal"
        )

        ax.plot(
            t_fast,
            alt_fast,
            color="black",
            linewidth=1.2,
            linestyle="--",
            alpha=0.85,
            label="Fast decay"
        )

        ax.plot(
            t_slow,
            alt_slow,
            color="black",
            linewidth=1.2,
            linestyle=":",
            alpha=0.85,
            label="Slow decay"
        )

        # -------------------------------------------------
        # Titles
        # -------------------------------------------------
        ax.set_title(
            f"Initial altitude: {h0} km",
            fontsize=12,
            pad=8
        )

        # -------------------------------------------------
        # Axes formatting
        # -------------------------------------------------
        ax.set_xlim(0, DECAY_DAYS)

        ax.grid(
            True,
            linestyle="--",
            linewidth=0.5,
            alpha=0.5
        )

        # Axis labels only where needed
        if idx % 2 == 0:
            ax.set_ylabel(
                "Altitude [km]",
                fontsize=12
            )

        if idx >= 8:
            ax.set_xlabel(
                "Time [days]",
                fontsize=12
            )

        # Tick label size
        ax.tick_params(
            axis="both",
            labelsize=10
        )

        # -------------------------------------------------
        # Disable scientific notation / offset notation
        # -------------------------------------------------
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)

        ax.yaxis.set_major_formatter(formatter)
        ax.xaxis.set_major_formatter(formatter)

        # -------------------------------------------------
        # Optional: cleaner integer ticks
        # -------------------------------------------------
        ax.ticklabel_format(
            style='plain',
            axis='both'
        )

        # -------------------------------------------------
        # Summary rows
        # -------------------------------------------------
        summary_rows.append([
            h0,

            f"{alt_fast[-1]:.2f}",
            "yes" if re_fast else "no",
            f"{tre_fast:.2f}" if re_fast else "-",

            f"{alt_nom[-1]:.2f}",
            "yes" if re_nom else "no",
            f"{tre_nom:.2f}" if re_nom else "-",

            f"{alt_slow[-1]:.2f}",
            "yes" if re_slow else "no",
            f"{tre_slow:.2f}" if re_slow else "-"
        ])

    # -----------------------------------------------------
    # Remove unused subplot (10th axis)
    # -----------------------------------------------------
    fig.delaxes(axes[-1])

    # -----------------------------------------------------
    # Global legend
    # -----------------------------------------------------
    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        fontsize=11,
        frameon=True,
        bbox_to_anchor=(0.5, 0.975)
    )

    # -----------------------------------------------------
    # Figure title
    # -----------------------------------------------------
    fig.suptitle(
        f"{DECAY_DAYS:.0f}-Day Orbital Decay Evolution",
        fontsize=16,
        fontweight="bold",
        y=0.995
    )

    # -----------------------------------------------------
    # Layout spacing
    # -----------------------------------------------------
    fig.subplots_adjust(
        left=0.10,
        right=0.97,
        top=0.93,
        bottom=0.06,
        hspace=0.25,
        wspace=0.20
    )

    # -----------------------------------------------------
    # Save figure
    # -----------------------------------------------------
    fig.savefig(
        os.path.join(OUT_DIR, "fig6_deorbit_decay.svg"),
        dpi=300
    )

    plt.close(fig)

    # -----------------------------------------------------
    # Export CSV summary
    # -----------------------------------------------------
    path = os.path.join(
        OUT_DIR,
        "deorbit_decay_summary.csv"
    )

    with open(path, "w", newline="") as f:

        w = csv.writer(f)

        w.writerow([
            "h0 [km]",
            "h_fast_10d [km]",
            "fast_reentry?",
            "fast_t_reentry [d]",
            "h_nominal_10d [km]",
            "nominal_reentry?",
            "nominal_t_reentry [d]",
            "h_slow_10d [km]",
            "slow_reentry?",
            "slow_t_reentry [d]"
        ])

        w.writerows(summary_rows)

    print(f"De-orbit decay summary written to {path}")


# ---------------------------------------------------------------------------
# Summary CSV table (representative altitudes)
# ---------------------------------------------------------------------------
def export_summary_table(zonal, drag, srp, tb):
    idx_map = {h: int(np.argmin(np.abs(ALT - h))) for h in REF_ALTS}
    path = os.path.join(OUT_DIR, "disturbance_budget_table.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Altitude [km]", "J2 [m/s^2]", "J3-J6 [m/s^2]",
                    "Drag_HarrisPriester_low [m/s^2]", "Drag_HarrisPriester_nominal [m/s^2]",
                    "Drag_HarrisPriester_high [m/s^2]",
                    "SRP_nominal [m/s^2]", "Moon [m/s^2]", "Sun [m/s^2]"])
        for h in REF_ALTS:
            i = idx_map[h]
            w.writerow([h,
                        f"{zonal['J2'][i]:.3e}", f"{zonal['J3-J6'][i]:.3e}",
                        f"{drag['drag_low'][i]:.3e}", f"{drag['drag_mean'][i]:.3e}",
                        f"{drag['drag_high'][i]:.3e}",
                        f"{srp['srp_nom'][i]:.3e}",
                        f"{tb['moon'][i]:.3e}", f"{tb['sun'][i]:.3e}"])
    print(f"Summary table written to {path}")


if __name__ == "__main__":
    print("Computing zonal harmonics profile...")
    zonal = fig_zonal_harmonics()
    print("Computing atmospheric drag profile (Harris-Priester)...")
    drag = fig_drag()
    print("Plotting Harris-Priester density profile...")
    fig_harris_priester_density_profile()
    print("Plotting Harris-Priester diurnal bulge shape...")
    fig_harris_priester_diurnal_bulge()
    print("Computing SRP profile...")
    srp = fig_srp()
    print("Computing third-body profile...")
    tb = fig_third_body()
    print("Building combined disturbance budget plot...")
    fig_disturbance_budget(zonal, drag, srp, tb)
    export_summary_table(zonal, drag, srp, tb)
    print("Computing 10-day de-orbit decay study...")
    fig_deorbit_decay()
    print("Done. Figures saved in:", os.path.abspath(OUT_DIR))