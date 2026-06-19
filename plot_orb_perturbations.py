"""
plot_orb_perturbations.py
==========================

Generates publication-quality plots (300 dpi) for the preliminary orbital
disturbance budget of a LEO Sun-Synchronous mission (200-1000 km).

New in this version: all drag-dependent figures are produced for three solar
activity levels (low / mean / high, F10.7 ≈ 75 / 150 / 230 SFU) so that the
solar-cycle sensitivity of the drag environment is visible alongside every
other perturbation source.

The low- and high-activity Harris-Priester tables are DERIVED engineering
estimates (F10.7-scaled from the Montenbruck & Gill mean table using
altitude-dependent sensitivity exponents from Emmert 2015); they are NOT
independently tabulated primary-source data.  See orbital_perturbations.py
for the full derivation and caveats.

Run:
    python3 plot_orb_perturbations.py

Outputs (in ./figures/):
    fig1_zonal_harmonics.png
    fig2_atmospheric_drag.png                 <- three activity curves
    fig2b_harris_priester_density_profile.png <- three activity curves
    fig2c_harris_priester_diurnal_bulge.png
    fig2d_drag_solar_activity_ratio.png       <- NEW: ratio high/low vs mean
    fig3_srp.png
    fig4_third_body.png
    fig5_disturbance_budget.png               <- three drag bands
    fig5b_disturbance_budget_solar_activity.png <- NEW: budget per activity
    fig6_deorbit_decay.svg
    disturbance_budget_table.csv              <- extended with activity columns
    deorbit_decay_summary.csv
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.integrate import solve_ivp

import orbital_perturbations as op

# ---------------------------------------------------------------------------
# Global styling
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.figsize":     (10, 6),
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "font.family":        "serif",
    "font.size":          15,
    "axes.titleweight":   "bold",
    "axes.grid":          True,
    "grid.linestyle":     "--",
    "grid.linewidth":     0.6,
    "grid.alpha":         0.3,
    "grid.color":         "gray",
    "legend.frameon":     True,
    "legend.framealpha":  0.92,
    "lines.linewidth":    2.0,
})

OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

PALETTE = {
    "J2": "#1f4e79", "J3": "#2e8b57", "J4": "#b8860b",
    "J5": "#8b4a8b", "J6": "#c0504d", "total_zonal": "#000000",
    "drag_low": "#9ecae1", "drag_mean": "#3182bd", "drag_high": "#08306b",
    "rho_min":  "#6baed6", "rho_max":   "#08306b",
    "n2": "#74c476",  "n6": "#cb181d",
    "srp_band": "#e6a817", "srp_nom": "#b8860b",
    "moon": "#7b7b7b", "sun": "#d9534f",
    # per-activity colours used in combined figures
    "act_low":  "#2ca02c",   # green
    "act_mean": "#1f77b4",   # blue
    "act_high": "#d62728",   # red
}

# Activity-level display metadata (mirrors op._HP_TABLES)
ACT_STYLE = {
    'low':  dict(color=PALETTE["act_low"],  ls=':', lw=1.8,
                 label='Low activity (F10.7≈75)',  fill='#98df8a'),
    'mean': dict(color=PALETTE["act_mean"], ls='-', lw=2.2,
                 label='Mean activity (F10.7≈150)', fill='#aec7e8'),
    'high': dict(color=PALETTE["act_high"], ls='--', lw=1.8,
                 label='High activity (F10.7≈230)', fill='#ffbb78'),
}

H_MIN, H_MAX, N_PTS = 200.0, 1000.0, 80
ALT = np.linspace(H_MIN, H_MAX, N_PTS)
REF_ALTS = [200, 300, 400, 500, 600, 700, 800, 900, 1000]

AM_GRID  = np.linspace(0.005, 0.04, 12)   # area-to-mass [m^2/kg]
CR_GRID  = np.linspace(1.0,   2.0,   6)   # SRP reflectivity

DECAY_DAYS     = 10.0
DECAY_CD       = 2.2
REENTRY_ALT_KM = 120.0


def style_axes(ax, ylabel="Perturbing acceleration [m/s²]",
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
# 1) Zonal harmonics (unchanged — no solar activity dependence)
# ---------------------------------------------------------------------------
def fig_zonal_harmonics():
    n_list = (2, 3, 4, 5, 6)
    series = {n: [] for n in n_list}
    total  = []
    for h in ALT:
        res = op.zonal_max_accel_profile(h, n_list=n_list)
        for n in n_list:
            series[n].append(res[n])
        total.append(res['total'])

    fig, ax = plt.subplots()
    for n in n_list:
        ax.plot(ALT, series[n], label=f"$J_{n}$", color=PALETTE[f"J{n}"])
    ax.plot(ALT, total, label="Total zonal (J2–J6)",
            color=PALETTE["total_zonal"], linestyle="-.", linewidth=2.3)
    style_axes(ax)
    ax.set_title("Earth Gravity Field Perturbations vs. Altitude\n")
    ax.legend(ncol=2, loc="upper right")
    fig.savefig(os.path.join(OUT_DIR, "fig1_zonal_harmonics.png"))
    plt.close(fig)
    return {"J2": series[2],
            "J3-J6": [t - j2 for t, j2 in zip(total, series[2])],
            "total_zonal": total}


# ---------------------------------------------------------------------------
# 2) Atmospheric drag — three solar activity levels
# ---------------------------------------------------------------------------
def _compute_drag_envelopes():
    """Return {activity: {'lo','nom','hi'}} arrays over ALT."""
    envelopes = {}
    for act in op.SOLAR_ACTIVITIES:
        lo, nom, hi = [], [], []
        for h in ALT:
            l, m, hh = op.drag_accel_envelope(h, AM_GRID, activity=act)
            lo.append(l); nom.append(m); hi.append(hh)
        envelopes[act] = {
            'lo':  np.array(lo),
            'nom': np.array(nom),
            'hi':  np.array(hi),
        }
    return envelopes


def fig_drag(envelopes):
    """
    Fig 2a — drag envelope for each activity level on a single axes.
    """
    fig, ax = plt.subplots()
    for act in op.SOLAR_ACTIVITIES:
        st = ACT_STYLE[act]
        d  = envelopes[act]
        ax.fill_between(ALT, d['lo'], d['hi'],
                        color=st['fill'], alpha=0.25)
        ax.plot(ALT, d['nom'], color=st['color'],
                linestyle=st['ls'], linewidth=st['lw'],
                label=f"Nominal – {st['label']}")

    style_axes(ax)
    ax.set_title("Atmospheric Drag Acceleration vs. Altitude\n"
                 "(Harris-Priester, shaded bands = A/m & ψ envelope)")
    ax.legend(loc="upper right", fontsize=10)
    ax.text(0.02, 0.03,
            "Low/high activity are F10.7-scaled engineering estimates\n"
            "(not independently tabulated; see orbital_perturbations.py)",
            transform=ax.transAxes, fontsize=7.5, color="#666666",
            va="bottom")
    fig.savefig(os.path.join(OUT_DIR, "fig2_atmospheric_drag.png"))
    plt.close(fig)


def fig_harris_priester_density_profile():
    """Fig 2b — min/max density profiles for all three activity levels."""
    fig, ax = plt.subplots()
    for act in op.SOLAR_ACTIVITIES:
        st  = ACT_STYLE[act]
        rho_min, rho_max = op.harris_priester_minmax_density(ALT, activity=act)
        ax.fill_between(ALT, rho_min, rho_max,
                        color=st['fill'], alpha=0.20)
        ax.plot(ALT, rho_min, color=st['color'],
                linestyle=st['ls'], linewidth=st['lw'],
                label=f"ρ_min – {st['label']}")
        ax.plot(ALT, rho_max, color=st['color'],
                linestyle=st['ls'], linewidth=st['lw'], alpha=0.55)

    ax.axvline(op._HP_H_MAX_VALID_KM, color="gray",
               linestyle=":", linewidth=1.2)
    ax.text(op._HP_H_MAX_VALID_KM + 30, 3e-13,
            "Table validity\nlimit (1000 km)", fontsize=7.5,
            color="#555555", va="top")
    style_axes(ax, ylabel="Atmospheric density [kg/m³]")
    ax.set_title("Harris-Priester Density vs. Altitude — Three Solar Activity Levels\n"
                 "(low/high: F10.7-scaled from M&G mean table; see caveats)")
    ax.legend(loc="upper right", fontsize=8.5)
    fig.savefig(os.path.join(OUT_DIR,
                             "fig2b_harris_priester_density_profile.png"))
    plt.close(fig)


def fig_harris_priester_diurnal_bulge():
    """Fig 2c — diurnal bulge shape (mean activity only; n=2 vs n=6)."""
    psi = np.linspace(0, 180, 181)
    ref_alts = [400, 700, 1000]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=False)
    for ax, h in zip(axes, ref_alts):
        rho_n2 = op.harris_priester_density(h, psi, n_exp=2, activity='mean')
        rho_n6 = op.harris_priester_density(h, psi, n_exp=6, activity='mean')
        ax.plot(psi, rho_n2, color=PALETTE["n2"], label="n=2 (low incl.)")
        ax.plot(psi, rho_n6, color=PALETTE["n6"], label="n=6 (near-polar/SSO)")
        ax.set_title(f"h = {h} km", fontsize=12)
        ax.set_xlabel("Diurnal-bulge angle ψ [deg]")
        ax.set_xlim(0, 180)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
    axes[0].set_ylabel("Density [kg/m³]")
    axes[0].legend(loc="upper right", fontsize=9)
    fig.suptitle(
        "Harris-Priester Diurnal Bulge Shape (mean solar activity)",
        fontweight="bold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(OUT_DIR,
                             "fig2c_harris_priester_diurnal_bulge.png"))
    plt.close(fig)


def fig_drag_solar_activity_ratio(envelopes):
    """
    Fig 2d — NEW: ratio of nominal drag (high/mean and low/mean) vs altitude.
    Communicates the solar-cycle density uncertainty directly as a multiplier.
    """
    nom_mean = envelopes['mean']['nom']
    nom_low  = envelopes['low']['nom']
    nom_high = envelopes['high']['nom']

    ratio_high = nom_high / nom_mean
    ratio_low  = nom_low  / nom_mean

    fig, ax = plt.subplots()
    ax.plot(ALT, ratio_high, color=PALETTE["act_high"], linestyle='--',
            linewidth=2.0, label="High / Mean  (F10.7≈230 / 150)")
    ax.axhline(1.0, color=PALETTE["act_mean"], linewidth=1.5,
               linestyle='-', label="Mean (reference, ratio = 1)")
    ax.plot(ALT, ratio_low,  color=PALETTE["act_low"],  linestyle=':',
            linewidth=2.0, label="Low / Mean   (F10.7≈75 / 150)")

    ax.fill_between(ALT, ratio_low, ratio_high,
                    color="#d9d9d9", alpha=0.35, label="Solar-cycle band")

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    style_axes(ax, ylabel="Drag acceleration ratio (vs. mean activity)", logy=True)
    ax.set_title("Solar-Cycle Sensitivity of Atmospheric Drag\n"
                 "(Harris-Priester, nominal A/m & ψ, F10.7-scaled estimates)")
    ax.legend(loc="upper left", fontsize=10)

    # Annotate key altitudes
    for h_ann in [400, 600, 800]:
        idx = np.argmin(np.abs(ALT - h_ann))
        ax.annotate(f"×{ratio_high[idx]:.1f}",
                    xy=(ALT[idx], ratio_high[idx]),
                    xytext=(ALT[idx] + 20, ratio_high[idx] * 1.15),
                    fontsize=8, color=PALETTE["act_high"],
                    arrowprops=dict(arrowstyle='->', color=PALETTE["act_high"],
                                   lw=0.8))

    ax.text(0.02, 0.03,
            "Ratios derived from F10.7-scaled engineering estimates\n"
            "(not independently tabulated; see orbital_perturbations.py)",
            transform=ax.transAxes, fontsize=7.5, color="#666666", va="bottom")
    fig.savefig(os.path.join(OUT_DIR,
                             "fig2d_drag_solar_activity_ratio.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3) SRP (altitude-independent; no solar activity effect on the model)
# ---------------------------------------------------------------------------
def fig_srp():
    lo, nom, hi = op.srp_accel_envelope(AM_GRID, CR_GRID)
    fig, ax = plt.subplots()
    ax.fill_between(ALT, np.full_like(ALT, lo), np.full_like(ALT, hi),
                    color=PALETTE["srp_band"], alpha=0.25,
                    label="Envelope (A/m: 0.005–0.04 m²/kg, Cᵣ: 1.0–2.0)")
    ax.plot(ALT, np.full_like(ALT, nom), color=PALETTE["srp_nom"],
            linewidth=2.3, label="Nominal (Cᵣ=1.3, median A/m)")
    style_axes(ax)
    ax.set_ylim(lo * 0.5, hi * 2.0)
    ax.set_title("Solar Radiation Pressure Acceleration vs. Altitude\n"
                 "(no eclipse, fully sunlit case)")
    ax.legend(loc="center right")
    fig.savefig(os.path.join(OUT_DIR, "fig3_srp.png"))
    plt.close(fig)
    lo_arr  = np.full_like(ALT, lo)
    nom_arr = np.full_like(ALT, nom)
    hi_arr  = np.full_like(ALT, hi)
    return {"srp_low": lo_arr, "srp_nom": nom_arr, "srp_high": hi_arr}


# ---------------------------------------------------------------------------
# 4) Third-body
# ---------------------------------------------------------------------------
def fig_third_body():
    moon, sun = [], []
    for h in ALT:
        moon.append(op.third_body_max_accel(h, op.MU_MOON, op.D_MOON))
        sun.append(op.third_body_max_accel(h, op.MU_SUN,  op.D_SUN))
    moon, sun = np.array(moon), np.array(sun)
    fig, ax = plt.subplots()
    ax.plot(ALT, moon, color=PALETTE["moon"],
            label="Moon (worst-case coplanar)")
    ax.plot(ALT, sun,  color=PALETTE["sun"],
            label="Sun (worst-case coplanar)")
    style_axes(ax)
    ax.set_title("Third-Body (Lunisolar) Perturbation vs. Altitude\n"
                 "(max over orbital phase, 0–360°)")
    ax.legend(loc="upper left")
    fig.savefig(os.path.join(OUT_DIR, "fig4_third_body.png"))
    plt.close(fig)
    return {"moon": moon, "sun": sun}


# ---------------------------------------------------------------------------
# 5a) Combined disturbance budget — mean activity (original figure)
# ---------------------------------------------------------------------------
def fig_disturbance_budget(zonal, envelopes, srp, tb):
    drag = envelopes['mean']
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    ax.plot(ALT, zonal["J2"], color=PALETTE["J2"],
            label="$J_2$ (gravity)", linewidth=2.4)
    ax.plot(ALT, zonal["J3-J6"], color=PALETTE["J4"], linestyle="--",
            label="$J_3$–$J_6$ combined")
    ax.fill_between(ALT, drag['lo'], drag['hi'],
                    color=PALETTE["drag_mean"], alpha=0.15)
    ax.plot(ALT, drag['nom'], color=PALETTE["drag_mean"],
            label="Drag nominal (mean activity)")
    ax.plot(ALT, srp["srp_nom"], color=PALETTE["srp_nom"],
            linestyle=":", label="SRP nominal")
    ax.plot(ALT, tb["moon"], color=PALETTE["moon"], label="Moon (3rd body)")
    ax.plot(ALT, tb["sun"],  color=PALETTE["sun"],  label="Sun (3rd body)")
    style_axes(ax)
    ax.set_title("Preliminary Disturbance Budget — LEO SSO Mission\n"
                 "(Drag: Harris-Priester, mean solar activity)")
    ax.legend(ncol=2, loc="upper right", fontsize=9)
    ax.axvspan(600, 800, color="gray", alpha=0.07)
    ax.text(700, ax.get_ylim()[1] * 0.4, "Typical\nSSO band",
            ha="center", fontsize=8.5, color="#555555")
    fig.savefig(os.path.join(OUT_DIR, "fig5_disturbance_budget.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5b) Combined disturbance budget — all three activity levels  (NEW)
# ---------------------------------------------------------------------------
def fig_disturbance_budget_solar_activity(zonal, envelopes, srp, tb):
    """
    Overlay drag nominal curves for low / mean / high solar activity on top of
    the full gravity + SRP + third-body budget, so the reader can see when and
    by how much solar activity promotes drag above or below the other sources.
    """
    fig, ax = plt.subplots(figsize=(10, 6.5))

    # Fixed perturbations (no activity dependence)
    ax.plot(ALT, zonal["J2"], color=PALETTE["J2"],
            linewidth=2.4, label="$J_2$ (gravity)", zorder=5)
    ax.plot(ALT, zonal["J3-J6"], color=PALETTE["J4"], linestyle="--",
            linewidth=1.6, label="$J_3$–$J_6$ combined", zorder=5)
    ax.plot(ALT, srp["srp_nom"], color=PALETTE["srp_nom"],
            linestyle=":", linewidth=2.0, label="SRP nominal", zorder=5)
    ax.plot(ALT, tb["moon"], color=PALETTE["moon"],
            linewidth=1.8, label="Moon (3rd body)", zorder=5)
    ax.plot(ALT, tb["sun"],  color=PALETTE["sun"],
            linewidth=1.8, label="Sun (3rd body)",  zorder=5)

    # Drag — three activity levels
    for act in op.SOLAR_ACTIVITIES:
        st = ACT_STYLE[act]
        d  = envelopes[act]
        ax.fill_between(ALT, d['lo'], d['hi'],
                        color=st['fill'], alpha=0.18)
        ax.plot(ALT, d['nom'], color=st['color'],
                linestyle=st['ls'], linewidth=st['lw'],
                label=f"Drag nominal – {st['label']}", zorder=4)

    style_axes(ax)
    ax.set_title("Disturbance Budget vs. Solar Activity — LEO SSO Mission\n"
                 "(drag bands span A/m & ψ envelope at each activity level)")
    ax.legend(ncol=2, loc="upper right", fontsize=8.5)
    ax.axvspan(600, 800, color="gray", alpha=0.06)
    ax.text(700, ax.get_ylim()[1] * 0.35, "Typical\nSSO band",
            ha="center", fontsize=8.0, color="#555555")
    ax.text(0.02, 0.03,
            "Low/high drag: F10.7-scaled engineering estimates\n"
            "(not independently tabulated; see orbital_perturbations.py)",
            transform=ax.transAxes, fontsize=7.5, color="#666666", va="bottom")
    fig.savefig(os.path.join(OUT_DIR,
                             "fig5b_disturbance_budget_solar_activity.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6) De-orbit decay — three activity levels per starting altitude
# ---------------------------------------------------------------------------
def _da_dt_fixed(t, y, area_to_mass, psi_deg, activity):
    a_km   = y[0]
    h_km   = a_km - op.R_EARTH
    a_drag = op.drag_accel(h_km, area_to_mass, psi_deg,
                           Cd=DECAY_CD, activity=activity)
    n = np.sqrt(op.MU_EARTH / a_km**3)
    return [-2.0 * (a_drag / 1000.0) / n]


def _da_dt_nominal(t, y, activity):
    a_km   = y[0]
    h_km   = a_km - op.R_EARTH
    _, a_drag_nom, _ = op.drag_accel_envelope(h_km, AM_GRID, activity=activity)
    n = np.sqrt(op.MU_EARTH / a_km**3)
    return [-2.0 * (a_drag_nom / 1000.0) / n]


def _reentry_event(t, y, *args):
    return (y[0] - op.R_EARTH) - REENTRY_ALT_KM
_reentry_event.terminal  = True
_reentry_event.direction = -1


def _propagate_decay(h0_km, rhs, args=()):
    a0     = op.R_EARTH + h0_km
    t_span = (0.0, DECAY_DAYS * 86400.0)
    t_eval = np.linspace(*t_span, 300)
    sol = solve_ivp(rhs, t_span, [a0], args=args, t_eval=t_eval,
                    events=_reentry_event, rtol=1e-9, atol=1e-6, method="RK45")
    t_days = sol.t / 86400.0
    alt_km = sol.y[0] - op.R_EARTH
    reentered  = len(sol.t_events[0]) > 0
    t_reentry  = None
    if reentered:
        t_event  = sol.t_events[0][0]
        a_event  = sol.y_events[0][0][0]
        t_days   = np.append(t_days, t_event / 86400.0)
        alt_km   = np.append(alt_km,  a_event - op.R_EARTH)
        t_reentry = t_event / 86400.0
    return t_days, alt_km, reentered, t_reentry


# Decay line styles per activity level
_DECAY_STYLE = {
    'low':  dict(color=PALETTE["act_low"],  ls=':', lw=1.5),
    'mean': dict(color=PALETTE["act_mean"], ls='-', lw=2.2),
    'high': dict(color=PALETTE["act_high"], ls='--', lw=1.5),
}
# am / psi combos for fast and slow legs
_AM_MIN, _AM_MAX = AM_GRID.min(), AM_GRID.max()


def fig_deorbit_decay():
    from matplotlib.ticker import ScalarFormatter

    fig, axes = plt.subplots(5, 2, figsize=(14, 18), sharex=True)
    axes = axes.ravel()
    summary_rows = []

    for idx, h0 in enumerate(REF_ALTS):
        ax = axes[idx]

        # For each activity level plot fast / nominal / slow
        first_act = True
        row = [h0]
        for act in op.SOLAR_ACTIVITIES:
            st = _DECAY_STYLE[act]

            # fast: max A/m, psi=0 (bulge)
            t_f, a_f, re_f, tr_f = _propagate_decay(
                h0, _da_dt_fixed, args=(_AM_MAX, 0.0, act))
            # slow: min A/m, psi=180 (anti-bulge)
            t_s, a_s, re_s, tr_s = _propagate_decay(
                h0, _da_dt_fixed, args=(_AM_MIN, 180.0, act))
            # nominal
            t_n, a_n, re_n, tr_n = _propagate_decay(
                h0, _da_dt_nominal, args=(act,))

            # Align arrays to a common time grid for fill_between
            t_common = np.linspace(0, DECAY_DAYS, 300)
            a_f_i = np.interp(t_common, t_f, a_f,
                              left=a_f[0], right=a_f[-1])
            a_s_i = np.interp(t_common, t_s, a_s,
                              left=a_s[0], right=a_s[-1])
            ax.fill_between(t_common, a_s_i, a_f_i,
                            color=st['color'], alpha=0.10)
            ax.plot(t_n, a_n, color=st['color'],
                    linestyle=st['ls'], linewidth=st['lw'],
                    label=ACT_STYLE[act]['label'] if first_act else None)
            ax.plot(t_f, a_f, color=st['color'],
                    linestyle='--', linewidth=0.9, alpha=0.7)
            ax.plot(t_s, a_s, color=st['color'],
                    linestyle=':', linewidth=0.9, alpha=0.7)

            row += [f"{a_n[-1]:.2f}", "yes" if re_n else "no",
                    f"{tr_n:.2f}" if re_n else "-"]
            first_act = False

        summary_rows.append(row)

        ax.set_title(f"Initial altitude: {h0} km", fontsize=11, pad=6)
        ax.set_xlim(0, DECAY_DAYS)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        if idx % 2 == 0:
            ax.set_ylabel("Altitude [km]", fontsize=11)
        if idx >= 8:
            ax.set_xlabel("Time [days]", fontsize=11)
        ax.tick_params(axis="both", labelsize=9)
        fmt = ScalarFormatter(useOffset=False)
        fmt.set_scientific(False)
        ax.yaxis.set_major_formatter(fmt)
        ax.xaxis.set_major_formatter(fmt)
        ax.ticklabel_format(style='plain', axis='both')

    fig.delaxes(axes[-1])

    # Build legend from the last populated axes
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               fontsize=10, frameon=True, bbox_to_anchor=(0.5, 0.975))
    fig.suptitle(
        f"{DECAY_DAYS:.0f}-Day Orbital Decay — Three Solar Activity Levels\n"
        "(solid = nominal, dashed = fast, dotted = slow at each level)",
        fontsize=14, fontweight="bold", y=0.998)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.93,
                        bottom=0.05, hspace=0.28, wspace=0.22)
    fig.savefig(os.path.join(OUT_DIR, "fig6_deorbit_decay.svg"), dpi=300)
    plt.close(fig)

    # CSV
    path = os.path.join(OUT_DIR, "deorbit_decay_summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "h0 [km]",
            "h_nom_low_10d [km]",  "low_reentry?",  "low_t_reentry [d]",
            "h_nom_mean_10d [km]", "mean_reentry?", "mean_t_reentry [d]",
            "h_nom_high_10d [km]", "high_reentry?", "high_t_reentry [d]",
        ])
        w.writerows(summary_rows)
    print(f"De-orbit decay summary -> {path}")


# ---------------------------------------------------------------------------
# Summary CSV — extended with solar activity columns
# ---------------------------------------------------------------------------
def export_summary_table(zonal, envelopes, srp, tb):
    idx_map = {h: int(np.argmin(np.abs(ALT - h))) for h in REF_ALTS}
    path = os.path.join(OUT_DIR, "disturbance_budget_table.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "Altitude [km]",
            "J2 [m/s^2]", "J3-J6 [m/s^2]",
            "Drag_low_nom [m/s^2]",  "Drag_mean_nom [m/s^2]",
            "Drag_high_nom [m/s^2]",
            "Drag_low_max [m/s^2]",  "Drag_mean_max [m/s^2]",
            "Drag_high_max [m/s^2]",
            "SRP_nominal [m/s^2]",
            "Moon [m/s^2]", "Sun [m/s^2]",
        ])
        for h in REF_ALTS:
            i = idx_map[h]
            w.writerow([h,
                f"{zonal['J2'][i]:.3e}",
                f"{zonal['J3-J6'][i]:.3e}",
                f"{envelopes['low']['nom'][i]:.3e}",
                f"{envelopes['mean']['nom'][i]:.3e}",
                f"{envelopes['high']['nom'][i]:.3e}",
                f"{envelopes['low']['hi'][i]:.3e}",
                f"{envelopes['mean']['hi'][i]:.3e}",
                f"{envelopes['high']['hi'][i]:.3e}",
                f"{srp['srp_nom'][i]:.3e}",
                f"{tb['moon'][i]:.3e}",
                f"{tb['sun'][i]:.3e}",
            ])
    print(f"Summary table -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Zonal harmonics...")
    zonal = fig_zonal_harmonics()

    print("Atmospheric drag envelopes (3 activity levels)...")
    envelopes = _compute_drag_envelopes()
    fig_drag(envelopes)

    print("Harris-Priester density profiles (3 activity levels)...")
    fig_harris_priester_density_profile()

    print("Harris-Priester diurnal bulge shape...")
    fig_harris_priester_diurnal_bulge()

    print("Solar-activity drag ratio plot...")
    fig_drag_solar_activity_ratio(envelopes)

    print("SRP...")
    srp = fig_srp()

    print("Third-body...")
    tb = fig_third_body()

    print("Combined disturbance budget (mean activity)...")
    fig_disturbance_budget(zonal, envelopes, srp, tb)

    print("Combined disturbance budget (all 3 activity levels)...")
    fig_disturbance_budget_solar_activity(zonal, envelopes, srp, tb)

    print("Extended disturbance budget CSV...")
    export_summary_table(zonal, envelopes, srp, tb)

    print("De-orbit decay study (3 activity levels)...")
    fig_deorbit_decay()

    print("Done. Figures in:", os.path.abspath(OUT_DIR))