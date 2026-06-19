"""
thrusters_sizing.py
====================

Preliminary sizing of the propulsion subsystem (8-thruster RCS configuration,
Section 3 of the docking report) and trade against a dedicated reaction-wheel
assembly (RWA) for attitude control, for the single-"atom" hexagonal module
of Table 7.

This script:
  1. Backs out an equivalent structural radius from the Table-7 inertia
     tensor (thin-disk consistency check), used as the RCS thruster moment
     arm for torque calculations.
  2. Computes the gravity-gradient disturbance torque vs. altitude and
     compares it with aerodynamic-torque and SRP-torque order-of-magnitude
     estimates (CoM/CoP offset assumption stated explicitly).
  3. Computes the torque required for representative attitude
     re-orientation (slew) maneuvers and compares it against (a) the
     torque available from the 8-thruster RCS and (b) a candidate
     reaction-wheel class -- this comparison is the basis for the
     RCS-vs-RWA actuator trade.
  4. Sizes the RCS thrust class against the phasing Delta-V budget
     (already established in phasing.py) for a chosen, finite burn
     duration.
  5. Builds a consolidated Delta-V budget (phasing + station-keeping +
     rendezvous & docking) and converts it into a propellant-mass
     estimate via the rocket equation, for a chosen propellant/Isp.

Run:
    python3 thrusters_sizing.py

Outputs (./figures/):
    fig9_disturbance_torques.png
    fig10_slew_torque_vs_actuators.png
    fig11_thrust_burn_duration_trade.png
    fig12_deltav_budget.png
    propulsion_sizing_summary.csv
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import orbital_perturbations as op




# ---------------------------------------------------------------------------
# 2) Slew torque requirement vs. RCS / RWA torque capability
# ---------------------------------------------------------------------------
def slew_torque(angle_deg, duration_s, I):
    """Torque [N*m] for a rest-to-rest, symmetric triangular (bang-bang)
    slew of angle_deg in duration_s, about an axis with inertia I:
        alpha = 4*theta / T^2 ,  T_req = I * alpha
    """
    theta = np.radians(angle_deg)
    alpha = 4.0 * theta / duration_s**2
    return I * alpha


def fig_slew_torque_vs_actuators(F_rcs_per_thruster, n_couple_thrusters=2):
    angles = np.linspace(5, 180, 60)
    durations = [120, 300, 600, 1200]  # seconds
    I_major = I_DIAG[2]   # worst case (largest inertia axis)

    T_rcs_available = n_couple_thrusters * F_rcs_per_thruster * R_EQ  # couple torque [N*m]
    T_rw_candidate = 0.20  # [N*m] candidate single reaction-wheel torque (see text)

    fig, ax = plt.subplots()
    for T in durations:
        Treq = slew_torque(angles, T, I_major)
        ax.plot(angles, Treq * 1e3, label=f"{T/60:.0f} min slew")

    ax.axhline(T_rw_candidate * 1e3, color="#2e8b57", linestyle="--", linewidth=2.2,
               label=f"Candidate RW torque ({T_rw_candidate:.2f} N$\\cdot$m)")
    ax.axhline(T_rcs_available * 1e3, color="#c0504d", linestyle="-.", linewidth=2.2,
               label=f"RCS couple torque @ {F_rcs_per_thruster:.0f} N/thruster "
                     f"({T_rcs_available:.1f} N$\\cdot$m)")
    ax.set_yscale("log")
    ax.set_xlabel("Slew angle [deg]")
    ax.set_ylabel("Required torque [mN$\\cdot$m]")
    ax.set_title("Required Slew Torque (major axis) vs. Actuator Torque Capability")
    ax.legend(fontsize=9, loc="lower right", ncol=1)
    fig.savefig(os.path.join(OUT_DIR, "fig10_slew_torque_vs_actuators.png"))
    plt.close(fig)
    return {"T_rcs_available": T_rcs_available, "T_rw_candidate": T_rw_candidate}


# ---------------------------------------------------------------------------
# 3) RCS thrust-class sizing against the phasing Delta-V (finite burn trade)
# ---------------------------------------------------------------------------
def fig_thrust_burn_duration_trade(dv_cases_m_s, m=M_ATOM):
    burn_s = np.linspace(30, 1200, 200)
    fig, ax = plt.subplots()
    for dv in dv_cases_m_s:
        F = m * dv / burn_s
        ax.plot(burn_s / 60.0, F, label=f"$\\Delta V$ = {dv:.1f} m/s")
    ax.axhspan(20, 30, color="#2e8b57", alpha=0.12,
               label="Candidate cluster thrust band (4 $\\times$ ~22 N)")
    ax.axhline(88.0, color="#c0504d", linestyle="--", linewidth=1.6,
               label="4-thruster cluster, 22 N class (88 N)")
    ax.set_xlabel("Burn duration [min]")
    ax.set_ylabel("Required combined thrust [N]")
    ax.set_title(f"RCS Thrust vs. Burn Duration for Representative Phasing $\\Delta V$\n"
                 f"(single-atom mass = {m:.0f} kg)")
    ax.set_ylim(0, 250)
    ax.legend(fontsize=9.5)
    fig.savefig(os.path.join(OUT_DIR, "fig11_thrust_burn_duration_trade.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4) Station-keeping Delta-V (drag make-up), extrapolated from the
#    de-orbit/decay model already used for Figure 4 of the perturbation study
# ---------------------------------------------------------------------------
def station_keeping_dv_rate(h_km, regime="nominal", am_grid=None):
    """Drag-compensation Delta-V rate [m/s per day] at altitude h_km.
    For near-circular orbits the tangential drag deceleration integrates
    directly into the velocity (and hence altitude) that must be restored
    by station-keeping burns: dV/dt ~= a_drag(h)."""
    if am_grid is None:
        am_grid = np.linspace(0.005, 0.04, 12)
    if regime == "fast":
        a_drag = op.drag_accel(h_km, am_grid.max(), 0.0, Cd=2.2)
    elif regime == "slow":
        a_drag = op.drag_accel(h_km, am_grid.min(), 180.0, Cd=2.2)
    else:
        _, a_drag, _ = op.drag_accel_envelope(h_km, am_grid)
    return a_drag * 86400.0  # [m/s per day]


# ---------------------------------------------------------------------------
# 5) Consolidated Delta-V budget and propellant mass
# ---------------------------------------------------------------------------
def fig_deltav_budget(mission_years=5.0, isp_s=220.0):
    am_grid = np.linspace(0.005, 0.04, 12)

    # --- Phasing (from phasing.py / Section 4 of the document) ---
    dv_phasing_worst = 55.0           # [m/s] worst case, 180 deg in <1 day, 500 km floor
    dv_phasing_recommended = 20.0     # [m/s] recommended few-day phasing window (Section 4)

    # --- Station-keeping at 600 km over the assumed mission life ---
    days = mission_years * 365.25
    dv_sk_fast = station_keeping_dv_rate(H_REF, "fast", am_grid) * days
    dv_sk_nom = station_keeping_dv_rate(H_REF, "nominal", am_grid) * days
    dv_sk_slow = station_keeping_dv_rate(H_REF, "slow", am_grid) * days

    # --- Rendezvous & docking, post entry-gate (far-range homing + close
    #     range + final approach), bottom-up engineering allocation from
    #     heritage closing-rate profiles (Fehse 2003; Table 6 of the
    #     document), NOT a 6-DOF trajectory simulation:
    #       far-range homing / trajectory shaping  ~5 m/s
    #       close-range braking sequence (hold points 1000/500/250/30 m)  ~4 m/s
    #       final approach & mating (cm/s-level closing)                 ~1.5 m/s
    #       CAM / contingency reserve (1-2 abort events)                 ~4 m/s
    #     + 20% engineering margin
    dv_rvd_nominal = (5.0 + 4.0 + 1.5)
    dv_rvd_with_cam = dv_rvd_nominal + 4.0
    dv_rvd_with_margin = dv_rvd_with_cam * 1.2

    categories = ["Phasing\n(worst case)", "Phasing\n(recommended)",
                  f"Station-keeping\n{mission_years:.0f} yr (slow)",
                  f"Station-keeping\n{mission_years:.0f} yr (nominal)",
                  f"Station-keeping\n{mission_years:.0f} yr (fast)",
                  "Rendezvous &\ndocking (w/ margin)"]
    values = [dv_phasing_worst, dv_phasing_recommended, dv_sk_slow, dv_sk_nom,
              dv_sk_fast, dv_rvd_with_margin]

    fig, ax = plt.subplots(figsize=(12, 6.5))
    colors = ["#c0504d", "#d99694", "#9ecae1", "#3182bd", "#08306b", "#2e8b57"]
    bars = ax.bar(categories, values, color=colors, width=0.62)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + max(values) * 0.015, f"{v:.1f}",
                ha="center", fontsize=10.5)
    ax.set_ylabel("$\\Delta V$ [m/s]")
    ax.set_title("Preliminary Delta-V Budget by Source - Single Atom, 600 km SSO")
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, fontsize=10.5)
    ax.set_ylim(0, max(values) * 1.18)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig12_deltav_budget.png"))
    plt.close(fig)

    # Total mission Delta-V (recommended phasing + nominal station-keeping
    # + rendezvous & docking w/ margin), and worst-case bound
    dv_total_recommended = dv_phasing_recommended + dv_sk_nom + dv_rvd_with_margin
    dv_total_worst = dv_phasing_worst + dv_sk_fast + dv_rvd_with_margin

    def prop_mass(dv, m0=M_ATOM, isp=isp_s):
        return m0 * (1.0 - np.exp(-dv / (isp * G0)))

    summary = {
        "dv_phasing_worst": dv_phasing_worst,
        "dv_phasing_recommended": dv_phasing_recommended,
        "dv_sk_fast": dv_sk_fast, "dv_sk_nom": dv_sk_nom, "dv_sk_slow": dv_sk_slow,
        "dv_rvd_nominal": dv_rvd_nominal, "dv_rvd_with_cam": dv_rvd_with_cam,
        "dv_rvd_with_margin": dv_rvd_with_margin,
        "dv_total_recommended": dv_total_recommended, "dv_total_worst": dv_total_worst,
        "mprop_recommended_kg": prop_mass(dv_total_recommended),
        "mprop_worst_kg": prop_mass(dv_total_worst),
    }
    return summary


# ---------------------------------------------------------------------------
# Export summary CSV
# ---------------------------------------------------------------------------
def export_summary(disturb, actuators, dv_budget):
    path = os.path.join(OUT_DIR, "propulsion_sizing_summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Parameter", "Value", "Unit"])
        w.writerow(["Equivalent structural radius R_eq", f"{R_EQ:.3f}", "m"])
        w.writerow(["Ixx check (thin-disk, 0.25*m*R_eq^2)", f"{IXX_CHECK:.1f}", "kg*m^2 (table value 3571)"])
        w.writerow(["Gravity-gradient torque @600km (max)", f"{disturb['Tgg_600']*1e3:.3f}", "mN*m"])
        w.writerow(["Aero torque @600km (nominal)", f"{disturb['Taero_nom_600']*1e3:.4f}", "mN*m"])
        w.writerow(["Aero torque @600km (fast)", f"{disturb['Taero_fast_600']*1e3:.4f}", "mN*m"])
        w.writerow(["SRP torque (nominal)", f"{disturb['Tsrp_600']*1e3:.4f}", "mN*m"])
        w.writerow(["RCS couple torque available", f"{actuators['T_rcs_available']:.2f}", "N*m"])
        w.writerow(["Candidate RW torque", f"{actuators['T_rw_candidate']:.2f}", "N*m"])
        for k, v in dv_budget.items():
            unit = "kg" if "mprop" in k else "m/s"
            w.writerow([k, f"{v:.2f}", unit])
    print(f"Summary written to {path}")


if __name__ == "__main__":
    print(f"Equivalent structural radius R_eq = {R_EQ:.3f} m "
          f"(Ixx check: {IXX_CHECK:.1f} kg*m^2 vs. table value 3571 kg*m^2)")

    print("Computing disturbance torque budget (gravity-gradient, aero, SRP)...")
    disturb = fig_disturbance_torques()
    for k, v in disturb.items():
        print(f"  {k} = {v*1e3:.4f} mN*m")

    print("Computing slew torque vs. actuator capability...")
    actuators = fig_slew_torque_vs_actuators(F_rcs_per_thruster=22.0, n_couple_thrusters=2)
    print(f"  RCS couple torque available  = {actuators['T_rcs_available']:.2f} N*m")
    print(f"  Candidate RW torque          = {actuators['T_rw_candidate']:.2f} N*m")

    print("Computing thrust-vs-burn-duration trade for phasing...")
    fig_thrust_burn_duration_trade(dv_cases_m_s=[12.0, 20.0, 28.0, 55.0])

    print("Building consolidated Delta-V budget...")
    dv_budget = fig_deltav_budget(mission_years=5.0, isp_s=220.0)
    for k, v in dv_budget.items():
        unit = "kg" if "mprop" in k else "m/s"
        print(f"  {k} = {v:.2f} {unit}")

    export_summary(disturb, actuators, dv_budget)
    print("Done. Figures saved in:", os.path.abspath(OUT_DIR))