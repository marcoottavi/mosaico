import numpy as np
import orbital_perturbations as op
import matplotlib.pyplot as plt
import os
from orbital_perturbations import R_EARTH, MU_EARTH


OUT_DIR = 'figures'
PHASING_BUDGET = 55 # m/s
WAYPOINTS = np.array([30000, 5000, 1000, 500])/1000 # km
A_M = 0.01 # m^2/kg
ORBIT_ALTITUDE = 600 # km
T_MISSION = 11 * 365*24*3600 # seconds


# Rendezvous and docking budget
def rendezvous_and_docking(waypoints, orbit_altitude, radius, mu):
    """
    Calculate the rendezvous and docking budget for a spacecraft.

    Parameters:
    waypoints (list): List of waypoints for the rendezvous and docking maneuver.
    orbit_altitude (float): Altitude of the orbit in meters.
    radius (float): Radius of the spacecraft in meters.
    mu (float): Standard gravitational parameter of the central body.

    Returns:
    float: Total delta-v required for the rendezvous and docking maneuver.
    """
    R_orbit = radius + orbit_altitude
    n = np.sqrt(mu / R_orbit**3)  # Mean motion
    total_delta_v = 0.0

    for i in range(len(waypoints) - 1):
        # Calculate the delta-v for each leg of the maneuver
        delta_x = np.abs(waypoints[i + 1] - waypoints[i])
        total_delta_v += n*delta_x/4 * 2 # Departure and Arrival burns are identical = n*delta_x/4 each
    
    return total_delta_v*1000 # Convert from km/s to m/s


# Station-keeping budget
def stationkeeping_budget(A_m, h, T_mission):
    """
    Calculate the station-keeping budget for a spacecraft.

    Parameters:
    A_m (float): Area-to-mass ratio of the spacecraft in m^2/kg.
    h (float): Altitude of the orbit in meters.
    radius (float): Radius of the spacecraft in meters.
    mu (float): Standard gravitational parameter of the central body.

    Returns:
    float: Total delta-v required for station-keeping.
    """
    drag_low = op.drag_accel(h, A_m, psi_deg = 90,
               Cd=2.2, n_exp=6, corotating_atm=False, activity='low')
    drag_mean = op.drag_accel(h, A_m, psi_deg = 90,
               Cd=2.2, n_exp=6, corotating_atm=False, activity='mean')
    drag_high = op.drag_accel(h, A_m, psi_deg = 90,
               Cd=2.2, n_exp=6, corotating_atm=False, activity='high')
    
    delta_v_sk_low = drag_low * T_mission
    delta_v_sk_mean = drag_mean * T_mission
    delta_v_sk_high = drag_high * T_mission

    return delta_v_sk_low, delta_v_sk_mean, delta_v_sk_high

def total_budget_histogram(A_m, h, T_mission, margin = 0.2):
    """
    Calculate the total budget histogram for a spacecraft.

    Parameters:
    A_m (float): Area-to-mass ratio of the spacecraft in m^2/kg.
    h (float): Altitude of the orbit in meters.
    T_mission (float): Duration of the mission in seconds.

    Returns:
    tuple: Total delta-v required for rendezvous and docking, station-keeping low, mean, and high.
    """
    r_and_d_budget = rendezvous_and_docking(WAYPOINTS, h, R_EARTH, MU_EARTH)*(1+margin)
    delta_v_sk_low, delta_v_sk_mean, delta_v_sk_high = stationkeeping_budget(A_m, h, T_mission)
    delta_v_sk_low *= (1+margin)
    delta_v_sk_mean *= (1+margin)
    delta_v_sk_high *= (1+margin)

    # Generate the histogram plot
    budgets = [r_and_d_budget, delta_v_sk_low, delta_v_sk_mean, delta_v_sk_high]
    labels = ['R&D', 'SK (Low)', 'SK (Mean)', 'SK (High)']
    plt.rcParams.update({'font.size': 12,
                         'font.family': 'serif'})
    plt.figure(figsize=(14, 6))
    bars = plt.bar(labels, budgets, color=['blue', 'orange', 'green', 'red'])
    plt.ylabel('Delta-v (m/s)')
    plt.title(f'Delta-V Budget with Margin of {margin*100:.0f}%')
    plt.yscale('log')
    plt.ylim(1, max(budgets)*5)
    plt.grid(which = 'both', linestyle = '--', alpha = 0.3)

    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height * 1.1,                # 10% above bar (works well for log scale)
            f'{height:.2f} m/s',
            ha='center',
            va='bottom'
        )

    plt.savefig(os.path.join(OUT_DIR, f'delta_v_budget.svg'), bbox_inches = 'tight')


if __name__ == "__main__":
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    print(R_EARTH, MU_EARTH)
    # Rendezvous and docking budget
    r_and_d_budget = rendezvous_and_docking(WAYPOINTS, ORBIT_ALTITUDE, R_EARTH, MU_EARTH)
    print(f"Rendezvous and docking budget: {r_and_d_budget:.2f} m/s")
    total_budget_histogram(A_M, ORBIT_ALTITUDE, T_MISSION, margin = 0.2)
