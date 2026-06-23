import numpy as np
import orbital_perturbations as op
import matplotlib.pyplot as plt
import os
from orbital_perturbations import R_EARTH, MU_EARTH


OUT_DIR = 'figures'
PHASING_MANEUVER = 55/2 # m/s
RECCOMMENDED_PHASING_MANEUVER = 20/2 # m/s
WAYPOINTS = np.array([30000, 5000, 1000, 500])/1000 # km
A_M = 0.01 # m^2/kg
ORBIT_ALTITUDE = 600 # km
T_MISSION = 10*365*24*3600 # seconds
I_SP = 300 # s
MASS = 1250 # kg (whole atom)
DELTA_V_MARGIN = 0.2 # 20% margin

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
    T_mission (float): Duration of the mission in seconds.

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
    phasing_budget = PHASING_MANEUVER*(1+margin)

    # Generate the histogram plot
    budgets = [phasing_budget, r_and_d_budget, delta_v_sk_low, delta_v_sk_mean, delta_v_sk_high]
    labels = ['Phasing', 'R&D', 'SK (Low)', 'SK (Mean)', 'SK (High)']
    plt.rcParams.update({'font.size': 12,
                         'font.family': 'serif'})
    plt.figure(figsize=(14, 6))
    bars = plt.bar(labels, budgets, color=['blue', 'orange', 'green', 'red', 'purple'])
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

    return r_and_d_budget, delta_v_sk_low, delta_v_sk_mean, delta_v_sk_high


def thrusters_selection_plot(mass, h_phasing, r_body, mu_body):
    
    # Orbit period
    R_orbit = r_body + h_phasing
    T_orbit = 2*np.pi*np.sqrt(R_orbit**3/mu_body)

    # Define a time grid
    time_grid = np.linspace(100, T_orbit, 1000)
    t_perc_grid = time_grid/T_orbit * 100
    
    # define a delta_v grid
    delta_v_grid = np.linspace(1, 100, 1000) # m/s

    # Create meshes
    Time_mesh, Delta_v_mesh = np.meshgrid(time_grid, delta_v_grid)
    Thrust_mesh = mass*Delta_v_mesh/Time_mesh # N

    # Contour plot with fmt
    levels = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    plt.rcParams.update({'font.size': 15,
                         'font.family': 'serif'})
    plt.figure(figsize=(14, 6))
    plt.vlines(x = [RECCOMMENDED_PHASING_MANEUVER], ymin = t_perc_grid.min(), ymax = t_perc_grid.max(), label = 'Reccommended phasing maneuver', color = 'orange')
    plt.vlines(x = [PHASING_MANEUVER], ymin = t_perc_grid.min(), ymax = t_perc_grid.max(), label = 'Worst-case phasing maneuver', color = 'blue')
    cp = plt.contour(Delta_v_mesh, Time_mesh/T_orbit * 100,  Thrust_mesh, levels=levels, colors = 'black')
    cp.clabel(fmt='%1.1f' + ' N', fontsize=12)
    plt.ylabel(r'Percentage of Orbit Period for Maneuver (%)')
    plt.xlabel(r'$\Delta V$ (m/s)')
    plt.minorticks_on()
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(which = 'both', linestyle = '--', alpha = 0.3)
    plt.legend(loc = 'center left', fontsize = 12, bbox_to_anchor = (-0.01, -0.15))
    plt.savefig(os.path.join(OUT_DIR, f'thrusters_selection.svg'), bbox_inches = 'tight')

if __name__ == "__main__":
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    # Rendezvous and docking budget
    r_and_d_budget, delta_v_sk_low, delta_v_sk_mean, delta_v_sk_high = total_budget_histogram(A_M, ORBIT_ALTITUDE, T_MISSION, margin = DELTA_V_MARGIN)

    # Trust level selection plot
    thrusters_selection_plot(MASS, ORBIT_ALTITUDE, R_EARTH, MU_EARTH)