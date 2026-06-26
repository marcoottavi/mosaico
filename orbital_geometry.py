import numpy as np
import os
from orbital_perturbations import MU_EARTH, R_EARTH, ALTITUDE, A_FRONTAL, A_LATERAL

# Configuration parameters for the orbital geometry calculations

OUT_DIR = 'figures'  # Directory to save figures
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

MASS_TOTAL = 1250  # Total mass of the spacecraft in kg

os.system('clear')


def gamma_angle(lambda_s, i_deg, delta_s=23.44):
    """
    gamma angle [rad] = arccos(h_hat . s_hat) for a dawn-dusk SSO.

    Parameters
    ----------
    lambda_s     :              float or ndarray  Sun ecliptic longitude [rad]
    i_deg   : float             Orbital inclination [deg]
    delta_s : float             Obliquity of the ecliptic [deg], default 23.44

    Returns
    -------
    gamma : same shape as lam, in radians
    """
    i   = np.radians(i_deg)
    delta_s = np.radians(delta_s)

    dot = (np.sin(i) * (np.cos(lambda_s)**2 + np.sin(lambda_s)**2 * np.cos(delta_s))
           + np.cos(i) * np.sin(lambda_s) * np.sin(delta_s))

    return np.arccos(np.clip(dot, -1.0, 1.0))


def sin_alpha(theta, lambda_s, inc, delta_s = 23.44):
    """
    Compute the absolute value of the sine of the gamma angle for a given
    orbital inclination, solar longitude, and obliquity.

    Parameters
    ----------
    inc     : float or ndarray  Orbital inclination [deg]
    theta   : float or ndarray  True anomaly [rad]
    lambda_s: float or ndarray  Sun ecliptic longitude [rad]
    delta_s : float             Obliquity of the ecliptic [deg], default 23.44

    Returns
    -------
    abs_sin_gamma : same shape as inputs, dimensionless
    """
    inc = np.radians(inc)
    delta_s = np.radians(delta_s)

    return -np.cos(inc)*np.cos(theta)*(np.cos(lambda_s)**2+np.cos(delta_s)*np.sin(lambda_s)**2)\
          + np.sin(lambda_s)*(np.cos(theta)*np.sin(delta_s)*np.sin(inc)+(1-np.cos(delta_s))*np.cos(lambda_s)*np.sin(theta))


def max_mean_min(x):
    """
    Compute the maximum, mean, and minimum of an array.

    Parameters
    ----------
    x : ndarray
        Input array.

    Returns
    -------
    max_val : float
        Maximum value of the array.
    mean_val : float
        Mean value of the array.
    min_val : float
        Minimum value of the array.
    """
    return np.max(x), np.mean(x), np.min(x)

def plot_angle(t, value, ylabel = ''):
    """
    Plot an angle over time with optional max, mean, and min lines.

    Parameters
    ----------
    t : ndarray
        Time array.
    value : ndarray
        Angle values to plot.
    max : float, optional
        Maximum value to plot as a horizontal line.
    mean : float, optional
        Mean value to plot as a horizontal line.
    min : float, optional
        Minimum value to plot as a horizontal line.
    ylabel : str, optional
        Label for the y-axis.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(14, 6))
    plt.rcParams.update({'font.size': 14,
                         'font.family': 'serif'})
    plt.plot(t / (86400), value, label='Angle', color='blue')
    
    plt.axhline(np.max(value), color='red', linestyle='--', label='Max')
    plt.axhline(np.mean(value), color='green', linestyle='--', label='Mean')
    plt.axhline(np.min(value), color='orange', linestyle='--', label='Min')

    plt.xlabel('Time [hours]')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(which = 'major', linestyle = '-', linewidth = 0.5, color = 'gray')
    plt.savefig(os.path.join(OUT_DIR, f'{ylabel.replace(" ", "_").lower()}.svg'), bbox_inches='tight', dpi=300)


# ── Example usage ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    i_deg = 97.79

    t_grid = np.arange(0, 365.2425* 24 * 3600, 60)  # Time grid for one day in seconds
    n_sc = np.sqrt(MU_EARTH / (R_EARTH + ALTITUDE)**3)  # mean motion [rad/s]
    print(n_sc)
    n_sun = 2 * np.pi / (365.25 * 24 * 3600)  # mean motion of the Sun [rad/s]
    theta = n_sc * t_grid  # True anomaly [rad]
    lambda_s = n_sun * t_grid  # Sun ecliptic longitude [rad]

    gamma = np.rad2deg(gamma_angle(lambda_s, i_deg))
    gamma_max, gamma_mean, gamma_min = max_mean_min(gamma)

    abs_sin_alpha = np.abs(sin_alpha(theta, lambda_s, i_deg))
    abs_cos_alpha = np.sqrt(1 - abs_sin_alpha**2)
    alpha = np.rad2deg(np.arcsin(abs_sin_alpha))

    alpha_max, alpha_mean, alpha_min = max_mean_min(alpha)
    sin_alpha_max, sin_alpha_mean, sin_alpha_min = max_mean_min(abs_sin_alpha)
    cos_alpha_max, cos_alpha_mean, cos_alpha_min = max_mean_min(abs_cos_alpha)

    # Plotting results
    plot_angle(t_grid, gamma, ylabel='Gamma Angle (deg)')
    plot_angle(t_grid, alpha, ylabel='Alpha Angle (deg)')

    # Print summary of results
    print(f"gamma Angle: Max = {gamma_max:.2f} deg, Mean = {gamma_mean:.2f} deg, Min = {gamma_min:.2f} deg")
    print(f"Alpha Angle: Max = {alpha_max:.2f} deg, Mean = {alpha_mean:.2f} deg, Min = {alpha_min:.2f} deg")
    print(f"Sin(Alpha): Max = {sin_alpha_max:.4f}, Mean = {sin_alpha_mean:.4f}, Min = {sin_alpha_min:.4f}")
    print(f"Cos(Alpha): Max = {cos_alpha_max:.4f}, Mean = {cos_alpha_mean:.4f}, Min = {cos_alpha_min:.4f}")

    # Calculations for Aerodynamic forces
    A_avg = A_FRONTAL * sin_alpha_mean + A_LATERAL * cos_alpha_mean
    A_min = A_FRONTAL * sin_alpha_min + A_LATERAL * cos_alpha_max
    A_max = A_FRONTAL * sin_alpha_max + A_LATERAL * cos_alpha_min
    print(f"Average Area: {A_avg:.4f} m^2, Min Area: {A_min:.4f} m^2, Max Area: {A_max:.4f} m^2")
    print(f"Average A/m: {A_avg/MASS_TOTAL:.6f} m^2/kg, Min A/m: {A_min/MASS_TOTAL:.6f} m^2/kg, Max A/m: {A_max/MASS_TOTAL:.6f} m^2/kg")

