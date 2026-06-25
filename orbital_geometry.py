import numpy as np

def beta_angle(lam, i_deg, eps_deg=23.44):
    """
    Beta angle [rad] = arccos(h_hat . s_hat) for a dawn-dusk SSO.

    Parameters
    ----------
    lam     : float or ndarray  Sun ecliptic longitude [rad]
    i_deg   : float             Orbital inclination [deg]
    eps_deg : float             Obliquity of the ecliptic [deg], default 23.44

    Returns
    -------
    beta : same shape as lam, in radians
    """
    i   = np.radians(i_deg)
    eps = np.radians(eps_deg)

    dot = (np.sin(i) * (np.cos(lam)**2 + np.sin(lam)**2 * np.cos(eps))
           + np.cos(i) * np.sin(lam) * np.sin(eps))

    return np.arccos(np.clip(dot, -1.0, 1.0))


def average_of_f(f, i_deg, eps_deg=23.44, n_samples=100_000):
    """
    Compute the annual average of a generic function f(beta) over one
    full solar year, sampling lambda uniformly in [0, 2*pi).

    Parameters
    ----------
    f        : callable         Any function of beta [rad] -> scalar or array
    i_deg    : float            Orbital inclination [deg]
    eps_deg  : float            Obliquity [deg]
    n_samples: int              Number of uniform lambda samples

    Returns
    -------
    mean_f   : float            (1/2pi) * integral_0^{2pi} f(beta(lambda)) d_lambda
    std_f    : float            Standard deviation across samples (uncertainty indicator)
    beta_arr : ndarray          Beta values at each sample [rad]
    """
    lam      = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    beta_arr = beta_angle(lam, i_deg, eps_deg)
    f_vals   = f(beta_arr)

    return np.mean(f_vals), np.std(f_vals), beta_arr

def abs_sin(x):

    return np.abs(np.sin(x))

def abs_cos(x):

    return np.abs(np.cos(x))




# ── Example usage ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    i_deg = 97.79

    # Average beta itself (in degrees)
    mean_beta, std_beta, beta_arr = average_of_f(np.degrees, i_deg)
    print(f"Mean beta          : {mean_beta:.3f} deg")
    print(f"Std  beta          : {std_beta:.3f} deg")
    print(f"Min  beta          : {np.degrees(np.min(beta_arr)):.3f} deg")
    print(f"Max  beta          : {np.degrees(np.max(beta_arr)):.3f} deg")

    # Average np.abs(np.sin(beta)) (dimensionless)
    mean_abs_sin_beta, std_abs_sin_beta, _ = average_of_f(abs_sin, i_deg)
    print(f"Mean |sin(beta)|   : {mean_abs_sin_beta:.3f}")
    print(f"Std  |sin(beta)|   : {std_abs_sin_beta:.3f}")

    # Average np.abs(np.cos(beta)) (dimensionless)
    mean_abs_cos_beta, std_abs_cos_beta, _ = average_of_f(abs_cos, i_deg)
    print(f"Mean |cos(beta)|   : {mean_abs_cos_beta:.3f}")
    print(f"Std  |cos(beta)|   : {std_abs_cos_beta:.3f}")