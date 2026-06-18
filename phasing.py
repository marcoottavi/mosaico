import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import os

OUT_DIR = 'figures'
NUM_STEPS = 20

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

def hohmann(mu, r1, r2):
    v1 = np.sqrt(mu / r1)
    v2 = np.sqrt(mu / r2)
    a = (r1 + r2) / 2

    v_transfer1 = np.sqrt(2 * mu / r1 - mu / a)
    v_transfer2 = np.sqrt(2 * mu / r2 - mu / a)

    delta_v1 = np.abs(v_transfer1 - v1)
    delta_v2 = np.abs(v2 - v_transfer2)

    return delta_v1, delta_v2


def delta_sma(mu, r2, phasing_time, initial_angle, k_val):
    n2 = np.sqrt(mu / r2**3)

    n1 = ((n2 * phasing_time - initial_angle + 2*np.pi*k_val)) / phasing_time

    # Avoid invalid values (negative or zero mean motion)
    n1 = np.where(n1 <= 0, np.nan, n1)

    a1 = (mu / n1**2)**(1/3)

    return a1 - r2


def total_delta_v(mu, r2, initial_angle, phasing_time, k_val):
    d_sma = delta_sma(mu, r2, phasing_time, initial_angle, k_val)
    r1 = r2 + d_sma

    # Avoid invalid radii
    r1 = np.where(r1 <= 0, np.nan, r1)

    delta_v1, delta_v2 = hohmann(mu, r1, r2)

    return delta_v1 + delta_v2


def fmt(x):
    return f'{x:.1f} m/s' if x < 100 else f'{x:.0f} m/s'

# Divide the total altitude change into multiple smaller maneuvers
def ladder(mu, r1, r2, num_steps):
    altitudes = np.linspace(r1, r2, num_steps + 1)
    delta_vs = []
    for i in range(num_steps):
        dv1, dv2 = hohmann(mu, altitudes[i], altitudes[i+1])
        delta_vs.append(dv1 + dv2)
    return np.sum(delta_vs)

if __name__ == '__main__':
    os.system('clear')

    # Constants
    mu = 3.986004418e5      # km^3/s^2
    Re = 6378               # km
    r2 = Re + 600           # target orbit radius (km)
    max_dv = 0.2            # km/s
    sma_min = Re + 500      # km

    # Grids
    initial_angle = np.linspace(-np.pi, 0, 1000)
    phasing_time = np.linspace(1, 10 * 24 * 3600, 1000)
    k = np.arange(-6,6)

    X, Y = np.meshgrid(initial_angle, phasing_time)
    Z_stack = np.full((X.shape[0], X.shape[1], len(k)), np.nan)
    for ii, k_val in enumerate(k):
        Z_stack[:,:,ii] = total_delta_v(mu, r2, X, Y, k_val)

    k_values = np.array(k)
    k_opt_idx = np.nanargmin(Z_stack, axis=2)
    k_opt = k_values[k_opt_idx]

    Z = np.nanmin(Z_stack, axis = 2)
    Z = np.where(Z > max_dv, np.nan, Z)
    Z*=1000  # Convert to m/s
    levels =np.array((0,0.5,1,2,3,4,5,7,10,15,20,30,40,50,100))

    # Compute delta_sma for optimal k
    d_sma_opt = delta_sma(mu, r2, Y, X, k_opt)
    sma_opt = r2 + d_sma_opt
    admissible = (sma_opt >= sma_min).astype(int)
    print("Max Admissible Delta-V (m/s):", np.nanmax(Z[admissible==1]))
    
    # Debug stats
    print("Z min =", np.nanmin(Z))
    print("Z max =", np.nanmax(Z))
    print("Z median =", np.nanmedian(Z))
    print("NaNs =", np.isnan(Z).sum())

    # Plot 1
    plt.rcParams.update({
        'font.size':15,
        'font.family':'serif',
    })
    fig, ax = plt.subplots(figsize=(8, 6))
    X_plot = Y/86400 
    Y_plot = np.rad2deg(X)

    CS = ax.contour(X_plot, Y_plot, Z, levels = levels, colors = "#000000")
    CS_mask = ax.contourf(X_plot, Y_plot, admissible, levels = [-0.5, 0.5,  1.5], colors = ['#FF0000', '#00FF00'], alpha = 0.3)
    ax.set_xlim(X_plot.min(), X_plot.max())
    ax.set_ylim(Y_plot.min()*1.05, Y_plot.max()*1.05)
    ax.grid(which = 'major', alpha = 0.3, color = 'gray')
    labels = ax.clabel(CS, inline=True, fontsize=8, fmt=fmt)

    ax.set_xlabel("Phasing time (days)")
    ax.set_ylabel("Initial phasing angle (deg)")
    legend_elements = [
    Patch(facecolor='#00FF00', edgecolor='k',
          alpha=0.3, label='Admissible'),
    Patch(facecolor='#FF0000', edgecolor='k',
          alpha=0.3, label='Non-admissible')
]
    ax.set_title(r"Phasing Orbit $\rightarrow$ Target Orbit Transfer $\Delta V$ (m/s)")
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(0, -0.2), ncol = 2)
    ax.minorticks_on()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'delta_v.svg'), dpi = 300, bbox_inches = 'tight')
    plt.close('all')

    # Plot 2
    fig, ax = plt.subplots(figsize = (10,6))
    cf = ax.pcolormesh(X_plot, Y_plot, k_opt, cmap = 'tab10')
    cbar = plt.colorbar(cf)
    cbar.set_label("Optimal k")
    plt.savefig(os.path.join(OUT_DIR, 'optimal_k.png'), dpi = 300, bbox_inches = 'tight')

    # Plot 3: Ladder Plot from minimum admissible sma to target orbit
    N_grid = np.linspace(1, NUM_STEPS, NUM_STEPS).astype(int)
    delta_vs = []
    for n in N_grid:
        delta_v = ladder(mu, sma_min, r2, n)
        delta_vs.append(delta_v*1000)  # Convert to m/s

    fig, ax = plt.subplots(figsize = (8,4))
    ax.plot(N_grid, (delta_vs-delta_vs[0])/delta_vs[0]*100, marker = 'o')
    from matplotlib.ticker import ScalarFormatter

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0,0))

    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.set_xlabel("Number of Steps")
    ax.set_ylabel("Delta-V cost increase (%)")
    ax.grid(which = 'major', alpha = 0.3, color = 'gray')
    plt.savefig(os.path.join(OUT_DIR, 'ladder_delta_v.svg'), dpi = 300, bbox_inches = 'tight')
    plt.close('all')
    print('Maximum relative Delta-V cost increase '+ f' {(np.max(delta_vs-delta_vs[0])/delta_vs[0])*100:.4f} %')
    