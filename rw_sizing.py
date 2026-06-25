# Sizing of the Reaction Wheels for GG Torque Control

"""
gg_multi_beta.py
================
GG torque components and angular momentum accumulation for multiple
beta values, held on the same axes.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# ── Configuration ─────────────────────────────────────────────────────────────
BETAS_DEG = [31.2, -15.7]
N_ORBITS  = 1
OUT_DIR = 'figures'
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

# ── Constants ─────────────────────────────────────────────────────────────────
MU    = 398600.4418e9
R_E   = 6378.137e3
H_M   = 600e3
R_M   = R_E + H_M
Ixx   = 5101.10
Iyy   = 5101.10
Izz   = 10168.86

N_ORB = np.sqrt(MU / R_M**3)
T_ORB = 2*np.pi / N_ORB
K     = 3*MU/R_M**3 * (Izz - Ixx)

N_PTS = 4000
t     = np.linspace(0, N_ORBITS * T_ORB, N_ORBITS * N_PTS, endpoint=False)
theta = N_ORB * t
t_norm = t / T_ORB
dt    = t[1] - t[0]

# ── Colours and line styles per case ─────────────────────────────────────────
STYLES = {
    31.2:  dict(color='#1f4e79', ls='-',  lw=2.0),
    -15.7: dict(color='#b8382d', ls='-',  lw=2.0),
    'avg': dict(color='#2e7d32', ls='--', lw=2.2),
}

# ── Compute for each beta + average ──────────────────────────────────────────
results = {}
for b_deg in BETAS_DEG:
    b = np.radians(b_deg)
    iv = np.pi/2 + b
    Tx = K/2  * np.cos(iv) * np.sin(2*theta)
    Ty = -K/4 * np.sin(2*iv) * (1 - np.cos(2*theta))
    results[b_deg] = dict(
        Tx=Tx, Ty=Ty,
        Hx=np.cumsum(Tx)*dt,
        Hy=np.cumsum(Ty)*dt,
        Ty_mean=-K/4*np.sin(2*iv),
        i_deg=np.degrees(iv),
    )

# Average torque = mean of the two beta cases
Tx_avg = (results[BETAS_DEG[0]]['Tx'] + results[BETAS_DEG[1]]['Tx']) / 2
Ty_avg = (results[BETAS_DEG[0]]['Ty'] + results[BETAS_DEG[1]]['Ty']) / 2
results['avg'] = dict(
    Tx=Tx_avg, Ty=Ty_avg,
    Hx=np.cumsum(Tx_avg)*dt,
    Hy=np.cumsum(Ty_avg)*dt,
    Ty_mean=(results[BETAS_DEG[0]]['Ty_mean'] + results[BETAS_DEG[1]]['Ty_mean']) / 2,
)

keys   = BETAS_DEG + ['avg']
labels = [rf'$\beta={b}°$' for b in BETAS_DEG] + [r'Average']

# ── Matplotlib style ──────────────────────────────────────────────────────────
plt.rcParams.update({'font.size': 12, 'font.family': 'serif',
                     'axes.grid': True, 'grid.alpha': 0.35,
                     'grid.linestyle': '--'})

# ══════════════════════════════════════════════════════════════════════════════
# Figure 1: Torque components
# ══════════════════════════════════════════════════════════════════════════════
fig1, axes1 = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

for key, lbl in zip(keys, labels):
    r  = results[key]
    st = STYLES[key]
    axes1[0].plot(t_norm, r['Tx']*1e3, label=lbl, **st)
    axes1[1].plot(t_norm, r['Ty']*1e3, label=lbl, **st)

# Secular mean lines on Ty
for key, lbl in zip(keys, labels):
    r  = results[key]
    st = STYLES[key]
    axes1[1].axhline(r['Ty_mean']*1e3, color=st['color'], ls=':',
                     lw=1.2, alpha=0.8)

axes1[0].axhline(0, color='gray', lw=0.7, alpha=0.5)
axes1[1].axhline(0, color='gray', lw=0.7, alpha=0.5)

axes1[0].set_ylabel(r'$T_x$ [mN·m]')
axes1[1].set_ylabel(r'$T_y$ [mN·m]')
axes1[1].set_xlabel('Time [orbital periods]')
axes1[0].set_xlim(0, N_ORBITS)

axes1[0].set_title(
    r'$T_x = \frac{K}{2}\cos i\;\sin 2\theta$', fontsize=12)
axes1[1].set_title(
    r'$T_y = -\frac{K}{4}\sin 2i\;(1-\cos 2\theta)$  '
    r'(dotted = orbital mean)', fontsize=12)

axes1[0].legend(fontsize=10)
axes1[1].legend(fontsize=10)

fig1.suptitle(
    f'Gravity Gradient Torque Components — MOSAICO 600 km SSO\n'
    rf'$K=(3\mu/r^3)(I_{{zz}}-I_{{xx}})={K*1e3:.2f}$ mN·m,  '
    rf'$T_z=0$ $(I_{{xx}}=I_{{yy}})$',
    fontweight='bold', fontsize=12)
plt.tight_layout()
fig1.savefig(os.path.join(OUT_DIR, 'gg_torque_multi_beta.svg'),
             bbox_inches='tight', dpi=200)
plt.close(fig1)

# ══════════════════════════════════════════════════════════════════════════════
# Figure 2: Angular momentum accumulation
# ══════════════════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

for key, lbl in zip(keys, labels):
    r  = results[key]
    st = STYLES[key]
    axes2[0].plot(t_norm, r['Hx'], label=lbl, **st)
    axes2[1].plot(t_norm, r['Hy'], label=lbl, **st)

for ax in axes2:
    ax.axhline(0, color='gray', lw=0.7, alpha=0.5)

axes2[0].set_ylabel(r'$H_x$ [N·m·s]')
axes2[1].set_ylabel(r'$H_y$ [N·m·s]')
axes2[1].set_xlabel('Time [orbital periods]')
axes2[0].set_xlim(0, N_ORBITS)

axes2[0].set_title(r'$H_x = \int T_x\,dt$  (cyclic, zero mean)', fontsize=12)
axes2[1].set_title(r'$H_y = \int T_y\,dt$  (secular drift)', fontsize=12)

axes2[0].legend(fontsize=10)
axes2[1].legend(fontsize=10)

# Annotate per-orbit H_y accumulation
for key, lbl in zip(keys, labels):
    r  = results[key]
    st = STYLES[key]
    Hy_per_orbit = r['Ty_mean'] * T_ORB
    axes2[1].annotate(
        rf'$\Delta H_y$={Hy_per_orbit:.2f} N·m·s/orbit',
        xy=(N_ORBITS, r['Hy'][-1]),
        xytext=(N_ORBITS - 0.55, r['Hy'][-1] + 0.3),
        fontsize=8, color=st['color'],
        arrowprops=dict(arrowstyle='->', color=st['color'], lw=0.8))

fig2.suptitle(
    'Angular Momentum Accumulation — MOSAICO 600 km SSO\n'
    r'$H_z = 0$ exactly $(I_{xx}=I_{yy})$',
    fontweight='bold', fontsize=12)
plt.tight_layout()
fig2.savefig(os.abortpath.join(OUT_DIR, 'gg_angular_momentum_multi_beta.svg'),
             bbox_inches='tight', dpi=200)
plt.close(fig2)

# ── Console summary ───────────────────────────────────────────────────────────
print(f"{'β':>8}  {'i':>7}  {'peak Tx':>10}  {'peak Ty':>10}  "
      f"{'<Ty>':>10}  {'ΔHy/orbit':>12}")
print("-"*65)
for key, lbl in zip(keys, labels):
    r = results[key]
    b_str = f"{key}°" if key != 'avg' else 'avg'
    i_str = f"{r.get('i_deg', float('nan')):.1f}°" if key != 'avg' else '—'
    print(f"{b_str:>8}  {i_str:>7}  "
          f"{np.max(np.abs(r['Tx']))*1e3:>9.3f}m  "
          f"{np.max(np.abs(r['Ty']))*1e3:>9.3f}m  "
          f"{r['Ty_mean']*1e3:>9.3f}m  "
          f"{r['Ty_mean']*T_ORB:>11.3f}")

print("\nFigures saved.")