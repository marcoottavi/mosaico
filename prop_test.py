from godot import cosmos
from godot.core import tempo, util
from godot.core import astro
import numpy as np



# avoid verbose output
util.suppressLogger()

uni_config = cosmos.util.load_yaml('universe.yml')
uni = cosmos.Universe(uni_config)
gm = {'gm': 398600.4418}


epoch0 = tempo.Epoch("2026-03-20T09:01:00 UTC")
dt = 365 * 24 * 3600  # seconds in a year
epoch_grid = tempo.EpochRange(epoch0, epoch0+dt).createGrid(60)

tol = 1e-8
pro = cosmos.BallisticPropagator(uni, 'SC', 'SunEarthMoonDynamics', epoch0, 'Earth', tol)
pro.setMaxSteps(int(1e6))

init_kepl = [6378 + 600,
             0.0,
             np.radians(97.79),
             np.radians(90),
             0.0,
             0.0]

in_state = astro.cartFromKep(init_kepl, gm)
pro.compute(in_state, 0.0, epoch0+dt)

state_cart = [uni.frames.vector6('SC', 'Earth', 'ICRF', epoch) for epoch in epoch_grid]
state_kepl = np.array([astro.kepFromCart(state, gm) for state in state_cart])

# Plot Keplerian Elements
import matplotlib.pyplot as plt
a = state_kepl[:, 0]
e = state_kepl[:, 1]
i = np.degrees(state_kepl[:, 2])
Omega = np.degrees(state_kepl[:, 3])
omega = np.degrees(state_kepl[:, 4])
nu = np.degrees(state_kepl[:, 5])

t_grid = np.array([epoch.jd('TDB', tempo.JulianDay.JD) - epoch0.jd('TDB', tempo.JulianDay.JD) for epoch in epoch_grid])

plt.figure(figsize=(10, 8))
plt.rcParams.update({'font.size': 15,
                     'font.family':'serif'})
plt.subplot(3, 2, 1)
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.plot(t_grid, a, label='Semi-major axis (km)')
plt.ylabel('Semi-major axis (km)')

plt.subplot(3, 2, 2)
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.plot(t_grid, e, label='Eccentricity')
plt.ylabel('Eccentricity')

plt.subplot(3, 2, 3)
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.plot(t_grid, i, label='Inclination (deg)')
plt.ylabel('Inclination (deg)')

plt.subplot(3, 2, 4)
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.plot(t_grid, Omega, label='RAAN (deg)')
plt.ylabel('RAAN (deg)')

plt.subplot(3, 2, 5)
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.plot(t_grid, omega, label='Argument of Perigee (deg)')
plt.ylabel('Argument of Perigee (deg)')

plt.tight_layout()
plt.savefig('keplerian_elements.png')