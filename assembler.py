import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import os


SQRT3 = np.sqrt(3.0)
OUT_DIR = 'figures'

# ============================================================
# Small-hex axial grid
# ============================================================

AXIAL_DIRS = [
    (1, 0),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (0, -1),
    (1, -1),
]


def axial_add(a, b):
    return a[0] + b[0], a[1] + b[1]


def axial_to_xy(q, r, edge):
    """
    Flat-top regular hexagonal grid.

    q, r are axial coordinates of the SMALL hexagons.
    """
    x = 1.5 * edge * q
    y = SQRT3 * edge * (r + 0.5 * q)
    return np.array([x, y], dtype=float)


def hex_vertices(center_xy, edge):
    """
    Vertices of a flat-top regular hexagon.
    """
    cx, cy = center_xy

    angles = np.deg2rad([0, 60, 120, 180, 240, 300])

    return np.column_stack([
        cx + edge * np.cos(angles),
        cy + edge * np.sin(angles),
    ])


# ============================================================
# Correct atom-center lattice
# ============================================================

def macro_to_small_axial(Q, R):
    """
    Convert macro atom-lattice coordinates to small-hex axial coordinates.

    This is the key correction.

    The basis vectors

        u = (2, 1)
        v = (-1, 3)

    generate a determinant-7 sub-lattice of the small-hex grid.
    Therefore each atom, made of 7 small hexagons, tiles the plane
    without overlap and without internal empty sites.

    Parameters
    ----------
    Q, R : int
        Macro axial coordinates of the atom.

    Returns
    -------
    q, r : int
        Small-hex axial coordinates of the atom center.
    """
    q = 2 * Q - R
    r = Q + 3 * R
    return q, r


def atom_hexes_from_macro_center(Q, R):
    """
    Return the 7 small hexagons composing one atom.

    The atom is one central hexagon plus its six direct small-grid neighbors.
    """
    cq, cr = macro_to_small_axial(Q, R)

    local_hexes = [
        (0, 0),
        (1, 0),
        (0, 1),
        (-1, 1),
        (-1, 0),
        (0, -1),
        (1, -1),
    ]

    return [(cq + dq, cr + dr) for dq, dr in local_hexes]


# ============================================================
# Radial macro-lattice filling
# ============================================================

def generate_atom_centers_macro(n_atoms):
    """
    Generate atom centers on the macro atom lattice, filled radially outwards.

    The sequence is:
        radius 0: 1 atom
        radius 1: 6 atoms
        radius 2: 12 atoms
        radius 3: 18 atoms
        ...

    Therefore:
        N = 1  -> only the center
        N = 5  -> center + first 4 positions of ring 1
        N = 7  -> center + complete ring 1
        N = 8  -> center + ring 1 + first atom of ring 2

    This version does NOT revisit the central atom.
    """
    if n_atoms <= 0:
        raise ValueError("n_atoms must be positive.")

    centers = [(0, 0)]

    if n_atoms == 1:
        return centers

    # Macro axial directions.
    # These must be consistent with the macro_to_small_axial() basis.
    macro_dirs = [
        (1, 0),
        (0, 1),
        (-1, 1),
        (-1, 0),
        (0, -1),
        (1, -1),
    ]

    radius = 1

    while len(centers) < n_atoms:

        # Start at one corner of the hexagonal ring.
        # Using direction 4 gives the lower corner of the ring.
        Q = macro_dirs[4][0] * radius
        R = macro_dirs[4][1] * radius

        # Walk around the ring, side by side.
        for dQ, dR in macro_dirs:
            for _ in range(radius):
                if len(centers) >= n_atoms:
                    return centers

                centers.append((Q, R))

                Q += dQ
                R += dR

        radius += 1

    return centers



def build_structure(edge, n_atoms):
    """
    Build the molecule as a union of small hexagons.

    Returns
    -------
    small_hex_axials : list of tuple[int, int]
        Small-hex axial coordinates of all hexagons.
    atom_centers_macro : list of tuple[int, int]
        Macro atom-lattice coordinates.
    atom_centers_small : list of tuple[int, int]
        Corresponding small-hex axial coordinates of atom centers.
    """
    atom_centers_macro = generate_atom_centers_macro(n_atoms)

    atom_centers_small = [
        macro_to_small_axial(Q, R)
        for Q, R in atom_centers_macro
    ]

    small_hex_set = set()

    for Q, R in atom_centers_macro:
        for h in atom_hexes_from_macro_center(Q, R):
            small_hex_set.add(h)

    small_hex_axials = sorted(list(small_hex_set))

    expected = 7 * n_atoms
    actual = len(small_hex_axials)

    if actual != expected:
        raise RuntimeError(
            f"Overlap detected: expected {expected} small hexagons, "
            f"but found {actual} unique hexagons."
        )

    return small_hex_axials, atom_centers_macro, atom_centers_small


# ============================================================
# Mass properties of one small hexagonal prism
# ============================================================

def single_hex_mass_properties(edge, thickness, mass_hex):
    """
    Mass properties of one regular hexagonal prism about its own centroid.

    The prism lies in the x-y plane.
    Thickness is along z.
    """
    a = edge
    t = thickness
    m = mass_hex

    area_hex = 3.0 * SQRT3 / 2.0 * a**2
    volume_hex = area_hex * t

    # Regular hexagonal lamina:
    #
    # Izz = m * 5/12 * a^2
    # Ixx = Iyy = m * 5/24 * a^2
    #
    # For a prism of thickness t:
    # add m * t^2 / 12 to Ixx and Iyy.
    Ixx = m * (5.0 / 24.0 * a**2 + t**2 / 12.0)
    Iyy = Ixx
    Izz = m * (5.0 / 12.0 * a**2)

    I_centroid = np.diag([Ixx, Iyy, Izz])

    return area_hex, volume_hex, I_centroid


# ============================================================
# Total mass properties
# ============================================================

def compute_mass_properties(edge, thickness, mass_hex, n_atoms):
    """
    Assemble the atoms and compute total mass properties.

    Parameters
    ----------
    edge : float
        Edge length of one small hexagon.
    thickness : float
        Thickness along the positive/negative z direction.
    mass_hex : float
        Mass of one small hexagonal prism.
    n_atoms : int
        Number of 7-hex atoms.

    Returns
    -------
    result : dict
        Contains:
        - total mass
        - centroid
        - projected area
        - external perimeter
        - external 3D surface area
        - inertia matrix about origin
        - inertia matrix about centroid
    """
    if edge <= 0:
        raise ValueError("edge must be positive.")

    if thickness <= 0:
        raise ValueError("thickness must be positive.")

    if mass_hex <= 0:
        raise ValueError("mass_hex must be positive.")

    if n_atoms <= 0:
        raise ValueError("n_atoms must be positive.")

    small_hex_axials, atom_centers_macro, atom_centers_small = build_structure(
        edge=edge,
        n_atoms=n_atoms,
    )

    n_hex = len(small_hex_axials)
    total_mass = n_hex * mass_hex

    area_hex, volume_hex, I_hex_centroid = single_hex_mass_properties(
        edge=edge,
        thickness=thickness,
        mass_hex=mass_hex,
    )

    centers_xy = np.array([
        axial_to_xy(q, r, edge)
        for q, r in small_hex_axials
    ])

    centers_xyz = np.column_stack([
        centers_xy,
        np.zeros(n_hex),
    ])

    # Equal mass per small hexagon.
    centroid = np.mean(centers_xyz, axis=0)

    projected_area_xy = n_hex * area_hex

    # Count exposed edges for the outer perimeter.
    small_hex_set = set(small_hex_axials)

    exposed_edges = 0

    for h in small_hex_axials:
        for d in AXIAL_DIRS:
            neighbor = axial_add(h, d)

            if neighbor not in small_hex_set:
                exposed_edges += 1

    external_perimeter = exposed_edges * edge

    # Full exposed surface:
    #
    # top + bottom + lateral exposed walls
    external_surface_area = (
        2.0 * projected_area_xy
        + external_perimeter * thickness
    )

    # Inertia about body-frame origin.
    I_origin = np.zeros((3, 3))

    for r_vec in centers_xyz:
        d2 = np.dot(r_vec, r_vec)
        I_origin += (
            I_hex_centroid
            + mass_hex * (d2 * np.eye(3) - np.outer(r_vec, r_vec))
        )

    # Inertia about total centroid.
    I_centroid_total = np.zeros((3, 3))

    for r_vec in centers_xyz:
        rel = r_vec - centroid
        d2 = np.dot(rel, rel)

        I_centroid_total += (
            I_hex_centroid
            + mass_hex * (d2 * np.eye(3) - np.outer(rel, rel))
        )

    return {
        "n_atoms": n_atoms,
        "n_hexagons": n_hex,
        "total_mass": total_mass,
        "centroid": centroid,
        "projected_area_xy": projected_area_xy,
        "external_perimeter": external_perimeter,
        "external_surface_area": external_surface_area,
        "inertia_about_origin": I_origin,
        "inertia_about_centroid": I_centroid_total,
        "small_hex_axials": small_hex_axials,
        "atom_centers_macro": atom_centers_macro,
        "atom_centers_small": atom_centers_small,
        "small_hex_centers_xy": centers_xy,
    }


# ============================================================
# Plotting helper
# ============================================================

def plot_structure(
    edge,
    n_atoms,
    thickness=1.0,
    mass_hex=1.0,
    ax=None,
    show_centroid=True,
    show_atom_centers=True,
    show_atom_labels=False,
):
    """
    Plot the whole assembled molecule in the body-frame x-y plane.

    The structure lies in the x-y plane.
    The normal direction is positive z.
    """
    props = compute_mass_properties(
        edge=edge,
        thickness=thickness,
        mass_hex=mass_hex,
        n_atoms=n_atoms,
    )

    small_hex_axials = props["small_hex_axials"]
    atom_centers_small = props["atom_centers_small"]
    centroid = props["centroid"]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    # Plot all small hexagons.
    for q, r in small_hex_axials:
        center = axial_to_xy(q, r, edge)
        vertices = hex_vertices(center, edge)

        patch = Polygon(
            vertices,
            closed=True,
            edgecolor="black",
            facecolor="#d9e8ff",
            linewidth=1.0,
        )

        ax.add_patch(patch)

    # Plot atom centers.
    if show_atom_centers:
        atom_centers_xy = np.array([
            axial_to_xy(q, r, edge)
            for q, r in atom_centers_small
        ])

        ax.scatter(
            atom_centers_xy[:, 0],
            atom_centers_xy[:, 1],
            s=45,
            c="tab:blue",
            label="Atom centers",
            zorder=5,
        )

        if show_atom_labels:
            for i, xy in enumerate(atom_centers_xy):
                ax.text(
                    xy[0],
                    xy[1],
                    str(i),
                    ha="center",
                    va="center",
                    color="white",
                    zorder=6,
                )

    # Plot geometric centroid.
    if show_centroid:
        ax.scatter(
            centroid[0],
            centroid[1],
            s=120,
            c="tab:red",
            marker="x",
            linewidths=2.5,
            label="Geometric centroid",
            zorder=10,
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Body-frame x")
    ax.set_ylabel("Body-frame y")
    ax.set_title(f"Assembly of {n_atoms} atom") if n_atoms == 1 else ax.set_title(f"Assembly of {n_atoms} atoms")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    return fig, ax, props


# ============================================================
# Example
# ============================================================

if __name__ == "__main__":
    edge = 1.65
    thickness = 0.4
    mass_hex = 1250
    n_atoms = 1

    props = compute_mass_properties(
        edge=edge,
        thickness=thickness,
        mass_hex=mass_hex,
        n_atoms=n_atoms,
    )

    plt.rcParams.update({'font.size': 15,
                         'font.family':'serif'})

    print("Number of atoms:", props["n_atoms"])
    print("Number of small hexagons:", props["n_hexagons"])
    print("Total mass:", props["total_mass"])

    print("\nCentroid [x, y, z]:")
    print(props["centroid"])

    print("\nProjected area in x-y:")
    print(props["projected_area_xy"])

    print("\nExternal perimeter:")
    print(props["external_perimeter"])

    print("\nExternal 3D surface area:")
    print(props["external_surface_area"])

    print("\nInertia about origin:")
    print(props["inertia_about_origin"])

    print("\nInertia about centroid:")
    print(props["inertia_about_centroid"])

    plot_structure(
        edge=edge,
        thickness=thickness,
        mass_hex=mass_hex,
        n_atoms=n_atoms,
        show_atom_labels=True,
    )

    plt.savefig(os.path.join(OUT_DIR, f"assembly_{n_atoms}_atoms.svg"), dpi=300)