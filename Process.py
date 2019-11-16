import ase.visualize
import matplotlib.pyplot as plt
import pandas as pd
from ase import Atoms

structures = pd.read_csv('input/champs-scalar-coupling/structures.csv')
train = pd.read_csv('input/champs-scalar-coupling/train.csv')

cpk = {
    'C': ("black", 2),
    'H': ("white", 1),
    'O': ("red", 2),
    'N': ("dodgerblue", 2),
    'F': ("green", 2)
}

bond_colors = {'1.0': 'black', '1.5': 'darkgreen', '2.0': 'green', '3.0': 'red'}


def bond_type_to_pair(bond_type):
    # Input bond type and return last 3
    return bond_type[3:]


def bond_type_to_n(bond_type):
    return bond_type[0:3]


def view3d_molecule(name, xsize="200px", ysize="200px"):
    # Mouse clickeble 3D view
    m = structures[structures.molecule_name == name]
    positions = m[['x', 'y', 'z']].values
    v = ase.visualize.view(Atoms(positions=positions, symbols=m.atom.values),
                           viewer="x3d")
    return v


def plot_molecule(name, ax=None, bonds=None, charges=None, elev=0, azim=-60):
    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    # Get info of atoms with input name and sort by index
    m = structures[structures.molecule == name].sort_values(by='atom_index')

    if charges is not None:
        # Get info of charges with input name and sort by index and compare if it is the same as m
        charges = charges[charges.molecule_name == name].sort_values(by='atom_index')
        if len(charges) != len(m):
            print(f"Warning bad charges data for molecule {name}")

    # Create molecule formula (C2H6, C2H5OH...)
    a_count = {a: 0 for a in cpk}
    formula = ""
    for a in m.atom:
        a_count[a] += 1
    for a in a_count:
        if a_count[a] == 1:
            # If atom has only 1 unit, just add the character
            formula += a
        elif a_count[a] > 1:
            formula += "%s_{%d}" % (a, a_count[a])
    print(formula)

    # Display coupling
    couples = train[train.molecule_name == name][['atom_index_0', 'atom_index_1', 'type', 'scalar_coupling_constant']]
    for c in couples.itertuples():
        m0 = m[m.atom_index == c.atom_index_0]  # Get the first atom from coupling train file
        m1 = m[m.atom_index == c.atom_index_1]  # Get the second atom from coupling train file
        ax.plot([float(m0.x), float(m1.x)], [float(m0.y), float(m1.y)], [float(m0.z), float(m1.z)],
                linestyle=['', '-', '--', 'dotted'][int(c.type[0])],  # Type 1 2 3 4 consequently
                color=['', 'black', 'green', 'red'][int(c.type[0])],
                linewidth=abs(float(c.scalar_coupling_constant)) / 5, alpha=0.2),

    if bonds is not None:
        #