from scipy.spatial import cKDTree as KDTree
from tqdm import tqdm_notebook as tqdm
import numpy as np
import pandas as pd

structures = pd.read_csv('../input/champs-scalar-coupling/structures.csv')
train = pd.read_csv('../input/champs-scalar-coupling/train.csv')
# "usual" valence of atoms
VALENCE_MAX = {'C': 4, 'H': 1, 'N': 4, 'O': 2, 'F': 1}
VALENCE_STD = {'C': 4, 'H': 1, 'N': 3, 'O': 2, 'F': 1}

# expected distances in [A] for covalence 1 bond: https://en.wikipedia.org/wiki/Atomic_radii_of_the_elements_(data_page)
BOND_DIST_C1 = {'C': 0.77, 'H': 0.38, 'N': 0.75, 'O': 0.73, 'F': 0.71}

# order used for finding bonds by atom type
BOND_ORDER = {'H': 0, 'F': 0, 'O': 1, 'N': 2, 'C': 3}


def add_bond(n_avail, nbond, a0, a1, d1=None):
    key = tuple(sorted((a0, a1)))
    if key in nbond:
        nbond[key][0] += 1.0
    elif d1 is not None:
        nbond[key] = [1.0, d1]
    else:
        raise Exception(f"{a0},{a1} added after phase 1")
    n_avail[a0] -= 1
    n_avail[a1] -= 1


def get_bonded_atoms(atoms, nbond, i):
    """returns: [sorted atoms list], [sorted atom index] )"""
    bonded = []
    for (a0, a1), (n, _) in nbond.items():
        if a0 == i:
            bonded.append((a1, atoms[a1]))
        elif a1 == i:
            bonded.append((a0, atoms[a0]))
    bonded = sorted(bonded, key=lambda b: b[1])
    return "".join([b[1] for b in bonded]), [b[0] for b in bonded]


def search_bonds(kdt, n_avail, nbond, connected, isleaf, coords, atoms, atoms_index, a0, connect_once=True,
                 VALENCE=VALENCE_STD):
    atom0 = atoms[a0]
    if n_avail[a0] == 0:
        return

    # Select closest atoms ORDERED BY DISTANCE: closest first
    # note: the first result (closest to itself) is the atom itself and must be removed
    next_dist, next_i = kdt.query(coords[a0], min(1 + VALENCE[atom0], len(atoms)))
    next_dist = next_dist[1:]  # remove a0 from list
    next_i = next_i[1:]

    # for each #VALENCE closest atoms
    found = False  # Init found

    for d1, a1 in zip(next_dist, next_i):
        if connect_once and (a1 in connected[a0]):
            continue  # enforce 1-bond only in STEP 1 and do nothing until wrong condition

        atom1 = atoms[a1]

        # Predicted bond equals expected distances of atom0 + atom1
        predicted_bond = BOND_DIST_C1[atom0] + BOND_DIST_C1[atom1]

        if abs(d1 / predicted_bond) < 1.2:  # keep only atoms in the 20% expected distance or closer
            if n_avail[a1] > 0:  # decease remaining valence and create bond
                add_bond(n_avail, nbond, a0, a1, d1)
                connected[a0][a1] = 1
                connected[a1][a0] = 1
                if (n_avail[a0] == 0) or (
                        n_avail[a1] == 0):  # mark both atoms as leaf if one of them has zero remaining valence
                    isleaf[a0] = 1
                    isleaf[a1] = 1
                found = True

            else:
                pass

        return found


def compute_bonds(structures, molecules):
    out_name = []
    out_a0 = []
    out_a1 = []
    out_n = []
    out_dist = []
    out_error = []
    out_type = []

    cycle_name = []
    cycle_index = []
    cycle_seq = []
    cycle_atom_index = []

    charge_name = []
    charge_atom_index = []
    charge_value = []

    for imol, name in tqdm(list(enumerate(molecules))):
        molecule = structures.loc[name]
        error = 0
        atoms = molecule.atom.values
        atoms_index = molecule.atom_index.values

        n_avail = np.asarray([VALENCE_STD[a] for a in atoms])  # Array of valence standard
        n_charge = np.zeros(len(atoms), dtype=np.float16)
        isleaf = np.zeros(len(atoms), dtype=np.bool)  # is the atom in the leafs of connection tree?
        coords = molecule[['x', 'y', 'z']].values
        kdt = KDTree(coords)  # use an optimized structure for closest match query
        nbond = {}  # Empty dict
        connected = {i: {} for i in atoms_index}  # Empty dict with key = atoms_index

        # select Hydrogen first to avoid butadyne-like ordering failures (molecule_name=dsgdb9nsd_000023)
        ordered_atoms_index = list(atoms_index)
        ordered_atoms_index.sort(key=lambda i: BOND_ORDER[atoms[i]])
        ordered_atoms_index = np.asarray(ordered_atoms_index)

        # STEP 1: 1-bond connect each atom with closest match
        #         only one bond for each atom pair is done in step 1

        for a0 in ordered_atoms_index:
            search_bonds(kdt, n_avail, nbond, connected, isleaf, coords, atoms, atoms_index, a0, connect_once=True,
                         VALENCE=VALENCE_STD)

        # STEP 2: greedy connect n-bonds, progressing from leafs of connection tree
        while (((n_avail > 0).sum() > 0) and isleaf).sum() > 0:
            progress = False
            for a0 in ordered_atoms_index:  # For each atoms a0
                if (n_avail[a0] > 0) and isleaf[a0]:  # If valence of a0 > 0 and isleaf
                    for a1 in connected[a0]:  # For each connected atoms a1 in array of a0
                        if (n_avail[a0] > 0) and (n_avail[a1] > 0):  # if both a0 a1 has valence > 0
                            # For each connected neighbour, add as many bonds as possible
                            add_bond(n_avail, nbond, a0, a1)
                            progress = True
                            if (n_avail[a0] == 0) or (n_avail[a1] == 0):  # If a0 or a1 has valence = 0
                                # Mark both connected atoms as leaf
                                isleaf[a0] = 1
                                isleaf[a1] = 1

            if not progress:
                break

            # gather remaining multiple bonds
            if n_avail.sum() > 0:
                for key in nbond.keys():
                    a0, a1 = key
                    while (n_avail[a0] > 0) and (n_avail[a1] > 0):
                        add_bond(n_avail, nbond, a0, a1)

            # STEP 3: search for known ionized radicals
            if n_avail.sum() > 0:
                for i, a in zip(atoms_index, atoms):
                    if a == 'N':  # If atom is Nitro
                        # NH3+
                        bonded_str, bonded_index = get_bonded_atoms(atoms, nbond, i)
                        if (bonded_str == 'HHH') and (n_avail[i] == 0):
                            # Add a valence unit and search a dangling bond nearby
                            n_avail[i] += 1
                            n_charge[i] += 1
                            if search_bonds(kdt, n_avail, nbond, connected, isleaf, coords, atoms, atoms_index, i,
                                            connect_once=False, VALENCE=VALENCE_MAX):
                                print(f"++ NH3+ found for {name} atom_index={i}")
                            else:
                                print(f"** NH3+ bonding failure for {name} atom_index={i}")
                        elif a == 'O' and n_avail[i] == 1:  # O with one available bond connected to CO
                            # COO-
                            bonded_str, bonded_index = get_bonded_atoms(atoms, nbond, i)
                            if bonded_str == "C":
                                C_i = bonded_index[0]
                                C_bonded_str, C_bonded_index = get_bonded_atoms(atoms, nbond, C_i)
                                if "OO" in C_bonded_str:
                                    has_2CO = False

                                    for a1, i1 in zip(C_bonded_str, C_bonded_index):
                                        key = tuple(sorted((C_i, i1)))
                                        if (a1 == 'O') and (nbond[key][0] == 2):
                                            has_2CO = True

                                    if (len(C_bonded_index) == 3) and has_2CO:  # If the number of bonded atoms is 3
                                        # found carboxyle COOH
                                        n_avail[i] -= 1
                                        print(f"** COO- found for {name} C_atom_index={C_i}")
                                        for a1, i1 in zip(C_bonded_str, C_bonded_index):
                                            if a1 == 'O':
                                                n_charge[i1] = -0.5
                                                key = tuple(sorted((C_i, i1)))
                                                nbond[key][0] = 1.5
        # Create DataFrame (Excel) with column
        bonds = pd.DataFrame({'molecule_name': out_name, 'atom_index_0': out_a0, 'atom_index_1': out_a1, 'nbond': out_n,
                              'L2dist': out_dist, 'error': out_error, 'bond_type': out_type})
        charges = pd.DataFrame({'molecule_name': charge_name, 'atom_index': charge_atom_index,
                                'charge': charge_value})

        # inputs for DataFrame bond
        for (a0, a1), (n, dist) in nbond.items():
            out_name.append(name)
            out_a0.append(a0)
            out_a1.append(a1)
            out_n.append(n)
            out_dist.append(dist)
            out_error.append(error)
            out_type.append(f"{n:0.1f}" + "".join(sorted(f"{atoms[a0]}{atoms[a1]}")))

        return bonds, charges


train_bonds, train_charges = compute_bonds(structures.set_index('molecule_name'), train.molecule_name.unique())

train_bonds.to_csv('train_bonds.csv', index=False)
train_charges.to_csv('train_charges.csv', index=False)
