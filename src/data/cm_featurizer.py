from dscribe.descriptors import CoulombMatrix
from ase import db


def get_max_number_of_atoms(indices):
    """
    Gets the maximum number of atoms for the MOBOQM9 model.
    
    args:
        indices: Indices for the MOBOQM9 model.
    
    returns:
        int: Maximum number of atoms for the MOBOQM9 model.
    """
    max_number_of_atoms = 0
    with db.connect("QM9_data.db") as qm9:
        for i, row in enumerate(qm9.select()):
            if i in indices:
                atoms = row.toatoms()
                max_number_of_atoms = max(max_number_of_atoms,
                                          atoms.get_number_of_atoms())
    return max_number_of_atoms

def get_coulomb_matrix(indices, targets):
    """
    Gets the coulomb matrix for the MOBOQM9 model.
    
    args:
        indices: Indices for the MOBOQM9 model.
        targets: Targets for the MOBOQM9 model.
    
    returns:
        features: Features for the MOBOQM9 model.
        targets: Targets for the MOBOQM9 model.
    """
    max_number_of_atoms = get_max_number_of_atoms(indices)
    cm = CoulombMatrix(n_atoms_max=max_number_of_atoms)
    
    atoms_list, targets = [], []
    with db.connect("QM9_data.db") as qm9:
        for i, row in enumerate(qm9.select()):
            if i in indices:
                atoms_list.append(row.toatoms())
                target_list = []
                for target in targets:
                    target_list.append(row[target])
                targets.append(target_list)
    features = cm.create(atoms_list, n_jobs=4)
    
    return features, targets