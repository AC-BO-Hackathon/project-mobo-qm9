from dscribe.descriptors import SOAP
from ase import db
from pathlib import Path
import numpy as np

def get_species(indices):
    """
    Get the species for the MOBOQM9 model.
    
    args:
        indices: Indices for the MOBOQM9 model.
    
    returns:
        int: Maximum number of atoms for the MOBOQM9 model.
    """
    species = []
    with db.connect(str(Path(__file__).parent / "QM9_data.db")) as qm9:
        for i, row in enumerate(qm9.select()):
            if i in indices:
                atoms = row.toatoms()
                species = list(set(atoms.get_chemical_symbols() + species))
    return species

def get_soap(indices, targets):
    """
    Gets the SOAP fingerprints for the MOBOQM9 model.
    
    args:
        indices: Indices for the MOBOQM9 model.
        targets: Targets for the MOBOQM9 model.
    
    returns:
        features: Features for the MOBOQM9 model.
        targets: Targets for the MOBOQM9 model.
    """
    species = get_species(indices)
    soap = SOAP(
        species=species,
        r_cut = 3.0,
        n_max = 4,
        l_max = 3,
        periodic=False,
        sparse=False,
        average="inner",
        rbf="gto",
    )
    
    atoms_list, computed_targets = [], []
    with db.connect(str(Path(__file__).parent / "QM9_data.db")) as qm9:
        for i, row in enumerate(qm9.select()):
            if i in indices:
                is_OK = True
                for target in targets:
                    try:
                        row[target]
                    except AttributeError:
                        is_OK = False
                        break
                if is_OK:
                    atoms_list.append(row.toatoms())
                    target_list = []
                    for target in targets:
                        target_list.append(row[target])
                    computed_targets.append(target_list)
    features = soap.create(atoms_list, n_jobs=4)
    
    return features, np.array(computed_targets)