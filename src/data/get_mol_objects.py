from rdkit import Chem
from pathlib import Path

TARGET_MAP = {
    "RC_A" : 1,
    "RC_B" : 2,
    "RC_C" : 3,
    "mu": 4,
    "alpha": 5,
    "homo": 6,
    "lumo": 7,
    "gap": 8,
    "r2": 9,
    "zpve": 10,
    "U_0" : 11,
    "U_298" : 12,
    "H_enthalpy" : 13,
    "G_free_energy" : 14,
    "Cv": 15
    }

def get_mol_data(idx, targets):
    """
    Gets the molecular data for the MOBOQM9 model.
    
    args:
        idx: Index of the data.
    
    returns:
        rdkit mol object and targets.
    """
    content = Path(f"raw_data/dsgdb9nsd_{idx:06d}.xyz").read_text().split("\n")
    props = content[1].split("\t")
    target_list = [float(props[TARGET_MAP[target]]) for target in targets]
    temp_file = Path("data.xyz")
    content = content[:-4]
    for i in range(2, len(content)):
        content[i] = "\t".join(content[i].split("\t")[:-1])
    temp_file.write_text("\n".join(content))
    mol = Chem.MolFromXYZFile(str(temp_file))
    temp_file.unlink()
    return mol, target_list


if __name__ == "__main__":
    print(get_mol_data(199, ["mu", "alpha"]))