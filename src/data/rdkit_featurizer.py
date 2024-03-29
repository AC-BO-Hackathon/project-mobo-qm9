from pathlib import Path
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import pandas as pd

df = pd.read_csv('../../src/data/target.csv')

def get_ecfp_molecule(index):
    """
    Gets the Morgan Fingerprint of a molecule using SMILES
    
    args:
        indices: Indices for the MOBOQM9 model.
    
    returns:
        int: Morgan Fingerprint of a molecule using SMILES
    """
    
    content = Path(f'../../../moboqm9/dataxyz/dsgdb9nsd_{index+1:06d}.xyz').read_text().split("\n")
    smiles = content[-3].split('\t')[0]
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetHashedMorganFingerprint(mol, 3, nBits=1024)
    array = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp,array)
        
    return array

def get_ecfp_matrix(indices,targets):
    """
    Gets the Morgan Fingerprint matrix using SMILES
    
    args:
        indices: Indices for the MOBOQM9 model.
    
    returns:
        int: Morgan Fingerprint matrix using SMILES
    """
    ecfp_matrix= np.zeros((0,1024), dtype=np.int8)
    samples_targets = np.zeros((0,len(targets)))
    for i in indices:
        ecfp_vector=get_ecfp_molecule(i)
        ecfp_matrix=np.vstack((ecfp_matrix,ecfp_vector))
        samples_targets=np.vstack((samples_targets,np.array(df[targets].iloc[i])))
        
    return ecfp_matrix,samples_targets