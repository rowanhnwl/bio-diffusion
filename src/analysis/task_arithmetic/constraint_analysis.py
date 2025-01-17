from rdkit import Chem
from rdkit.Chem import Descriptors

from rdkit.Chem.PandasTools import LoadSDF

import os

def eval_mol_weight(
    threshold: float,
    smiles: list
):
    pass_mols = []
    mws = []

    # Calculate the molecular weight
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)

        if mol:
            mw = Descriptors.MolWt(mol)
        else:
            print(f"Invalid SMILES string: {smi}")
            continue

        if mw >= threshold:
            pass_mols.append(smi)

        mws.append(mw)

    pass_rate = len(pass_mols) / len(smiles)

    print(f"Molecular weight pass rate: {pass_rate} ({len(pass_mols)} / {len(smiles)})")
    print(f"Average molecular weight: {sum(mws) / len(mws)}")

# Function to demultiplex the constraint
def constraint_eval(
    constraint: str,
    threshold: float,
    sdf_path: str,
):
    # Load the molecules
    sdf_path = os.path.join(sdf_path, os.listdir(sdf_path)[0])
    df = LoadSDF(sdf_path, smilesName='SMILES')

    try:
        smiles = df["SMILES"]
    except:
        print("No available SMILES strings")
    
    #if "Molecular Weight" in constraint:
    eval_mol_weight(threshold, smiles)