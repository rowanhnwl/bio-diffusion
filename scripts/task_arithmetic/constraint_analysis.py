from rdkit import Chem
from rdkit.Chem import Descriptors

from rdkit.Chem.PandasTools import LoadSDF
from rdkit.Chem import rdMolDescriptors

import numpy as np

import argparse
import os
import json

def eval_mol_weight(
    smiles: list
):
    mws = []

    # Calculate the molecular weight
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)

        if mol:
            mw = Descriptors.MolWt(mol)
        else:
            print(f"Invalid SMILES string: {smi}")
            continue

        mws.append(mw)

    return np.mean(mws), np.std(mws)

def eval_tpsa(
    smiles: list
):
    tpsas = []

    # Calculate the TPSAs
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)

        if mol:
            tpsa = rdMolDescriptors.CalcTPSA(mol)
        else:
            print(f"Invalid SMILES string: {smi}")
            continue

        tpsas.append(tpsa)

    return np.mean(tpsas), np.std(tpsas)


# Function to demultiplex the constraint
def constraint_eval(
    constraints: list,
    sdf_dir_path: str,
    out_path: str
):
    
    sdf_dirs = os.listdir(sdf_dir_path)

    out_dict = {}

    for dir in sdf_dirs:
        full_sdf_path = os.path.join(sdf_dir_path, dir)
        sdf_fn = os.listdir(full_sdf_path)[0]
        full_sdf_path = os.path.join(full_sdf_path, sdf_fn)

        out_dict[dir] = {}

        df = LoadSDF(full_sdf_path, smilesName='SMILES')

        try:
            smiles = df["SMILES"]
        except:
            print("No available SMILES strings")
    
        for constraint in constraints:
            mean, std = 0, 0

            if constraint == "Molecular Weight":
                mean, std = eval_mol_weight(smiles)
            elif constraint == "TPSA":
                mean, std = eval_tpsa(smiles)

            out_dict[dir][constraint] = {
                "mean": mean,
                "std": std
            }

    with open(out_path, "w") as f:
        json.dump(out_dict, f, indent=3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config_path = args.config

    with open(config_path, "r") as cf:
        config = json.load(cf)

    constraints = config["eval_constraints"]
    sdf_dir_path = config["sdf_dir_path"]
    out_path = config["out_path"]

    constraint_eval(
        constraints,
        sdf_dir_path,
        out_path
    )