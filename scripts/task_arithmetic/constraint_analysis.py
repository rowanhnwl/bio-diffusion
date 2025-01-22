from rdkit import Chem
from rdkit.Chem import Descriptors

from rdkit.Chem.PandasTools import LoadSDF
from rdkit.Chem import rdMolDescriptors

import numpy as np

import argparse
import os
import json

def eval_mol_weight(
    smiles: list,
    threshold: float,
    bound_type: str
):
    mws = []
    good_mws_count = 0
    valid_smiles = 0

    # Calculate the molecular weight
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)

        if mol:
            mw = Descriptors.MolWt(mol)
            valid_smiles += 1
        else:
            print(f"Invalid SMILES string: {smi}")
            continue

        if bound_type == "upper" and mw <= threshold:
            good_mws_count += 1
        elif bound_type == "lower" and mw >= threshold:
            good_mws_count += 1

        mws.append(mw)

    pass_rate = good_mws_count / len(mws)

    return np.mean(mws), np.std(mws), valid_smiles, pass_rate

def eval_tpsa(
    smiles: list,
    threshold: float,
    bound_type: str
):
    tpsas = []
    good_tpsas_count = 0
    valid_smiles = 0

    # Calculate the TPSAs
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)

        if mol:
            tpsa = rdMolDescriptors.CalcTPSA(mol)
            valid_smiles += 1
        else:
            print(f"Invalid SMILES string: {smi}")
            continue

        if bound_type == "upper" and tpsa <= threshold:
            good_tpsas_count += 1
        elif bound_type == "lower" and tpsa >= threshold:
            good_tpsas_count += 1

        tpsas.append(tpsa)

    pass_rate = good_tpsas_count / len(tpsas)

    return np.mean(tpsas), np.std(tpsas), valid_smiles, pass_rate


def eval_xlogp(
    smiles: list,
    threshold: float,
    bound_type: str
):
    xlogps = []
    good_xlogps_count = 0
    valid_smiles = 0

    # Calculate the XLogPs
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)

        if mol:
            xlogp, mr = rdMolDescriptors.CalcCrippenDescriptors(mol)
            valid_smiles += 1
        else:
            print(f"Invalid SMILES string: {smi}")
            continue

        if bound_type == "upper" and xlogp <= threshold:
            good_xlogps_count += 1
        elif bound_type == "lower" and xlogp >= threshold:
            good_xlogps_count += 1

        xlogps.append(xlogp)

    pass_rate = good_xlogps_count / len(xlogps)

    return np.mean(xlogps), np.std(xlogps), valid_smiles, pass_rate

def eval_rbc(
    smiles: list,
    threshold: float,
    bound_type: str
):
    rbcs = []
    good_rbcs_count = 0
    valid_smiles = 0

    # Calculate the RBCs
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)

        if mol:
            rbc = rdMolDescriptors.CalcNumRotatableBonds(mol)
            valid_smiles += 1
        else:
            print(f"Invalid SMILES string: {smi}")
            continue

        if bound_type == "upper" and rbc <= threshold:
            good_rbcs_count += 1
        elif bound_type == "lower" and rbc >= threshold:
            good_rbcs_count += 1

        rbcs.append(rbc)

    pass_rate = good_rbcs_count / len(rbcs)

    return np.mean(rbcs), np.std(rbcs), valid_smiles, pass_rate

# Function to demultiplex the constraint
def constraint_eval(
    constraints: list,
    threshold_info_list: list,
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
            smiles = []
    
        for i, constraint in enumerate(constraints):
            mean, std, valid_smiles, pass_rate = 0, 0, 0, 0

            threshold_info = threshold_info_list[i]

            threshold = threshold_info["threshold"]
            bound_type = threshold_info["bound"] # Upper or lower

            if len(smiles) == 0:
                pass
            elif constraint == "Molecular Weight":
                mean, std, valid_smiles, pass_rate = eval_mol_weight(smiles, threshold, bound_type)
            elif constraint == "TPSA":
                mean, std, valid_smiles, pass_rate = eval_tpsa(smiles, threshold, bound_type)
            elif constraint == "XLogP":
                mean, std, valid_smiles, pass_rate = eval_xlogp(smiles, threshold, bound_type)
            elif constraint == "Rotatable Bond Count":
                mean, std, valid_smiles, pass_rate = eval_rbc(smiles, threshold, bound_type)

            out_dict[dir][constraint] = {
                "mean": mean,
                "std": std,
                "n_valid": valid_smiles,
                "pass_rate": pass_rate
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

    threshold_path = "src/models/components/json/thresholds.json"
    with open(threshold_path, "r") as f:
        thresholds = json.load(f)
    threshold_info = [thresholds[c] for c in constraints]

    constraint_eval(
        constraints,
        threshold_info,
        sdf_dir_path,
        out_path
    )