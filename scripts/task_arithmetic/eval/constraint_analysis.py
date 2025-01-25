from rdkit import Chem
from rdkit.Chem import Descriptors

from rdkit.Chem.PandasTools import LoadSDF
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs

import numpy as np

import argparse
import os
import json

from tqdm import tqdm
from heapq import nlargest

from multiprocessing import Pool, Manager
manager = Manager()

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

        if bound_type == "upper" and tpsa <= threshold and tpsa > 0:
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

# Structure-based evaluation
def tdc_evaluation(
    args
):
    (
        list_of_gen_mols,
        meet_dataset_mols,
        fail_dataset_mols,
        shared_met_list,
        shared_fail_list,
    ) = args

    fpgen = AllChem.GetRDKitFPGenerator()
    
    # Iterate through the generated molecules
    for smiles in list_of_gen_mols:
        gen_mol = Chem.MolFromSmiles(smiles)

        if gen_mol:
    
            met_sim = []
            fail_sim = []
    
            gen_fp = fpgen.GetFingerprint(gen_mol)
            
            # Iterate through the dataset molecules that met the constraint
            for met_smiles in meet_dataset_mols:
            
                met_mol = Chem.MolFromSmiles(met_smiles)
                if met_mol:

                    # Get the similarity score between the generated and dataset molecule
                    met_fp = fpgen.GetFingerprint(met_mol)
                    sim = DataStructs.TanimotoSimilarity(gen_fp,met_fp)            
                    met_sim.append(sim)
            
            # Iterate through the dataset molecules that did not meet the constraint
            for fail_smiles in fail_dataset_mols:
            
                fail_mol = Chem.MolFromSmiles(fail_smiles)
                if fail_mol:

                    # Get the similarity score between the generated and dataset molecule
                    fail_fp = fpgen.GetFingerprint(fail_mol)
                    sim = DataStructs.TanimotoSimilarity(gen_fp,fail_fp)            
                    fail_sim.append(sim)
            
            # Append the list of similarity scores to each list
            shared_met_list.append(met_sim)            
            shared_fail_list.append(fail_sim)

def split_smiles(
    smiles: list,
    n_processes: int
):

    sublists = [[] for _ in range(n_processes)]
    
    # Distribute the SMILES strings across the sublists
    for i, smi in enumerate(smiles):
        sublists[i % n_processes].append(smi)
    
    return sublists

def get_structure_eval(
    smiles: list,
    mols_that_meet_threshold: list,
    mols_that_fail_threshold: list
):
    n_cpu = os.cpu_count()

    # Split the SMILES strings into separate lists for each process
    smiles_lists = split_smiles(
        smiles,
        n_cpu
    )

    gen_mol_met_sim = manager.list()
    gen_mol_fail_sim = manager.list()

    with Pool(n_cpu) as p:
        p.map(
            tdc_evaluation,
            [
                (
                    smiles_list,
                    mols_that_meet_threshold,
                    mols_that_fail_threshold,
                    gen_mol_met_sim, # SHARED
                    gen_mol_fail_sim, # SHARED
                )
                for smiles_list in smiles_lists
            ]
        )

    # Get the rates at which the generated-real pairs have a similarity over a given threshold
    gen_mol_met_sim_vec = np.array(list(gen_mol_met_sim)).flatten()
    gen_mol_fail_sim_vec = np.array(list(gen_mol_fail_sim)).flatten()

    sim_threshold = 0.2

    high_sim_fail_count = 0
    high_sim_pass_count = 0

    for sim_score in gen_mol_met_sim_vec:
        if sim_score >= sim_threshold:
            high_sim_pass_count += 1

    for sim_score in gen_mol_fail_sim_vec:
        if sim_score >= sim_threshold:
            high_sim_fail_count += 1

    # Get the number of generated molecules that have AT LEAST ONE pairing with a similarity score over the threshold
    n_good = 0
    for mol_sim_scores in list(gen_mol_met_sim):
        for sim_score in mol_sim_scores:
            if sim_score >= sim_threshold:
                n_good += 1
                break

    return high_sim_pass_count / len(gen_mol_met_sim_vec), high_sim_fail_count / len(gen_mol_fail_sim_vec), n_good

def eval_dataset(
    constraint: str,
    bound: str,
    threshold: float,
    datasets_dir: str
):
    # Get the dataset file name
    dataset_names = os.listdir(datasets_dir)
    dataset_filename = ""
    for dname in dataset_names:
        if constraint in dname:
            dataset_filename = dname

    assert dataset_filename != "", "No corresponding dataset found"

    dataset_filepath = os.path.join(datasets_dir, dataset_filename)
    with open(dataset_filepath, "r") as f:
        dataset_dict = json.load(f)

    mols_that_meet_threshold = []
    mols_that_fail_threshold = []
    
    # Go through the dataset, and separate the molecules based on whether or not they meet the constraint
    if bound == 'upper':
        for mol in dataset_dict.keys():
            if dataset_dict[mol] <= threshold:
                mols_that_meet_threshold.append(mol)
            else:
                mols_that_fail_threshold.append(mol)
    elif bound == 'lower':
        for mol in dataset_dict.keys():
            if dataset_dict[mol] >= threshold:
                mols_that_meet_threshold.append(mol)
            else:
                mols_that_fail_threshold.append(mol)
    else:
        raise ValueError('Need to specify upper or lower bound')
    
    return mols_that_meet_threshold, mols_that_fail_threshold

# Function to demultiplex the constraint
def constraint_eval(
    constraints: list,
    threshold_info_list: list,
    sdf_dir_path: str,
    datasets_dir: str,
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

            calc = True

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
            else: # NEED TO USE STRUCTURAL SIMILARITY

                # Split the dataset by passing/failing the threshold
                mols_that_meet_threshold, mols_that_fail_threshold = eval_dataset(
                    constraint,
                    bound_type,
                    threshold,
                    datasets_dir
                )

                high_sim_pass_count, high_sim_fail_count, n_good = get_structure_eval(
                    smiles,
                    mols_that_meet_threshold,
                    mols_that_fail_threshold
                )

                calc = False

            if calc:
                out_dict[dir][constraint] = {
                    "mean": mean,
                    "std": std,
                    "n_valid": valid_smiles,
                    "pass_rate": pass_rate
                }
            else:
                out_dict[dir][constraint] = {
                    "high_sim_pass_rate": high_sim_pass_count,
                    "high_sim_fail_rate": high_sim_fail_count,
                    "n_valid": len(smiles),
                    "n_good": n_good
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
    datasets_dir = config["datasets_dir"]

    threshold_path = "src/models/components/json/thresholds.json"
    with open(threshold_path, "r") as f:
        thresholds = json.load(f)
    threshold_info = [thresholds[c] for c in constraints]

    constraint_eval(
        constraints,
        threshold_info,
        sdf_dir_path,
        datasets_dir, # For structure-based eval
        out_path
    )