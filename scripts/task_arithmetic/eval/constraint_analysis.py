from rdkit import Chem
from rdkit.Chem import Descriptors

from rdkit.Chem.PandasTools import LoadSDF
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs

from scipy.stats import ttest_ind

import numpy as np

import argparse
import os
import json

from copy import deepcopy

from multiprocessing import Pool, Manager
manager = Manager()

def eval_calc_constraint(
    smiles: list,
    constraint: str,
    threshold: float,
    bound_type: str
):
    smiles_results_dict = {smi: 0 for smi in smiles}

    # Calculate the molecular weight
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)

        if mol:
            if constraint == "Molecular Weight":
                val = Descriptors.MolWt(mol)
            elif constraint == "TPSA":
                val = rdMolDescriptors.CalcTPSA(mol)
            elif constraint == "XLogP":
                val, mr = rdMolDescriptors.CalcCrippenDescriptors(mol)
            elif constraint == "Rotatable Bond Count":
                val = rdMolDescriptors.CalcNumRotatableBonds(mol)
        else:
            print(f"Invalid SMILES string: {smi}")
            continue

        if bound_type == "upper" and val <= threshold:
            if not (constraint == "TPSA" and val == 0):
                smiles_results_dict[smi] = 1
        elif bound_type == "lower" and val >= threshold:
            smiles_results_dict[smi] = 1

    return smiles_results_dict

# Structure-based evaluation
def tdc_evaluation(
    args
):
    (
        list_of_gen_mols,
        meet_dataset_mols,
        fail_dataset_mols,
        shared_met_dict,
        shared_fail_dict,
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
            shared_met_dict[smiles] = met_sim       
            shared_fail_dict[smiles] = fail_sim

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

    gen_mol_met_sim = manager.dict()
    gen_mol_fail_sim = manager.dict()

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
    gen_mol_met_sim = dict(gen_mol_met_sim)
    gen_mol_fail_sim = dict(gen_mol_fail_sim)

    smiles_results_dict = {smi: 0 for smi in smiles}
    sim_thresh = 0.5

    for smi in smiles:
        if np.max(gen_mol_met_sim[smi]) >= sim_thresh:
            smiles_results_dict[smi] = 1

    return smiles_results_dict

def eval_dataset(
    constraint: str,
    bound: str,
    threshold: float,
    datasets_dir: str
):
    
    with open("eval_datasets/convert/constraint_to_dataset.json", "r") as f:
        constraint_to_dataset = json.load(f)

    constraint_dataset_substring = constraint_to_dataset[constraint]

    # Get the dataset file name
    dataset_names = os.listdir(datasets_dir)
    dataset_filename = ""
    for dname in dataset_names:
        if constraint_dataset_substring in dname and "fixed" not in dname:
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

# Get the number of generated molecules that satisfy ALL constraints
def get_multi_constraint_success_rate(
    results_dicts,
    n_molecules  
):
    smiles = list(results_dicts[0].keys())

    good_count = 0

    for smi in smiles:
        satisfies = True
        
        for r_dict in results_dicts:
            if r_dict[smi] == 0:
                satisfies = False

        if satisfies:
            good_count += 1

    return good_count / n_molecules

def constraint_summary(
    constraints: list, 
    results_dict: dict,
    gen_unconstrained: bool
):
    constraint_name_key = ":".join(constraints)
    unconstrained_key = "unconstrained"

    lists_dict = {
        constraint: {
            "unconstrained": [],
            "task_arithmetic": []
        } for constraint in constraints
    }
    if len(constraints) > 1:
        lists_dict[constraint_name_key] = {
            "unconstrained": [],
            "task_arithmetic": []
        }

    summary_dict = {}
    for constraint in constraints:
        summary_dict[constraint] = {}
    if len(constraints) > 1:
        summary_dict[constraint_name_key] = {}

    for k in results_dict.keys():
        if unconstrained_key in k:
            for ck in results_dict[k].keys():
                lists_dict[ck]["unconstrained"].append(results_dict[k][ck]["P(meets threshold) total"])
        elif constraint_name_key.lower() in k:
            for ck in results_dict[k].keys():
                lists_dict[ck]["task_arithmetic"].append(results_dict[k][ck]["P(meets threshold) total"])

    for k in lists_dict.keys():
        summary_dict[k]["unconstrained_mean"] = np.mean(lists_dict[k]["unconstrained"]) if gen_unconstrained else 0
        summary_dict[k]["unconstrained_std"] = np.std(lists_dict[k]["unconstrained"]) if gen_unconstrained else 0
        summary_dict[k]["task_arithmetic_mean"] = np.mean(lists_dict[k]["task_arithmetic"])
        summary_dict[k]["task_arithmetic_std"] = np.std(lists_dict[k]["task_arithmetic"])

    results_dict["summary"] = summary_dict

def compute_ttest_scores(results_dict):
    ttest_dict = {
        "unconstrained": {},
        "constrained": {}
    }

    constraint_keys = set()

    for k in results_dict.keys():
        if "unconstrained" in k:
            for constraint_key in results_dict[k]:
                constraint_keys.add(constraint_key)
                p_total = results_dict[k][constraint_key]["P(meets threshold) total"]
                try:
                    ttest_dict["unconstrained"][constraint_key].append(p_total)
                except:
                    ttest_dict["unconstrained"][constraint_key] = [p_total]
        elif k != "summary":
            for constraint_key in results_dict[k]:
                constraint_keys.add(constraint_key)
                p_total = results_dict[k][constraint_key]["P(meets threshold) total"]
                try:
                    ttest_dict["constrained"][constraint_key].append(p_total)
                except:
                    ttest_dict["constrained"][constraint_key] = [p_total]

    for constraint_key in constraint_keys:
        constrained_p_totals = ttest_dict["constrained"][constraint_key]
        unconstrained_p_totals = ttest_dict["unconstrained"][constraint_key]

        _, pval = ttest_ind(constrained_p_totals, unconstrained_p_totals, equal_var=False)

        results_dict["summary"][constraint_key]["p_val"] = pval

# Function to demultiplex the constraint
def constraint_eval(
    constraints: list,
    threshold_info_list: list,
    sdf_dir_path: str,
    datasets_dir: str,
    out_path: str,
    gen_unconstrained: bool,
    total_molecules: int=250
):
    
    sdf_dirs = os.listdir(sdf_dir_path)

    out_dict = {}

    for dir in sdf_dirs:
        full_sdf_path = os.path.join(sdf_dir_path, dir)
        sdf_fn = os.listdir(full_sdf_path)[0]
        full_sdf_path = os.path.join(full_sdf_path, sdf_fn)

        out_dict[dir] = {}

        df = LoadSDF(full_sdf_path, smilesName='SMILES')

        results_dicts = []

        try:
            smiles = list(set(df["SMILES"])) # Make sure there are no duplicate SMILES strings
        except:
            print("No available SMILES strings")
            smiles = []

        for i, constraint in enumerate(constraints):

            threshold_info = threshold_info_list[i]

            threshold = threshold_info["threshold"]
            bound_type = threshold_info["bound"] # Upper or lower
            weight = threshold_info["weight"]

            if len(smiles) == 0:
                pass
            elif constraint in ["Molecular Weight", "TPSA", "XLogP", "Rotatable Bond Count"]:
                results_dict = eval_calc_constraint(
                    smiles,
                    constraint,
                    threshold,
                    bound_type
                )

                results_dicts.append(deepcopy(results_dict))

            else: # NEED TO USE STRUCTURAL SIMILARITY

                # Split the dataset by passing/failing the threshold
                mols_that_meet_threshold, mols_that_fail_threshold = eval_dataset(
                    constraint,
                    bound_type,
                    threshold,
                    datasets_dir
                )

                results_dict = get_structure_eval(
                    smiles,
                    mols_that_meet_threshold,
                    mols_that_fail_threshold
                )

                results_dicts.append(deepcopy(results_dict))

            if len(smiles) != 0:

                n_passing = np.sum(list(results_dict.values()))

                out_dict[dir][constraint] = {
                    "P(meets threshold) distinct": n_passing / len(smiles),
                    "P(meets threshold) total": n_passing / total_molecules,
                    "n_distinct": len(smiles),
                    "Threshold": threshold,
                    "Bound type": bound_type
                }
            else:

                out_dict[dir][constraint] = {
                    "P(meets threshold)": 0,
                    "P(meets threshold) total": 0,
                    "n_distinct": 0,
                    "Threshold": threshold,
                    "Bound type": bound_type
                }

            if dir != "unconstrained_benchmark":
                out_dict[dir][constraint]["Weight"] = weight

        # Create a new field to show the success rate across BOTH constraints
        if len(constraints) > 1 and len(smiles) > 0:
            multi_constraint_p = get_multi_constraint_success_rate(results_dicts, len(smiles))

            joined_constraint_name = ":".join(constraints)
            out_dict[dir][joined_constraint_name] = {
                "P(meets threshold) distinct": multi_constraint_p,
                "P(meets threshold) total": multi_constraint_p * (len(smiles) / total_molecules)
            }
        elif len(smiles) == 0:
            joined_constraint_name = ":".join(constraints)
            out_dict[dir][joined_constraint_name] = {
                "P(meets threshold) distinct": 0,
                "P(meets threshold) total": 0
            }

    constraint_summary(constraints, out_dict, gen_unconstrained)
    
    if gen_unconstrained:
        compute_ttest_scores(out_dict)

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