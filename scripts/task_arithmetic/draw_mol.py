import rdkit
from rdkit import Chem

import os
import json

from heapq import nlargest

from eval.constraint_analysis import constraint_eval

if __name__ == "__main__":
    sdf_path = "tmp_mol/caco2 permeability_GEOM_ts-1000_iw-0.25_fw-0.01_ai-5_am-mean_sm-exp_caco2 permeability_GEOM_ts-1000_iw-0.25_fw-0.01_ai-5_am-mean_sm-exp"

    constraints = ["Caco2 Permeability"]
    threshold_info = [
    {
        "threshold": -6.0,
        "weight": 1,
        "bound": "lower"
    }]
    datasets_dir = "eval_datasets/geom"
    out_path = "task_arithmetic_eval/smiles/caco2_geom.json"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    constraint_eval(
        constraints,
        threshold_info,
        sdf_path,
        datasets_dir,
        out_path,
        False
    )
    os.sync()

    with open(out_path, "r") as f:
        results_dict = json.load(f)

    constraint_key = ":".join(constraints)
    smiles = results_dict["smiles"][constraint_key]

    smiles_frequency_dict = {}
    for smi in smiles:
        smiles_frequency_dict[smi] = 0

    for smi in smiles:
        smiles_frequency_dict[smi] += 1

    candidate_smiles = nlargest(n=1, iterable=list(smiles_frequency_dict.items()), key=lambda x: x[1])[0][0]
    print(candidate_smiles)

    mol = Chem.MolFromSmiles(candidate_smiles)
    Chem.Draw.MolToFile(mol, "figs/mol_geom_caco2.png", size=(500, 500))