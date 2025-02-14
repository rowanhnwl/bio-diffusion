import json
import os
from itertools import product

constraints_dict = {
    "Caco2 Permeability": {
        "threshold": -6.0,
        "weight": 1,
        "bound": "lower"
    },
    "Lipophilicity": {
        "threshold": 2.0,
        "weight": 1,
        "bound": "lower"
    },
    "Solubility": {
        "threshold": 0.0,
        "weight": 1,
        "bound": "lower"
    },
    "Volume Distribution at Steady State": {
        "threshold": 5.0,
        "weight": 1,
        "bound": "upper"
    },
    "Acute Toxicity": {
        "threshold": 2.0,
        "weight": 1,
        "bound": "upper"
    },
    "TPSA": {
        "threshold": 50.0,
        "weight": 1,
        "bound": "upper"
    },
    "XLogP": {
        "threshold": 0.0,
        "weight": 1,
        "bound": "upper"
    },
    "Rotatable Bond Count": {
        "threshold": 1.0,
        "weight": 1,
        "bound": "upper"
    }
}

def set_config_name(
    param_config,
    constraint_name
):
    
    (
        timesteps,
        _,
        init_weight,
        final_weight,
        add_interval,
        add_method,
        schedule_method,
    ) = param_config

    sdf_name_list = []
    sdf_name_list.append(constraint_name.lower())
    sdf_name_list.append(sampled_dataset)
    sdf_name_list.append(f"ts-{timesteps}")
    sdf_name_list.append(f"iw-{init_weight}")
    sdf_name_list.append(f"fw-{final_weight}")
    sdf_name_list.append(f"ai-{add_interval}")
    sdf_name_list.append(f"am-{add_method}")
    sdf_name_list.append(f"sm-{schedule_method}")

    return "_".join(sdf_name_list)

def gen_config_dict(
    output_dir,
    eval_out_dir,
    sampled_dataset,
    datasets_dir,
    min_train_smiles_length,
    n_iterations,
    gen_unconstrained,
    constraints,
    param_config        
):
    (
        timesteps,
        molecules,
        init_weight,
        final_weight,
        add_interval,
        add_method,
        schedule_method
    ) = param_config
    
    config_dict = {}

    config_dict["timesteps"] = timesteps
    config_dict["molecules"] = molecules
    config_dict["init_weight"] = init_weight
    config_dict["final_weight"] = final_weight
    config_dict["add_interval"] = add_interval
    config_dict["add_method"] = add_method
    config_dict["schedule_method"] = schedule_method

    constraint_list = [{c: constraints_dict[c]} for c in constraints]
    config_dict["constraint_info"] = constraint_list

    config_dict["output_dir"] = output_dir
    config_dict["eval_out_dir"] = eval_out_dir
    config_dict["sampled_dataset"] = sampled_dataset
    config_dict["datasets_dir"] = datasets_dir
    config_dict["min_train_smiles_length"] = min_train_smiles_length
    config_dict["n_iterations"] = n_iterations
    config_dict["gen_unconstrained"] = gen_unconstrained

    return config_dict

configs_dir = "configs/task_arithmetic/gen/geom_no_constraint_caco2"
os.makedirs(configs_dir, exist_ok=True)

timesteps_list = [500]
molecules_list = [250]
init_weight_list = [0]
final_weight_list = [0]
add_interval_list = [5]
add_method_list = ["add"]
schedule_method_list = ["none"]

output_dir = "output/geom_no_constraint_caco2"
eval_out_dir = "task_arithmetic_eval/geom_no_constraint_caco2"
sampled_dataset = "GEOM"
datasets_dir = "eval_datasets"
min_train_smiles_length = 50
n_iterations = 1
gen_unconstrained = False
constraints = ["Caco2 Permeability"]

param_configs = product(
    timesteps_list,
    molecules_list,
    init_weight_list,
    final_weight_list,
    add_interval_list,
    add_method_list,
    schedule_method_list
)

for pc in param_configs:

    constraint_name = ":".join(constraints)
    config_name = set_config_name(pc, constraint_name) + ".json"

    out_path = os.path.join(configs_dir, config_name)

    config_dict = gen_config_dict(
        output_dir,
        eval_out_dir,
        sampled_dataset,
        datasets_dir,
        min_train_smiles_length,
        n_iterations,
        gen_unconstrained,
        constraints,
        pc
    )

    with open(out_path, "w") as f:
        json.dump(config_dict, f, indent=3)