import subprocess
import json
import argparse
from itertools import product

# Params to add
# init_w, final_w, gap, add_method, timesteps, weight_schedule_method
# constraint_matrices_file_path

def set_sdf_filename(
    param_config
):
    
    (
        constraint_name,
        timesteps,
        init_weight,
        final_weight,
        add_interval,
        add_method,
        schedule_method,
        _ # Unused path string
    ) = param_config

    sdf_name_list = []
    sdf_name_list.append(constraint_name)
    sdf_name_list.append(f"ts-{timesteps}")
    sdf_name_list.append(f"iw-{init_weight}")
    sdf_name_list.append(f"fw-{final_weight}")
    sdf_name_list.append(f"ai-{add_interval}")
    sdf_name_list.append(f"am-{add_method}")
    sdf_name_list.append(f"sm-{schedule_method}")

    return "_".join(sdf_name_list)

def gen_molecule(
    param_config      
):
    (
        constraint_name,
        timesteps,
        init_weight,
        final_weight,
        add_interval,
        add_method,
        schedule_method,
        constraint_matrices_json_path
    ) = param_config

    # ADD HYPERPARAMS TO THE YAML FILES

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config_path = args.config

    with open(config_path, "r") as cf:
        config = json.load(cf)

    constraint_matrices_json_path = config["constraint_matrices_json_path"]
    with open(constraint_matrices_json_path, "r") as cmf:
        constraint_matrices_dict = json.load(cmf)

    constraint_names_l = list(constraint_matrices_dict.keys())
    
    timesteps_l = config["timesteps"]

    init_weight_l = config["init_weight"]    
    final_weight_l = config["final_weight"]
    add_interval_l = config["add_interval"]

    add_method_l = config["add_method"]
    schedule_method_l = config["schedule_method"]

    params_configs = product(
        constraint_names_l,
        timesteps_l,
        init_weight_l,
        final_weight_l,
        add_interval_l,
        add_method_l,
        schedule_method_l,
        [constraint_matrices_json_path]
    )

    for param_config in params_configs:
        gen_molecule(param_config)