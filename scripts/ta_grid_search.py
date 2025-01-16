import subprocess
import json
import argparse
from itertools import product
import os
from tqdm import tqdm

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
    sdf_name_list.append(constraint_name.lower())
    sdf_name_list.append(f"ts-{timesteps}")
    sdf_name_list.append(f"iw-{init_weight}")
    sdf_name_list.append(f"fw-{final_weight}")
    sdf_name_list.append(f"ai-{add_interval}")
    sdf_name_list.append(f"am-{add_method}")
    sdf_name_list.append(f"sm-{schedule_method}")

    return "_".join(sdf_name_list)

def gen_molecule(
    param_config,
    out_dir  
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

    # Get the number of atoms
    with open(constraint_matrices_json_path, "r") as cmf:
        matrices = json.load(cmf)
    n_atoms = len(matrices[constraint_name])

    # Get the full output path
    out_path = os.path.join("./output", out_dir)
    out_path = f"'{out_path}'" # Escape ':' for hydra parsing

    # Fix the constraint name for hydra parsing
    constraint_name = f"'{constraint_name}'"

    subprocess.run(
        f"  python3 src/mol_gen_sample.py \
            datamodule=edm_qm9 \
            model=qm9_mol_gen_ddpm \
            logger=csv \
            trainer.accelerator=gpu \
            trainer.devices=[0] \
            ckpt_path=\"checkpoints/QM9/Unconditional/model_1_epoch_979-EMA.ckpt\" \
            num_samples=1 \
            num_nodes={n_atoms} \
            all_frags=true \
            sanitize=false \
            relax=false \
            num_resamplings=1 \
            jump_length=1 \
            num_timesteps={timesteps} \
            output_dir={out_path} \
            seed=123 \
            constraint_name={constraint_name} \
            init_weight={init_weight} \
            final_weight={final_weight} \
            add_interval={add_interval} \
            add_method={add_method} \
            schedule_method={schedule_method} \
            constraint_matrices_json_path={constraint_matrices_json_path}",
            shell=True,
            stdout=subprocess.DEVNULL # Avoid printing
    )

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

    for param_config in tqdm(params_configs):
        out_dir = set_sdf_filename(param_config)
        gen_molecule(param_config, out_dir)