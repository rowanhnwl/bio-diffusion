import subprocess
import json
import argparse
import os
from tqdm import tqdm

from eval.constraint_analysis import *

def set_sdf_dirname(
    param_config,
    constraint_name
):
    
    (
        timesteps,
        init_weight,
        final_weight,
        add_interval,
        add_method,
        schedule_method,
        _,
        _
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
        timesteps,
        init_weight,
        final_weight,
        add_interval,
        add_method,
        schedule_method,
        matrix_path,
        molecules
    ) = param_config

    # Get the number of atoms
    with open(matrix_path, "r") as cmf:
        matrix_dict = json.load(cmf)
    
    constraint_name_r = list(matrix_dict.keys())[0]
    n_atoms = len(matrix_dict[constraint_name_r])

    out_path_r = os.path.join(out_dir, set_sdf_dirname(param_config, constraint_name_r), set_sdf_dirname(param_config, constraint_name_r))
    out_path = f"'{out_path_r}'" # Escape ':' for hydra parsing

    # Fix the constraint name for hydra parsing
    constraint_name = f"'{constraint_name_r}'"

    subprocess.run(
        f"  python3 src/mol_gen_sample.py \
            datamodule=edm_qm9 \
            model=qm9_mol_gen_ddpm \
            logger=csv \
            trainer.accelerator=gpu \
            trainer.devices=[0] \
            ckpt_path=\"checkpoints/QM9/Unconditional/model_1_epoch_979-EMA.ckpt\" \
            num_samples={molecules} \
            num_nodes={n_atoms} \
            all_frags=true \
            sanitize=false \
            relax=false \
            num_resamplings=1 \
            jump_length=1 \
            num_timesteps={timesteps} \
            output_dir={out_path} \
            seed=42 \
            constraint_name={constraint_name} \
            init_weight={init_weight} \
            final_weight={final_weight} \
            add_interval={add_interval} \
            add_method={add_method} \
            schedule_method={schedule_method} \
            constraint_matrices_json_path={matrix_path}",
            shell=True,
            stdout=subprocess.DEVNULL # Avoid printing
    )

    return os.path.dirname(out_path_r), constraint_name_r

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config_path = args.config

    with open(config_path, "r") as cf:
        config = json.load(cf)

    constraint_matrices_json_paths = config["constraint_matrices_json_paths"]

    output_dir = config["output_dir"]
    molecules = config["molecules"] # Number of samples per param configuration

    timesteps = config["timesteps"]

    init_weight = config["init_weight"]    
    final_weight = config["final_weight"]
    add_interval = config["add_interval"]

    add_method = config["add_method"]
    schedule_method = config["schedule_method"]

    eval_out_dir = config["eval_out_dir"]
    datasets_dir = config["datasets_dir"]

    threshold_path = "src/models/components/json/thresholds.json"
    with open(threshold_path, "r") as f:
        thresholds = json.load(f)

    if not os.path.exists(eval_out_dir):
        os.makedirs(eval_out_dir)

    params_configs = [
        (
            timesteps,
            init_weight,
            final_weight,
            add_interval,
            add_method,
            schedule_method,
            path,
            molecules
        ) for path in constraint_matrices_json_paths
    ]

    for param_config in tqdm(params_configs):
        sdf_dir_path, constraint_name = gen_molecule(param_config, output_dir)
        matrix_path = param_config[6]

        eval_out_path = os.path.join(eval_out_dir, os.path.basename(matrix_path))

        constraint_name_list = constraint_name.split(":")
        threshold_info = [thresholds[c] for c in constraint_name_list]

        constraint_eval(
            constraint_name_list,
            threshold_info,
            sdf_dir_path,
            datasets_dir,
            eval_out_path
        )
