import subprocess
import json
import argparse
import os
from shutil import rmtree
from tqdm import tqdm
import time

from eval.constraint_analysis import *
from gen_binary_matrix import generate_binary_matrix, rm_fixed_datasets
from gen_single_matrix import generate_single_matrix

BM_IW = 0
BM_FW = 0
BM_AI = 5
BM_AM = "add"
BM_SM = "none"

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
        constraint_info,
        _,
        sampled_dataset
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

def gen_molecule(
    param_config,
    out_dir,
    iteration,
    config_name,
    benchmark=False,
    preset_dir=None
):
    (
        timesteps,
        init_weight,
        final_weight,
        add_interval,
        add_method,
        schedule_method,
        matrix_path,
        constraint_info,
        molecules,
        sampled_dataset
    ) = param_config
    
    constraints = [list(d.keys())[0] for d in constraint_info]
    constraint_name_r = ":".join(constraints)
    
    with open(f"{matrix_path}/matrix.json", "r") as f:
        matrix_dict = json.load(f)

    constraint_matrix = matrix_dict["matrix"]
    n_atoms = len(constraint_matrix)

    if not benchmark:
        out_path_r = os.path.join(out_dir, config_name + "_" + set_sdf_dirname(param_config, constraint_name_r), set_sdf_dirname(param_config, constraint_name_r) + "_" + str(iteration))
    else:
        out_path_r = os.path.join(out_dir, preset_dir, "unconstrained_benchmark_" + str(iteration))
    out_path = f"'{out_path_r}'" # Escape ':' for hydra parsing

    # Remove any existing molecules
    if os.path.exists(out_path_r) and len(os.listdir(out_path_r)) > 0:
        os.remove(os.path.join(out_path_r, (os.listdir(out_path_r))[0]))

    # Fix the constraint name for hydra parsing
    constraint_name = f"'{constraint_name_r}'"

    datamodule = f"edm_{sampled_dataset.lower()}"
    model = f"{sampled_dataset.lower()}_mol_gen_ddpm"
    ckpt_path = "checkpoints/GEOM/Unconditional/36hq94x5_model_1_epoch_76-EMA.ckpt" if sampled_dataset == "GEOM" \
        else "checkpoints/QM9/Unconditional/model_1_epoch_979-EMA.ckpt"

    subprocess.run(
        f"  python3 src/mol_gen_sample.py \
            datamodule={datamodule} \
            model={model} \
            logger=csv \
            trainer.accelerator=gpu \
            trainer.devices=[0] \
            ckpt_path=\"{ckpt_path}\" \
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
            matrix_path={matrix_path}",
            shell=True,
            stdout=subprocess.DEVNULL, # Avoid printing
            stderr=subprocess.DEVNULL
    )

    return os.path.dirname(out_path_r), constraint_name_r

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config_path = args.config

    with open(config_path, "r") as cf:
        config = json.load(cf)

    config_name = os.path.basename(config_path).removesuffix(".json")

    constraint_info = config["constraint_info"]
    min_smiles_len = config["min_train_smiles_length"]
    binary = len(constraint_info) > 1

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

    n_iters = config["n_iterations"]

    sampled_dataset = config["sampled_dataset"]

    gen_unconstrained = config["gen_unconstrained"]

    threshold_info = [list(d.values())[0] for d in constraint_info]

    if not os.path.exists(eval_out_dir):
        os.makedirs(eval_out_dir)

    for n in tqdm(range(n_iters)):
        # Get the constraint matrix
        if binary:
            constraint_matrix = generate_binary_matrix(
                constraint_info,
                min_smiles_len,
                datasets_dir,
                sampled_dataset
            )
        else:
            constraint_matrix = generate_single_matrix(
                constraint_info,
                min_smiles_len,
                datasets_dir,
                sampled_dataset
            )

        print(f"Number of atoms: {len(constraint_matrix)}")

        tmp_matrix_path = "tmp_matrix"
        os.makedirs(tmp_matrix_path, exist_ok=True)
        tmp_dict = {}
        tmp_dict["matrix"] = constraint_matrix
        with open(f"{tmp_matrix_path}/matrix.json", "w") as f:
            json.dump(tmp_dict, f, indent=3)

        # Sync the filesystem and wait
        os.sync()
        time.sleep(1)

        param_config = (
            timesteps,
            init_weight,
            final_weight,
            add_interval,
            add_method,
            schedule_method,
            tmp_matrix_path,
            constraint_info,
            molecules,
            sampled_dataset
        )

        bm_param_config = ( # Benchmark param config
            timesteps,
            BM_IW,
            BM_FW,
            BM_AI,
            BM_AM,
            BM_SM,
            tmp_matrix_path,
            constraint_info,
            molecules,
            sampled_dataset
        )

        # Generate the molecules
        sdf_dir_path, constraint_name = gen_molecule(param_config, output_dir, n, config_name)
        if gen_unconstrained:
            gen_molecule(bm_param_config, output_dir, n, config_name, benchmark=True, preset_dir=os.path.basename(sdf_dir_path)) # Benchmark generation

        eval_out_path = os.path.join(eval_out_dir, config_name + "_" + constraint_name + ".json")

        constraint_name_list = constraint_name.split(":")

        # Remove the matrix file
        if os.path.exists(tmp_matrix_path):
            rmtree(tmp_matrix_path)

    # Sync the filesystem and wait
    os.sync()
    time.sleep(1)

    new_datasets_dir = os.path.join(datasets_dir, sampled_dataset.lower())

    constraint_eval(
        constraint_name_list,
        threshold_info,
        sdf_dir_path,
        new_datasets_dir,
        eval_out_path,
        gen_unconstrained
    )

    rm_fixed_datasets(new_datasets_dir)