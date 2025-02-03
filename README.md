# Bio-diffusion with Task Arithmetic

## Description
Latent space steering to constrain molecular generation. Based on [*In-context Vectors: Making In Context Learning More Effective and Controllable Through Latent Space Steering*](https://doi.org/10.48550/arXiv.2311.06668). Originally forked from https://github.com/BioinfoMachineLearning/bio-diffusion, code for [*Geometry-Complete Diffusion for 3D Molecule Generation and Optimization*](https://arxiv.org/abs/2302.04313).

The task arithmetic logic is implemented for the QM9 Unconditional Generation task, specified in the original GCDM repository. Other parameters are listed [here](https://github.com/BioinfoMachineLearning/bio-diffusion?tab=readme-ov-file#generate-new-unconditional-3d-molecules-qm9).

## Installation
First, follow the steps outlined in the original repository (up to the end of the **Installation guide** section)

Next, install additional packages

```
pip install pubchempy transformers pytdc
```

Finally, change the Matplotlib version (ignore errors)

```
pip install matplotlib==3.3.4 --no-deps
pip uninstall seaborn scanpy
pip install seaborn==0.11.0 scanpy==1.8.0
```

## Molecule Generation
Create a config file with the following format

```
{
    "timesteps": 1000,
    "molecules": 250,
    "init_weight": 0.75,
    "final_weight": 0.01,
    "add_interval": 5,
    "add_method": "mean",
    "schedule_method": "exp",
    "constraint_info": [
        {
            "Caco2 Permeability": {
                "weight": 5,
                "threshold": -6.0,
                "bound": "lower"
            }
        },
        {
            "TPSA": {
                "weight": 1,
                "threshold": 50.0,
                "bound": "upper"
            }
        }
    ],
    "output_dir": "output/task_arithmetic_molecules",
    "eval_out_dir": "task_arithmetic_eval/",
    "datasets_dir": "eval_datasets",
    "min_train_smiles_length": 20,
    "n_iterations": 3
}
```

Note that the above example uses the optimal hyperparameters. The only fields that should be changed are:

`constraint_info`: Contains all constraint-related parameters \
`output_dir`: Output directory of SDF files containing the RDKit molecules \
`min_train_smiles_length`: Higher -> more atoms per molecule \
`n_iterations`: The number of iterations (default `n=3`)

Generation can be run with the following command

```
python scripts/task_arithmetic/ta_generate.py --config <PATH_TO_CONFIG>
```

### Evaluation output
The output JSON will report the performance for each iteration (for both task arithmetic and unconstrained), along with a `summary` dictionary that contains the aggregate means and standard deviations.

### Grid search for molecule generation
To run a grid search, create a JSON config with the following form
```
{
    "timesteps": [ ... ],
    "init_weight": [ ... ],
    "final_weight": [ ... ],
    "add_interval": [ ... ],
    "add_method": [ ... ],
    "schedule_method": [ ... ],
    "constraint_matrices_json_path": <PATH_TO_CONSTRAINT_MATRICES>,
    "output_dir": <OUTPUT_DIRECTORY>
}
```

The config can be placed in `configs/task_arithmetic/gen`, and the grid search can be run with

```
python3 scripts/task_arithmetic/ta_grid_search.py --config <PATH_TO_CONFIG>
```

An example/default config file can be found at `configs/task_arithmetic/gen/ta_grid_search.json`

To generate molecules without task arithmetic, run the same script but using the config file `configs/task_arithmetic/gen/no_constraint_grid_search.json`

### Description of parameters
`timesteps`: The number of denoising iterations \
`init_weight`: The initial weight scalar applied to the constraint matrix \
`final_weight`: The final weight scalar applied to the constraint matrix \
`add_interval`: The time interval at which the constraint matrix is added to the latent representation \
`add_method`: Either `"add"` for $\boldsymbol{z}_{t-1} \sim p(\boldsymbol{z}_{t-1} | \boldsymbol{z}_t) + w_{ta}\boldsymbol{z}_{ta}$ or `"mean"` for $\boldsymbol{z}_{t-1} \sim p(\boldsymbol{z}_{t-1} | \boldsymbol{z}_t)(1 - w_{ta}) + w_{ta}\boldsymbol{z}_{ta}$ where $w_{ta}$ is the weight of the constraint matrix \
`schedule_method`: Either `"lin"` for a linear decrease in $w_{ta}$ over time or `"exp"` for exponential decay over time \
`constraint_matrices_json_path`: Path to the JSON file supplying the constraint matrix for each constraint or constraint combination. All single and binary constraint matrices are located in the `matrices` directory
`output_dir`: Path to the desired output directory

### Description of output
In the specified `output_dir`, a separate directory for each molecule's `.sdf` file will be created. The naming convention is as follows
```
<constraint>_ts-<timesteps>_iw-{init_weight}_fw-{final_weight}_ai-{add_interval}_am-{add_method}_sm-{schedule_method}
```

## Evalutation
To evaluate the molecules generated from a grid search, create a new config file with the form

```
{
    "eval_constraints": [
        ...
    ],
    "sdf_dir_path": <PATH_TO_GENERATED_MOLECULES>,
    "out_path": <PATH_TO_EVAL_RESULTS>
}
```

This config should be placed in `configs/task_arithmetic/eval`, and the evaluation can be run using

```
python3 scripts/task_arithmetic/constraint_analysis.py --config <PATH_TO_CONFIG>
```

### Description of parameters
`eval_constraints`: This is a list that contains the constraints to be evaluated, for example `["Molecular Weight", "XLogP"]`
`sdf_dir_path`: The path to the outputted SDF files containing the generated molecules
`out_path`: The path to the JSON which will contain the evaluation results (must contain the name of the JSON)

### Description of output
At the `out_path`, there will be a JSON file with the following form

```
{
    <PARAMETER_CONFIG_1>: {
        <CONSTRAINT_1>: {
            "mean": ...,
            "std": ...,
            "valid_smiles": ...,
            "n_valid": ...
        },
        <CONSTRAINT_2>: {
            ...
        }
        ...
    }
    <PARAMETER_CONFIG_2>: {
        ...
    }
    ...
}
```

where `n_valid` is the number of generated molecules that can be converted to an RDKit `mol` object, and `pass_rate` is the fraction of generated molecules whose constraint value was above/below the specified threshold (depending on upper/lower bound)

## Important files
Task arithmetic utility functions are located in `src/models/components/task_arithmetic.py`, and are applied to the diffusion logic in `src/models/component/variational_diffusion.py`

The diffusion model for the QM9 dataset is located at `src/models/qm9_mol_gen_ddpm.py`

To add more grid search parameters for task arithmetic, change the config file `configs/mol_gen_sample.yaml`

A JSON containing the thresholds and bound types (upper/lower) of the constraints can be found at `src/models/components/json/thresholds.json`

The grid search script is located at `scripts/task_arithmetic/ta_grid_search.py`
The evalutation script is located at `scripts/task_arithmetic/constraint_analysis.py`