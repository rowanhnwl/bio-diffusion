# Bio-diffusion with Task Arithmetic

## Description
Latent space steering to constrain molecular generation. Based on [*In-context Vectors: Making In Context Learning More Effective and Controllable Through Latent Space Steering*](https://doi.org/10.48550/arXiv.2311.06668). Originally forked from https://github.com/BioinfoMachineLearning/bio-diffusion, code for [*Geometry-Complete Diffusion for 3D Molecule Generation and Optimization*](https://arxiv.org/abs/2302.04313).

The task arithmetic logic is implemented for the QM9 Unconditional Generation task, specified in the original GCDM repository. Other parameters are listed [here](https://github.com/BioinfoMachineLearning/bio-diffusion?tab=readme-ov-file#generate-new-unconditional-3d-molecules-qm9).

## Installation
Follow the steps outlined in the original repository

## Running
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

The config can be placed in `configs/task_arithmetic`, and the grid search can be run with

```
python3 scripts/ta_grid_search.py --config <PATH_TO_CONFIG>
```

An example/default config file can be found at `configs/task_arithmetic/ta_grid_search.json`

### Description of parameters
`timesteps`: The number of denoising iterations \
`init_weight`: The initial weight scalar applied to the constraint matrix \
`final_weight`: The final weight scalar applied to the constraint matrix \
`add_interval`: The time interval at which the constraint matrix is added to the latent representation \
`add_method`: Either `"add"` for $\boldsymbol{z}_{t-1} \sim p(\boldsymbol{z}_{t-1} | \boldsymbol{z}_t) + w_{ta}\boldsymbol{z}_{ta}$ or `"mean"` for $\boldsymbol{z}_{t-1} \sim p(\boldsymbol{z}_{t-1} | \boldsymbol{z}_t)(1 - w_{ta}) + w_{ta}\boldsymbol{z}_{ta}$ where $w_{ta}$ is the weight of the constraint matrix \
`schedule_method`: Either `"lin"` for a linear decrease in $w_{ta}$ over time or `"exp"` for exponential decay over time \
`constraint_matrices_json_path`: Path to the JSON file supplying the constraint matrix for each constraint or constraint combination. Two files are provided in `src/models/components/json`. One contains the matrices for the single constraints, and the other for the combined constraints
`output_dir`: Path to the desired output directory

### Description of output
In the specified `output_dir`, a separate directory for each molecule's `.sdf` file will be created. The naming convention is as follows \
```
<constraint>_ts-<timesteps>_iw-{init_weight}_fw-{final_weight}_ai-{add_interval}_am-{add_method}_sm-{schedule_method}
```

## Important files
Task arithmetic utility functions are located in `src/models/components/task_arithmetic.py`, and are applied to the diffusion logic in `src/models/component/variational_diffusion.py`

The diffusion model for the QM9 dataset is located at `src/models/qm9_mol_gen_ddpm.py`

To add more grid search parameters for task arithmetic, change the config file `configs/mol_gen_sample.yaml`

The grid search script is located at `scripts/ta_grid_search.py`