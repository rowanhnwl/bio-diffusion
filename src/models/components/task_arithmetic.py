### UTILITY FUNCTIONS FOR TASK ARITHMETIC COMPATABILITY

import json
import torch
import numpy as np

def add_ta_latent_vec(
    z: torch.Tensor,
    z_ta: torch.Tensor,
    task_arithmetic_weight: float, # Weight of the task arithmetic vector when summing
    method: str="add" # Weighted addition or mean
):
    """
    Add the task arithmetic latent vector to z
    Afterwards, shift the coordinate means to zero
    """

    assert (task_arithmetic_weight >= 0 and task_arithmetic_weight <= 1.0), "Improper task arithmetic weighting"

    n_samples = int(z.shape[0] / z_ta.shape[0])
    n_atoms = z_ta.shape[0]

    # For multiple samples, the number of rows in the latent space is n_samples * n_atoms
    for i in range(n_samples):
        if method == "add":
            z[i * n_atoms:(i + 1) * n_atoms, :] = z[i * n_atoms:(i + 1) * n_atoms, :] + z_ta * task_arithmetic_weight
        else:
            z[i * n_atoms:(i + 1) * n_atoms, :] = z[i * n_atoms:(i + 1) * n_atoms, :] * (1 - task_arithmetic_weight) + z_ta * task_arithmetic_weight

    return z

def get_scheduled_weight(
  t: int,
  timesteps: int,
  init_w: float=0.5,
  final_w: float=0.01,
  gap: int=3,
  method: str="lin" # Method of scheduling (linear, exponential, last, or none)
):
    """
    Progressively decrease the task arithmetic weight for convergence purposes
    """

    # Return 0 if the current time is in the gap
    if method != "last" and t % gap != 0:
        return 0

    if method == "lin": # Linear scheduling
        w_delta = init_w - final_w
        t_progress = t / timesteps

        w_t = w_delta * (1 - t_progress)
    elif method == "exp": # Exponential scheduling
        b = - timesteps / np.log(final_w / init_w)

        w_t = init_w * np.exp(-t / b)
    elif method == "last": # Add the full matrix only at the last timestep
        w_t = 0 if t < timesteps - 1 else init_w
    elif method == "none":
        w_t = init_w

    return w_t

def get_rand_ta_mat(
    z0: torch.Tensor
):
    """
    Get a random task arithmetic matrix for testing purposes
    Randomly generated values in [-absmax(z0), absmax(z0)]
    """

    absmax_z0 = torch.max(torch.abs(z0))
    
    z_ta = 0.5 * torch.ones(z0.shape, device=z0.device) - torch.rand(z0.shape, device=z0.device)
    z_ta *= absmax_z0

    return z_ta

def get_preset_ta_mat(
    matrix_path: str,
    device: str
):
    """
    Get the constraint matrix
    """

    with open(f"{matrix_path}/matrix.json", "r") as f:
        matrix_dict = json.load(f)

    constraint_matrix = list(matrix_dict.values())[0]

    # Preset matrix
    ta_nonzero_mean = torch.tensor(
        constraint_matrix,
        device=device
    )

    return ta_nonzero_mean