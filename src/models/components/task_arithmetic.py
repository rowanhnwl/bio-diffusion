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

    z = torch.add(
        z if method == "add" else z * (1 - task_arithmetic_weight), # Latent space representation
        z_ta * task_arithmetic_weight # Task arithmetic vector
    )

    return z

def get_scheduled_weight(
  t: int,
  timesteps: int,
  init_w: float=0.5,
  final_w: float=0.01,
  gap: int=3,
  method: str="lin" # Method of scheduling (linear, exponential, sigmoid, none)
):
    """
    Progressively decrease the task arithmetic weight for convergence purposes
    """

    # Return 0 if the current time is in the gap
    if t % gap != 0:
        return 0

    if method == "lin": # Linear scheduling
        w_delta = init_w - final_w
        t_progress = t / timesteps

        w_t = w_delta * (1 - t_progress)
    elif method == "exp": # Exponential scheduling
        b = - timesteps / np.log(final_w / init_w)

        w_t = init_w * np.exp(-t / b)

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
    constraint_name: str,
    constraint_matrices_json_path: str,
    device: str,
    space_dims: int=3 # Number of dimensions (X vector)
):
    """
    Create a preset matrix for testing purposes
    Set the X, Y, Z means to zero before returning
    """

    # Load the JSON
    with open(constraint_matrices_json_path, "r") as cmf:
        matrices = json.load(cmf)

    assert (constraint_name in matrices.keys()), "Invalid constraint name"

    # Preset matrix
    ta_nonzero_mean = torch.tensor(
        matrices[constraint_name],
        device=device
    )
    
    # Reset the mean of the X, Y, Z coordinates to zero
    mean_coords = torch.sum(ta_nonzero_mean[:, :space_dims], dim=0, keepdim=True) / ta_nonzero_mean.shape[0]
    ta_nonzero_mean[:, :space_dims] = ta_nonzero_mean[:, :space_dims] - mean_coords

    return ta_nonzero_mean