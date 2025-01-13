### UTILITY FUNCTIONS FOR TASK ARITHMETIC COMPATABILITY

import torch

def add_ta_latent_vec(
    z: torch.Tensor,
    z_ta: torch.Tensor,
    task_arithmetic_weight: float, # Weight of the task arithmetic vector when summing
    space_dims: int=3 # Number of dimensions (X vector)
):
    """
    Add the task arithmetic latent vector to z
    Afterwards, shift the coordinate means to zero
    """

    assert (task_arithmetic_weight >= 0 and task_arithmetic_weight <= 1.0), "Improper task arithmetic weighting"

    z = torch.add(
        z * (1 - task_arithmetic_weight), # Latent vector
        z_ta * task_arithmetic_weight # Task arithmetic vector
    )

    # Reset the mean of the X, Y, Z coordinates to zero
    mean_coords = torch.sum(z[:, :space_dims], dim=0, keepdim=True) / z.shape[0]
    z[:, :space_dims] = z[:, :space_dims] - mean_coords

    return z

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