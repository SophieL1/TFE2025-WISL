import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float
from typing import Any
from differt.scene import TriangleScene
from differt.rt import rays_intersect_any_triangle
from src.utils import DEFAULT_EPSILON

def initialization(
    scene: TriangleScene, 
    measurements: Float[Array, " "], 
    m: int=50, 
    n: int=50,
    nb_init: int=5
) -> Float[Array, "2"]:
    """
    Initialization heuristic for the transmitter position based on scene and measurements.

    Args:
        scene: TriangleScene containing the geometry and receivers' positions.
        measurements: The power measurements from the receivers.
        args: Additional arguments for the loss function.
        m: Number of rows in the grid.
        n: Number of columns in the grid.
        nb_init: Number of initial candidates to consider.

    Returns:    
        An array of 2D candidate transmitter positions.
    """
    # Set a regular grid of candidates in the scene (no candidate on the outside walls)
    batch = (m+1, n+1)
    scene_grid = scene.with_receivers_grid(*batch, height=0.0)
    x, y, _ = jnp.unstack(scene_grid.receivers, axis=-1)
    x = x[:-1, :-1]
    y = y[:-1, :-1]
    x = x + 0.5 * (x[0, 1] - x[0, 0])
    y = y + 0.5 * (y[1, 0] - y[0, 0])
    candidates = jnp.column_stack([x.ravel(), y.ravel()])

    # Sort candidates by distance to the 3 most powerful receivers, weighted by the measurements
    rx_coords = scene.receivers[:, :2]
    rx_powerful = jnp.argsort(measurements)[-3:]
    rx_powerful_coords = rx_coords[rx_powerful]
    distances = []
    for i in range(len(candidates)):
        distance = 0
        for j in range(len(rx_powerful_coords)):
            distance += jnp.linalg.norm(candidates[i] - rx_powerful_coords[j])*measurements[rx_powerful[j]]
        distances.append(distance)
    distances = jnp.array(distances)
    candidates = candidates[jnp.argsort(distances)]

    # Filter and remove the candidates that are too close to the receivers to be plausible.
    # If the true power measurement at a receiver is lower than what is simulated considering only LOS for the transmitter at a candidate position,
    # it is not plausible that the transmitter is at this candidate position (too close to the considered receiver). 
    # To keep enough candidates, we introduce a 1/4 factor.
    candidates_after_LOS = []
    LOS_tol_factor = 0.25
    for i in range(len(candidates)):
        far_enough_from_all_rx = True
        for j in range(len(measurements)):
            rx = jnp.array([jnp.concatenate((rx_coords[j], jnp.array([0.0])))])
            cand = jnp.array([jnp.concatenate((candidates[i], jnp.array([0.0])))])
            vector = rx - cand
            length_LOS = jnp.linalg.norm(vector)
            if not rays_intersect_any_triangle(cand, vector, scene.mesh.triangle_vertices):
                LOS_power = 1 / (length_LOS**2 + DEFAULT_EPSILON)
                if measurements[j] < LOS_tol_factor*LOS_power:
                    far_enough_from_all_rx = False
                    break
        if far_enough_from_all_rx:
            candidates_after_LOS.append(candidates[i])

    # Keep only the first nb_init candidates (at most)
    n = min(nb_init, len(candidates_after_LOS))
    candidates_after_LOS = candidates_after_LOS[:n]
    
    return jnp.array(candidates_after_LOS)