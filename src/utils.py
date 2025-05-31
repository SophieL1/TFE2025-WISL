from differt.scene import TriangleScene
from differt.geometry import path_lengths
from jaxtyping import Array, Float, Int
import jax.numpy as jnp
import equinox as eqx

DEFAULT_EPSILON = 1e-12   # small value to avoid division by zero
DEFAULT_R_COEF = 0.125      # reflection coefficient of the walls 0.5
DEFAULT_EPS_POWER = 1e-10 # epsilon to add to power in [W] to avoid log(0) when converting to dB
DEFAULT_EPS_ROOM = 1e-3   # distance to the wall when projecting inside the room 

def get_measurements(
    scene: TriangleScene, 
    max_order: Int,
    approx: bool = False, 
    alpha: float = 1.0, 
    r_coef: float = DEFAULT_R_COEF,
) -> Float[Array, " "]:
    """
    Compute the received power at each receiver in the scene.

    Args:
        scene: TriangleScene containing the geometry and receivers' positions.
        max_order: Maximum order of reflections to consider.
        approx: Whether to use the physics-based smoothing in DiffeRT for power calculation.
        alpha: Smoothing factor for the received power.

    Returns:
        An array of received power at each receiver.
    """
    powers = 0.0
    for order in range(max_order + 1):
        paths = scene.compute_paths(order=int(order), approx=approx, alpha=alpha)
        masks = paths.mask 
        if approx:
            weighted_powers = masks * received_power(paths.vertices, r_coef)
            powers += jnp.sum(weighted_powers, axis=-1)
        else:
            powers += jnp.sum(received_power(paths.vertices, r_coef), axis=-1, where=masks)
    return powers

@eqx.filter_jit
def received_power(
    vertices: Float[Array, " "], 
    r_coef: float = DEFAULT_R_COEF
) -> Float[Array, " "]:
    """
    Compute the received power for given paths defined by vertices, following a simplified physics model.

    Args:
        vertices: Array of vertices defining the paths.
        
    Returns:
        An array of received power for each path.
    """
    paths_order = vertices.shape[-2] - 2
    
    d = path_lengths(vertices)

    r_coeff = r_coef**paths_order
    
    return r_coeff / (d * d + DEFAULT_EPSILON)