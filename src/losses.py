import jax.numpy as jnp
import jax
import equinox as eqx
from jaxtyping import Array, Float, Int
from differt.scene import TriangleScene
from src.utils import get_measurements, DEFAULT_EPS_POWER, DEFAULT_EPS_ROOM

@eqx.filter_jit
def loss(
    tx_coords: Float[Array, "2"],
    scene: TriangleScene, 
    measurements: Float[Array, " "],
    max_order: Int, 
    approx: bool = False,
    alpha: float = 1.0,
) -> Float:
    """
    Returns the value of the MSE loss function evaluated at position tx_coords.

    Args:
        tx_coords: Coordinates of the transmitter.
        scene: TriangleScene containing the geometry and receivers' positions.
        measurements: The power measurements at the receivers.
        max_order: Maximum order of reflections to consider.
        approx: Whether to use the physics-based smoothing in DiffeRT for power calculation.
        alpha: Smoothing factor for the received power.

    Returns:
        The value of the MSE loss function.
    """
    # Set transmitter position
    scene = eqx.tree_at(
        lambda s: s.transmitters, scene, scene.transmitters.at[:2].set(tx_coords)
    )
    # Get simulated received power at receivers
    received_power_per_receiver = jnp.array(get_measurements(scene, max_order, approx, alpha))

    # Difference between received powers and measurements in dB ( 10*log(p1) - 10*log(p2) = 10*log(p1/p2) )
    difference_of_dB = 10 * jnp.log10((received_power_per_receiver + DEFAULT_EPS_POWER) / (measurements + DEFAULT_EPS_POWER))
    
    # MSE
    err = jnp.sum(difference_of_dB**2)
    return err / measurements.size


@eqx.filter_jit
def gaussian_smooth_loss(
    tx_coords: Float[Array, "2"],
    scene: TriangleScene,
    measurements: Float[Array, " "],
    max_order: Int,
    sigma: float = 0.1, 
    N: int = 10,
    key: jax.random.PRNGKey = jax.random.PRNGKey(1234),
) -> Float:
    """
    Returns the value of the loss function evaluated at position tx_coords with Gaussian smoothing (LGS).

    Args:
        tx_coords: Coordinates of the transmitter.
        scene: TriangleScene containing the geometry and receivers' positions.
        measurements: The power measurements from the receivers.
        max_order: Maximum order of reflections to consider.
        sigma: Standard deviation for the Gaussian smoothing of the power.
        N: Number of samples for the Gaussian smoothing of the power.
        key: Random key for generating samples.

    Returns:
        The value of the loss function.
    """
    # Set transmitter position
    scene = eqx.tree_at(
        lambda s: s.transmitters, scene, scene.transmitters.at[:2].set(tx_coords)
    )
    # Generate normal samples
    v = jax.random.normal(key, (N, *tx_coords.shape))
    u = sigma * v

    # Compute estimate of the GS-loss function
    tx_translated = (tx_coords - u).clip(-5.0 + DEFAULT_EPS_ROOM, 5.0 - DEFAULT_EPS_ROOM)
    loss_vals = jax.vmap(lambda x: loss(x, scene, measurements, max_order), in_axes=(0))(tx_translated)
    f = jnp.mean(loss_vals, axis=0)

    return jnp.array(f)


@eqx.filter_jit
def smooth_loss_value_and_grad(tx_coords, measurements, scene, max_order, sigma=0.01, N=10, key=jax.random.PRNGKey(1234), 
                               symmetric=True, forward_findiff=False, stocha_rx=False, nb_rx=9, receivers=None, key_rx=jax.random.PRNGKey(1234)):
    """
    Returns the value of the loss function and its gradient evaluated at position tx_coords with Gaussian smoothing (LGS).

    Args:
        tx_coords: Coordinates of the transmitter.
        measurements: The power measurements from the receivers.
        scene: TriangleScene containing the geometry and receivers' positions.
        max_order: Maximum order of reflections to consider.
        sigma: Standard deviation for the Gaussian smoothing of the power.
        N: Number of samples for the Gaussian smoothing of the power.
        key: Random key for generating samples.
        symmetric: Whether to use symmetric sampling for the gradient estimation.
        forward_findiff: Whether to use forward finite differences for the gradient estimation.
        stocha_rx: Whether to select a random subset of receivers for the loss computation.
        nb_rx: Number of receivers to select if stocha_rx is True.
        receivers: The coordinates of the receivers to sample from if stocha_rx is True.
        key_rx: Random key for selecting the receivers if stocha_rx is True.
        
    Returns:
        f: The value of the loss function.
        df: The gradient of the loss function.
    """
    # Set transmitter position
    scene = eqx.tree_at(
        lambda s: s.transmitters, scene, scene.transmitters.at[:2].set(tx_coords)
    )
    if stocha_rx:
        select_idx = jax.random.permutation(key_rx, receivers.shape[0])[:nb_rx]
        truncated_receivers = receivers[select_idx]
        meas = measurements[select_idx]
        scene = eqx.tree_at(
            lambda s: s.receivers, scene, scene.receivers.at[:].set(truncated_receivers)
        )
    else:
        meas = measurements

    # Generate normal samples
    if forward_findiff:
            v= jax.random.normal(key, (N//2, *tx_coords.shape))
    else:
        if symmetric:
            v = jax.random.normal(key, (N//2, *tx_coords.shape))
            v = jnp.concatenate((v, -v), axis=0)
        else:
            v = jax.random.normal(key, (N, *tx_coords.shape))
    u = sigma * v

    tx_translated = (tx_coords - u).clip(-5.0 + DEFAULT_EPS_ROOM, 5.0 - DEFAULT_EPS_ROOM)
    loss_vals = jax.vmap(lambda x: loss(x, scene, meas, max_order), in_axes=(0))(tx_translated)

    if forward_findiff:
        t_loss_val = loss(tx_coords, scene, meas, max_order)

    f = jnp.mean(loss_vals)

    # Compute the estimate of the gradient
    loss_vals = loss_vals[:, None]

    if forward_findiff:
        df = jnp.mean((loss_vals - t_loss_val) * (-u) / sigma**2, axis=0)
    else:
        df = jnp.mean(loss_vals * (-u) / sigma**2, axis=0)

    return f, df


@eqx.filter_jit
def transmitters_power(tx_coords, scene, max_order):
    """
    Function used in vectorized map of function tt_smooth_loss_grad.
    """
    scene = eqx.tree_at(
        lambda s: s.transmitters, scene, scene.transmitters.at[:2].set(tx_coords)
    )
    powers = jnp.array(get_measurements(scene, max_order))
    return powers


@eqx.filter_jit
def tt_smooth_loss_grad(
    tx_coords: Float[Array, "2"],
    scene: TriangleScene,
    measurements: Float[Array, " "],
    max_order: Int, 
    sigma: float = 0.1,
    N: int = 50, 
    key: jax.random.PRNGKey = jax.random.PRNGKey(1234),
    symmetric: bool = True,
    forward_findiff: bool = False,
) -> tuple[Float, Float]:
    """
    Returns the gradient of the loss function evaluated at position tx_coords with Gaussian smoothing (GS).

    Args:
        tx_coords: Coordinates of the transmitter.
        scene: TriangleScene containing the geometry and receivers' positions.
        measurements: The power measurements from the receivers.
        max_order: Maximum order of reflections to consider.
        sigma: Standard deviation for the Gaussian smoothing of the power.
        N: Number of samples for the Gaussian smoothing of the power.
        key: Random key for generating samples.
        symmetric: Whether to use symmetric sampling for the gradient estimation.
        forward_findiff: Whether to use forward finite differences for the gradient estimation.

    Returns:
        df: The gradient of the loss function.
    """
    scene = eqx.tree_at(
        lambda s: s.transmitters, scene, scene.transmitters.at[:2].set(tx_coords)
    )
    key1, key2 = jax.random.split(key)
    m = scene.receivers.shape[0]

    # Generate normal samples (serie 1)
    if symmetric:
        v = jax.random.normal(key1, (N//2, *tx_coords.shape))
        v = jnp.concatenate((v, -v), axis=0)
    else:
        v = jax.random.normal(key1, (N, *tx_coords.shape)) # (N, 2)
    u = sigma * v

    # Get received power at perturbed receivers
    eps = 1e-3
    tx_translated = (tx_coords - u).clip(-5.0+eps, 5.0-eps)
    powers = jax.vmap(lambda x: transmitters_power(x, scene, max_order), in_axes=(0))(tx_translated) # (N, m)

    # Compute estimate of the gradient of MSE evaluated at the measurements with smoothed powers
    a_bar = jnp.mean(powers, axis=0) 
    a = 2 / m * 10 * jnp.log10((a_bar + DEFAULT_EPS_POWER) / (measurements + DEFAULT_EPS_POWER)) # (m,)

    # Generate normal samples (serie 2)
    if forward_findiff:
        v = jax.random.normal(key2, (1, *tx_coords.shape))
    else:
        if symmetric:
            v = jax.random.normal(key2, (N//2, *tx_coords.shape))
            v = jnp.concatenate((v, -v), axis=0)
        else:
            v = jax.random.normal(key2, (N, *tx_coords.shape)) # (N, 2)
    u = sigma * v

    # Get received power at perturbed receivers
    tx_translated = (tx_coords - u).clip(-5.0+eps, 5.0-eps)
    powers = jax.vmap(lambda x: transmitters_power(x, scene, max_order), in_axes=(0))(tx_translated) # (N, m)
    powers = 10 * jnp.log10((powers + DEFAULT_EPS_POWER)) # in dB

    if forward_findiff:
        t_power = transmitters_power(tx_coords, scene, max_order)
        t_power = 10 * jnp.log10((t_power + DEFAULT_EPS_POWER))
        powers_times_u = jnp.einsum("ij,ik->ijk", powers - t_power, -u/sigma**2)
    else:
        powers_times_u = jnp.einsum("ij,ik->ijk", powers, -u/sigma**2) # (N, m, 2)

    # Compute estimate of the jacobian of the smooth power measurements
    J = jnp.mean(powers_times_u, axis=0) # (m, 2)

    # Compute estimate of the gradient of the smooth loss function
    df = a @ J # (2,)

    return df