import jax
import jax.numpy as jnp
import equinox as eqx
from differt.scene import TriangleScene
from typing import Literal
import numpy as np

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from src.scenes import (basic_scene, choose_receivers)
from src.initialization import initialization
from src.utils import (get_measurements)
from src.losses import (loss, smooth_loss_value_and_grad, tt_smooth_loss_grad)
from src.optimization import (physics_smoothing_minimize, gaussian_smoothing_minimize, transmitter_smoothing_minimize)

# Regularly spaced receivers for testing in benchmarks.
rx1 = jnp.array([[0.0, 0.0]])
rx2 = jnp.array([[-3.0, 0.0], [3.0, 0.0]])
rx3 = jnp.array([[-3.0, 0.0], [3.0, 3.0], [3.0, -3.0]])
rx4 = jnp.array([[-3.0, 0.0], [3.0, 0.0], [0.0, -3.0], [0.0, 3.0]])
rx5 = jnp.array([[-3.0, 0.0], [3.0, 0.0], [0.0, -3.0], [0.0, 3.0], [0.0, 0.0]])
rx6 = jnp.array([[-3.0, 0.0], [3.0, 0.0], [-3.0, -3.0], [3.0, 3.0], [-3.0, 3.0], [3.0, -3.0]])
rx7 = jnp.array([[-3.0, 0.0], [3.0, 0.0], [-3.0, -3.0], [3.0, 3.0], [-3.0, 3.0], [3.0, -3.0], [0.0, 0.0]])
rx8 = jnp.array([[-3.0, 0.0], [3.0, 0.0], [-3.0, -3.0], [3.0, 3.0], [-3.0, 3.0], [3.0, -3.0], [0.0, 3.0], [0.0, -3.0]])
rx9 = jnp.array([[-3.0, 0.0], [3.0, 0.0], [-3.0, -3.0], [3.0, 3.0], [-3.0, 3.0], [3.0, -3.0], [0.0, 3.0], [0.0, -3.0], [0.0, 0.0]])

def find_emitter(
    scene: TriangleScene, 
    measurements: jnp.ndarray,
    nb_init: int = 5, 
    smooth_method: Literal["gaussian", "physics", "transmitter"] = "gaussian",
    nb_iter: int = 100, 
    N: int = 100,
    sym_sampling: bool = True,
    fin_diff: bool = False,
    stochastic_rx: bool = False,
    nb_rx: int = 4,
    receivers: jnp.ndarray = rx9
):
    """
    Find the emitter position using an optimization method, with different smoothing techniques.

    Args:
        scene: TriangleScene containing the geometry and receivers' positions.
        measurements: Power measurements at the receivers.
        nb_init: Number of initial candidates to consider.
        smooth_method: Smoothing method to use for optimization.
        nb_iter: Number of optimization steps.
        N: Number of Monte Carlo samples for Gaussian smoothing.
        sym_sampling: Whether to use symmetric sampling for Gaussian smoothing.
        fin_diff: Whether to use finite difference for gradient computation.
        stochastic_rx: Whether to use a stochastic subset of receivers (only added for Gaussian smoothing).
        nb_rx: Number of receivers to consider in the stochastic case.
        receivers: Receiver positions to use in the optimization (if stochastic_rx is True).

    Returns:
        success: Boolean indicating if the distance from the candidate to the true minimizer was below 0.3.
        success_idx: Index of the successful initialization.
        tx_list: List of transmitter positions found during optimization.
        loss_list: List of loss values corresponding to each transmitter position.
    """

    scene = scene.set_assume_quads()
    success = False
    success_idx = -1
    tx_list = []
    loss_list = []
    true_tx = scene.transmitters[:-1]

    # Compute the initial points for the optimization
    init_points = initialization(scene, measurements)

    for exp in range(nb_init):

        # If there wasn't enough possible initial points, continue
        if exp >= len(init_points):
            tx_list.append(jnp.zeros(2))
            loss_list.append(1e10)
            continue

        # Apply optimization method
        x0 = init_points[exp]

        # LGS
        if smooth_method == "gaussian":
            if stochastic_rx:
                scene = eqx.tree_at(lambda s: s.receivers, scene, receivers[:nb_rx])

            x, losses = gaussian_smoothing_minimize(smooth_loss_value_and_grad, x0, args=(measurements, scene, 2), steps=nb_iter,
                                                    sigmas=0.01 * jnp.ones(nb_iter),
                                                    N=N, optimizer=optax.sgd(0.01), symmetric=sym_sampling, fin_diff=fin_diff, 
                                                    stocha_rx=stochastic_rx, nb_rx=nb_rx, receivers=receivers)
            tx = x[jnp.argmin(losses)] # iterate with minimum loss

        # TTS
        elif smooth_method == "transmitter":
            x, losses = transmitter_smoothing_minimize(tt_smooth_loss_grad, x0, args=(scene, measurements, 2), steps=nb_iter, 
                                               sigmas=0.01 * jnp.ones(nb_iter),
                                               N=N//2, optimizer=optax.chain(optax.sgd(0.01)), symmetric=sym_sampling, fin_diff=fin_diff)
            tx = x[jnp.argmin(losses)] # iterate with minimum loss

        # PBS
        elif smooth_method == "physics":
            x, losses = physics_smoothing_minimize(loss, x0, args=(scene, measurements, 2, True), steps=nb_iter, 
                                                   optimizer=optax.chain(optax.sgd(0.01), optax.zero_nans(), optax.add_noise(1e-7, 0.9, 0)), 
                                                   alphas=jnp.concatenate([jnp.logspace(2, 5, nb_iter - nb_iter//4), 1e10*jnp.ones(nb_iter//4)]))
            tx = x[jnp.argmin(losses)] # iterate with minimum loss
        
        # No smoothing method, vanilla GD
        elif smooth_method is None:
            x, losses = physics_smoothing_minimize(loss, x0, args=(scene, measurements, 2, False), steps=nb_iter, 
                                                   optimizer=optax.chain(optax.sgd(0.01), optax.zero_nans(), optax.add_noise(1e-7, 0.9, 0)))
            tx = x[jnp.argmin(losses)] # iterate with minimum loss

        scene = eqx.tree_at(lambda s: s.receivers, scene, receivers)
        tx_list.append(tx)
        loss_value = loss(tx, scene, measurements, 2)
        loss_list.append(loss_value)
    
    best_tx = tx_list[jnp.argmin(jnp.array(loss_list))]
    if jnp.linalg.norm(best_tx - true_tx) < 0.3:
        success = True
        success_idx = jnp.argmin(jnp.array(loss_list))

    return success, success_idx, tx_list, loss_list


def success_percentage(rxs, mesh, name_txt, name_npy, nb_experiments, seed, nb_receivers, 
                       smooth_method='gaussian', 
                       N_MonteCarlo=100, 
                       nb_iter=100,
                       sym_sampling=True,
                       fin_diff=False,
                       max_order_measurements=2, 
                       stochastic_rx=False,
                       nb_rx=4
                       ):
    """
    Evaluate the success percentage of emitter localization using a given number of experiments.

    Args:
        rxs: Receiver positions.
        mesh: TriangleMesh object representing the scene.
        name_txt: Name of the output text file for success rates.
        name_npy: Name of the output numpy file for estimates.
        nb_experiments: Number of experiments to run.
        seed: Random seed for reproducibility.
        nb_receivers: Number of receivers to use in the experiment.
        smooth_method: Smoothing method to use for optimization.
        N_MonteCarlo: Number of Monte Carlo samples for Gaussian smoothing.
        nb_iter: Number of optimization steps.
        sym_sampling: Whether to use symmetric sampling for Gaussian smoothing.
        fin_diff: Whether to use finite difference for gradient computation.
        max_order_measurements: Maximum order of reflections to consider.
        stochastic_rx: Whether to use a stochastic subset of receivers.
        nb_rx: Number of receivers to consider in the stochastic case.

    Returns:
        None
    """

    tx_positions = jax.random.uniform(jax.random.PRNGKey(seed), (nb_experiments, 2), minval=-5.0, maxval=5.0)
    nb_init = 5
    estimates = np.zeros((nb_experiments, 6, 3))
    i = nb_receivers

    with open(name_txt+".txt", "a") as f:
        total_success = 0

        for exp, tx in enumerate(tx_positions):
            if rxs is not None:
                rx = jnp.concatenate([rxs, jnp.zeros((rxs.shape[0], 1))], axis=1)
            else:
                rx = choose_receivers(exp, i, 10.0, 10.0, 5, 0.5)

            tx = jnp.concatenate((tx, jnp.array([0.0])))
            scene = TriangleScene(tx, rx, mesh)
            scene = eqx.tree_at(lambda s: s.transmitters, scene, tx)
            scene = eqx.tree_at(lambda s: s.receivers, scene, rx)

            measurements = get_measurements(scene, max_order_measurements)

            success, success_idx, tx_list, loss_list = find_emitter(
                scene, measurements, nb_init=nb_init, smooth_method=smooth_method, nb_iter=nb_iter, N=N_MonteCarlo, 
                sym_sampling=sym_sampling, fin_diff=fin_diff, stochastic_rx=stochastic_rx, nb_rx=nb_rx, receivers=rx
            )

            if success:
                total_success += 1

            estimates[exp, 0, :2] = tx[:2].tolist()
            estimates[exp, 1:, :2] = tx_list
            estimates[exp, 1:, 2] = loss_list
            print(f"Emitter position: {tx}, Estimated: {tx_list[0]}, {tx_list[1]}, {tx_list[2]}, {tx_list[3]}, {tx_list[4]}")
            print(f"Success: {success}, Success Index: {success_idx}")

        success_rate = total_success / len(tx_positions)
        f.write(f"{i} receivers: {success_rate:.2%}\n")
        np.save(name_npy+".npy", np.array(estimates))
