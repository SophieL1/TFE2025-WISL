from collections.abc import Callable
from functools import partial
from typing import Any, Concatenate, ParamSpec

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, ArrayLike, Num

from src.losses import loss
from src.utils import DEFAULT_EPS_ROOM

OptState = Any
P = ParamSpec("P")

@eqx.filter_jit
def physics_smoothing_minimize(
    fun: Callable[Concatenate[Num[Array, " n"], P], Num[Array, " "]],
    x0: Num[ArrayLike, "*batch n"],
    args: tuple[Any, ...] = (),
    steps: int = 100,
    optimizer: optax.GradientTransformation | None = None,
    ad_mode: str = "forward",
    alphas: Num[Array, " *batch"] = None,
) -> tuple[Num[Array, "*batch n"], Num[Array, " *batch"]]:
    """
    Minimize a scalar function of one or more variables.

    Function adapted from DiffeRT repository minimize function to allow for PBS smoothing and choice between forward and backward
    mode for the gradient computation.

    Args:
        fun: Function to be minimized.
        x0: Initial point for the optimization.
        args: Additional arguments to be passed to the function fun.
        steps: Number of optimization steps.
        optimizer: Optimizer to be used for the optimization. If None, defaults to Adam with learning rate 0.1.
        ad_mode: Automatic differentiation mode, either "forward" or "backward".

    Returns:
        x: Final point after optimization.
        losses: Loss value at each step.
    """

    if alphas is None:
        alphas1 = jnp.logspace(1, 10, steps//2)
        alphas2 = 1e10 * jnp.ones(steps//2)
        alphas = jnp.concatenate([alphas1, alphas2])

    x0 = jnp.asarray(x0)
    if x0.ndim > 1 and args:
        chex.assert_tree_has_only_ndarrays(args, exception_type=TypeError)
        chex.assert_tree_shape_prefix(
            (x0, *args), x0.shape[:-1], exception_type=TypeError
        )

    optimizer = optimizer or optax.adam(learning_rate=0.1)
    opt_state = optimizer.init(x0)

    if ad_mode == "backward":
        f_and_df = jax.value_and_grad(fun)
        for _ in x0.shape[:-1]:
            f_and_df = jax.vmap(f_and_df)

        def f(
            carry: tuple[Num[Array, "*batch n"], OptState],
            alpha: float,
        ) -> tuple[tuple[Num[Array, "*batch n"], OptState], Num[Array, " *batch"]]:
            x, opt_state = carry
            args_f = (*args, alpha)
            _, grads = f_and_df(x, *args_f)
            loss_val = loss(x, *args)

            updates, opt_state = optimizer.update(grads, opt_state)
            def normalize_if_needed(u):
                return jax.lax.cond(
                    jnp.linalg.norm(u) > 1.0,
                    lambda v: v / jnp.linalg.norm(v),
                    lambda v: v,
                    u
                )

            updates = normalize_if_needed(updates)
            x = x + updates
            x = jnp.clip(x, -5.0 + DEFAULT_EPS_ROOM, 5.0 - DEFAULT_EPS_ROOM) # Ensure the iterates stay in the domain
            carry = (x, opt_state)
            return carry, loss_val

    elif ad_mode == "forward":
        def f(
            carry: tuple[Num[Array, "*batch n"], OptState],
            alpha: float,
        ) -> tuple[tuple[Num[Array, "*batch n"], OptState], Num[Array, " *batch"]]:
            x, opt_state = carry
            args_f = (*args, alpha)

            def fct(tx_coords):
                return loss(tx_coords, *args_f)

            loss_val = fct(x)
            grads = jax.jacfwd(fct)(x).reshape(-1)
            updates, opt_state = optimizer.update(grads, opt_state)
            def normalize_if_needed(u):
                return jax.lax.cond(
                    jnp.linalg.norm(u) > 1.0,
                    lambda v: v / jnp.linalg.norm(v),
                    lambda v: v,
                    u
                )

            updates = normalize_if_needed(updates)
            x = x + updates
            x = jnp.clip(x, -5.0 + DEFAULT_EPS_ROOM, 5.0 - DEFAULT_EPS_ROOM) # Ensure the iterates stay in the domain
            carry = (x, opt_state)
            return carry, (x, loss_val)
        
    else:
        raise ValueError("ad_mode must be either 'forward' or 'backward'")
    
    (_, _), (all_x, losses) = jax.lax.scan(f, init=(x0, opt_state), xs=alphas, length=steps)

    return all_x, losses


@eqx.filter_jit
def gaussian_smoothing_minimize(
    smooth_f_and_df: Callable[Concatenate[Num[Array, " n"], P], Num[Array, " "]],
    x0: Num[ArrayLike, "*batch n"],
    args: tuple[Any, ...] = (),
    steps: int = 100,
    sigmas: Num[Array, " "] = None,
    N: int = 100,
    optimizer: optax.GradientTransformation | None = None,
    seed: int = 1234,
    symmetric: bool = True,
    fin_diff: bool = False,
    stocha_rx: bool = False,
    nb_rx: int = 9, 
    receivers: Num[Array, " *batch 3"] | None = None,
) -> tuple[Num[Array, "steps *batch n"], Num[Array, "steps *batch"]]:
    """
    Minimize a scalar function of one or more variables using Gaussian smoothing (GS). 
    
    Args:
        smooth_f_and_df: Function returning an estimate of the GS-loss and its gradient.
        x0: Initial point for the optimization.
        args: Additional arguments to be passed to the function smooth_f_and_df.
        steps: Number of optimization steps.
        sigmas: Standard deviations for the Gaussian smoothing. If None, defaults to 0.01 * ones(steps).
        N: Number of samples for the Gaussian smoothing.
        optimizer: Optimizer to be used for the optimization. If None, defaults to Adam with learning rate 0.1.
        seed: Random seed for reproducibility.
        symmetric: Whether to use symmetric sampling for the gradient estimation.
        fin_diff: Whether to use finite differences for the gradient estimation.
        stocha_rx: Whether to use stochastic subset of receivers for the gradient estimation.
        nb_rx: Number of receivers to consider if stocha_rx is True.
        receivers: Optional array of receivers' positions. If None, the function will use the receivers from the scene.
            If provided, it should be of shape (*batch, 3) where the last dimension contains the x, y, z coordinates of the receivers.
            Should be provided if `stocha_rx` is True.

    Returns:
        all_x: Array of shape (steps, 2) containing the iterates.
        all_losses: Array of shape (steps,) containing the loss value at each step.
    """
    # Necessary checks
    if sigmas is None:
        sigmas = 0.01 * jnp.ones(steps)

    x0 = jnp.asarray(x0)
    if x0.ndim > 1 and args:
        chex.assert_tree_has_only_ndarrays(args, exception_type=TypeError)
        chex.assert_tree_shape_prefix(
            (x0, *args), x0.shape[:-1], exception_type=TypeError
        )

    optimizer = optimizer or optax.adam(learning_rate=0.1)

    for _ in x0.shape[:-1]:
        smooth_f_and_df = jax.vmap(smooth_f_and_df)

    opt_state = optimizer.init(x0)

    # Iteration of the 1st order optimization algorithm
    def f(
        carry: tuple[Num[Array, "*batch n"], OptState],
        xs: tuple[Num[Array, ""], jax.Array],
    ) -> tuple[tuple[Num[Array, "*batch n"], OptState], tuple[Num[Array, "*batch n"], Num[Array, "*batch"]]]:
        (x, opt_state) = carry
        sigma, key = xs

        # Estimate of GS-loss and gradient
        args_f = (*args, sigma, N, key, symmetric, fin_diff, stocha_rx, nb_rx, receivers, key)
        loss, grads = smooth_f_and_df(x, *args_f) 

        # Optimization step
        updates, opt_state = optimizer.update(grads, opt_state)
        x = x + updates
        x = jnp.clip(x, -5.0 + DEFAULT_EPS_ROOM, 5.0 - DEFAULT_EPS_ROOM) # Ensure the iterates stay in the domain

        carry = (x, opt_state)
        return carry, (x, loss)

    base_key = jax.random.PRNGKey(seed)
    keys = jax.random.split(base_key, steps)

    # Optimization
    (_, _), (all_x, all_losses) = jax.lax.scan(f, init=(x0, opt_state), xs=(sigmas, keys), length=steps)

    return all_x, all_losses


@eqx.filter_jit
def transmitter_smoothing_minimize(
    smooth_df,
    x0: Num[ArrayLike, "*batch n"],
    args: tuple[Any, ...] = (),
    steps: int = 100,
    sigmas: Num[Array, " "] = None,
    N: int = 50,
    optimizer: optax.GradientTransformation | None = None,
    seed: int = 1234,
    symmetric: bool = True,
    fin_diff: bool = False,
) -> tuple[Num[Array, "steps *batch n"], Num[Array, "steps *batch"]]:
    """
    Minimize a scalar function of one or more variables using Gaussian smoothing (GS). 
    
    Args:
        smooth_df: Function returning an estimate of the gradient of the TTS-loss.
        x0: Initial point for the optimization.
        args: Additional arguments to be passed to the function smooth_df.
        steps: Number of optimization steps.
        sigmas: Standard deviations for the Gaussian smoothing of the power. If None, defaults to 0.01 * ones(steps).
        N: Number of samples for the Gaussian smoothing of the power.
        optimizer: Optimizer to be used for the optimization. If None, defaults to Adam with learning rate 0.1.
        seed: Random seed for reproducibility.

    Returns:
        all_x: Array of shape (steps, 2) containing the iterates.
        all_losses: Array of shape (steps,) containing the loss value at each step.
    """
    if sigmas is None:
        sigmas = 0.01 * jnp.ones(steps)

    x0 = jnp.asarray(x0)
    if x0.ndim > 1 and args:
        chex.assert_tree_has_only_ndarrays(args, exception_type=TypeError)
        chex.assert_tree_shape_prefix(
            (x0, *args), x0.shape[:-1], exception_type=TypeError
        )

    optimizer = optimizer or optax.adam(learning_rate=0.1)

    for _ in x0.shape[:-1]:
        smooth_df = jax.vmap(smooth_df)

    opt_state = optimizer.init(x0)

    def f(
        carry: tuple[Num[Array, "*batch n"], OptState],
        xs: tuple[Num[Array, ""], jax.Array],
    ) -> tuple[tuple[Num[Array, "*batch n"], OptState], tuple[Num[Array, "*batch n"], Num[Array, " *batch"]]]:
        x, opt_state = carry
        sigma, key = xs
        args_f = (*args, sigma, N, key, symmetric, fin_diff)
        grads = smooth_df(x, *args_f)
        updates, opt_state = optimizer.update(grads, opt_state)
        x = x + updates
        x = jnp.clip(x, -5.0 + DEFAULT_EPS_ROOM, 5.0 - DEFAULT_EPS_ROOM) # Ensure the iterates stay in the domain
        loss_val = loss(x, *args)
        carry = (x, opt_state)
        return carry, (x, loss_val)

    base_key = jax.random.PRNGKey(seed)
    keys = jax.random.split(base_key, steps)

    (_, _), (all_x, all_losses) = jax.lax.scan(f, init=(x0, opt_state), xs=(sigmas, keys), length=steps)

    return all_x, all_losses