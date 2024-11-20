"""MIT License.

Copyright (c) 2024 Phillip Lippe

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from typing import Any, Callable, Dict, Tuple, Sequence, Optional

from ml_collections import ConfigDict
import jax
import jax.numpy as jnp
import numpy as np
from flax.struct import dataclass
from flax.training import train_state
import flax.linen as nn
import functools

# Type aliases
PyTree = Any
Metrics = Dict[str, Tuple[jax.Array, ...]]
Parameter = jax.Array | nn.Partitioned


class TrainState(train_state.TrainState):
    rng: jax.Array


@dataclass
class Batch:
    inputs: jax.Array
    labels: jax.Array

def sync_gradients(
    grads: PyTree,
    shard_axis_names: Sequence[str],
) -> PyTree:
    """Synchronize gradients across devices.

    Gradients for parameters that are replicated over a given axis are averaged across devices.
    Parameters that are partitioned over a given axis are considered to already have a mean of
    the gradients on each device, and hence do not need to be altered.

    Args:
        grads: The gradients to synchronize.
        shard_axis_names: The axis names to synchronize gradients across.

    Returns:
        The gradients averaged over the specified axes if they are replicated.
    """

    def sync_grad(g: Parameter) -> Parameter:
        if isinstance(g, nn.Partitioned):
            replication_axis_names = [
                name for name in shard_axis_names if name not in jax.tree_util.tree_leaves(g.names)
            ]
            if len(replication_axis_names) == 0:
                # Parameters partitioned over all axes.
                return g
            else:
                # Average over remaining replicated axes.
                return g.replace(value=jax.lax.pmean(g.value, axis_name=replication_axis_names))
        else:
            # Parameters are replicated over all axes.
            return jax.lax.pmean(g, axis_name=shard_axis_names)
    return jax.tree.map(sync_grad, grads, is_leaf=lambda x: isinstance(x, nn.Partitioned))


def accumulate_gradients_loop(
    state: TrainState,
    batch: Batch,
    rng: jax.random.PRNGKey,
    num_minibatches: int,
    loss_fn: Callable,
) -> Tuple[PyTree, Metrics]:
    """Calculate gradients and metrics for a batch using gradient accumulation.

    Args:
        state: Current training state.
        batch: Full training batch.
        rng: Random number generator to use.
        num_minibatches: Number of minibatches to split the batch into. Equal to the number of gradient accumulation steps.
        loss_fn: Loss function to calculate gradients and metrics.

    Returns:
        Tuple with accumulated gradients and metrics over the minibatches.
    """
    batch_size = batch.inputs.shape[0]
    minibatch_size = batch_size // num_minibatches
    rngs = jax.random.split(rng, num_minibatches)
    # Define gradient function for single minibatch.
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    # Prepare loop variables.
    grads = None
    metrics = None
    for minibatch_idx in range(num_minibatches):
        with jax.named_scope(f"minibatch_{minibatch_idx}"):
            # Split the batch into minibatches.
            start = minibatch_idx * minibatch_size
            end = start + minibatch_size
            minibatch = jax.tree_map(lambda x: x[start:end], batch)
            # Calculate gradients and metrics for the minibatch.
            (_, step_metrics), step_grads = grad_fn(
                state.params, state.apply_fn, minibatch, rngs[minibatch_idx]
            )
            # Accumulate gradients and metrics across minibatches.
            if grads is None:
                grads = step_grads
                metrics = step_metrics
            else:
                grads = jax.tree_map(jnp.add, grads, step_grads)
                metrics = jax.tree_map(jnp.add, metrics, step_metrics)
    # Average gradients over minibatches.
    grads = jax.tree_map(lambda g: g / num_minibatches, grads)
    return grads, metrics


def accumulate_gradients_scan(
    state: TrainState,
    batch: Batch,
    rng: jax.random.PRNGKey,
    num_minibatches: int,
    loss_fn: Callable,
) -> Tuple[PyTree, Metrics]:
    """Calculate gradients and metrics for a batch using gradient accumulation.

    In this version, we use `jax.lax.scan` to loop over the minibatches. This is more efficient in terms of compilation time.

    Args:
        state: Current training state.
        batch: Full training batch.
        rng: Random number generator to use.
        num_minibatches: Number of minibatches to split the batch into. Equal to the number of gradient accumulation steps.
        loss_fn: Loss function to calculate gradients and metrics.

    Returns:
        Tuple with accumulated gradients and metrics over the minibatches.
    """
    batch_size = batch.inputs.shape[0]
    minibatch_size = batch_size // num_minibatches
    rngs = jax.random.split(rng, num_minibatches)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    def _minibatch_step(minibatch_idx: jax.Array | int) -> Tuple[PyTree, Metrics]:
        """Determine gradients and metrics for a single minibatch."""
        minibatch = jax.tree_map(
            lambda x: jax.lax.dynamic_slice_in_dim(  # Slicing with variable index (jax.Array).
                x, start_index=minibatch_idx * minibatch_size, slice_size=minibatch_size, axis=0
            ),
            batch,
        )
        (_, step_metrics), step_grads = grad_fn(
            state.params, state.apply_fn, minibatch, rngs[minibatch_idx]
        )
        return step_grads, step_metrics

    def _scan_step(
        carry: Tuple[PyTree, Metrics], minibatch_idx: jax.Array | int
    ) -> Tuple[Tuple[PyTree, Metrics], None]:
        """Scan step function for looping over minibatches."""
        step_grads, step_metrics = _minibatch_step(minibatch_idx)
        carry = jax.tree_map(jnp.add, carry, (step_grads, step_metrics))
        return carry, None

    # Determine initial shapes for gradients and metrics.
    grads_shapes, metrics_shape = jax.eval_shape(_minibatch_step, 0)
    grads = jax.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), grads_shapes)
    metrics = jax.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), metrics_shape)
    # Loop over minibatches to determine gradients and metrics.
    (grads, metrics), _ = jax.lax.scan(
        _scan_step, init=(grads, metrics), xs=jnp.arange(num_minibatches), length=num_minibatches
    )
    # Average gradients over minibatches.
    grads = jax.tree_map(lambda g: g / num_minibatches, grads)
    return grads, metrics


def accumulate_gradients(
    state: TrainState,
    batch: Batch,
    rng: jax.random.PRNGKey,
    num_minibatches: int,
    loss_fn: Callable,
    use_scan: bool = False,
) -> Tuple[PyTree, Metrics]:
    """Calculate gradients and metrics for a batch using gradient accumulation.

    This function supports scanning over the minibatches using `jax.lax.scan` or using a for loop.

    Args:
        state: Current training state.
        batch: Full training batch.
        rng: Random number generator to use.
        num_minibatches: Number of minibatches to split the batch into. Equal to the number of gradient accumulation steps.
        loss_fn: Loss function to calculate gradients and metrics.
        use_scan: Whether to use `jax.lax.scan` for looping over the minibatches.

    Returns:
        Tuple with accumulated gradients and metrics over the minibatches.
    """
    if use_scan:
        return accumulate_gradients_scan(
            state=state, batch=batch, rng=rng, num_minibatches=num_minibatches, loss_fn=loss_fn
        )
    else:
        return accumulate_gradients_loop(
            state=state, batch=batch, rng=rng, num_minibatches=num_minibatches, loss_fn=loss_fn
        )


def print_metrics(metrics: Metrics, title: str | None = None) -> None:
    """Prints metrics with an optional title."""
    metrics = jax.device_get(metrics)
    lines = [f"{k}: {v[0] / v[1]:.6f}" for k, v in metrics.items()]
    if title:
        title = f" {title} "
        max_len = max(len(title), max(map(len, lines)))
        lines = [title.center(max_len, "=")] + lines
    print("\n".join(lines))


def get_num_params(state: TrainState) -> int:
    """Calculate the number of parameters in the model."""
    return sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(state.params))


def loss_fn_tp(
    params: PyTree,
    apply_fn: Any,
    batch: Batch,
    rng: jax.Array,
    data_axis_name: str,
    model_axis_name: str,
) -> Tuple[jax.Array, Dict[str, Any]]:
    # Since dropout masks vary across the batch dimension, we want each device to generate a
    # different mask. We can achieve this by folding the rng over the data axis, so that each
    # device gets a different rng and thus mask.
    dropout_rng = fold_rng_over_axis(rng, (data_axis_name, model_axis_name))
    # Remaining computation is the same as before for single device.
    logits = apply_fn(
        {"params": params},
        batch.inputs,
        train=True,
        rngs={"dropout": dropout_rng},
    )
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch.labels)
    correct_pred = jnp.equal(jnp.argmax(logits, axis=-1), batch.labels)
    batch_size = np.prod(batch.labels.shape)
    # Mask out loss and accuracy for model devices except first one.
    model_idx = jax.lax.axis_index(model_axis_name)
    loss = jnp.where(model_idx != 0, 0.0, loss)
    correct_pred = jnp.where(model_idx != 0, False, correct_pred)
    batch_size = jnp.where(model_idx != 0, 0, batch_size)
    # Collect metrics and return loss.
    step_metrics = {
        "loss": (loss.sum(), batch_size),
        "accuracy": (correct_pred.sum(), batch_size),
    }
    loss = loss.mean()
    return loss, step_metrics

def init_tp(rng: jax.random.PRNGKey, x: jax.Array, model: nn.Module) -> TrainState:
    init_rng, rng = jax.random.split(rng)
    variables = model.init({"params": init_rng}, x, train=False)
    params = variables.pop("params")
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        rng=rng,
    )
    return state

def train_step_tp(
    state: TrainState,
    metrics: Metrics | None,
    batch: Batch,
    model_axis_name: str,
    data_axis_name: str,
    num_minibatches: int,
    loss_fn: Callable = loss_fn_tp,
) -> Tuple[TrainState, Metrics]:
    rng, step_rng = jax.random.split(state.rng)
    grads, step_metrics = accumulate_gradients(
        state,
        batch,
        step_rng,
        num_minibatches,
        loss_fn=functools.partial(loss_fn,data_axis_name=data_axis_name,model_axis_name=model_axis_name)
    )
    # Update parameters. We need to sync the gradients across devices before updating.
    with jax.named_scope("sync_gradients"):
        grads = sync_gradients(grads, (data_axis_name, model_axis_name))
    new_state = state.apply_gradients(grads=grads, rng=rng)
    # Sum metrics across replicas. Alternatively, we could keep the metrics separate
    # and only synchronize them before logging. For simplicity, we sum them here.
    with jax.named_scope("sync_metrics"):
        step_metrics = jax.tree_map(
            lambda x: jax.lax.psum(x, axis_name=(data_axis_name, model_axis_name)),
            step_metrics,
        )
    if metrics is None:
        metrics = step_metrics
    else:
        metrics = jax.tree_map(jnp.add, metrics, step_metrics)
    return new_state, metrics

def find_multiple(n: int, k: int) -> int:
    assert k > 0
    if n % k == 0:
        return n
    return n + k - (n % k)

def num_parameters(module: nn.Module, requires_grad: Optional[bool] = None) -> int:
    total = 0
    for p in module.parameters():
        if requires_grad is None or p.requires_grad == requires_grad:
            if hasattr(p, "quant_state"):
                # bitsandbytes 4bit layer support
                total += math.prod(p.quant_state.shape)
            else:
                total += p.numel()
    return total


def reset_parameters(module: nn.Module) -> None:
    """Calls `reset_parameters` on the module and all its submodules."""
    for mod in module.modules():
        if callable(getattr(mod, "reset_parameters", None)):
            mod.reset_parameters()
