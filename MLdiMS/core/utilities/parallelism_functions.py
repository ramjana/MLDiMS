
import functools
from typing import Any, Dict, Tuple, Callable, Sequence
from pprint import pprint


import numpy as np
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import lax
from flax.struct import dataclass
from flax.training import train_state
from jax.experimental.shard_map import shard_map

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

def scale_fn(initializer_fn: Callable,
             scale_factor: float = 1.0):

    """
       function used for parameter intialization of module tensor parallelism
       sharded.

       tensor parallelism use two strategies:
       Gather: 
            Input(s) are communicated to all devices, weights are sharded along
            first dimension , each device compute part of output
            need all_gather primitive to collect inputs before applying computation
       scatter:
            calculate sub-results of each input on output indepedently on each device,
            each device communiate partial results of output  to do final reduction 
            of each device to get final output
        for example:
           Weights= Aij
           inputs = xi
           gather:
              Aij = Ai/devicesxj
              x = all_gather(xj)
              yi = Aij dot x
           scatter:
              Aij = Aixj/devices
              yj(i) = Aixj dot xi
              yj = psum(yj(i))
              yi = sum(jy(i))

        scale the initialization of weights for scatter strategy by 1/num_devices
    """
    def _scale_fn(rng, *args, **kwargs):
        return scale_factor * initializer_fn(rng, *args, **kwargs)

    return _scale_fn


def stack_weights(
    weights: PyTree,
    shard_axis_name: str,
    axis: int = 0,
    nonmask_model_idx: jax.Array | int | None = None) -> PyTree:

    """ Stacks sharded parameters along a given axis name

    args:
       weights: PyTree of weights 
       shard_axis_name : name of the axis to be stack along
       axis: index of axis to stack along
       nonmask_model_idx: If not None, only the mask_except 
    """ 
    def _stack(x: Parameter) -> Parameter:
        if isinstance(x, nn.Partitioned):
            value, names = x.value, x.names
        else:
            value, names = x, (None,)*x.ndim

        if nonmask_model_idx is not None:
            axis_index = jax.lax.axis_index(shard_axis_name)
            value = jnp.where(axis_index == nonmask_model_idx,value,0.0)
        value = jnp.expand_dims(value,axis)
        names = names[:axis] + (shard_axis_name,) + names[axis:]
        return nn.Partitioned(value,names=names)

    return jax.tree.map(_stack,weights, is_leaf=lambda x: isinstance(x,nn.Partitioned))

def unstack_weights(
    weights: PyTree, shard_axis_name: str) -> PyTree:
    """  unstacks weights along a given axis name

    args:
       params: PyTree of parameters
       shard_axis_name: name of the axis to unstack along
    """

    def _unstack(x: Parameter) -> Parameter:
        if isinstance(x, nn.Partitioned) and shard_axis_name in x.names:
            value = x.value
            names = x.names
            axis_idx = names.index(shard_axis_name)
            value = value.squeeze(axis_idx)
            names = names[:axis_idx] + names[axis_idx+1:]
            if all([n is None for n in names]):
                return value
            else:
                return nn.Partitioned(value,names=names)
        else:
            return x
    return jax.tree.map(_unstack,weights,is_leaf=lambda x: isinstance(x,nn.Partitioned))

def shard_module_weights(
    target: nn.Module | Callable,
    shard_axis_name: str,
    min_weight_size: int = 2**10) -> nn.Module | Callable:

    """shard parameters for module across replicas
    Args:
       target: the modulle to shard
       shard_axis_name: axis name to shard over
       weight_size: minimum shard size , parameters with fewer values will not be sharded
    """

    return nn.map_variables(
        target,
        trans_in_fn=functools.partial(gather_weights,shard_axis_name=shard_axis_name),
        trans_out_fn=functools.partial(
             shard_weights,
             shard_axis_name=shard_axis_name,
             min_size=min_weight_size,
        ),
        mapped_collections="params",
        mutable=True
    )

##weights sharding

#setup sharding of weights
#shard weights function takes parameters pytee and axis name . function determines
#sharding axis for each parameter. To scale for different model sizes , pick axis that has largest
# devices

@jax.named_scope("shard_weights")
def shard_weights(weights : PyTree, shard_axis_name: str , min_size: int = 2**13) -> PyTree:
    """shard weights across provided mesh axis name
      weights: pytree weights of the model
      shard_axis_name : mesh axis over weights are sharded
      min_size : minimum weights size for weights sharding
    """

    # get Id(s) of sharded compute resources , axis_name = sharded axis name
    axis_idx = jax.lax.axis_index(shard_axis_name)
    # total number of compute resources
    shard_axis_size = jax.lax.psum(1,shard_axis_name)
    def split_fn(x: Parameter) -> Parameter:
        #if already partitioned  do nothing
        #print(f"Called axisname = {shard_axis_name}")
        if isinstance(x,nn.Partitioned):
            value, names = x.value, x.names
        else:
            value  = x
            #create tuple of input weights shape (none,none,none) for three dimensional weights shape
            names  = (None,)*value.ndim
        if shard_axis_name in names:
            print(f"Parameter {value.shape} with names {names} already has sharded on axis {shard_axis_name}")
            return x
        elif value.size < min_size:
            print(f"Parameter {value.shape} with names {names} too small to shard, size {value.size} < {min_size}")
            return x
        else:
            shape = value.shape
            #sort weight dimension(S) in reverse order  (2,1,0) first one is fast changing dimension
            idxes = np.argsort(shape)[::-1]
            #print(f"shard {value.shape} with names {names} on axis {shard_axis_name}")
            #print(f"axis-size = {shard_axis_size}")
            #print(idxes)
            #iterate through dimensions
            for i in idxes:
                #choose largest dimension (in size) to shard
                if shape[i] % shard_axis_size == 0 and names[i] is None:
                    #print("hello....")
                    split_size = shape[i]//shard_axis_size
                    #nn.Partitioned partiton parameter in the axis, split size,
                    p_sharded = nn.Partitioned(
                        value = lax.dynamic_slice_in_dim(
                            value, axis_idx*split_size,split_size,axis=i
                        ),
                        names = names[:i] + (shard_axis_name,) + names[i+1:],
                    )
                    #print(p_sharded.names)
                    return p_sharded
            print(f"Could not shard {value.shape} with names {names} on axis {shard_axis_name} no suitable axis found")
            return x
    return jax.tree_util.tree_map(
            split_fn,
            weights,
            is_leaf = lambda x: isinstance(
                x,nn.Partitioned
            ),
    )

#function to gather shard pararmeters for two reasons.
#1. forward pass last layer to porject output from single device
#2. in backward pass non shard parameter are mean across sharded axis and scatter back

def gather_array_with_mean_grads(gradients: jax.Array, axis: int, shard_axis_name: str):
    """Gathering with averaging gradients across replicas."""
    axis_size = jax.lax.psum(1, shard_axis_name)

    # Define a custom gradient for the gather operation.
    @jax.custom_gradient
    def f(grads):
        def grad_fn(g):
            # pmean_scatter
            # average the sharded parameter gradients and return back to compute resources
            return (
                jax.lax.psum_scatter(g, shard_axis_name, scatter_dimension=axis, tiled=True) / axis_size
            )
        #gather all non sharded parameter gradients from sharded dimension to average them
        return jax.lax.all_gather(grads, shard_axis_name, axis=axis, tiled=True), grad_fn

    return f(gradients)

#gather function to get parameters back to single device
@jax.named_scope("gather_weights")
def gather_weights(weights: PyTree, shard_axis_name: str) -> PyTree:
    """Gather parameters from all replicas across the given axis.
        weights: The parameters to gather.
        shard_axis_name: The axis to gather parameters across.
    Returns:
        PyTree of same structure as params, but with leaves gathered if they were a nn.Partitioned object.
    """

    def gather_fn(p: Parameter) -> Parameter:
        if isinstance(p, nn.Partitioned) and shard_axis_name in p.names:
            weight_shard = p.names
            shard_axis = weight_shard.index(shard_axis_name)
            value = gather_array_with_mean_grads(p.value, axis=shard_axis, shard_axis_name=shard_axis_name)
            # If there are any other axes that are sharded, we need to keep the partitioned structure.
            # Otherwise, we can return the value directly.
            weight_shard = weight_shard[:shard_axis] + (None,) + weight_shard[shard_axis + 1 :]
            if any([name is not None for name in weight_shard]):
                return nn.Partitioned(value, weight_shard)
            else:
                return value
        else:
            return p

    return jax.tree_util.tree_map(gather_fn, weights, is_leaf=lambda x: isinstance(x, nn.Partitioned))

#@functools.partial(shard_map, mesh=mesh, in_specs=P("i", "j"), out_specs=P("i", "j"))
def parallel_normalize(x: jax.Array, shard_axis_name) -> jax.Array:
    mean = jax.lax.pmean(x, axis_name=shard_axis_name)
    std = jax.lax.pmean((x - mean) ** 2, axis_name=shard_axis_name) ** 0.5
    return (x - mean) / std

def split_array_over_mesh(x: jax.Array, axis_name: str, split_axis: int) -> jax.Array:
    axis_size = jax.lax.psum(1, axis_name)
    axis_index = jax.lax.axis_index(axis_name)
    slice_size = x.shape[split_axis] // axis_size
    x = jax.lax.dynamic_slice_in_dim(
        x,
        axis_index * slice_size,
        slice_size,
        axis=split_axis,
    )
    return x

def prep_module(
        layer: Callable[..., nn.Module],
        layer_name: str,
        fsdp_modules: tuple,
        shard_size: int,
        shard_parameter:bool,
        axis_name: str = "data",
        checkpoint_en: bool = False,
) -> Callable[...,nn.Module]:

    """ prepares module for sharding and checkpointing parameters

    function to prepare layer function in a remat/checkpoint and/or sharding parameter 

    args:
      layer: layer to prepare
      layer_name: name of the layer
      shard_parameter: shard parameter (bool yes or no)
      modules: tuple containing layer names for data/tensor sharding
      checkpoint_en: remat/checkpoint for memory foot print optimization
      shard_size: minimun weight size to shard

    """

    #axis_name = "data"
    #shard parameters over shard_axis nam
    if shard_parameter and layer_name in fsdp_modules:
        #print(f" prep_module::layer_name= {layer_name}")
        #print(f" prep_module::shard_parameter = {shard_parameter}")
        #print(f" prep_module::axis_name = {axis_name}")
        layer = shard_module_weights(
                layer, shard_axis_name=axis_name,min_weight_size=shard_size
        )
    if checkpoint_en:
        layer = nn.remat(layer, prevent_cse=False)
 
    return layer

def fold_rng_over_axis(rng: jax.random.PRNGKey, axis_name: str) -> jax.random.PRNGKey:
    """Folds the random number generator over the given axis.

    This is useful for generating a different random number for each device
    across a certain axis (e.g. the model axis).

    Args:
        rng: The random number generator.
        axis_name: The axis name to fold the random number generator over.

    Returns:
        A new random number generator, different for each device index along the axis.
    """
    axis_index = jax.lax.axis_index(axis_name)
    return jax.random.fold_in(rng, axis_index)


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
