
import yaml
import json
import functools


import numpy as np
import jax.numpy as jnp

import jax
from jax.experimental import mesh_utils

import optax

import flax
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
from flax.training import train_state

def create_device_mesh(config, devices=None):
    # create device mesh based on input configuration
    if devices is None:
        devices = jax.devices()

    num_devices = len(devices)
    num_slices = config.num_slices
    assert(num_slices == 1)

    device_parallelism = [config.data_parallelism, config.fsdp_parallelism, config.sequence_parallelism, config.tensor_parallelism, config.append_parallelism]

    mesh = mesh_utils.create_device_mesh(
            device_parallelism,
            devices,
        )

    print(f"Num devices: {num_devices}, shape {mesh.shape}")
    return mesh

def unbox_logicallypartioned(boxed_pytree):
    """Unboxes the flax.LogicallyPartitioned pieces

    Args:
      boxed_pytree: a pytree that includes LogicallyPartitioned
      leaves.
    Returns:
      a pytree where all all LogicallyPartitioned leaves have been unboxed.
    """
    return jax.tree_util.tree_map(
        lambda x: x.unbox() if isinstance(x, flax.linen.spmd.LogicallyPartitioned) else x,
        boxed_pytree,
        is_leaf=lambda k: isinstance(k, flax.linen.spmd.LogicallyPartitioned),
    )


def init_decode_state(apply_fn, params):
    """Init train state with null opt state for decode."""
    state = train_state.TrainState(step=0, apply_fn=apply_fn, params=params, tx=None, opt_state={})
    return state

def init_kv_cache(model,config,rng):
    input_shape = (config.micro_batch_size_to_train_on, config.max_prefill_predict_len)

    model_vars = model.init(
            {"params": rng, "dropout": rng},
            jnp.ones(input_shape),
            jnp.ones(input_shape),
            mode="prefill",
    )

    return model_vars["kvcache"]

def get_kvcache_annotations(model,config,rng,mesh):
    
    with nn_partitioning.axis_rules(config.logical_axis_rules):
        init_kvcache_partial = functools.partial(init_kv_cache,model,config,rng)
        abstract_state = jax.eval_shape(init_kvcache_partial)

    state_logical_annotations = nn.get_partition_spec(abstract_state)

    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
        state_mesh_annotations = nn.logical_to_mesh(state_logical_annotations)

    return state_mesh_annotations



def init_training_state(apply_fn, params, tx):
    """Init train state with null opt state for decode."""
    state = train_state.TrainState.create(apply_fn=apply_fn, params=params, tx=tx)
    return state


def init_initial_state(model,config, key, tx, is_training:bool=False):
    """
    We pass in "static" objects like model, tx, config as JAX compares them by
    object hash, and instantiating them inside causes pjit top-level annotations
    to fail to match as pytree prefixes if we re-instantiate.

    Args: model, tx, config, is_training, key
    """
    input_shape = (config.micro_batch_size_to_train_on, config.max_seq_length)
    model_vars = model.init(
        {"params": key, "dropout": key},
        np.ones(input_shape, dtype=jnp.int32),
        np.ones(input_shape, dtype=jnp.int32),
    )
    #if is_training:
    #  return init_training_state(model.apply, model_vars, tx)
    state = init_decode_state(model.apply, model_vars)
    return state

def get_abstract_state(model,config,rng,mesh,tx:optax.GradientTransformation=None, is_training=False):

    """ get shaped abstrction of state """

    init_state_partial = functools.partial(init_initial_state, model, config, rng, tx, is_training)
    with nn_partitioning.axis_rules(config.logical_axis_rules):
        abstract_state = jax.eval_shape(init_state_partial)

    state_logical_annotations = nn.get_partition_spec(abstract_state)

    state_mesh_shardings = nn.logical_to_mesh_sharding(state_logical_annotations,mesh,config.logical_axis_rules)

    abstract_shared_state = jax.jit(init_state_partial,
        in_shardings=None,
        out_shardings=state_mesh_shardings).eval_shape()

    unboxed_abstract_shared_state = unbox_logicallypartioned(abstract_shared_state)

    with nn_partitioning.axis_rules(config.logical_axis_rules):
        state_mesh_annotations = nn.logical_to_mesh(state_logical_annotations)

    return ( unboxed_abstract_shared_state, state_mesh_annotations,state_mesh_shardings)



def setup_initial_state(model,config,rng,mesh,tx: optax.GradientTransformation =None, is_training=False):
    """initialize the model and optimizer

    Args:
       model : jax model
       config : config object
       tx: optx.gradientTransformation
       mesh: jax.mesh()
    returns:
       state : initialized model state
       state_mesh_annotations: the mesh annotations for the train state
    """

    if is_training == True:
        raise ValueError ("Training mode is not supported yet")

    _abstract_state, state_mesh_annotations, state_mesh_shardings = get_abstract_state(
        model, config, rng, mesh, tx, is_training)


    #initialization
    with nn_partitioning.axis_rules(config.logical_axis_rules):
        init_state_partial = functools.partial(init_initial_state, model, config, rng, tx) #, is_training)
        #state = init_initial_state(model, config, rng, tx, False)
        init_state_partial.__name__ = "initialize_state"
        state = jax.jit(
           init_state_partial,
           in_shardings=None,
           out_shardings=state_mesh_shardings,
        )(is_training=is_training)

    state = unbox_logicallypartioned(state)
    return state, state_mesh_annotations



def setup_decode_state(model, config, rng, mesh, checkpoint_manager):
    """Setup decode state by loading params from a checkpoint.
    Args:
      model: the flax model to initialize
      config: config object
      rng: jax.prng key
      mesh: jax.devices() mesh

    Returns:
      state: state with decode params loaded/initialized(random) from the checkpoint
      state_mesh_annotations: the mesh annotations for the state
    """
    if not config.parameters_path:
      # generate random params
      print(f"No decode checkpoint specified - generating random weights.")
      state, state_mesh_annotations = setup_initial_state(model, config, rng, mesh, None, False)
    else:
      assert(false, "Not supported yet")
      # Load params from checkpoint
      print(f"Loading decode params from {config.load_parameters_path}")
      unboxed_abstract_state, state_mesh_annotations, _ = get_abstract_state(model, config, rng, mesh, None, False)
      with nn_partitioning.axis_rules(config.logical_axis_rules):
        params = load_params_from_path(config.load_parameters_path, unboxed_abstract_state.params)
      state = init_decode_state(None, params)

    state = unbox_logicallypartioned(state)
    return state, state_mesh_annotations
