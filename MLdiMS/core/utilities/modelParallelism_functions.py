
import functools
from pprint import pprint
from typing import Any, Callable, Dict, Tuple, List, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core.frozen_dict import FrozenDict
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from ml_collections import ConfigDict

from .parallelism_functions import (
        fold_rng_over_axis,
        stack_weights,
        unstack_weights,
)

PyTree = Any
Parameter = jax.Array | nn.Partitioned


##############################################################
## Wrapper function to shard nn.Module for (Tensor,Pipeline)
## Model Parallelism. Wrapper adds sharding over the model axis 
## to the model parameters and initializes the module with 
## different parameters across the model aixs
##############################################################

class ModelParallelism(nn.Module):
    """ Wrapper class for applying modelParallelism to nn.Module 

    args:
      axis_name : name of the axis over which model parameters are sharded
      module_fn : Callable function returning flax.nn.Module to wrap
      nonmask_model_idx : if not None, nonmask_model_idx'shard will be non-zero; used for 
                          controlling output/input of the sharded module
      create_rngs_for_shard_axis: if true, create separate RNGS across the model axis
      module_kwargs: if not None, additional args for nn.Module function

    """

    module_fn: Callable[...,nn.Module]
    nonmask_model_idx: int | None = None
    create_rngs_for_shard_axis: bool = True
    module_kwargs: FrozenDict[str,Any] =  FrozenDict({})
    shard_axis_name: str = "model"


    #@shard_decorator("ModelParallelism",attrgetter('shard_axis'),attrgetter('module_fn'))
    @nn.compact
    def __call__(self, *args,**kwargs):
        if self.is_initializing() and self.create_rngs_for_shard_axis:
            #initialize each module across the model axis with different parameters
            self.scope.rngs["params"] = self.scope.rngs["params"].replace(
                rng=fold_rng_over_axis(self.scope.rngs["params"].rng, self.shard_axis_name))

        #wrap variables in nn.Partitioned objecs to add sharding over the model axis
        #map_variables: transformation function applied on parameters before and after the
        #               module is called. gather parameters before the nn.module is called
        #               shard parameters after the nn.module is called
        #pprint(self.module_kwargs)
        module = nn.map_variables(
            target=functools.partial(
                self.module_fn,
                name="sharded",
                **self.module_kwargs,
            ),
            trans_in_fn = functools.partial(unstack_weights, shard_axis_name=self.shard_axis_name),
            trans_out_fn = functools.partial(stack_weights,shard_axis_name=self.shard_axis_name,nonmask_model_idx=self.nonmask_model_idx,),
            mapped_collections="params",
            mutable=True,
        )()
        return module(
            *args,
            **kwargs,
        )



#step_pipeline implementation
# two step approach: do stages on micro-batch (input) and communicate the last output between devices to get the next input
# communcaiton is dones using jax.lax.ppermute transfers an array over a ring stsage 1 sends its output to stage 2,,..

#nn.scan implementing pipeline loop, keeps the parameters for all steps in device but allows for differnt inputs and outputs

def execute_pipeline_step(
    module: nn.Module,
    state: jax.Array,
    input: jax.Array,
    *args,
    shard_axis_name: str,
    **kwargs,
) -> Tuple[jax.Array,jax.Array]:

    """ single micro-batch pipeline step

    Args:
        module: flax module representing the stage to execute
        state:  last state of the stage used as input to the device stages for all stages except the first.
        input: original micro-batch input to the pipeline stage
        *args: additional arguments to the module
        shard_axis_name: name of the model axis in the mesh/shard_map
        **kwargs: Additiona keyword arguments to the module

    Returns:
       Tuple of the new state

    """

    num_stages = jax.lax.psum(1,shard_axis_name)
    stage_index = jax.lax.axis_index(shard_axis_name)


    #conditional output , stage=0 input is micro-batch 
    #stage>0 input is previous stage output
    state = jnp.where(stage_index == 0, input,state)
    state = module(state, *args, **kwargs)
    
    # for the last stage , we return the state as output
    # for all other stages we return zeors
    #conditional output, for all stages except last stage, output = 0, 
    output = jnp.where(
        stage_index == num_stages -1,
        state,
        jnp.zeros_like(state),
    )

    ##communicate the last state to the next stage
    state = jax.lax.ppermute(
        state,
        model_axis_name,
        perm=[(i, (i+1)%num_stages) for i in range(num_stages)]
    )

    return(state,output)

#implement pipeline using execute_pipeline_step as a function, microbatch as input vector to nn.scan

@jax.named_scope("pipeline")
def execute_pipeline(
        module : nn.Module,
        x: jax.Array,
        *args, 
        num_microbatches: int,
        shard_axis_name: str,
        **kwargs
) -> jax.Array:

    """
    Execute Model pipeline of #stages on a batch of data.
    split batch into #micro-batches of micro-batch size and runnig the stages in parallel

    Args:
       Module : flax nn.Module
       x: input
       args: additional arguments to th emodule
       num_microbatches : number of micor-batches to split the batch into.
       shard_axis_name: name of th emodl axis in the mesh/shard map
       **kwargs : additional keyword argument

    returns:
       output of the last stage of the pipeline

    """

    num_stages = jax.lax.psum(1, shard_axis_name)

    #shape of the input data
    batch_size = x.shape[0]

    assert(  batch_size % num_microbatches == 0 ), f"Batch size {batch_size} must be divisible of num_microbatches {num_microbatches}"

    microbatch_size = batch_size//num_microbatches
    #reshape batch_size, _,_ into num_microbatches,microbatch_size, _,_
    microbatches = x.reshape(num_microbatches, microbatch_size, *x.shape[1:])
    inputs = jnp.concatenate( # concatenate zeros for unused slots for first stage
         [
            microbatches,
            jnp.zeros((num_stages-1, *microbatches.shape[1:]),dtype=x.dtype),
         ],
         axis=0,
    )

    #initial state for nn.scan 
    init_state = jnp.zeros_like(microbatches[0])
    #input vector 
    #microbatches
    #function()
    #module
    num_iterations = inputs.shape[0]
    _,outputs = nn.scan(
            #function with arguments using functools.partial
            functools.partial(
                execute_pipeline_step,
                *args,
                shard_axis_name=shard_axis_name,
                **kwargs
            ),
            variable_broadcast = {"params": True},
            split_rngs = {"params": False, "dropout": True},
            length=num_iterations,
            in_axes  = 0,
            out_axes = 0,
    )(module,init_state,inputs)

    #Take the last #num_microbatches of the output
    outputs = jnp.concatenate(outputs[-num_microbatches:],axis=0)
    return outputs

class PipelineModule(nn.Module):
    """
       constructor class for placing nn.Module into pipeline and executing
       before calling PipelineModule sharding of parameters over model parallelism
       axis should be done
       args:
        axis_name : str name of the axis over which model parameters are sharded
        module_fn : Callable[..., nn.Module]
        num_microbatches: int  number of microbatches for the pipeline
    """
    axis_name: str
    num_microbatches: int
    module_fn: Callable[...,nn.Module]

    @nn.compact
    def __call__(self, *args, **kwargs):
        module = self.module_fn()
        out = execute_pipe(
            module,
            *args,
            **kwargs,
            num_microbatches=self.num_microbatches,
            shard_axis_name=self.shard_axis_name,
        )
        return out

#utilities for Async gather communication/compute overlap
#communicate each device (subset of) inputs to next device while doing 
#computation using jax.lax.ppermute function

#communication example for devices[4] 
#------------------------------------------------------------------
#        step 0    step 1     step 2     step 3
#
# GPU0  [x0    ]  [x0x3  ]   [x0x3x2 ]  [x0x3x2x1]
#
# GPU1  [x1    ]  [x1x0  ]   [x1x0x3 ]  [x1x0x3x2]
#
# GPU2  [x2    ]  [x2x1  ]   [x2x1x0 ]  [x2x1x0x3]
#
# GPU3  [x3    ]  [x3x2  ]   [x3x2x1 ]  [x3x2x1x0]


def async_gather(x: PyTree,
                 shard_axis_name: str,
                 shift_left: bool = True ) -> List[PyTree]:

    """ All gather using ring permutation..

    Args:
       x: input to gather
       shard_axis_name: the axis name to gather along
       shift_left: shift left or right (left device 0->1  right device 1->0

    returns List of gathered inputs.

    """

    num_devices = jax.lax.psum(1,shard_axis_name)

    #communcation pattern
    if shift_left == True:
        shift_permute = [(p , (p+1)%num_devices) for p in range(num_devices)]
    else:    
        shift_permute = [(p , (p-1)%num_devices) for p in range(num_devices)]

    inp_stack = [x]
    inp = x
    for idx in range(1, num_devices):
        inp = jax.lax.ppermute(inp,shard_axis_name,perm=shift_permute) 
        inp_stack.append(inp)
    return inp_stack


def async_gather_bidir(x: PyTree,
                 shard_axis_name: str,
                 shift_left: bool = True ) -> List[PyTree]:

    """ All gather using ring permutation..

    Args:
       x: input to gather
       shard_axis_name: the axis name to gather along
       shift_left: shift left or right (left device 0->1  right device 1->0), controls the order of 
                   gather tensors 

    returns List of gathered inputs.

    """

    num_devices = jax.lax.psum(1,shard_axis_name)

    #communcation pattern
    shift_up_permute  = [(p , (p+1)%num_devices) for p in range(num_devices)]
    shift_down_permute  = [(p , (p-1)%num_devices) for p in range(num_devices)]

    shift_up_stack = []
    shift_down_stack = []
    inp_up = x
    inp_down = x

    for idx in range(1,num_devices):
        if idx%2 == 0:
            inp_down = jax.lax.ppermute(inp_down,shard_axis_name,shift_down_permute)
            shift_down_stack.append(inp_down)
        else:
            inp_up = jax.lax.ppermute(inp_up,shard_axis_name,shift_up_permute)
            shift_up_stack.append(inp_up)

    #concatenate tensors on both directions
    if shift_left:
        inp = [x] + shift_up_stack + shift_down_stack[::-1]
    else:
        inp = [x] + shift_down_stack + shift_up_stack[::-1]

    return inp


## aysnc scatter
#  devices have all inputs to start computation and require outputs from other device to do reduction

def async_scatter(xs: Sequence[PyTree],
                  shard_axis_name: str,
                  shift_left: bool = True) -> PyTree:

    """ scatter sum using ring permutation..

    Args:
       x: input to gather
       shard_axis_name: the axis name to gather along
       shift_left: shift left or right (left device 0->1  right device 1->0), controls the order of 
                   gather tensors 

    return of scatter summed outputs.

    """

    num_devices = jax.lax.psum(1,shard_axis_name)

    assert (
        len(xs) == num_devices
    ), f"Number of shards needs to match axis size got {len(x) } with {shard_axis_name} number devices {num_devices}"

    if shift_left:
        shift_permute = [(j , (j+1)%num_devices) for j in range(num_devices)]
    else:    
        shift_permute = [(j , (j-1)%num_devices) for j in range(num_devices)]

    outp = xs[0]
    for x in xs[1:]:
        outp = jax.lax.ppermute(outp,shard_axis_name,perm=shift_permute)
        outp = jax.tree_map(jnp.add,outp,x)
    return outp


def async_scatter_bidir(xs: Sequence[PyTree],
     shard_axis_name: str) -> PyTree:

    """ scatter sum using ring permutation.. bi-directional

    Args:
       x: input to gather
       shard_axis_name: the axis name to gather along

    return of scatter summed outputs.

    """

    num_devices = jax.psum(1,shard_axis_name)

    assert (
        len(xs) == num_devices
    ), f"Number of shards needs to match axis size got {len(x) } with {shard_axis_name} number devices {num_devices}"

    def _splitInputs(outputs: Sequence[PyTree]): 
        return ( jax.tree_map(lambda x: x[..., : x.shape[-1]//2], outputs), jax.tree_map(lambda x: x[..., x.shape[-1]//2,:], outputs))

    shift_up_permute = [(p, (p+1)%num_devices) for p in range(num_devices)]
    shift_down_permute = [(p, (p-1)%num_devices) for p in range(num_devices)]

    out_up, out_down = _splitInputs(xs[0])
    for x in xs[1:]:
        y_up = jax.lax.ppermute(out_up,shard_axis_name,x)
        y_down = jax.lax.ppermute(out_down,shard_axis_name,x)
        inp_up , inp_down = _splitInputs(x)
        y_up = jax.tree_map(jnp.add,y_up,inp_up)
        y_down = jax.tree_map(jnp.add,y_down,inp_down)
    return jax.tree_map(lambda y1, y2: jnp.concatenate([y1, y2], axis=-1), y_up, y_down)

