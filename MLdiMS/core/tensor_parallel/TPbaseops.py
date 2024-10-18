###############################################
# Tensor parallelism support for ML ops
###############################################

import functools
from typing import Any, Dict, Tuple, Callable, Literal
from pprint import pprint

import numpy as np

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import core

from ..utilities.parallelism_functions import split_array_over_mesh,shard_module_weights,scale_fn
from core.utilities.modelParallelism_functions import (
        ModelParallelism,
        async_gather,
        async_gather_bidir,
        async_scatter,
        async_scatter_bidir
    )
from ..layers.baseops import Dense,RMSNorm
from ..layers.blocks import MLPBlockInput,MLPBlockOutput,InputEmbedding

PyTree = Any
Parameter = jax.Array | nn.Partitioned

class TPDense(nn.Module):
    """ Tensor parallelism support for Dense op

       Args:
          dense_fn : Any Constructor function for denseOp
          shard_axis_name: str name of the model axis
          tp_strategy: str  Tensor parallelism strategy for dense op ["gather", "scatter","Auto"]
          kernel_init_scale_factor: int
          kernel_init_fn : Callable 
          skip_communication: bool 
          module_name: str Name of the module
    """

    dense_fn: Any
    shard_axis_name: str
    data_type: jnp.dtype
    tp_strategy: Literal["gather","scatter","Auto"] = "Auto"
    kernel_init_scale_factor : float = 1.0
    kernel_init_fn : Callable = nn.initializers.lecun_normal()
    skip_communication: bool = False
    module_name: str = "TPDense"

    @nn.compact
    def __call__(self,x: jax.Array) -> jax.Array:

       num_devices = jax.lax.psum(1,self.shard_axis_name)
       strategy = tp_strategy if num_devices>1 else "Auto"
       Dense_fn = functools.partial(
            ModelParallelism,
            shard_axis_name=self.shard_axis_name,
            module_fn = functools.partial(
                self.dense_fn,
                scale_fn(kernel_init_fn,kernel_init_scale_factor)
                ),
            name=module_name,
       )
        
       if _strategy == "Auto":
           ##vanilla dense op no sharding
           x = self.dense_fn(kernel_init=kernel_init_fn)(x)
           return x
       elif _strategy == "gather":
           ## gather communicate input x to all devices before computation
           if not skip_communication:
               x = jax.lax.all_gather(x,shard_axis_name,axis=-1,tiled=True)
           x = Dense_fn()(x)
       elif _strategy == "scatter":
           ## scatter perform denseop first and communicate results to all devices for 
           ## reduction
           x = Dense_fn()(x)
           if not skip_communication:
               x = jax.lax.psum_scatter(
                   x, shard_axis_name,scatter_dimension=x.ndim-1,tiled=True)
       else:
           raise ValueError(f"Unknown tensor parallelism strategy {_strategy}")

       return x

##########################################################################
# Tensor parallelism support for MLP input and output block
# gather primitive embedded in MLPinput to collect inputs before computation
# scatter primitive embedded in MLPouput to collect outputs of compute for reduction
##########################################################################

class TPMLPBlock(nn.Module):
     hidden_dim: int
     embedding_dim: int
     shard_axis_name: str
     data_type: jnp.dtype
     use_norm: bool = False
     use_bias: bool = True
     tp_strategy: Literal["gather","scatter"] = "gather"

     def setup(self):
         super().setup()

     @nn.compact
     def __call__(self, x: jax.Array) -> jax.Array:
         input_features = x[-1]
         assert(input_features == embedding_dim) 
         num_devices = jax.lax.psum(1,shard_axis_name)
         assert(num_devices>1)
         x = RMSNorm(data_type=self.data_type,name="MLPPrenorm")(x)
         ## model parallelism Dense function with async bi-directional communication
         x = TPDense(
             dense_fn = functools.partial(
                 MLPBLockInput,
                 features=self.hidden_dim//num_devices,
                 use_norm=self.use_norm,
                 data_type=self.data_type),
             shard_axis_name = self.shard_axis_name,
             tp_strategy=self.tp_strategy,
             name="MLPInput",
         )(x)
         ##MLP output
         x = TPDense(
             dense_fn = functools.partial(
                 MLPBLockOutput,
                 features=self.hidden_dim*num_devices,
                 use_bias=self.use_bias,
                 data_type=self.data_type),
             shard_axis_name = self.shard_axis_name,
             tp_strategy="scatter",
             kernel_init_scale_factor = num_devices**-0.5,
             name="MLPOutput",
         )(x)
         return x

class TPRMSNorm(nn.Module):
    data_type: jnp.dtype
    shard_axis_name: str

    def setup(self):
        super().setup()

    @nn.compact
    def __call__(self,x: jax.Array) -> jax.Array:

        ##call modelParallelsim to wrap RMSnorm op for model parallelism
        x = ModelParallelism(
            shard_axis_name=self.shard_axis_name,
            module_fn = functools.partial(
                RMSNorm,
                dtype=self.data_type,
                axis_name=self.shard_axis_name),
            name="RMSNorm",
            )(x)
        return x


class TPAsyncDense(nn.Module):
    """ Tensor parallelism for Dense Op with async compute/communcation 

    dense_fn : wrapper function  dense op
    shard_axis_name: axis on which dense layer op is sharded
    tp_strategy : sharding strategy ['gather','scatter','Auto'] 
    kernel_init_fn : callable kernel init function 
    dense_name : layer name
    use_bidirectional: use bidirectional communication for feature transfer
    """

    dense_fn: Any
    shard_axis_name: str
    tp_strategy: Literal['gather','scatter','Auto'] = 'Auto'
    kernel_init_fn : Callable = nn.initializers.lecun_normal()
    kernel_init_scale_factor: float = 1.0
    #name: str = 'AsyncDenseOp'
    use_bidirectional: bool = True
    
    def setup(self):
        super().setup()


    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:

        num_devices = jax.lax.psum(1,self.shard_axis_name)
        _strategy = self.tp_strategy if num_devices > 1  else 'Auto'
        #print(f" TPAsyncDense= {x.shape}")
        dense_op = functools.partial(
             ModelParallelism,
             shard_axis_name= self.shard_axis_name,
             module_fn = functools.partial(
                 self.dense_fn,
                 kernel_init=scale_fn(self.kernel_init_fn,self.kernel_init_scale_factor),
             ),
             name= self.name
        )

        if _strategy == "Auto":
            output = self.dense_fn(kernel_init=self.kernel_init_fn,name=self.name)(x)
            return output
        elif _strategy == 'gather':
            #gather all input features
            gather_op = async_gather_bidir if self.use_bidirectional else async_gather
            inputs = gather_op(x,shard_axis_name=self.shard_axis_name)
            #do dense op
            outputs = [ 
                 dense_op(
                     module_kwargs={'use_bias' : (idx == 0)},
                     name=f'shard_{idx}',
                 )(inp)
                 for idx,inp in enumerate(inputs)
            ]
            #do sum of all outputs
            output = jax.tree_map(lambda *args: sum(args), *outputs)
            return output
        elif _strategy == 'scatter':
            #do dense op
            outputs = [
                    dense_op(
                        module_kwargs={'use_bias': (idx==0)},
                        name=f"shard_{idx}",
                    )(x)
                    for idx in range(num_devices)
            ]
            #async scatter results to devices for reduction
            output = async_scatter(outputs,shard_axis_name=self.shard_axis_name)
            return output
        else:
            raise ValueError(f"Unknown tp_strategy for model parallelism given {self.tp_strategy}")

####################################################################################
# Tensor parallelism async compute/communication support MLP input and output block 
# gather primitive embedded in MLPinput to collect inputs before computation
# scatter primitive embedded in MLPouput to collect outputs of compute for reduction
####################################################################################

class TPAsyncMLPBlock(nn.Module):
     shard_axis_name: str
     embedding_dim: int
     hidden_dim: int
     data_type: jnp.dtype
     use_norm: bool = False
     use_bias: bool = True
     tp_strategy: Literal['gather','scatter','auto'] = 'auto'
     kernel_init_scale_factor: float = 1.0
     communication_bidir: bool = True,
     name: str ="MLPBlock"

     def setup(setup):
         super().setup()


     @nn.compact
     def __call__(self,x: jax.Array) -> jax.Array:
         num_devices = jax.lax.psum(1,self.shard_axis_name)
         feature_dim = x.shape[-1]
         #print(f"axis_name = {self.shard_axis_name}")
         #Normalize across devices before  MLP 
         if self.use_norm:
             x = TPRMSNorm(shard_axis_name=self.shard_axis_name, data_type = self.data_type, name="PreNorm")(x)

         x = TPAsyncDense(
             dense_fn = functools.partial(
                  MLPBlockInput,
                  features=self.hidden_dim//num_devices,
                  use_norm=False,
                  data_type=self.data_type
             ),
             tp_strategy = self.tp_strategy,
             shard_axis_name=self.shard_axis_name,
             kernel_init_scale_factor=self.kernel_init_scale_factor*(num_devices**-0.5),
             name="MLPInput",
         )(x)
         #output MLP layer
         x = TPAsyncDense(
             dense_fn = functools.partial(
                  MLPBlockOutput,
                  features=feature_dim,
                  use_bias=self.use_bias,
                  data_type=self.data_type
             ),
             tp_strategy = "scatter",
             shard_axis_name=self.shard_axis_name,
             kernel_init_scale_factor=self.kernel_init_scale_factor*(num_devices**-0.5),
             name="MLPOutput",
         )(x)
         return x

#outputlayer produce [batch-size,num_tokens,-] tensor shape, 
#shard sequence length over TP axis. 

class TPOutputLayer(nn.Module):
    shard_axis_name: str
    num_outputs: int
    datatype: jnp.dtype
    shard_min_size: int

    def setup(self):
        super().setup()

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        #apply Gather op over feaure dimension
        #apply shard over sequence length

        x = jax.lax.all_gather(x,axis_name=self.shard_axis_name,axis=-1, tiled=True)
        x = split_array_over_mesh(x,axis_name=self.shard_axis_name,split_axis=1)

        #shard parameters over model axis
        norm_fn = shard_module_weights(
            nn.RMSNorm,
            shard_axis_name=self.shard_axis_name,
            min_weight_size=self.shard_min_size,
        )

        dense_fn = shard_module_weights(
            nn.Dense,
            shard_axis_name=self.shard_axis_name,
            min_weight_size=self.shard_min_size,
        )
        x = norm_fn(dtype=self.datatype, name="outRMSnorm")(x)
        x = dense_fn(
            features=self.num_outputs,
            dtype=self.datatype,
            name="output_layer",
        )(x)
        return x

class TPInputEmbedding(nn.Module):
    seq_len:int
    shard_axis_name: str
    data_type: jnp.dtype
    embedding_dim: int
    vocab_size: int
    encoding_type: Literal["learned","sinusoidal"] = "sinusoidal"

    def setup(self):
        super().setup()

    @nn.compact
    def __call__(self,x: jax.Array) -> jax.Array:

        return ModelParallelism(
                shard_axis_name=self.shard_axis_name,
                module_fn = functools.partial(
                    InputEmbedding,
                    seq_len=self.seq_len,
                    shard_axis_name=self.shard_axis_name,
                    data_type=self.data_type,
                    vocab_size=self.vocab_size,
                    embedding_dim=self.embedding_dim,
                    encoding_type=self.encoding_type),
                name="module",
        )(x)
