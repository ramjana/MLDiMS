#Initial version

import functools
from typing import Any, Dict, Tuple, Callable, Optional, Mapping
from pprint import pprint

import numpy as np
from operator import attrgetter

import flax.linen as nn
import jax
import jax.numpy as jnp
#from jax import core
from flax.linen import partitioning as nn_partitioning

from core.utilities.utils import print_shapes
from kernels.decorators import gemm_call

import dataclasses

Pytree = Any



def perf(message1,get_attrvalue):
    def decorator(LayerOp):
        #result_cache = {}
        print(message1)
        def wrapper(self,*args, **kwargs):
            #cache_key = (*args, *kwargs.items())
            print(message2)
            M,K = jnp.shape(args[0])
            N =  get_attrvalue(self)
            print(f"flops = {2*N*M*K}")
            result = LayerOp(self,*args,**kwargs)
            print(f"LayerOp {self.__class__.__name__} took { 10: .2f} seconds to execute")
            #result_cache[cache_key] = result
            return result
        return wrapper
    return decorator


@dataclasses.dataclass
class ShardMixIn:
    """Adds parameter sharding constraints for any flax.linen Module.

    This is a mix-in class that overrides the `param` method of the
    original Module, to selectively add sharding constraints as specified
    in `shard_axes`"""

    shard_axes: Optional[Mapping[str, Tuple[str, ...]]] = None

    # Modifies off https://github.com/google/flax/blob/main/flax/linen/partitioning.py#L304
    def param(self, name: str, *init_args):
        # Initialize using the original Module's `param` method
        param = super().param(name, *init_args)

        # If `shard_axes` specified and param name in the dict, apply constraint
        if self.shard_axes and (name in self.shard_axes.keys()):
            axes = self.shard_axes[name]

            # Apply the sharding constraint (e.g. axes=('embedding', 'hidden'))
            param = nn.with_logical_constraint(param, axes)

            # Sow this, to have the AxisMetadata available at initialization.
            self.sow(
                "params_axes",
                f"{name}_axes",
                nn_partitioning.AxisMetadata(axes),
                reduce_fn=nn_partitioning._param_with_axes_sow_reduce_fn,
            )

        return param

class Dense(ShardMixIn,nn.Dense):
    """ A linear dense layer inherited from jax.liner.nn.dense
    Orignal Source code is at
    https://flax.readthedocs.io/en/v0.5.3/_modules/flax/linen/linear.html#Dense
    """

    def setup(self):
        super().setup()

    #@perf("Dense",attrgetter('features'),attrgetter('dtype'))
    @print_shapes("Dense",attrgetter('name'))
    @nn.compact
    def __call__(self,x: jax.Array) -> jax.Array:
        # update cycles 
        return (super().__call__(x))

class Linear(ShardMixIn,nn.Dense):
    """ A liner dense layer inherited from jax.liner.nn.dense
    Orignal Source code is at
    https://flax.readthedocs.io/en/v0.5.3/_modules/flax/linen/linear.html#Dense
    """

    def setup(self):
        super().setup()

    #@perf("DenseOp",attrgetter('features'),attrgetter('dtype'))
    @nn.compact
    def __call__(self,x: jax.Array) -> jax.Array:
        # update cycles 
        return (super().__call__(x))

class Embed(nn.Embed):
    """ Embedding op inherited from jax.liner.nn.Embedding
    """

    def setup(self):
        super().setup()

    #@perf("EmbedOp",attrgetter('name'),attrgetter('features'),attrgetter('num_embeddings'),attrgetter('dtype'))
    @nn.compact
    def __call__(self,x: jax.Array) -> jax.Array:
        return (super().__call__(x))

class RMSNorm(ShardMixIn,nn.RMSNorm):
    """
    normalization layer inherited from jax.liner.nn.
    """
    def setup(self):
        super().setup()

    #@perf("RMSNorm",attrgetter('name'),attrgetter('dtype'))
    @print_shapes("RMSNorm",attrgetter('name'))
    @nn.compact
    def __call__(self,x: jax.Array) -> jax.Array:
        return (super().__call__(x))

class LayerNorm(ShardMixIn,nn.LayerNorm):
    """
    normalization layer inherited from jax.liner.nn.
    """
    def setup(self):
        super().setup()

    #@perf("LayerNorm",attrgetter('name'),attrgetter('dtype'))
    @nn.compact
    def __call__(self,x: jax.Array) -> jax.Array:
        return (super().__call__(x))

class softmax(nn.Module):
    """
      softmax layer inherited from jax.liner.nn

    """
    def setup(self):
        super().setup()

    #@perf("softmax",attrgetter('name'),attrgetter('dtype'))
    @nn.compact
    def __call__(self,x: jax.Array) -> jax.Array:
        #return (super().__call__(x))
        return (nn.softmax(x))


class dot_product_attention(nn.Module): 
    """
      use same algorithm as https://flax.readthedocs.io/en/latest/api_reference/flax.linen/layers.html#flax.linen.dot_product_attention
      based on attention is all you need paper

      use f32 for softmax

      args:
         query : query jax.array of shape [..., seq_len,num_heads,attn_dim]
         key :  key jax.array of shape [..., seq_len,num_heads,attn_dim]
         value : value jax.array of shape [..., seq_len,num_heads,attn_dim]
         mask: boolean mask array (0 for mask 1 for non-mask)

     returns attention output jax array of shape [...,seq_len,num_heads,attn-dim]
    """
    def setup(self):
        super().setup()

    #@perf("dot_product_attention",attrgetter('name'),attrgetter('dtype'))
    @nn.compact
    def __call__(self,query: jax.Array, key: jax.Array, value: jax.Array, mask: jax.Array, attn_bias: bool = False, bias: jax.Array = None) -> jax.Array:
       attn_dim = query.shape[-1]
       dtype = query.dtype
       scale = attn_dim**-0.5
       #convert to float32
       query = query.astype(jnp.float32)
       key = key.astype(jnp.float32)
       value = value.astype(jnp.float32)
       query = query * scale
       qk  = jnp.einsum("...qhd,...khd->...hqk", query,key)
       if attn_bias is not None and attn_bias == True:
           bias = bias.astype(jnp.float32)
           qk = qk + jnp.log2(bias)
       if mask is not None:
           qk = jnp.where(mask,qk,jnp.finfo(jnp.float32).min)
       ## do softmax    
       qk = nn.softmax(qk,axis=-1)
       #down convert to original tensor data type
       qk = qk.astype(dtype)
       P = jnp.einsum("...hqk,...khd->...qhd",qk,value)
       P = P.astype(dtype)
       return P

class DenseGeneral(ShardMixIn,nn.DenseGeneral):
    """ linear dense layer with flexible axes inherited from jax.liner.nn.DenseGeneral
    Orignal Source code is at
    https://flax.readthedocs.io/en/v0.5.3/_modules/flax/linen/linear.html#DenseGeneral
    """

    def setup(self):
        super().setup()

    #@perf("DenseGeneral",attrgetter('features'))
    @gemm_call('features','dtype')
    @print_shapes("DenseGeneral",attrgetter('name'))
    def __call__(self,x: jax.Array) -> jax.Array:
        # update cycles 
        return (super().__call__(x))

class LearnedPositionalEncoding(nn.Module):

    """
    module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """
    seq_len: int
    embedding_dim: int

    def setup(self):
        super().setup() 

    #@perf("LearnedPositionalEncoding",attrgetter('name'),attrgetter('dtype'))
    @nn.compact
    def __call__(self,x: jax.Array) -> jax.Array:
       self.seq_len,self.embedding_dim = x.shape[-2:]
       pos_embed = self.param(
           "positional_embedding", 
           nn.initializers.normal(stddev=self.embedding_dim**-0.5),
           (self.seq_len,self.embedding_dim),
       )
       pos_emb = pos_embed.astype(
          x.dtype
       )
       pos_embed = jnp.expand_dims(pos_embed,axis=range(x.ndim-2))
       x = x + pos_embed
       return x     


class SinusoidalPositionalEncoding(nn.Module):

    """
       sinusoidal positional embedding
    """
    seq_len: int
    embedding_dim: int
    shard_axis_name: str
    padding_idx: int = None
    max_positions: int = None

    def setup(self):
        super().setup() 

    #@perf("SinuSoidalPositionalEncoding",attrgetter('name'),attrgetter('dtype'))
    @nn.compact
    def __call__(self,x: jax.Array) -> jax.Array:
        num_devices = jax.lax.psum(1,self.shard_axis_name)
        tp_index = jax.lax.axis_index(self.shard_axis_name)
        seq_len,embedding_dim = x.shape[-2:]
        position = jnp.arange(0,seq_len,dtype=jnp.float32)[:,None]
        div_val = jnp.exp(jnp.arange(tp_index*embedding_dim,(tp_index+1)*embedding_dim,2) * (-np.log(10000.0) / (num_devices*embedding_dim)))
        sinu_embed = jnp.stack([jnp.sin(position * div_val), jnp.cos(position * div_val)], axis=-1) 
        sinu_embed = jnp.reshape(sinu_embed,(seq_len,embedding_dim))
        sinu_embed = sinu_embed.astype(x.dtype)
        sinu_embed = jnp.expand_dims(sinu_embed,axis=range(x.ndim-2))
        x = x + sinu_embed
        return x

class AttnOutput(nn.Module):
     """
      module to flatten out last two dimension of the input [...,num_heads,attn-dim] before 
      applying linear transformation 

     """
     embedding_dim : int  #model dimension 
     data_type: jnp.dtype
     use_bias: bool = True
     kernel_init:  Callable = nn.initializers.lecun_normal()

     def setup(self):
         super().setup()

     @nn.compact
     def __call__(self,x: jax.Array) -> jax.Array:
         x = DenseGeneral(
             features=self.embedding_dim,
             axis=(-2,-1),
             kernel_init=self.kernel_init,
             use_bias=self.use_bias,
             dtype=self.data_type,
             name="AttnOut"
         )(x)
         return x 

class silu(nn.Module):
     """ 
        activation module inherited from jax module
        silu(x) = x*sigmoid(x)

     """
     def setup(self):
         super().setup()

     #@perf("silu",attrgetter('name'),attrgetter('dtype'))
     @print_shapes("silu",attrgetter('name'))
     @nn.compact
     def __call__(self,x : jax.Array) -> jax.Array:
         return nn.silu(x)

class relu(nn.Module):
     """ 
        activation module inherited from jax module
        relu(x) = max(0,x)

     """

     def setup(self):
         super().setup()

     #@perf("relu",attrgetter('name'),attrgetter('dtype'))
     @nn.compact
     def __call__(self,x : jax.Array) -> jax.Array:
         return nn.relu(x) 

class gelu(nn.Module):
     """ 
        gaussian error linear module inherited from jax module
        original source at https://github.com/google/jax/blob/main/jax/_src/nn/functions.py#L218-L241
     """

     def setup(self):
         super().setup()

     #@perf("gelu",attrgetter('name'),attrgetter('dtype'))
     @nn.compact
     def __call__(self,x : jax.Array) -> jax.Array:
         return nn.gelu(x)

class Dropout(nn.Dropout):
     """
        stochastic normalization technique randomly removes hidden and visible
        units in network
     """

     def setup(self):
         super().setup()

     #@perf("dropout")
     @nn.compact
     def __call__(self, x: jax.Array) -> jax.Array:
         return super().__call(x)


class RotaryPositionalEncoding(nn.Module):
    embedding_dim: int #attention dim 
    min_timescale: int = 1
    max_timescale: int = 10000

    def setup(self):
        assert(self.embedding_dim%2 == 0)

        self.timescale = self.min_timescale * (self.max_timescale/self.min_timescale) ** (2 * jnp.arange(0,self.embedding_dim//2) / self.embedding_dim)

    @nn.compact
    def __call__(self, x: jax.Array, position : jax.Array) -> jax.Array:

        assert(len(x.shape) == 4)  # shape (b,seq_len,n,attn_dim)
        assert(x.shape[-1] == self.embedding_dim)
        
        if position is None:
            raise ValueError("require positional input")

        ##add two dimensions to position shape for sin,cos
        position = position[:,:,jnp.newaxis,jnp.newaxis]
        scaled_position = position/self.timescale
        sin_x = jnp.sin(scaled_position).astype(x.dtype)
        cos_x = jnp.cos(scaled_position).astype(x.dtype)
        upper_segment, lower_segment = jnp.split(x,2,axis=-1)
        upper_segment = upper_segment * cos_x - lower_segment* sin_x
        lower_segment = upper_segment * cos_x + lower_segment* sin_x
        x_out = jnp.concatenate((upper_segment,lower_segment),axis=-1)
  
        return x_out

    @staticmethod
    def rotate_half(self, x: jax.Array) -> jax.Array:
        x1, x2 = jnp.split(x,2, axis=-1)
        return jnp.concatenate((-x2,x1),axis=-1)

# Based on https://github.com/google-research/google-research/blob/master/scaling_transformer_inference_efficiency/attention.py
def normalize_attention(local_outs, local_maxes, local_sums):
    """Normalize across multiple localized attentions

    Args:
        local_outs (list): List of unnormalized outputs entries for each local attention
        local_maxes (list): List of max exponentials entries for each local attention
        local_sums (list): List of exponential sum entries for each local attention

    Returns:
        Array: Combined attention that has been normalized
    """
    global_max = functools.reduce(jnp.maximum, local_maxes)
    global_sum = sum(
        [jnp.exp(local_max - global_max) * local_sum for (local_sum, local_max) in zip(local_sums, local_maxes)]
    )
    attn_out = 0
    for local_max, local_out in zip(local_maxes, local_outs):
        local_normalizer = jnp.exp(local_max - global_max) / global_sum
        attn_out += local_normalizer * local_out
    return attn_out


class fadd(nn.Module):
    def setup(self):
        super().setup()
    @nn.compact
    def __call__(self,x, y):
        out = x + y
        return out

class fmul(nn.Module):
    def setup(self):
        super().setup()
    @nn.compact
    def __call__(self,x, y):
        out = x * y
        return out
