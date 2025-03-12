
import functools
from typing import Any, Dict, Tuple, Callable, Optional
from pprint import pprint
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import core
from ml_collections import ConfigDict
from ..utilities.parallelism_functions import prep_module
from .TPbaseops import (
        TPDense,
        TPMLPBlock,
        TPRMSNorm,
        TPAsyncDense,
        TPAsyncMLPBlock,
        TPOutputLayer,
        TPInputEmbedding
     )

from core.layers.blocks import MLPBlockInput,QKVProjection
from core.layers.baseops import AttnOutput, dot_product_attention,RotaryPositionalEncoding
from core.layers.kvcache import llama_kvcache

PyTree = Any
Parameter= jax.Array | nn.Partitioned



class TPllamaAttention(nn.Module):
     embedding_dim : int  #model dimension 
     num_heads: int
     head_dim: int
     use_bias_qkv: bool
     max_seq_len: int
     dtype: jnp.dtype
     kvcache_prefill_axis_name: str
     kvcache_append_axis_name: str
     qkv_shard_axis_name: str
     out_shard_axis_name: str
     batch_size: int
     train: bool = False   #FIXME deprecate

     def setup(self):
         super().setup()
         num_devices = jax.lax.psum(1,self.qkv_shard_axis_name)
         if (self.train == False):
             self.kvcache = llama_kvcache(self.max_seq_len,
             self.kvcache_prefill_axis_name,
             self.kvcache_append_axis_name,
             (0,1,2,3),
             (0,1,2,3),
             self.dtype,self.num_heads//num_devices,self.batch_size,self.head_dim,"prefill")
         else:
             self.kvcache = None


     @nn.compact
     def flashattention(self,
         query: jax.Array,
         key: jax.Array,
         value: jax.Array,
         start_pos: int,
         mask: jax.Array):
         #mode: str = "prefill"):

         #if mode == "prefill":
         #    keys,values,start_pos = self.kvcache.llama_prefill(key,value,start_pos)
         #else:
             ##write to KV cache
         #    keys,values,start_pos = self.kvcache.llama_prefill(key,value,start_pos)
         #    #read start_pos+seqlen (read-out)
         #    keys,values,start_pos = self.kvcache.llama_append(key,value,start_pos)

         query = jnp.transpose(query, axes=(0,2,1,3))
         key = jnp.transpose(key, axes=(0,2,1,3))
         value = jnp.transpose(value, axes=(0,2,1,3))

         key = jnp.transpose(key, axes=(0,1,3,2))
         qk = jnp.matmul(query,key) / jnp.sqrt(self.head_dim)
         if mask is not None:
             qk = qk + mask
         qk = qk.astype(jnp.float32)
         s_qk = nn.softmax(qk, axis=-1)
         output = jnp.matmul(s_qk,value)
         output = jnp.transpose(value, axes=(0,2,1,3))
         return output

     @nn.compact
     def __call__(self,x: jax.Array, start_pos: int, mode:str, mask: Optional[jax.Array]) -> jax.Array:

         num_devices = jax.lax.psum(1,self.qkv_shard_axis_name)
         bs, seqlen, _ = x.shape
         positions = jnp.expand_dims(jnp.arange(0,seqlen),0)
         #QKV projection
         #densegeneral  builtin functions shards embedding dimension inot num_heads, head_dim
         #x = x.reshape(-1,self.num_heads,head_dim)
         _query,_key,_value = TPAsyncDense(
             dense_fn = functools.partial(
                 QKVProjection, 
                 head_dim = self.head_dim,
                 num_heads = self.num_heads//num_devices,
                 use_bias = self.use_bias_qkv,
                 normalize_qk = False,
                 data_type = self.dtype,
                 name=self.name+"QKVProjection",
            ),
            shard_axis_name= self.qkv_shard_axis_name,
            tp_strategy = "Auto",
            kernel_init_scale_factor = num_devices**-0.5,
         )(x)

         if (len(_query.shape) != 4):
            _query = jnp.reshape(_query,(bs, seqlen, self.num_heads//num_devices, self.head_dim))
            _key =   jnp.reshape(k,(bs, seqlen, self.num_heads//num_devices, self.head_dim))
            _value = jnp.reshape(v,(bs, seqlen, self.num_heads//num_devices, self.head_dim))

         _query = RotaryPositionalEncoding(self.head_dim)(_query,positions)
         _key = RotaryPositionalEncoding(self.head_dim)(_key,positions)

  
         if mode == "prefill":
             keys,values,start_pos = self.kvcache(_key,_value,start_pos,mode="prefill")
             x = self.flashattention(_query,keys,values,start_pos,mask)
         elif mode == "append":
             ##write and read to KV cache
             keys,values,start_pos = self.kvcache(_key,_value,start_pos,"append")
             x = self.flashattention(_query,keys,values,start_pos,mask)
         else:
             x = self.flashattention(_query,_key,_value,start_pos,mask)

         x = TPAsyncDense(
             dense_fn = functools.partial(
                 AttnOutput,
                 embedding_dim=self.embedding_dim,
                 data_type = self.dtype,
                 name=self.name+"attnoutput",
            ),
            shard_axis_name= self.out_shard_axis_name,
            tp_strategy = "scatter",
            kernel_init_scale_factor = num_devices**-0.5,
         )(x)
         return x
