##############################################i######
# Tensor parallelism support for LLM building blocks
##############################################i######


import functools
from typing import Any, Dict, Tuple, Callable, Optional
from pprint import pprint

import numpy as np
#import torch

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
        TPAsyncllamaMLPBlock,
        TPOutputLayer,
        TPInputEmbedding
     )

from ..layers.blocks import MLPBlockInput,QKVProjection
from ..layers.baseops import AttnOutput, dot_product_attention

PyTree = Any
Parameter= jax.Array | nn.Partitioned


class TPMultiHeadAttention(nn.Module):
     embedding_dim : int  #model dimension 
     num_heads: int
     head_dim: int
     use_bias_qkv: bool
     #attention: Callable
     data_type: jnp.dtype
     shard_axis_name: str
     normalize_qk: bool = False
     train: bool = False   #FIXME deprecate

     def setup(self):
         super().setup()

     @nn.compact
     def __call__(self,x: jax.Array, mask: jax.Array) -> jax.Array:
         num_devices = jax.lax.psum(1,self.shard_axis_name)
         features = x.shape[-1]
         x = TPRMSNorm(data_type=self.data_type,shard_axis_name=self.shard_axis_name,name="PreNorm")(x)
         #QKV projection
         #densegeneral  builtin functions shards embedding dimension inot num_heads, head_dim
         #x = x.reshape(-1,self.num_heads,head_dim)
         q,k,v = TPAsyncDense(
             dense_fn = functools.partial(
                 QKVProjection, 
                 head_dim = self.head_dim,
                 num_heads = self.num_heads//num_devices,
                 use_bias = self.use_bias_qkv,
                 normalize_qk = self.normalize_qk,
                 data_type = self.data_type,
            ),
            shard_axis_name= self.shard_axis_name,
            tp_strategy = "gather",
            kernel_init_scale_factor = num_devices**-0.5,
            name="QKVProjection",
         )(x)
         x = dot_product_attention()(q,k,v,mask,attn_bias=False)
         x = TPAsyncDense(
             dense_fn = functools.partial(
                 AttnOutput,
                 embedding_dim=features,
                 data_type = self.data_type,
            ),
            shard_axis_name= self.shard_axis_name,
            tp_strategy = "scatter",
            name="AttnOutput",
            kernel_init_scale_factor = num_devices**-0.5,
         )(x)
         return x

class TPTransformerBlock(nn.Module):
    llmConfig: ConfigDict
    train: bool

    def setup(self):
        self.data_type =  self.llmConfig.dtype
        self.dropout_rate = self.llmConfig.dropout_rate
        self.embedding_dim =  self.llmConfig.model_dim
        self.num_heads = self.llmConfig.num_heads
        self.head_dim = self.llmConfig.head_dim
        self.normalize_qk = self.llmConfig.normalize_qk
        self.use_bias_qkv = self.llmConfig.bias_qkv
        self.use_mlpout_bias = self.llmConfig.bias_mlpout
        self.use_mlpin_norm = self.llmConfig.normalize_mlp
        self.hidden_dim = self.llmConfig.hidden_dim
        self.shard_axis_name = self.llmConfig.model_axis_name
        self.checkpoint_en = False  #FIMME this later for checkpoint enable
        self.shard_parameter = self.llmConfig.get("fsdp",False)
        self.fsdp_modules = self.llmConfig.fsdp.get("modules",None) if self.llmConfig.get("fsdp",None) is not None else None
        self.fsdp_axis_name = self.llmConfig.fsdp.get("axis_name",None) if self.llmConfig.get("fsdp",None) is not None else None
        self.shard_min_weight_size = self.llmConfig.fsdp.get("min_weight_size",1) if self.llmConfig.get("fsdp",None) is not None  else None 
        self.remat_modules= None
        super().setup()

    @nn.compact
    def __call__(self,x: jax.Array, mask: jax.Array) -> jax.Array:

        #blayer 
        if self.llmConfig.get("fsdp",None) is not None and "MultiHeadAttn" in self.llmConfig.fsdp.modules:
            shard_parameter = True
        else: 
            shard_parameter = False
        attn_layer = prep_module(
                         layer=TPMultiHeadAttention,
                         layer_name="MultiHeadAttn",
                         axis_name=self.fsdp_axis_name,
                         checkpoint_en=self.checkpoint_en,
                         shard_size=self.shard_min_weight_size,
                         shard_parameter=shard_parameter,
                         fsdp_modules = self.fsdp_modules,
                         #remat_modules = self.remat_modules,
        )

        attn_out = attn_layer(
                       embedding_dim=self.embedding_dim,
                       num_heads=self.num_heads,
                       head_dim=self.head_dim,
                       use_bias_qkv=self.use_bias_qkv,
                       data_type=self.data_type,
                       normalize_qk=self.normalize_qk,
                       train=self.train,
                       shard_axis_name=self.shard_axis_name,
                       name="Attention"
        )(x,mask)

        attn_out = nn.Dropout(
                       rate=self.dropout_rate,
                       deterministic=not self.train,
        )(attn_out)

        x = x + attn_out

        if self.llmConfig.get("fsdp",None) is not None and "MLP" in self.llmConfig.fsdp.modules:
            shard_parameter = True
        else: 
            shard_parameter = False
        mlp_layer = prep_module(
                    layer=TPAsyncllamaMLPBlock,
                    layer_name="MLP",
                    axis_name=self.fsdp_axis_name,
                    checkpoint_en=self.checkpoint_en,
                    shard_size=self.shard_min_weight_size,
                    shard_parameter=shard_parameter,
                    fsdp_modules = self.fsdp_modules,
                    #remat_modules = self.remat_modules,
        )                        

        mlp_out = mlp_layer(
                      shard_axis_name=self.shard_axis_name,
                      embedding_dim=self.embedding_dim,
                      hidden_dim=self.hidden_dim,
                      tp_strategy ="gather",
                      data_type = self.data_type,
                      use_bias = self.use_mlpout_bias,
                      use_norm = self.use_mlpin_norm,
                      name="mlp",
        )(x)
        mlp_out = nn.Dropout(
                       rate=self.dropout_rate,
                       deterministic=not self.train,
        )(mlp_out)
        x = x + mlp_out

        return x


## parallelize MLP and QKV dense blocks (source: https://github.com/kingoflolz/mesh-transformer-jax), both operating on the 
## same input , main benefit is communication of input x once to each device and perform QKV dense and MLP block computation 
## indepedently (is this valid architecture????)

class QKVMLPDenseParallel(nn.Module):
    embedding_dim : int  #model dimension 
    num_heads: int
    head_dim: int
    use_bias_qkv: bool
    data_type: jnp.dtype
    use_bias_mlp : bool
    hidden_dim: int # hiddne dimension

    def setup(self):
        super().setup()

    @nn.compact
    def __call__(self, x: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
        h = MLPBlockInput(
            features=self.hidden_dim,
            data_type=self.data_type,
            use_norm=False,
            name="MLPInput",
        )(x)
        q,k,v = QKVProjection(
           head_dim = self.head_dim,
           num_heads = self.num_heads,
           use_bias = self.use_bias_qkv,
           normalize_qk = True,
           data_type=self.data_type,
           name="QKVProjection")(x)
        return h,(q,k,v)

class AttnMLPOutParallel(nn.Module):
    data_type: jnp.dtype
    embedding_dim : int  #model dimension 
    use_bias_mlp : bool
    use_bias: bool = True
    kernel_init: Callable = nn.initializers.lecun_normal()

    @nn.compact
    def __call__(self, x: Tuple[jax.Array, jax.Array]) -> jax.Array:
        mlp_h, attn_v = x
        mlp_out = MLPBlockOutput(
            features=self.embedding_dim,
            use_bias=self.use_bias_mlp,
            data_type=self.data_type,
            name="MLPOut",
        )(mlp_h)
        attn_out = AttnOut(
            features=self.embedding_dim,
            use_bias=self.use_bias,
            name="AttnOut",
        )(attn_v)
        out = mlp_out + attn_out
        return out

class TPTransformerParallelBlock(nn.Module):
    llmConfig: ConfigDict
    train: bool

    def setup(self):
        self.data_type =  llmConfig.dataType
        self.dropout_rate = llmConfig.dropoutRate
        self.embedding_dim =  llmConfig.model_dim
        self.num_heads = llmConfig.num_heads
        self.head_dim = llmConfig.head_dim
        self.normalize_qk = llmConfig.normalize_qk
        self.use_bias_qkv = llmConfig.bias_qkv
        self.use_bias_attn = llmConfig.bias_attn
        self.use_bias_mlpout = llmConfig.bias_mlpout
        self.hidden_dim = llmConfig.mlp_factor*llmConfig.model_dim
        self.shard_axis_name = llmConfig.shard_axis_name
        self.checkpoint_en = llmConfig.checkpoint_enable
        self.fsdp= llmConfig.fsdp
        super().setup()

    @nn.compact
    def __call__(self,x: jax.Array, mask: jax.Array) -> jax.Array:

        num_devices = jax.lax.psum(1,self.shard_axis_name)
        residual    = x
        x = TPNorm(shard_axis_name=self.shard_axis_name, name="PreNorm")(x)

        seq_lem,feature_dim = x.shape[-2:]

        h,(q,k,v) = TPAyncDense(
            dense_fn = functools.partial(
                 QKVMLPDenseParallel,
                 embedding_dim = feature_dim,
                 num_heads=self.num_heads,
                 head_dim = self.head_dim,
                 use_bias_qkv = self.bias_qkv,
                 hidden_dim = self.hidden_dim // num_devices,
                 data_type = self.data_type,
                 use_bias_mlp = self.use_bias_mlpout,
           ),
           shard_axis_name=self.shard_axis_name,
           tp_strategy = "gather",
           kernel_init_scale_factor=num_deivces**-0.5,
           dense_name="QKVMLPParallel",
        )(x)
            
        v = dot_product_attention(q,k,v,mask)

        # MLP and attention layer with async scatter.
        block_out = TPAsyncDense(
            dense_fn=functools.partial(
                AttnMLPOutParallel,
                embedding_dim=self.embedding_dim,
                use_bias_mlp=self.use_bias_mlp,
                use_bias=self.use_bias,
            ),
            model_axis_name=self.config.model_axis_name,
            tp_strategy = "scatter",
            kernel_init_scale_factor=num_deivces**-0.5,
           dense_name="AttnMLPOutParallel",
        )((h, v))
        # Apply dropout and add residual.
        block_out = nn.Dropout(rate=self.config.dropout_rate, deterministic=not self.train)(
            block_out
        )
        out = residual + block_out
        return out

class TransformerBackbone(nn.Module):
    llmConfig: ConfigDict
    train: bool
    block_fn: Any = TPTransformerBlock

    @nn.compact
    def __call__(self, x: jax.Array, mask: jax.Array) -> jax.Array:
        if self.llmConfig.model.get("fsdp",None) is not None and "Block" in self.llmConfig.model.fsdp.modules:
            shard_parameter = True
        else:
            shard_parameter = False
        if self.llmConfig.train.remat is not None and "Block" in self.llmConfig.train.remat:
            checkpoint_en = True
        else:
            checkpoint_en = False
        block_fn = prep_module(
            layer=self.block_fn,
            layer_name="Block",
            axis_name=self.llmConfig.model.fsdp.axis_name,
            checkpoint_en=checkpoint_en,
            shard_size=self.llmConfig.model.fsdp.min_weight_size,
            shard_parameter=shard_parameter,
            fsdp_modules=self.llmConfig.model.fsdp.modules,
        )
        block = block_fn(llmConfig=self.llmConfig.model, train=self.train,name="block")
        # Scan version
        x, _ = nn.scan(
            lambda module, carry, _: (module(carry,mask), None),
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            length=self.llmConfig.model.num_layers,
            metadata_params={
                "partition_name": None
            },  # We do not need to partition over the layer axis.
        )(block, x, ())
        return x

class Transformer(nn.Module):
    #model config Dict
    llmConfig: ConfigDict
    block_fn: Any = TPTransformerBlock
    train: bool = True

    @nn.compact
    def __call__(self, x: jax.Array, train: bool, mask: jax.Array | None = None) -> jax.Array:
        if mask is None and self.llmConfig.model.causal_mask:
            mask = nn.make_causal_mask(x, dtype=bool)
        x = TPInputEmbedding(
            shard_axis_name=self.llmConfig.model_axis_name,
            embedding_dim=self.llmConfig.model.model_dim,
            data_type=self.llmConfig.model.dtype,
            vocab_size=self.llmConfig.model.vocab_size,
            tp_strategy="gather",
            name="input_embedding",
        )(x)
        x = TransformerBackbone(
            llmConfig=self.llmConfig,
            train=self.train,
            block_fn=self.block_fn,
            name="backbone",
        )(x,mask)
        x = TPOutputLayer(
            shard_axis_name=self.llmConfig.model_axis_name,
            num_outputs=self.llmConfig.model.num_outputs,
            dtype = self.llmConfig.model.dtype,
            shard_min_size=self.llmConfig.model.fsdp.min_weight_size,
            name="output_layer",
        )(x)
        return x
