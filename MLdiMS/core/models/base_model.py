##  


""" Transformer model. """

from typing import Any, Callable, Optional


from flax import linen as nn
import functools
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec
from configs.mlconfig import llmConfig

from core.layers.baseops import (
        RMSNorm,
        DenseGeneral,
        Embed
    )
from core.layers.blocks import (
        GenericAttention,
        MLPBlock
    ) 

import sys
import os

class DecodeBlock(nn.Module):
    """ Trasformer decode layers """

    config: llmConfig
    mesh: Mesh

    @nn.compact
    def __call__(
            self,
            x,
            decoder_segment_ids,
            decoder_positions,
            deterministic,
            mode,
        ):

        #shard inputs to (batch, seqlen,embedding_dim) partitions
        #x = nn.with_logical_constraint(x,("batch","sequence_length","embedding_dim"))
        x = nn.with_logical_constraint(x,self.config.shard_axes["Input"])
        residual = x

        #print(f" residual shape {x.shape}")

        #normalize inputs using RMSNorm (openAi/meta models use RMSNorm )
        layerNormx = RMSNorm(
            dtype = self.config.dtype,
            param_dtype = self.config.weight_dtype,
            name="pre_norm",
            epsilon = self.config.normalization_epsilon,
            shard_axes = {"pre_norm": ("norm",)},
        )(x)

        #print(f" residual shape {layerNormx.shape}")
        #map output LN activations with sharding axes
        #layerNormx = nn.with_logical_constraint(layerNormx,("batch","sequence_length","embedding_dim"))
        layerNormx = nn.with_logical_constraint(layerNormx,self.config.shard_axes["LayerNorm"])

        #attention layer
        attn_op = GenericAttention(
            num_heads = self.config.num_heads,
            kv_div_factor=self.config.num_kv_heads,
            attn_dim=self.config.attn_dim,
            dtype=self.config.dtype,
            weight_dtype=self.config.weight_dtype,
            attention_bias=self.config.attention_bias,
            max_seq_len=self.config.max_seq_length,
            max_prefill_predict_len=self.config.max_prefill_predict_len,
            shard_axes = self.config.shard_axes,
            dropout_rate= self.config.dropout_rate,
            qkv_bias=self.config.qkv_bias,
            rope_min_timescale=self.config.rope_min_timescale,
            rope_max_timescale=self.config.rope_max_timescale)
        attn_out = attn_op(x_q=layerNormx,
            x_kv=layerNormx,
            x_position=decoder_positions,
            decoder_segment_ids=decoder_segment_ids,
            mode=mode)
        #print(f" attn_out shape {attn_out.shape}")

        #attn_out = nn.with_logical_constraint(attn_out,("batch","sequence_length","embedding_dim"))
        attn_out = nn.with_logical_constraint(attn_out,self.config.shard_axes["AttentionOut"])
        
        #dropout layer

        #attn_drop_out = nn.Dropout(rate=self.config.dropout_rate,broadcast_dims=(-2,))(attn_out,deterministic=deterministic)

        hidden_states = attn_out + residual
        hidden_states = nn.with_logical_constraint(hidden_states,self.config.shard_axes["HiddenStates"])
        residual = hidden_states
        #LayerNorm
        #normalize inputs using RMSNorm (openAi/meta models use RMSNorm )
        hidden_states = RMSNorm(
            dtype = self.config.dtype,
            param_dtype = self.config.weight_dtype,
            name="post_attn_norm",
            epsilon = self.config.normalization_epsilon,
            shard_axes = {"post_attn_norm": ("norm",)},
        )(hidden_states)
        
        hidden_states = nn.with_logical_constraint(hidden_states,self.config.shard_axes["HiddenStates"])
        #MLP block
        hidden_states = MLPBlock(
            hidden_dim=self.config.hidden_dim,
            embedding_dim=self.config.model_dim,
            in_shard_axes = self.config.shard_axes["MLPIn"],
            out_shard_axes = self.config.shard_axes["MLPOut"],
            data_type = self.config.dtype,
            weight_dtype=self.config.weight_dtype)(hidden_states)

        output = hidden_states + residual
        output = nn.with_logical_constraint(output,self.config.shard_axes["OutPut"])
        return output, None

        #attention output layer


class Decoder(nn.Module):
    """ Decoder block of Transformer, build using decodeBlock or specific model decodeBlock"""

    config: llmConfig
    mesh: Mesh


    def setup(self):
        super().setup()


    @nn.compact
    def __call__(self,
            inp_tokens,
            inp_positions,
            decoder_segment_ids=None,
            deterministic=False,
            mode="Prefill"
        ):

        assert(len(inp_tokens.shape) == 2)

        out = Embed(
            num_embeddings = self.config.vocab_size,
            features = self.config.model_dim,
            dtype=self.config.dtype,
            name="TokenEmbedding",
        )(inp_tokens.astype("int32"))

        #out = nn.with_logical_constraint(out, self.config.shard_axes["InputEmbed"])
 
        initializing = self.is_mutable_collection("params")
        params_spec = self.config.param_scan_axis if initializing else nn.partitioning.ScanIn(self.config.param_scan_axis)
        scan_fn  = nn.scan(
            DecodeBlock,
            variable_axes={ "params" : params_spec,
                            "kvcache": 0,
                            "intermediates": 0},
            split_rngs = {"params" : True, "dropout" : True},
            in_axes=(nn.broadcast,nn.broadcast,nn.broadcast,nn.broadcast),
            length=self.config.num_layers,
            metadata_params={
               "partition_name": "layers"
            },  # We do not need to partition over the layer axis.
        )

        out,_ = scan_fn(config=self.config,mesh=self.mesh,name="layers")(
            out,
            decoder_segment_ids,
            inp_positions,
            deterministic,
            mode,
        )

        out = RMSNorm(
            dtype = self.config.dtype,
            param_dtype = self.config.weight_dtype,
            name="pre_norm",
            epsilon = self.config.normalization_epsilon,
            shard_axes = {"post_norm": ("norm",)},
        )(out)

        #out = nn.Dropout(rate=self.config.dropout_rate, broadcast_dims=(-2,))(out, deterministic=deterministic)
        out = DenseGeneral(
            self.config.vocab_size,
            param_dtype=self.config.weight_dtype,
            dtype=self.config.dtype,
            shard_axes = {"OutPut" : self.config.shard_axes["OutPut"]},
            name="logits_out",
        )(out)

        out = nn.with_logical_constraint(out, PartitionSpec("pipeline","data","fsdp"))

        return out

class Transformer(nn.Module):
    """ Transformer Model (Decoder only)"""
    
    config: llmConfig
    mesh: Mesh

    def setup(self):
        super().setup()
        self.decoder = Decoder(config=self.config,mesh=self.mesh)

    @nn.compact
    def __call__(self,
            inp_tokens,
            inp_positions,
            decoder_segment_ids=None,
            mode="prefill"
        ):

        output = self.decoder(inp_tokens=inp_tokens,
            inp_positions=inp_positions,
            decoder_segment_ids=decoder_segment_ids,
            deterministic=False,
            mode=mode)
        return output
    

if __name__ == "__main__":

    from configs import mlconfig
    mlconfig.initialize(sys.argv)

    from jax.experimental import mesh_utils

    USE_CPU_ONLY = True
    # Set XLA flags to simulate a CPU with a given number of devices
    flags = os.environ.get("XLA_FLAGS", "")
    flags += f" --xla_force_host_platform_device_count={1}"
    os.environ["XLA_FLAGS"] = flags
    # Disable CUDA to force XLA to use the CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # Check for packages to be installed if needed. On Colab, the following packages are not installed by default:
    # - ml_collections
    try:
        import ml_collections
    except ImportError:
        install_package("ml_collections")


    config = mlconfig.config
    device_parallelism = [config.data_parallelism, config.pipeline_parallelism, config.fsdp_parallelism, config.sequence_parallelism, config.tensor_parallelism]
    devices = jax.devices()
    mesh = mesh_utils.create_device_mesh(
            device_parallelism,
            devices,
        )

    _axes = {
                  "Input" : ("act_batch","act_seqlen","act_embed_dim"),
                  "LayerNorm" : ("act_batch","act_seqlen","act_embed_dim"),
                  "AttentionOut" : ("act_batch","act_seqlen","act_embed_dim"),
                  "HiddenStates" : ("act_batch","act_seqlen","act_embed_dim"),
                  "OutPut" : ("act_batch","act_seqlen","act_embed_dim"),
                  "prefill_axis" : ("kvcache_batch","kvcache_seqlen", "kvcache_heads","kvcache_embed_dim"),
                  "append_axis" : ("kvcache_batch","kvcache_seqlen", "kvcache_heads","kvcache_embed_dim"),
                  "QKVProj" : ("embed","qkv","heads","kv"),
                  "Qrope" : ("act_kv_batch","act_seqlen","act_kv_heads","act_kv_headdim"),
                  "Krope" : ("act_kv_batch","act_seqlen","act_kv_heads","act_kv_headdim"),
                  "Vrope" : ("act_kv_batch","act_seqlen","act_kv_heads","act_kv_headdim"),
                  "OutProj" : ("act_heads","kv","act_embed_dim"),
                  "AttnOut" : ("act_heads","kv","act_embed_dim"),
                  "MLPIn" : ("embed","mlp"),
                  "MLPOut" : ("mlp","embed"),
            }
    config.shard_axes = _axes
    model = Transformer(config,mesh)
