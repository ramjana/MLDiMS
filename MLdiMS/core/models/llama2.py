
from ml_collections import ConfigDict
import functools
from pprint import pprint
from typing import Any, Callable, Dict, Tuple, Optional, List

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from ml_collections import ConfigDict
from tqdm.auto import tqdm

PyTree = Any
Parameter = jax.Array | nn.Partitioned
Metrics = Dict[str, Tuple[jax.Array, ...]]


from core.tensor_parallel.blocks import(
     TPTransformerBlock,
     Transformer,
     TPTransformerParallelBlock
)
from core.tensor_parallel.TPbaseops import TPInputEmbedding, TPAsyncDense, TPOutputLayer

from core.layers.baseops import Dense,RMSNorm
from core.layers.blocks import MLPBlockInput,MLPBlockOutput,InputEmbedding
from core.tensor_parallel.flashattention import TPllamaAttention


from core.layers.baseops import fmul,fadd,silu
from core.layers import baseops
from core.utilities.parallelism_functions import prep_module,fold_rng_over_axis,split_array_over_mesh 
from core.utilities.modelParallelism_functions import (
     ModelParallelism,
     async_gather,
     async_gather_bidir,
     async_scatter,
     async_scatter_bidir
)
from core.utilities.utils import *
from utils.train_utils import TrainState,Batch,train_step_tp,loss_fn_tp,init_tp,print_metrics
from configs.mlconfig import llmConfig


class TPAsyncllamaMLP(nn.Module):
     shard_axis_name: str
     embedding_dim: int
     hidden_dim: int
     multiple: int
     dtype: jnp.dtype
     kernel_init_scale_factor: float = 1.0
     communication_bidir: bool = True,

     def setup(self):
         super().setup()
         self.mul_block = fmul(name="mlpblock")
         self.silu_block = silu(name="MlpBlock")

     @nn.compact
     def __call__(self,x: jax.Array) -> jax.Array:
         num_devices = jax.lax.psum(1,self.shard_axis_name)
         feature_dim = x.shape[-1]
         #print(f"axis_name = {self.shard_axis_name}")

         assert(feature_dim == self.embedding_dim)

         hidden_dim = int(2 * self.hidden_dim / 3)
         hidden_dim = self.multiple * ((hidden_dim + self.multiple -1) // self.multiple)

         mlp1_out = TPAsyncDense(
             dense_fn = functools.partial(
                  MLPBlockInput,
                  features=hidden_dim//num_devices,
                  use_norm=False,
                  use_bias=False,
                  data_type=self.dtype,
                  name=self.name+"::mlp1",
             ),
             use_bidirectional = True,
             tp_strategy = 'gather',
             shard_axis_name=self.shard_axis_name,
             kernel_init_scale_factor=self.kernel_init_scale_factor*(num_devices**-0.5),
         )(x)

         mlp2_out = TPAsyncDense(
             dense_fn = functools.partial(
                  MLPBlockInput,
                  features=hidden_dim//num_devices,
                  use_norm=False,
                  use_bias=False,
                  data_type=self.dtype,
                  name=self.name+"::mlp2",
             ),
             tp_strategy = 'gather',
             shard_axis_name=self.shard_axis_name,
             use_bidirectional = True,
             kernel_init_scale_factor=self.kernel_init_scale_factor*(num_devices**-0.5),
         )(x)
         x = self.mul_block(mlp1_out, mlp2_out)
         x = self.silu_block(x)

         mlp3_out = TPAsyncDense(
             dense_fn = functools.partial(
                  MLPBlockOutput,
                  features=self.embedding_dim,
                  use_bias=False,
                  data_type=self.dtype,
                  name=self.name+"::mlp3",
             ),
             use_bidirectional = True,
             tp_strategy = 'scatter',
             shard_axis_name=self.shard_axis_name,
             kernel_init_scale_factor=self.kernel_init_scale_factor*(num_devices**-0.5),
         )(x)

         return mlp3_out

class llamaTransformerBlock(nn.Module):
    config: llmConfig 
    def setup(self):
        super().setup()
        self.dtype =  self.config.dtype
        self.dropout_rate = self.config.dropout_rate
        self.embedding_dim =  self.config.model_dim
        self.num_heads = self.config.num_heads
        self.head_dim = self.config.attn_dim
        self.hidden_dim = self.config.hidden_dim
        self.multiple = self.config.multiple
        self.shard_axis_name = self.config.model_axis_name
        self.norm_eps = self.config.normalization_epsilon
        self.checkpoint_en = False  #FIMME this later for checkpoint enable
        self.shard_parameter = self.config.fsdp
        self.fsdp_modules = self.config.fsdp_modules if self.shard_parameter == True else None
        self.fsdp_axis_name = self.config.fsdp_axis_name if self.shard_parameter == True else None
        self.shard_min_weight_size = self.config.fsdp_min_weight_size if self.shard_parameter == True else None
    @nn.compact
    def __call__(self,x: jax.Array, start_pos: int, mode:str, mask: Optional[jax.Array]) -> jax.Array:

        #blayer
        #print(f"TPtransformer head_dim= {self.head_dim}")
        #print(f"TPtransformer num_heads= {self.num_heads}")
        #print(f"TPtransformer model_dim= {self.embedding_dim}")
        if self.config.fsdp and "MultiHeadAttn" in self.config.fsdp_modules:
            shard_parameter = True
        else:
            shard_parameter = False
        train = False
        if mode == "Training":
            train= True
        bsz, seqlen,_ = x.shape
        attn_layer = prep_module(
                         layer=TPllamaAttention,
                         layer_name="MultiHeadAttn",
                         axis_name=self.fsdp_axis_name,
                         checkpoint_en=self.checkpoint_en,
                         shard_size=self.shard_min_weight_size,
                         shard_parameter=shard_parameter,
                         fsdp_modules = self.fsdp_modules,
                         #remat_modules = self.remat_modules,
        )
        residual = x
        hidden_state = RMSNorm(
            dtype=self.dtype,
            epsilon=self.norm_eps,
            name="preattnnorm")(x)
        attn_out = attn_layer(
                       embedding_dim=self.embedding_dim,
                       max_seq_len=self.config.max_seq_length,
                       num_heads=self.num_heads,
                       head_dim=self.head_dim,
                       use_bias_qkv=False,
                       dtype=self.dtype,
                       qkv_shard_axis_name=self.shard_axis_name,
                       kvcache_prefill_axis_name=None,
                       kvcache_append_axis_name=None,
                       out_shard_axis_name=self.shard_axis_name,
                       name="AttentionBlock",
                       train = train,
                       batch_size = bsz,   #HACK 
        )(hidden_state,start_pos,mode,mask)
        fadd_block = fadd(name="residual_add")
        hidden_state = fadd_block(attn_out,residual)
        norm_out = RMSNorm(
            dtype=self.dtype,
            epsilon=self.norm_eps,
            name="postattnnorm")(hidden_state)
        if self.config.fsdp and "MLP" in self.config.fsdp_modules:
            shard_parameter = True
        else:
            shard_parameter = False
        mlp_layer = prep_module(
                    layer=TPAsyncllamaMLP,
                    layer_name="MLP",
                    axis_name=self.fsdp_axis_name,
                    checkpoint_en=self.checkpoint_en,
                    shard_size=self.shard_min_weight_size,
                    shard_parameter=shard_parameter,
                    fsdp_modules = self.fsdp_modules,
        )
 
        mlp_out = mlp_layer(
                      shard_axis_name=self.shard_axis_name,
                      embedding_dim=self.embedding_dim,
                      hidden_dim=self.hidden_dim,
                      multiple = self.multiple,
                      dtype = self.dtype,
                      name="mlpblock",
        )(norm_out)
        out = fadd_block(mlp_out,hidden_state)
        return out 

class TransformerBackbone(nn.Module):
    config: llmConfig 
    block_fn: Any = llamaTransformerBlock

    def setup(self):
        super().setup()
    @nn.compact
    def __call__(self, x: jax.Array, start_pos: int, mode:str,  mask: Optional[jax.Array]) -> jax.Array:
        shard_parameter = False
        if self.config.fsdp and  "Block" in self.config.fsdp_modules:
            shard_parameter = True
        checkpoint_en = False
        block_fn = prep_module(
            layer=self.block_fn,
            layer_name="Block",
            axis_name=self.config.fsdp_axis_name,
            checkpoint_en=checkpoint_en,
            shard_size=self.config.fsdp_min_weight_size,
            shard_parameter=shard_parameter,
            fsdp_modules=self.config.fsdp_modules,
        )
        block = block_fn(config=self.config,name="block")
        # Scan version
        #x, _ = nn.scan(
        #    lambda module, carry,_: (module(carry,start_pos,mode,mask), None),
        #    variable_axes={"params": 0, "kvcache": 0},
        #    split_rngs={"params": True, "dropout": False},
        #    length=self.config.num_layers,
        #    metadata_params={
        #        "partition_name": None
        #    },  # We do not need to partition over the layer axis.
        #)(block, x,())
        for idx in range(self.config.num_layers):
            block = block_fn(config=self.config,name=f"BlockLayer_{idx}")
            x = block(x,start_pos,mode,mask)
        return x

class llama2(nn.Module):
    #model config Dict
    config : llmConfig 
    block_fn: Any = llamaTransformerBlock 

    @nn.compact
    def __call__(self, tokens: jax.Array, start_pos:int, mode: str) -> jax.Array:
        bs, seqlen = tokens.shape
        hidden_state = TPInputEmbedding(
            shard_axis_name=self.config.model_axis_name,
            embedding_dim=self.config.model_dim,
            data_type=self.config.dtype,
            vocab_size=self.config.vocab_size,
            tp_strategy = "gather",
            name="input_embedding",
        )(tokens)

        ##generate mask or tokensize>1

        mask = None
        if seqlen > 1:
            mask = jnp.full((seqlen,seqlen), float("-inf"))
            mask = jnp.triu(mask,k=1)
            mask = jnp.hstack([
                jnp.zeros((seqlen,start_pos)), 
                mask]).astype(hidden_state)

        layer_out = TransformerBackbone(
            config=self.config,
            block_fn=self.block_fn,
            name="backbone",
        )(hidden_state,start_pos,mode,mask)

        norm_out = RMSNorm(
            dtype=self.config.dtype,
            epsilon=self.config.normalization_epsilon,
            name="outnorm")(layer_out)

        out = TPOutputLayer(
            shard_axis_name=self.config.model_axis_name,
            num_outputs=self.config.vocab_size,
            dtype = self.config.dtype,
            shard_min_size=self.config.fsdp_min_weight_size,
            norm_en=False,
            name="output",
        )(norm_out)

        #out = TPAsyncDense(
        #    dense_fn = functools.partial(
        #        Dense,
        #        features=self.config.model_dim//num_devices,
        #        use_bias=False,
        #        dtype=self.config.dtype
        #     ),
        #     shard_axis_name = self.config.model_axis_name,
        #     tp_strategy="gather",
        #     name="output",
        # )(norm_out)
        return out

def get_transformer_module(config: Any):
    module_class = llama2 
    if config.fsdp:
        shard_parameter = True
    else:
        shard_parameter = False
    module_class = prep_module(
            layer=module_class,
            layer_name="Transformer",
            axis_name=config.data_axis_name,
            fsdp_modules=config.fsdp_modules,
            checkpoint_en=False,
            shard_parameter=shard_parameter,
            shard_size=config.fsdp_min_weight_size,
    )
    block_func = llamaTransformerBlock 
    return module_class(config=config,block_fn=block_func)


if __name__ == "__main__":
    from configs import mlconfig
    mlconfig.initialize(sys.argv)

    from jax.sharding import Mesh

    from jax.experimental import mesh_utils
    USE_CPU_ONLY = True
    # Set XLA flags to simulate a CPU with a given number of devices
    flags = os.environ.get("XLA_FLAGS", "")
    flags += f" --xla_force_host_platform_device_count={8}"
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
    array_devices = np.array(jax.devices()).reshape(-1,config.model_axis_size)
    print(jax.devices())
    mesh= Mesh(array_devices,(config.data_axis_name,config.model_axis_name))

    model = get_transformer_module(config)

    def init_transformer(rng: jax.random.PRNGKey, x: jax.Array) -> TrainState:
        init_rng, rng = jax.random.split(rng)
        #variables = model.init({"params" : init_rng, "kvcache" : init_rng}, x, start_pos=0, mode="prefill")
        variables = model.init({"params": init_rng}, x, start_pos=0, mode="prefill")
        params = variables.pop("params")
        state = TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=optimizer_transformer,
            rng=rng,
        )
        return state

        ##create optimizer below
    optimizer_transformer = optax.adam(
        learning_rate = optax.warmup_exponential_decay_schedule(
            init_value=0,
            peak_value=1e-3,
            warmup_steps=10,
            transition_steps=1,
            decay_rate=0.99,
        )
    )

    ##create optimizer below
    rng = jax.random.PRNGKey(config.seed)
    model_init_rng, data_inputs_rng = jax.random.split(rng)

    tokens = jax.random.randint(
        data_inputs_rng,
        (config.max_batch_size, config.max_seq_length),
        1,
        config.vocab_size,
    )
    batch_transformer = Batch(
        inputs=jnp.pad(tokens[:, :-1], ((0, 0), (1, 0)), constant_values=0),
        labels=tokens,
    )

    init_transformer_fn = jax.jit(
        shard_map(
            init_transformer,
            mesh,
            in_specs=(P(), P(config.data_axis_name)),
            out_specs=P(),
            check_rep=False,
        ),
    )
    state_transformer_shapes = jax.eval_shape(
        init_transformer_fn, model_init_rng, batch_transformer.inputs
    )
    state_transformer_specs = nn.get_partition_spec(state_transformer_shapes)

    init_transformer_fn = jax.jit(
        shard_map(
            init_transformer,
            mesh,
            in_specs=(P(), P(config.data_axis_name)),
            out_specs=state_transformer_specs,
            check_rep=False,
        ),
    )
    state_transformer = init_transformer_fn(model_init_rng, batch_transformer.inputs)
