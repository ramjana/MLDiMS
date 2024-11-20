from ml_collections import ConfigDict
import functools
from pprint import pprint
from typing import Any, Callable, Dict, Tuple

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


def make_modelConfig(fsdp:ConfigDict,vocab_size:int) -> ConfigDict:
    """
      FIXME:: pass training arguments to this config to generate model_config 
    """
    model_config = ConfigDict(
        dict(
            model_dim=256,
            hidden_dim=1024,
            dropout_rate=0.1,
            mlp_factor=4,
            num_layers=6,
            head_dim=32,
            normalize_qk=True,
            bias_qkv=True,
            bias_attn=True,
            bias_mlpout=True,
            normalize_mlp=True,
            positional_encoding_type="learned",
            parallel_block=False,
            causal_mask=True,
            vocab_size=vocab_size,
            num_outputs=vocab_size,
            dtype=jnp.float16,
            data_axis_name="data",
            model_axis_name="model",
            model_axis_size=4,
            fsdp=fsdp,
        )
    )
    return model_config

def make_dataConfig() -> ConfigDict:
    data_config = ConfigDict(
        dict(
            batch_size=8,
            vocab_size=100,
            seq_len=32,
        )
    )
    return data_config

def make_fsdp() -> ConfigDict:
    fsdp = ConfigDict(
        dict(
            modules=("Transformer",),
            axis_name="data",
            min_weight_size=2**8,
        )
    )
    return fsdp

def make_trainConfig() -> ConfigDict:
    train_config = ConfigDict(
        dict(
            data_type=jnp.float16,
            remat=("Block",),
        )
    )
    return train_config


def get_transformer_module(config: ConfigDict):
    module_class = Transformer
    if config.model.fsdp is not None:
        shard_parameter = True
    else:
        shard_parameter = False
    if config.train.remat is not None and "Transformer" in config.train.remat:
        checkpoint_enable = True
    else:
        checkpoint_enable = False
    module_class = prep_module(
            layer=module_class,
            layer_name="Transformer",
            axis_name=config.data_axis_name,
            fsdp_modules=config.model.fsdp.modules,
            checkpoint_en=checkpoint_enable,
            shard_parameter=shard_parameter,
            shard_size=config.model.fsdp.min_weight_size,
    )
    block_func = TPTransformerParallelBlock if config.model.parallel_block else TPTransformerBlock
    return module_class(llmConfig=config,block_fn=block_func)



##TRaining functions
def loss_fn_transformer(
    params: PyTree,
    apply_fn: Any,
    batch: Batch,
    rng: jax.Array,
    data_axis_name: str,
    model_axis_name: str
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
    labels = split_array_over_mesh(batch.labels, axis_name=model_axis_name, split_axis=1)
    assert (
        logits.shape[:-1] == labels.shape
    ), f"Logits and labels shapes do not match: {logits.shape} vs {labels.shape}"
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    correct_pred = jnp.equal(jnp.argmax(logits, axis=-1), labels)
    batch_size = np.prod(labels.shape)
    # Collect metrics and return loss.
    step_metrics = {
        "loss": (loss.sum(), batch_size),
        "accuracy": (correct_pred.sum(), batch_size),
    }
    loss = loss.mean()
    return loss, step_metrics


def main():

    ## generate configs for run
    fsdp_config = make_fsdp()
    data_config = make_dataConfig()
    model_config = make_modelConfig(fsdp_config,data_config.vocab_size)

    optimizer_config = ConfigDict(
        dict(
            learning_rate=1e-3,
            num_minibatches=1,
        )
    )
    training_config = make_trainConfig()
    model_config.num_heads = model_config.model_dim // model_config.head_dim

    config = ConfigDict(
        dict(
            model=model_config,
            optimizer=optimizer_config,
            data=data_config,
            train=training_config,
            data_axis_name=model_config.data_axis_name,
            model_axis_name=model_config.model_axis_name,
            model_axis_size=model_config.model_axis_size,
            seed=42,
        )
    )
    simulate_CPU_devices(8)
    print("Build nD mesh for distributed training ")
    #define 2D mesh of (dataParallelism,TensorParallelism)
    array_devices = np.array(jax.devices()).reshape(-1,config.model_axis_size)
    mesh = Mesh(array_devices,(config.data_axis_name,config.model_axis_name))

    ##create model
    model_transformer = get_transformer_module(config=config)

    def init_transformer(rng: jax.random.PRNGKey, x: jax.Array) -> TrainState:
        init_rng, rng = jax.random.split(rng)
        variables = model_transformer.init({"params" : init_rng}, x, mask=None, train=False)
        #variables = model_transformer.init({"params": init_rng}, x, train=False)
        params = variables.pop("params")
        state = TrainState.create(
            apply_fn=model_transformer.apply,
            params=params,
            tx=optimizer_transformer,
            rng=rng,
        )
        return state

    ##create optimizer below
    optimizer_transformer = optax.adam(
        learning_rate = optax.warmup_exponential_decay_schedule(
            init_value=0,
            peak_value=config.optimizer.learning_rate,
            warmup_steps=10,
            transition_steps=1,
            decay_rate=0.99,
        )
    )
    rng = jax.random.PRNGKey(config.seed)
    model_init_rng, data_inputs_rng = jax.random.split(rng)

    print("Initialize tokens for training")

    tokens = jax.random.randint(
        data_inputs_rng,
        (config.data.batch_size, config.data.seq_len),
        1,
        config.data.vocab_size,
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

    #print("Input Embedding")
    #pprint(state_transformer_specs.params["input_embedding"])

    #print("Output Layer")
    #pprint(state_transformer_specs.params["output_layer"])

    #print("Transformer Block")
    #pprint(state_transformer_specs.params["backbone"])

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

    #print("Transformer Block - Output Layer, Shard 0")
    if config.model.parallel_block:
        shard_0_params = state_transformer.params["backbone"]["block"]["out"]["shard_0"]["sharded"]
    else:
        shard_0_params = {
            "attn": state_transformer.params["backbone"]["block"]["Attention"]["AttnOutput"]["shard_0"]["sharded"],
            "mlp": state_transformer.params["backbone"]["block"]["mlp"]["MLPInput"]["shard_0"]["sharded"],
        }
    #pprint(
    #    jax.tree.map(
    #        lambda x: x.shape,
    #        shard_0_params,
    #    )
    #)

    train_step_transformer_fn = jax.jit(
       shard_map(
           functools.partial(train_step_tp,
                             loss_fn=loss_fn_transformer, 
                             model_axis_name=config.model_axis_name,
                             data_axis_name=config.data_axis_name,
                             num_minibatches=config.optimizer.num_minibatches),
           mesh,
           in_specs=(state_transformer_specs, P(), P(config.data_axis_name)),
           out_specs=(state_transformer_specs, P()),
           check_rep=False,
       ),
       #donate_argnames=("state", "metrics"),
    )

    state_shapes, metric_shapes = jax.eval_shape(
        train_step_transformer_fn,
        state_transformer,
        None,
        batch_transformer,
    )
    metrics_transformer = jax.tree.map(lambda x: jnp.zeros(x.shape, dtype=x.dtype), metric_shapes)
    state_transformer, metrics_transformer = train_step_transformer_fn(
        state_transformer, metrics_transformer, batch_transformer
    )

    for _ in tqdm(range(50)):
        state_transformer, metrics_transformer = train_step_transformer_fn(
            state_transformer, metrics_transformer, batch_transformer
        )

    final_metrics_transformer = jax.tree.map(
        lambda x: jnp.zeros(x.shape, dtype=x.dtype), metric_shapes
    )

    state_transformer, final_metrics_transformer = train_step_transformer_fn(
        state_transformer, final_metrics_transformer, batch_transformer
    )
    print_metrics(final_metrics_transformer, title="Final Metrics - Transformer")


if __name__ == "__main__":
    main()
