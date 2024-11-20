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
from configs import mlconfig
from configs.mlconfig import llmConfig
from jax.experimental import mesh_utils

PyTree = Any
Parameter = jax.Array | nn.Partitioned
Metrics = Dict[str, Tuple[jax.Array, ...]]

from core.models.llama2 import get_transformer_module
from core.utilities.utils import *
from utils.train_utils import TrainState,Batch,train_step_tp,loss_fn_tp,init_tp,print_metrics
from core.utilities.parallelism_functions import prep_module,fold_rng_over_axis,split_array_over_mesh
from utils import tokenizer

##TRaining functions
def loss_fn_transformer(
    params: PyTree,
    apply_fn: Any,
    batch: Batch,
    rng: jax.Array,
    model_axis_name: str,
    data_axis_name: str
) -> Tuple[jax.Array, Dict[str, Any]]:
    # Since dropout masks vary across the batch dimension, we want each device to generate a
    # different mask. We can achieve this by folding the rng over the data axis, so that each
    # device gets a different rng and thus mask.
    dropout_rng = fold_rng_over_axis(rng, (data_axis_name, model_axis_name))
    # Remaining computation is the same as before for single device.
    logits = apply_fn(
        {"params": params},
        batch.inputs,
        start_pos=0,
        mode="Training",
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

def get_tokenizer(tokenizer_path):
    # Load tokenizer
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(dir_path,"../../","configs")
    tokenizer_model = tokenizer.build_tokenizer(os.path.join(file_path,tokenizer_path), True,False)
    return tokenizer_model


def main(userArgs):

    mlconfig.initialize(sys.argv)
    config = mlconfig.config

    print("Build nD mesh for distributed training ")
    simulate_CPU_devices(config.num_devices)


    devices = np.array(jax.devices()).reshape(-1,config.model_axis_size)
    mesh = Mesh(devices,(config.data_axis_name,config.model_axis_name))
    print(mesh)
    ##create model
    model = get_transformer_module(config=config)

    def init_transformer(rng: jax.random.PRNGKey, x: jax.Array) -> TrainState:
        init_rng, rng = jax.random.split(rng)
        variables = model.init({"params": init_rng}, x, start_pos=0, mode="Training")
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
            peak_value=config.learning_rate,
            warmup_steps=config.warmup_steps,
            transition_steps=1,
            decay_rate=0.99,
        )
    )
    rng = jax.random.PRNGKey(config.seed)
    model_init_rng, data_inputs_rng = jax.random.split(rng)

    print("Initialize tokens for training")

    tokens = jax.random.randint(
        data_inputs_rng,
        (config.max_batch_size, config.max_seq_length),
        1,
        config.vocab_size,
    )
    print(tokens.shape)
    
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

    train_step_transformer_fn = jax.jit(
       shard_map(
           functools.partial(train_step_tp,
                             loss_fn=loss_fn_transformer, 
                             data_axis_name=config.data_axis_name,
                             model_axis_name=config.model_axis_name,
                             num_minibatches=1),
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
    
    print(f" Train for 10 steps")

    for _ in tqdm(range(10)):
        state_transformer, metrics_transformer = train_step_transformer_fn(
            state_transformer, metrics_transformer, batch_transformer
    )

    print_metrics(metrics_transformer, title="Metrics - llama2 Transformer")
    final_metrics_transformer = jax.tree.map(
        lambda x: jnp.zeros(x.shape, dtype=x.dtype), metric_shapes
    )

    print("collecting final metrics from across devices")
    state_transformer, final_metrics_transformer = train_step_transformer_fn(
        state_transformer, final_metrics_transformer, batch_transformer
    )
    print_metrics(final_metrics_transformer, title="Final Metrics - llama2 Transformer")


if __name__ == "__main__":
    main(sys.argv)
