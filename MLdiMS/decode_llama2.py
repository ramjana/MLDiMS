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

def get_tokenizer(tokenizer_path):
    # Load tokenizer
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(dir_path,"../../","configs")
    tokenizer_model = tokenizer.build_tokenizer(os.path.join(file_path,tokenizer_path), True,False)
    return tokenizer_model


def main(userArgs):

    mlconfig.initialize(sys.argv)
    config = mlconfig.config

    print("Build nD mesh for distributed inference ")
    simulate_CPU_devices(config.num_devices)

    if config.single_layer_sim:
        config.num_layers = 1
        print("simulating single layer ...")

    devices = np.array(jax.devices()).reshape(-1,config.model_axis_size)
    mesh = Mesh(devices,(config.data_axis_name,config.model_axis_name))
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

    def prefill_step(state,batch,start_pos):
        print("prefill -step")
        rng, step_rng = jax.random.split(state.rng)
        logits = model.apply({'params' : state.params},
                 batch.inputs,
                 start_pos=0,
                 mode="prefill",
                 rngs={"params": step_rng},
                 mutable=["kvcache"],
                 )
        first_token = jnp.argmax(logits[0], axis=-1)
        labels = split_array_over_mesh(batch.labels, axis_name=config.model_axis_name, split_axis=1)
        assert (
            logits[0].shape[:-1] == labels.shape
        ), f"Logits and labels shapes do not match: {logits[0].shape} vs {labels.shape}"

        token_logprobs = optax.softmax_cross_entropy_with_integer_labels(logits[0], labels)
        return token_logprobs

    def generate_step(state,batch,cur_pos):
        print("generation -step")
        rng, step_rng = jax.random.split(state.rng)
        logits = model.apply({'params' : state.params},
                 batch.inputs,
                 start_pos=cur_pos,
                 mode="generate",
                 rngs={"params": step_rng},
                 mutable=["kvcache"],
                 )
        logits = jax.lax.all_gather(logits,axis_name="data",axis=0, tiled=True)
        logits = logits[0]
        next_token = jnp.argmax(logits[:, -1], axis=-1)
        return next_token

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

    print("Initialize tokens for inference")

    tokens = jax.random.randint(
        data_inputs_rng,
        (config.max_batch_size, int(config.prompt_len)+int(config.gen_len)),
        1,
        config.vocab_size,
    )
    
    batch_transformer = Batch(
        inputs=jnp.pad(tokens[:, :config.prompt_len-1], ((0, 0), (1, 0)), constant_values=0),
        labels=tokens[:, :config.prompt_len]
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
            init_transformer_fn, model_init_rng, batch_transformer.inputs[:, :config.prompt_len]
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
    state_transformer = init_transformer_fn(model_init_rng, batch_transformer.inputs[:, :config.prompt_len])

    prefill_step_transformer_fn = jax.jit(
       shard_map(
           functools.partial(prefill_step),
           mesh,
           in_specs=(state_transformer_specs, P(config.data_axis_name),P()),
           out_specs=(P()),
           check_rep=False,
       ),
       #donate_argnames=("state", "metrics"),
    )

    #prefill case
    print(f"running prefill stage for prompt-len = {config.prompt_len}")
    prev_pos  = 0
    for _ in tqdm(range(1)):
        #batch_transformer.inputs = batch_transformer.inputs[:, :prompt_len]
        token_logprobs = prefill_step_transformer_fn(
            state_transformer,batch_transformer,prev_pos
        )

    generate_step_transformer_fn = jax.jit(
       shard_map(
           functools.partial(generate_step),
           mesh,
           in_specs=(state_transformer_specs,P(config.data_axis_name),P()),
           out_specs=(P()),
           check_rep=False,
       ),
       #donate_argnames=("state", "metrics"),
    )
    #rework on appending generated token to input
    #
    print(f"running generation stage for gen = {config.gen_len}")
    prev_pos = int(config.prompt_len)-1 
    total_len = int(config.prompt_len) + int(config.gen_len)
    for cur_pos in range(config.prompt_len,total_len):
        batch_transformer = Batch(
             inputs=tokens[:, prev_pos:cur_pos],
             labels=tokens[:, cur_pos:cur_pos]
        )
        mask = None
        #target = batch_transformer.labels[:, prev_pos+1:cur_pos+1]
        next_token = generate_step_transformer_fn(
                state_transformer,batch_transformer,prev_pos
        )
        tokens = tokens.at[:, cur_pos].set(next_token)
        prev_pos = cur_pos
        print("next token...")

if __name__ == "__main__":
    main(sys.argv)

