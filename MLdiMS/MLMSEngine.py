import functools
import sys
import os
from pathlib import Path

import flax
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
from flax import struct


import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as PS

from typing import Any, Callable, Optional

from utils.engine_utils import (
    create_device_mesh,
    setup_decode_state,
    get_kvcache_annotations,
    unbox_logicallypartioned,
)

from utils import tokenizer

from core.models.base_model import Transformer
from configs import mlconfig
from configs.mlconfig import llmConfig
from core.utilities.utils import simulate_CPU_devices

Prefix = Any
Params = Any


@struct.dataclass
class DecodeState:
  """The inputs into a generation step."""

  prefill_cache: jax.Array
  generate_cache: jax.Array
  generate_cache_index: int
  generate_lengths: jax.Array
  generated_token: jax.Array

class MLMSEngine():
    """
       Core simulation engine 

    """

    def __init__(self,cfg: llmConfig):
        self.config = cfg

        #initialize jax devices
        simulate_CPU_devices(self.config.num_devices)

        #create mesh based on user input parallelism
        devices_array = create_device_mesh(self.config)
        self.mesh = jax.sharding.Mesh(devices_array, self.config.mesh_axes)

        #model definition
        self.model = Transformer(config=self.config,mesh=self.mesh)

        self.replicatedsharding = jax.sharding.NamedSharding(self.mesh,PS(None))

        self.anbstract_params= None
        self.kvcache_shardings = None
        self.kvcache_annotations = None
        self.kvcache_annotations_named = None
        self.state_mesh_annotations = None

    def load_params(self,*args, rng : Optional[jax.random.PRNGKey] = None, **kwargs) -> Params:

        """ Load parameters  """

        if rng is None:
            rng = jax.random.PRNGKey

        rng1,rng2,rng3 = jax.random.split(rng,3)

        print("Calling setup_decode_State")
        state, self.state_mesh_annotations = setup_decode_state(self.model,self.config,rng1,self.mesh,None)

        self.abstract_params = jax.tree_util.tree_map(
            lambda x:jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype,sharding=x.sharding), state.params
        )
        self.kvcache_annotations = get_kvcache_annotations(self.model,self.config,rng2,self.mesh)

        self.kvcache_shardings = jax.tree_util.tree_map(
            lambda x:jax.sharding.NamedSharding(self.mesh,x), self.kvcache_annotations
        )

        params = state.params
        #print_mem_stats("After load_params")
        return params

    @functools.partial(jax.jit, static_argnums=(0,))
    def prefill(self,
            *,
            params: Params,
            padded_tokens: jax.Array,
            true_length: int,
            sampler: Optional[Callable[[Any], Any]] = None,
            rng: Optional[jax.random.PRNGKey] = None):
        """ returns KVcache for a new request  """

        if rng is None:
            rng = jax.random.PRNGKey(0)

        input_tokens = jnp.expand_dims(padded_tokens,0)   # batch,seq
        positions = jnp.expand_dims(jnp.arange(0,input_tokens.shape[1]),0)

        zero_2_n = jnp.arange(0,padded_tokens.shape[0])
        ones_to_keep = zero_2_n<true_length
        sequence_indicator = jnp.expand_dims(ones_to_keep*1,0)

        rng,new_rng = jax.random.split(rng)

        with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
            _logits, new_kvstates = self.model.apply(
                    params,
                    input_tokens,
                    positions,
                    decoder_segment_ids = sequence_indicator,
                    mode="prefill",
                    rngs={"params": new_rng},
                    mutable=["kvcache"],
            )
        first_token = jnp.argmax(_logits, axis=-1)
        generated_tokens = jnp.zeros((1, 1), dtype=jnp.int32)
        next_pos = jnp.full((1,1), true_length, dtype=jnp.int32)
        return {
            "logits" : _logits,
            "kvcache" : new_kvstates["kvcache"],
            "next_pos" : next_pos,
            "generated_tokens": generated_tokens,
            "tokens": first_token,
        }, first_token

    @functools.partial(jax.jit, static_argnums=(0,), donate_argnums=(2,))
    def append(
        self,
        params: Params,
        decode_state: DecodeState,
        rng: Optional[jax.random.PRNGKey] = None,
        ) -> DecodeState:
        """Run one generate step"""
        if rng is None:
            rng = jax.random.PRNGKey(0)

        previous_token = decode_state["tokens"]

        rng, new_rng = jax.random.split(rng)

        # run one step generation
        with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
            _logits, kvcache_states = self.model.apply(
                params | {"kvcache": decode_state["kvcache"]},
                previous_token,
                decode_state["next_pos"],
                mode="append",
                rngs={"params": new_rng},
                mutable=["kvcache"],
            )
        new_cache = jax.lax.with_sharding_constraint(kvcache_states["kvcache"], self.kv_cache_shardings)
        new_token = jnp.argmax(_logits, axis=-1)
        all_valid = jnp.ones(new_token.shape, dtype=jnp.int8)
        return {
            "logits" : _logits,
            "kvcache" : new_cache,
            "next_pos" : decode_state["next_pos"]+1,
            "generated_tokens": decode_state["generated_tokens"] + 1,
            "tokens": new_token }, new_token

    @functools.partial(
    jax.jit,
    static_argnums=(0,),
    donate_argnums=(
        1,
        2,
    ),
    )
    def insert(
        self,
        prefix: Prefix,
        decode_state: DecodeState,
        slot: int,
        ) -> DecodeState:
        """Insert into KV cache"""
        unboxed_prefix = unbox_logicallypartioned(prefix)

        def copy(path, partial_cache, full_cache, annotations):
            path_key = path[-1].key
            if path_key in ["kvcached_append_index", "kvcached_append_key", "kvcached_append_value"]:
                return full_cache  # we don't even zero these out because we can mask them out.

            batch_idx = -1
            if "kvcache_batch" in annotations:
                batch_idx = annotations.index("kvcache_batch")

            if batch_idx < 0:
                raise ValueError(f"Batch index {batch_idx=} shouldn't be less than zero for {path_key}, got {annotations=}")

            if path_key == "kvcached_append_segment_id":
                ### goal: zero this out in case there is existing data
                s = list(full_cache.shape)
                s[batch_idx] = 1
                zeros = jnp.zeros(tuple(s), dtype=jnp.int32)
                return jax.lax.dynamic_update_index_in_dim(full_cache, zeros, slot, batch_idx)
            elif path_key == "kvcached_prefill_segment_id":
                s = list(full_cache.shape)
                s[batch_idx] = 1
                zeros = jnp.zeros(tuple(s), dtype=jnp.int32)
                ## zero out in case prefill cache is too small to cover
                full_cache = jax.lax.dynamic_update_index_in_dim(full_cache, zeros, slot, batch_idx)
                ## copy prefill cachce
                full_cache = jax.lax.dynamic_update_index_in_dim(full_cache, partial_cache, slot, batch_idx)
                return full_cache
            elif path_key == "kvcached_append_lengths":
                return full_cache.at[slot].set(0)
            elif path_key in [ "kvcached_prefill_key","kvcached_prefill_value"]:
                return jax.lax.dynamic_update_index_in_dim(full_cache, partial_cache, slot, batch_idx)
            else:
                raise ValueError(f"We don't have a strategy for inserting {path_key}")

    
        #inserted_cache = jax.tree_util.tree_map_with_path(
        #    copy, unboxed_prefix["kvcache"], decode_state["kvcache"], self.kvcache_annotations_named
        #)
        inserted_cache = unboxed_prefix["kvcache"]
        inserted_logits = jax.lax.dynamic_update_index_in_dim(decode_state["logits"], unboxed_prefix["logits"], slot, 0)
        inserted_next_pos = jax.lax.dynamic_update_index_in_dim(decode_state["next_pos"], unboxed_prefix["next_pos"], slot, 0)
        inserted_generated_tokens = jax.lax.dynamic_update_index_in_dim(
            decode_state["generated_tokens"], unboxed_prefix["generated_tokens"], slot, 0
        )
        inserted_tokens = jax.lax.dynamic_update_index_in_dim(decode_state["tokens"], unboxed_prefix["tokens"], slot, 0)

        inserted_logits = jax.lax.with_sharding_constraint(inserted_logits, self.replicated_sharding)
        inserted_generated_tokens = jax.lax.with_sharding_constraint(inserted_generated_tokens, self.replicated_sharding)
        inserted_next_pos = jax.lax.with_sharding_constraint(inserted_next_pos, self.replicated_sharding)
        inserted_tokens = jax.lax.with_sharding_constraint(inserted_tokens, self.replicated_sharding)
        inserted_cache = jax.lax.with_sharding_constraint(inserted_cache, self.kvcache_shardings)

        return {
            "logits": inserted_logits,
            "kvcache": inserted_cache,
            "next_pos": inserted_next_pos,
            "generated_tokens": inserted_generated_tokens,
            "tokens": inserted_tokens,
        }

    def get_prefix_destination_sharding(self) -> Any:
        return jax.sharding.NamedSharding(mesh=self.mesh, spec=jax.sharding.PartitionSpec())

    def get_tokenizer(self):
        # Load tokenizer
        dir_path = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(dir_path,"configs") 
        tokenizer_model = tokenizer.build_tokenizer(os.path.join(file_path,self.config.tokenizer_path), self.config.add_bos, self.config.add_eos)
        return tokenizer_model

    def init_decode_state(
        self,
        abstract_params: Params,
        rng: Optional[jax.random.PRNGKey] = None,
        ) -> DecodeState:
        """Initialises any state which a generation step transforms."""

        if rng is None:
            rng = jax.random.PRNGKey(0)

        def init(abstract_params):
            x = jnp.ones(
                (int(self.config.per_device_batch_size * jax.device_count()), self.config.max_prefill_predict_len),
                dtype=jnp.int32,
            )
            _, cache = self.model.apply(
                abstract_params,
                x,
                x,
                decoder_segment_ids=jnp.zeros(x.shape, dtype=jnp.int32) + 1,
                mode="prefill",
                rngs={"params": rng},
                mutable=["kvcache"],
            )

            next_pos = jnp.zeros((int(self.config.per_device_batch_size * jax.device_count()), 1), dtype=jnp.int32)
            generated_tokens = jnp.zeros((int(self.config.per_device_batch_size * jax.device_count()), 1), dtype=jnp.int32)
            tokens = jnp.zeros((int(self.config.per_device_batch_size * jax.device_count()), 1), dtype=jnp.int32)
            return {
                 "logits": jnp.zeros((int(self.config.per_device_batch_size * jax.device_count()), 1, self.config.vocab_size)),
                 "kvcache": cache["kvcache"],
                 "next_pos": next_pos,
                 "generated_tokens": generated_tokens,
                 "tokens": tokens,
            }

        with nn_partitioning.axis_rules(self.config.logical_axis_rules):
            abstract_outputs = jax.eval_shape(init, self.abstract_params)
            logical_annotations = nn.get_partition_spec(abstract_outputs)

        with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
            mesh_annotations = nn.logical_to_mesh(logical_annotations)

        shardings = jax.tree_util.tree_map(
            lambda mesh_annotation: jax.sharding.NamedSharding(self.mesh, mesh_annotation), mesh_annotations
        )

        @functools.partial(jax.jit, out_shardings=shardings)
        def initialize():
            return jax.tree_util.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), abstract_outputs)

        cache = initialize()["kvcache"]

        def is_lp(k):
            return isinstance(k, flax.linen.spmd.LogicallyPartitioned)

        #self.kv_cache_annotations_named = jax.tree_util.tree_map(lambda x: tuple(x['decoder']), cache, is_leaf=is_lp)
        del cache
        zeroed = unbox_logicallypartioned(initialize())
        return zeroed

