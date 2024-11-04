
import sys
import jax
import os


from configs import mlconfig
from configs.mlconfig import llmConfig
from utils import engine_utils
import MLMSEngine

import jax.numpy as jnp


def main(config:llmConfig):
    
    engine = MLMSEngine.MLMSEngine(config)
    rng = jax.random.PRNGKey(1234)
    rng,rng_load_params = jax.random.split(rng)
    params = engine.load_params(rng=rng_load_params)

    text = config.prompt
    tokenizer_model = engine.get_tokenizer()
    tokens,true_length = tokenizer_model.encode(text, prefill_lengths=config.max_prefill_predict_len)
    assert true_length <= config.max_prefill_predict_len, "cannot take too many tokens"

    #split RNG before calling prefill
    rng,rng_prefill = jax.random.split(rng)
    print("calling prefill")
    tokens = jnp.array(tokens)
    prefill_result, first_token = engine.prefill(params=params,padded_tokens=tokens,true_length=true_length,rng=rng_prefill)

    print("completed prefill")

    rng,rng_init_decode = jax.random.split(rng)
    print("calling init_decode_state")
    decode_state = engine.init_decode_state(rng_init_decode)
    print("calling engine.insert")
    #decode_state = engine.insert(prefill_result,decode_state,slot=0)

    steps = range(config.max_prefill_predict_len,config.max_seq_length)
    sampled_tokens_list = []
    sampled_tokens_list.append(first_token)

    print("calling engine.append")
    for _ in steps:
        rng, rng_append = jax.random.split(rng)
        decode_state, sampled_tokens = engine.append(params,decode_state,rng=rng_append)
        sampled_tokens_list.append(sampled_tokens)

    results = [sampled_token for sampled_token in sampled_tokens_list]

    output = tokenizer_model.decode(results)
    print(f" input `{text}` - `{output}`")

if __name__ == "__main__":
    mlconfig.initialize(sys.argv)
    cfg = mlconfig.config

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
    cfg.shard_axes = _axes
    main(cfg)
