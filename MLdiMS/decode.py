
import jax
import engine_utils
import MLMSEngine

import os
from configs import mlconfig
import sys


def main(config:llmConfig):
    
    engine = MLMSEngine.MLMSEngine(config)
    rng = jax.random.PRNGKey(1234)
    rng,rng_load_params = jax.random.split(rng)
    params = engine.load_params(rng_load_params)

    text = config.prompt
    tokenizer_model = engine.get_tokenizer()
    tokens,true_length = tokenizer_model.encode(test, is_bos=True,prefill_lengths=[config.max_prefill_predict_length])
    assert true_length <= config.max_prefill_predict_length, "cannot take too many tokens"

    #split RNG before calling prefill
    rng,rng_prefill = jax.random.split(rng)
    prefill_result, first_token = engine.prefill(params=params,padded_tokens=tokens,true_length=true_length,rng=rng_prefill)

    rng,rng_init_decode = jax.random.split(rng)
    decode_state = engine.init_decode_state(rng_init_decode)
    decode_state = engine.insert(prefill_result,decode_state,slot=0)

    steps = range(config.max_prefill_predict_len,config.max_output_len)
    sampled_tokens_list = []
    sampled_tokens_list.append(first_token)

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

    main(cfg)
