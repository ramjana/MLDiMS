from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Type, Union

import torch
import yaml
from typing_extensions import Self
from train_utils import find_multiple


@dataclass
class Config:
    name: str = ""
    hf_config: dict = field(default_factory=dict)
    scale_embeddings: bool = False
    attention_scores_scalar: Optional[int] = None
    block_size: int = 4096
    sliding_window_size: Optional[int] = None
    sliding_window_layer_placing: Optional[Literal["all", "interleaved"]] = None
    vocab_size: int = 50254
    padding_multiple: int = 512
    padded_vocab_size: Optional[int] = None
    n_layer: int = 16
    n_head: int = 32
    head_size: Optional[int] = None
    n_embd: int = 4096
    rotary_percentage: float = 0.25
    parallel_residual: bool = True
    bias: bool = True
    lm_head_bias: bool = False
    n_query_groups: Optional[int] = None
    shared_attention_norm: bool = False
    norm_class_name: Literal["LayerNorm", "RMSNorm"] = "LayerNorm"
    post_attention_norm: bool = False
    post_mlp_norm: bool = False
    norm_eps: float = 1e-5
    mlp_class_name: Literal["LLaMAMLP","LLaMAMoE"] = "GptNeoxMLP"
    gelu_approximate: str = "none"
    intermediate_size: Optional[int] = None
    rope_condense_ratio: int = 1
    rope_base: int = 10000
    rope_adjustments: Optional[dict] = None
    n_expert: int = 0
    n_expert_per_token: int = 0
    attention_logit_softcapping: Optional[float] = None
    final_logit_softcapping: Optional[float] = None

    def __post_init__(self):
        if not self.name:
            self.name = self.hf_config.get("name", self.name)

        if self.head_size is None:
            assert self.n_embd % self.n_head == 0
            self.head_size = self.n_embd // self.n_head

        # vocab size should be a power of 2 to be optimal on hardware. compute the closest value
        if self.padded_vocab_size is None:
            self.padded_vocab_size = find_multiple(self.vocab_size, self.padding_multiple)
        else:
            # vocab size shouldn't be larger than padded vocab size
            self.vocab_size = min(self.vocab_size, self.padded_vocab_size)

        # compute the number of query groups
        if self.n_query_groups is not None:
            assert self.n_head % self.n_query_groups == 0
        else:
            self.n_query_groups = self.n_head

        # compute the intermediate size for MLP if not set
        if self.intermediate_size is None:
            if self.mlp_class_name == "LLaMAMLP":
                raise ValueError(f"The config {self.name!r}, needs to set the `intermediate_size`")
            self.intermediate_size = 4 * self.n_embd

        self.rope_n_elem = int(self.rotary_percentage * self.head_size)

        if self.sliding_window_size is not None:
            self.sliding_window_layer_placing = (
                1 if (self.sliding_window_layer_placing is None or self.sliding_window_layer_placing == "all") else 2
            )

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Optional[Self]:
        if name not in name_to_config:
            # search through all `config['hf_config']['name']`
            try:
                conf_dict = next(
                    config
                    for config in configs
                    if name == config["hf_config"]["name"]
                    or config["hf_config"]["org"] + "/" + config["hf_config"]["name"] == name
                )
            except StopIteration:
                raise ValueError(f"{name!r} is not a supported config name")
        else:
            conf_dict = name_to_config[name]

        conf_dict = conf_dict.copy()
        conf_dict.update(kwargs)
        return cls(**conf_dict)

    @classmethod
    def from_file(cls, path: Union[str, Path], **kwargs: Any) -> Self:
        with open(path, encoding="utf-8") as fp:
            file_kwargs = yaml.safe_load(fp)
            if file_kwargs is None:
                raise ValueError(f"{path} is empty which is likely unexpected.")
        file_kwargs.update(kwargs)
        return cls(**file_kwargs)

    @classmethod
    def from_checkpoint(cls, path: Path, **kwargs: Any) -> Self:
        """Automatically load `model_config.yaml` and if it doesn't exist - a matching config from `litgpt/config.py`."""
        if (config_path := path / "model_config.yaml").is_file():
            return cls.from_file(config_path, **kwargs)
        if (model_name := path.name) in name_to_config:
            return cls.from_name(model_name, **kwargs)
        raise FileNotFoundError(f"For {str(path)!r} neither 'model_config.yaml' nor matching config exists.")

    @property
    def mlp_class(self) -> Type:
        # `self.mlp_class_name` cannot be the type to keep the config serializable
        import litgpt.model
        return getattr(litgpt.model, self.mlp_class_name)

    @property
    def norm_class(self) -> Type:
        # `self.norm_class_name` cannot be the type to keep the config serializable
        if self.norm_class_name == "RMSNorm":
            from functools import partial

            from core.layers.baseops import RMSNorm
            import flax.linen as nn

            return partial(RMSNorm, add_unit_offset="Gemma" in self.name)
        return getattr(nn, self.norm_class_name)



configs = []

###############
# Meta LLaMA 2
###############
llama_2 = [
    # https://huggingface.co/meta-llama/Llama-2-7b-hf/blob/main/config.json
    dict(
        name="Llama-2-7b{}-hf",
        hf_config=dict(org="meta-llama", name="Llama-2-7b{}-hf"),
        vocab_size=32000,
        padding_multiple=64,
        n_layer=32,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=11008,
    ),
    # https://huggingface.co/meta-llama/Llama-2-13b-hf/blob/main/config.json
    dict(
        name="Llama-2-13b{}-hf",
        hf_config=dict(org="meta-llama", name="Llama-2-13b{}-hf"),
        vocab_size=32000,
        padding_multiple=64,
        n_layer=40,
        n_head=40,
        n_embd=5120,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=13824,
    ),
    # https://huggingface.co/meta-llama/Llama-2-70b-hf/blob/main/config.json
    dict(
        name="Llama-2-70b{}-hf",
        hf_config=dict(org="meta-llama", name="Llama-2-70b{}-hf"),
        vocab_size=32000,
        padding_multiple=64,
        n_layer=80,
        n_head=64,
        n_embd=8192,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=28672,
    ),
]
for c in llama_2:
    for kind in ("", "-chat"):
        copy = deepcopy(c)
        copy["name"] = c["name"].format(kind)
        copy["hf_config"]["name"] = c["hf_config"]["name"].format(kind)
        configs.append(copy)


###############
# Meta LLaMA 3
###############
llama_3 = [
    # https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/config.json
    dict(
        name="Llama-3-8B{}",
        hf_config=dict(org="meta-llama", name="Meta-Llama-3-8B{}"),
        block_size=8192,
        vocab_size=128000,
        padded_vocab_size=128256,
        n_layer=32,
        n_head=32,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=14336,
        rope_base=500000,
    ),
    # https://huggingface.co/meta-llama/Meta-Llama-3.1-8B/blob/main/config.json
    dict(
        name="Llama-3.1-8B{}",
        hf_config=dict(org="meta-llama", name="Meta-Llama-3.1-8B{}"),
        block_size=131072,
        vocab_size=128000,
        padded_vocab_size=128256,
        n_layer=32,
        n_head=32,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=14336,
        rope_base=500000,
        rope_adjustments=dict(factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_seq_len=8192)
    ),
    # https://huggingface.co/meta-llama/Meta-Llama-3-70B/blob/main/config.json
    dict(
        name="Llama-3-70B{}",
        hf_config=dict(org="meta-llama", name="Meta-Llama-3-70B{}"),
        block_size=8192,
        vocab_size=128000,
        padded_vocab_size=128256,
        n_layer=80,
        n_head=64,
        n_embd=8192,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=28672,
        rope_base=500000,
    ),
    # https://huggingface.co/meta-llama/Meta-Llama-3.1-70B/blob/main/config.json
    dict(
        name="Llama-3.1-70B{}",
        hf_config=dict(org="meta-llama", name="Meta-Llama-3.1-70B{}"),
        block_size=131072,
        vocab_size=128000,
        padded_vocab_size=128256,
        n_layer=80,
        n_head=64,
        n_embd=8192,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=28672,
        rope_base=500000,
        rope_adjustments=dict(factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_seq_len=8192)
    ),
    # https://huggingface.co/meta-llama/Meta-Llama-3.1-405B/blob/main/config.json
    dict(
        name="Llama-3.1-405B{}",
        hf_config=dict(org="meta-llama", name="Meta-Llama-3.1-405B{}"),
        block_size=131072,
        vocab_size=128000,
        padded_vocab_size=128256,
        n_layer=126,
        n_head=128,
        n_embd=16384,
        n_query_groups=16,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=53248,
        rope_base=500000,
        rope_adjustments=dict(factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_seq_len=8192)
    ),
    # https://huggingface.co/meta-llama/Llama-3.2-1B/blob/main/config.json
    dict(
        name="Llama-3.2-1B{}",
        hf_config=dict(org="meta-llama", name="Llama-3.2-1B{}"),
        block_size=131072,
        vocab_size=128000,
        padded_vocab_size=128256,
        n_layer=16,
        n_embd=2048,
        n_head=32,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=8192,
        rope_base=500000,
        rope_adjustments=dict(factor=32.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_seq_len=8192)
    ),
    # https://huggingface.co/meta-llama/Llama-3.2-3B/blob/main/config.json
    dict(
        name="Llama-3.2-3B{}",
        hf_config=dict(org="meta-llama", name="Llama-3.2-3B{}"),
        block_size=131072,
        vocab_size=128000,
        padded_vocab_size=128256,
        n_layer=28,
        n_embd=3072,
        n_head=24,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=8192,
        rope_base=500000,
        rope_adjustments=dict(factor=32.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_seq_len=8192)
    ),
]

for c in llama_3:
    for kind in ("", "-Instruct"):
        copy = deepcopy(c)
        copy["name"] = c["name"].format(kind)
        copy["hf_config"]["name"] = c["hf_config"]["name"].format(kind)
        configs.append(copy)

name_to_config = {config["name"]: config for config in configs}
