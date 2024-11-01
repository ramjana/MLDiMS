
import functools

from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from typing import Mapping, Optional, List, Tuple

from lightning_utilities.core.imports import RequirementCache

import numpy as np
import torch

import sys
import os
from pathlib import Path

from typing import TYPE_CHECKING, Any

from convert_hf_checkpoints import convert_hf_checkpoint


if TYPE_CHECKING:
    from jsonargparse import ArgumentParser

_SAFETENSORS_AVAILABLE = RequirementCache("safetensors")
_HF_TRANSFER_AVAILABLE = RequirementCache("hf_transfer")


def gen_parser(**kwargs: Any) -> "ArgumentParser":
    from jsonargparse import ActionConfigFile, ArgumentParser

    parser = ArgumentParser(**kwargs)
    parser.add_argument(
        "-c", "--config", action=ActionConfigFile, help="Path to a configuration file in json or yaml format."
    )
    return parser

def generate_base():
    pass

def main(userArgs):
    """ main entry point to the program"""

    from jsonargparse import ActionConfigFile, ArgumentParser

    _parser = gen_parser(prog="MLdiMS")

    generate_base_fn = generate_base

    parser_data = {
        "download": {"help": "Download weights or tokenizer data from the Hugging Face Hub.", "fn": download_from_hub},
    }

    from jsonargparse import set_config_read_mode, set_docstring_parse_options

    set_docstring_parse_options(attribute_docstrings=True)
    set_config_read_mode(urls_enabled=True)


    ## register level1 & level2 sub commands.. 
    subcommands = _parser.add_subcommands()
    subcommand_to_parser = dict()

    for k, v in parser_data.items():
        subcommand_parser = gen_parser()
        if "fn" in v:
            subcommand_parser.add_function_arguments(v["fn"])
        else:
            subcommand_to_parser[k] = subcommand_parser
        subcommands.add_subcommand(k, subcommand_parser, help=v["help"])
    for subcommand, parser in subcommand_to_parser.items():
        subcommands = parser.add_subcommands()
        for k, v in parser_data[subcommand].items():
            if k == "help":
                continue
            subsubcommand_parser = gen_parser()
            subsubcommand_parser.add_function_arguments(v["fn"])
            subcommands.add_subcommand(k, subsubcommand_parser, help=v["help"])

    args = _parser.parse_args()
    args = _parser.instantiate_classes(args)
    subcommand = args.get("subcommand")
    subargs = args.get(subcommand)
    subsubcommand = subargs.get("subcommand")
    subsubargs = subargs.get(subsubcommand) if isinstance(subsubcommand, str) else None

    level_1 = parser_data[subcommand]
    if subsubcommand is None:
        fn = level_1["fn"]
        kwargs = subargs
    else:
        fn = level_1[subsubcommand]["fn"]
        kwargs = subsubargs
    kwargs.pop("config")

    torch.set_float32_matmul_precision("high")

    fn(**kwargs)


def get_torch_state(model_name: str):
    """Downloads the model weights corresponding to model_name"""

    # Download the core model weights and config.
    fname_weights = f"https://dl.fbaipublicfiles.com/fair-esm/models/{model_name}.pt"
    torch_state = torch.hub.load_state_dict_from_url(fname_weights, map_location="cpu")

    return torch_state



def download_from_hub(
    repo_id: str,
    access_token: Optional[str] = os.getenv("HF_TOKEN"),
    convert_checkpoint: bool = True,
    dtype: Optional[str] = None,
    checkpoint_dir: Path = Path("checkpoints"),
    model_name: str = None
) -> None:
    """Download weights or tokenizer data from the Hugging Face Hub.

    Arguments:
        repo_id: The repository ID in the format ``org/name`` or ``user/name`` as shown in Hugging Face.
        access_token: Optional API token to access models with restrictions.
        dtype: The data type to convert the checkpoint files to. If not specified, the weights will remain in the
            dtype they are downloaded in.
        checkpoint_dir: Where to save the downloaded files.
            existing architectures.
    """
    #options = [f"{config['hf_config']['org']}/{config['hf_config']['name']}" for config in configs]

    #if model_name is None and repo_id not in options:
    #    print(f"Unsupported `repo_id`: {repo_id}."
    #    "\nIf you are trying to download models from  huggingface hub"
    #    "weights for a supported model, please specify the corresponding model via the `--model_name` option, "
    #    "for example, `meta-llama/Llama-2-7b`.")
    #    return

    from huggingface_hub import snapshot_download

    download_files = ["tokenizer*", "generation_config.json", "config.json"]
    from_safetensors = False
    bins, safetensors = find_weight_files(repo_id, access_token)
    if bins:
        # covers `.bin` files and `.bin.index.json`
        download_files.append("*.bin*")
    elif safetensors:
        if not _SAFETENSORS_AVAILABLE:
             raise ModuleNotFoundError(str(_SAFETENSORS_AVAILABLE))
        download_files.append("*.safetensors*")
        from_safetensors = True
    else:
        raise ValueError(f"Couldn't find weight files for {repo_id}")

    import huggingface_hub._snapshot_download as download
    import huggingface_hub.constants as constants

    previous = constants.HF_HUB_ENABLE_HF_TRANSFER
    if _HF_TRANSFER_AVAILABLE and not previous:
        print("Setting HF_HUB_ENABLE_HF_TRANSFER=1")
        constants.HF_HUB_ENABLE_HF_TRANSFER = True
        download.HF_HUB_ENABLE_HF_TRANSFER = True

    directory = checkpoint_dir / repo_id
    with gated_repo_catcher(repo_id, access_token):
        snapshot_download(
            repo_id,
            local_dir=directory,
            allow_patterns=download_files,
            token=access_token,
        )

    constants.HF_HUB_ENABLE_HF_TRANSFER = previous
    download.HF_HUB_ENABLE_HF_TRANSFER = previous

    if from_safetensors:
        print("Converting .safetensor files to PyTorch binaries (.bin)")
        safetensor_paths = list(directory.glob("*.safetensors"))
        with ProcessPoolExecutor() as executor:
            executor.map(convert_safetensors_file, safetensor_paths)

    if convert_checkpoint:
        print("Converting checkpoint files to LitGPT format.")
        convert_hf_checkpoint(checkpoint_dir=directory, dtype=dtype, model_name=model_name)


def convert_safetensors_file(safetensor_path: Path) -> None:
    from safetensors import SafetensorError
    from safetensors.torch import load_file as safetensors_load

    bin_path = safetensor_path.with_suffix(".bin")
    try:
        result = safetensors_load(safetensor_path)
    except SafetensorError as e:
        raise RuntimeError(f"{safetensor_path} is likely corrupted. Please try to re-download it.") from e
    print(f"{safetensor_path} --> {bin_path}")
    torch.save(result, bin_path)
    try:
        os.remove(safetensor_path)
    except PermissionError:
        print(
            f"Unable to remove {safetensor_path} file. "
            "This file is no longer needed and you may want to delete it manually to save disk space."
        )


def find_weight_files(repo_id: str, access_token: Optional[str]) -> Tuple[List[str], List[str]]:
    from huggingface_hub import repo_info
    from huggingface_hub.utils import filter_repo_objects

    with gated_repo_catcher(repo_id, access_token):
        info = repo_info(repo_id, token=access_token)
    #print(info)  
    filenames = [f.rfilename for f in info.siblings]
    bins = list(filter_repo_objects(items=filenames, allow_patterns=["*.bin*"]))
    safetensors = list(filter_repo_objects(items=filenames, allow_patterns=["*.safetensors*"]))
    return bins, safetensors


@contextmanager
def gated_repo_catcher(repo_id: str, access_token: Optional[str]):
    try:
        yield
    except OSError as e:
        err_msg = str(e)
        if "Repository Not Found" in err_msg:
            raise ValueError(
                f"Repository at https://huggingface.co/api/models/{repo_id} not found."
                " Please make sure you specified the correct `repo_id`."
            ) from None
        elif "gated repo" in err_msg:
            if not access_token:
                raise ValueError(
                    f"https://huggingface.co/{repo_id} requires authentication, please set the `HF_TOKEN=your_token`"
                    " environment variable or pass `--access_token=your_token`. You can find your token by visiting"
                    " https://huggingface.co/settings/tokens."
                ) from None
            else:
                raise ValueError(
                    f"https://huggingface.co/{repo_id} requires authentication. The access token provided by `HF_TOKEN=your_token`"
                    " environment variable or `--access_token=your_token` may not have sufficient access rights. Please"
                    f" visit https://huggingface.co/{repo_id} for more information."
                ) from None
        raise e from None



def extract(
    key: str, torch_params: Mapping[str, torch.Tensor], delete: bool = False
) -> np.ndarray:
    """Extract the param accessed by key from the torch_params Mapping.

    Args:
        torch_params (Mapping[str, torch.Tensor]): Mapping with original torch params.
        key (str): Name of the parameter to extract.
        delete (bool, optional): If True, delete the original torch param, to
        avoid duplicate memory use. Defaults to False.

    Returns:
        param: The extracted param as a numpy array.
    """
    param = torch_params[key].numpy()

    # Delete torch param to avoid double use of memory
    # by both torch and numpy version of the param.
    if delete:
        del torch_params[key]

    return param


def convert_self_attn(torch_params, cfg, layer_num: int, delete: bool = False):
    """Returns nested dictionary with params for the self-attention module of
    the specified layer."""

    # Obtain the self-attention layer weight specs.
    embed_dim = cfg["model"].encoder_embed_dim
    num_heads = cfg["model"].encoder_attention_heads
    head_dim = embed_dim // num_heads

    # Specify prefix key and initialize nested dict.
    self_attn_key = f"encoder.sentence_encoder.layers.{layer_num}.self_attn"
    proj_names = ["k_proj", "q_proj", "v_proj", "out_proj"]
    params = {key: {} for key in proj_names}

    # Begin extraction.
    extract_fn = functools.partial(extract, torch_params=torch_params, delete=delete)

    for proj_name in proj_names:
        # The out projection, and query/key/value projections have different param shapes.
        if proj_name == "out_proj":
            weight_shape = (num_heads, head_dim, embed_dim)
            bias_shape = (embed_dim,)

        else:
            weight_shape = (embed_dim, num_heads, head_dim)
            bias_shape = (num_heads, head_dim)

        # Get the key, and extract the weight and bias for the projection.
        proj_key = f"{self_attn_key}.{proj_name}"
        weight = extract_fn(f"{proj_key}.weight").T.reshape(weight_shape)
        bias = extract_fn(f"{proj_key}.bias").reshape(bias_shape)

        params[proj_name]["kernel"] = weight
        params[proj_name]["bias"] = bias

    return params


def convert_encoder_layer(torch_params, cfg, layer_num: int, delete: bool = False):
    """Returns nested dictionary with params for the specified layer."""

    # Specify prefix key and initialize nested dict.
    layer_key = f"encoder.sentence_encoder.layers.{layer_num}"
    sublayer_names = ["fc1", "fc2", "self_attn_layer_norm", "final_layer_norm"]
    params = {key: {} for key in sublayer_names}

    # Begin extraction.
    extract_fn = functools.partial(extract, torch_params=torch_params, delete=delete)

    for sublayer_name in sublayer_names:
        sublayer_key = f"{layer_key}.{sublayer_name}"
        weight = extract_fn(f"{sublayer_key}.weight")
        bias = extract_fn(f"{sublayer_key}.bias")

        # If LayerNorm, the weight is a vector to be renamed `scale`.
        if "norm" in sublayer_name:
            params[sublayer_name]["scale"] = weight

        # Else, its a matrix requiring a transpose.
        else:
            weight = weight.T
            params[sublayer_name]["kernel"] = weight
        params[sublayer_name]["bias"] = bias

    # Extract the params for the self attention layer.
    params["self_attn"] = convert_self_attn(torch_params, cfg, layer_num=layer_num)

    return params


def convert_lm_head(torch_params: Mapping[str, torch.Tensor], delete: bool = False):
    """Returns nested dictionary of params needed for the language model head."""
    params = {"lm_head_fc": {}, "lm_head_layer_norm": {}}

    # Begin extraction.
    extract_fn = functools.partial(extract, torch_params=torch_params, delete=delete)

    params["lm_head_fc"]["kernel"] = extract_fn("encoder.lm_head.dense.weight").T
    params["lm_head_fc"]["bias"] = extract_fn("encoder.lm_head.dense.bias")
    params["lm_head_layer_norm"]["scale"] = extract_fn(
        "encoder.lm_head.layer_norm.weight"
    )
    params["lm_head_layer_norm"]["bias"] = extract_fn("encoder.lm_head.layer_norm.bias")
    params["logit_bias"] = extract_fn("encoder.lm_head.bias")

    return params


def convert_encoder(
    torch_params: Mapping[str, torch.Tensor],
    cfg,
    delete: bool = False,
    lm_head: bool = False,
):
    """Returns nested dictionary with params for the full encoder network.

    Args:
        torch_params (Mapping[str, torch.Tensor]): Mapping containing the torch params.
        cfg (dict): Config dict obtained when loading the torch state.
        delete (bool, optional): If True, will remove loaded torch weight from memory once
            converted, to alleviate memory pressure when loading large models. Defaults to False.
        lm_head (bool, optional): If True, `torch_params` will also have params for the language
            model head. Defaults to False.

    Returns:
        params: Nested dictionary of np.ndarrays containing the converted weights.
    """
    # Specify prefix key and initialize nested dict.
    num_layers = cfg["model"].encoder_layers
    params = {"embedding": {}, "post_norm": {}}
    prefix = "encoder.sentence_encoder"

    # Begin extraction.
    extract_fn = functools.partial(extract, torch_params=torch_params, delete=delete)

    # Extract the initial embedding vectors.
    emb_prefix = f"{prefix}.embed_tokens"
    params["embedding"]["embedding"] = extract_fn(f"{emb_prefix}.weight")

    # Extract the weights of the encoder layers.
    for idx in range(num_layers):
        params[f"{idx}"] = convert_encoder_layer(torch_params, cfg, layer_num=idx)

    # Extract the params for the final layer LayerNorm.
    norm_prefix = f"{prefix}.emb_layer_norm_after"
    params["post_norm"]["scale"] = extract_fn(f"{norm_prefix}.weight")
    params["post_norm"]["bias"] = extract_fn(f"{norm_prefix}.bias")

    if lm_head:
        params.update(convert_lm_head(torch_params, delete=delete))

    return params


if __name__ == "__main__":
    main(sys.argv[1])
