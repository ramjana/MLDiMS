import sys
import time
import os
import pickle
import torch
import yaml
import gc
import json

from pathlib import Path
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, List, Literal, Optional, Union, Tuple, Dict
from typing import IO, TYPE_CHECKING,Set, Protocol, runtime_checkable, Iterator
from typing_extensions import override, overload
from tqdm import tqdm
import torch.nn as nn
from pprint import pprint
import numpy as np
from dataclasses import asdict, is_dataclass
from collections import defaultdict
from functools import partial
from hf_config import Config
from io import BytesIO
import warnings

_PATH = Union[str, Path]

_PYTHON_GREATER_EQUAL_3_10_0 = (sys.version_info.major, sys.version_info.minor) >= (3, 10)

def extend_checkpoint_dir(checkpoint_dir: Path) -> Path:
    new_checkpoint_dir = "checkpoints" / checkpoint_dir
    should_return_new_dir = (not checkpoint_dir.is_dir() and
                             checkpoint_dir.parts[0] != "checkpoints" and
                             not checkpoint_dir.is_absolute() and
                             new_checkpoint_dir.exists())
    return new_checkpoint_dir if should_return_new_dir else checkpoint_dir




class LazyLoadingUnpickler(pickle.Unpickler):
    def __init__(self, file: IO, file_reader: torch.PyTorchFileReader) -> None:
        super().__init__(file)
        self.file_reader = file_reader

    @override
    def find_class(self, module: str, name: str) -> Any:
        return super().find_class(module, name)

    @override
    def persistent_load(self, pid: tuple) -> "TypedStorage":
        from torch.storage import TypedStorage

        _, cls, _, _, _ = pid
        with warnings.catch_warnings():
            # The TypedStorage APIs have heavy deprecations in torch, suppress all these warnings for now
            warnings.simplefilter("ignore")
            storage = TypedStorage(dtype=cls().dtype, device="meta")
        storage.archiveinfo = pid
        return storage


def lazy_load(filename: _PATH) -> Any:
    if not os.path.isfile(filename):
        raise FileNotFoundError("Path {str(filename)!r} does not exist or is not a file.")
    file_reader = torch.PyTorchFileReader(str(filename))
    with BytesIO(file_reader.get_record("data.pkl")) as pkl:
       fp = LazyLoadingUnpickler(pkl,file_reader)
       return fp.load()

 
def save_config(config: "Config", checkpoint_dir: Path) -> None:
    config_dict = asdict(config)
    with open(checkpoint_dir / "model_config.yaml", "w", encoding="utf-8") as fp:
        yaml.dump(config_dict, fp)


def load_checkpoint(model: nn.Module, checkpoint_path: Path, strict: bool = True) -> None:
    ##FIXME replace lazy_load with torch.load here
    state_dict = lazy_load(checkpoint_path)
    state_dict = state_dict.get("model", state_dict)
    model.load_state_dict(state_dict, strict=strict)

class incremental_save:
    def __init__(self, name):
        self.name = name
        self.zipfile = torch._C.PyTorchFileWriter(str(name))
        self.has_saved = False
        self.next_key = 0

    def __enter__(self):
        return self

    def store_early(self, tensor):
        if isinstance(tensor, torch.Tensor):
            return SavingProxyForTensor(tensor, self)
        raise TypeError(f"can only store tensors early, not {type(tensor)}")

    def save(self, obj):
        if self.has_saved:
            raise RuntimeError("have already saved")
        # Write the pickle data for `obj`
        data_buf = BytesIO()
        pickler = IncrementalPyTorchPickler(self, data_buf, protocol=5)
        pickler.dump(obj)
        data_value = data_buf.getvalue()
        self.zipfile.write_record("data.pkl", data_value, len(data_value))
        self.has_saved = True

    def _write_storage_and_return_key(self, storage):
        if self.has_saved:
            raise RuntimeError("have already saved")
        key = self.next_key
        self.next_key += 1
        name = f"data/{key}"
        if storage.device.type != "cpu":
            storage = storage.cpu()
        num_bytes = storage.nbytes()

        current_version = version.parse(torch.__version__)
        threshold_version = version.parse("2.2.2")
        if current_version <= threshold_version:
            self.zipfile.write_record(name, storage.data_ptr(), num_bytes)
        else:
            self.zipfile.write_record(name, storage, num_bytes)

        return key

    def __exit__(self, type, value, traceback):
        self.zipfile.write_end_of_file()


def copy_weights_hf_llama(
    config: Config,
    qkv_weights: Dict[int, List[Optional[torch.Tensor]]],
    state_dict: Dict[str, torch.Tensor],
    hf_weights: Dict[str, torch.Tensor],
    saver: Optional[incremental_save] = None,
    dtype: Optional[torch.dtype] = None,
    pbar: Optional[tqdm] = None,
    progress_per_file: Optional[float] = None,
    debug_mode: Optional[bool] = False
) -> None:
    weight_map = {
        "model.embed_tokens.weight": "transformer.wte.weight",
        "model.layers.{}.input_layernorm.weight": "transformer.h.{l}.norm_1.weight",
        "model.layers.{}.input_layernorm.bias": "transformer.h.{l}.norm_1.bias",
        "model.layers.{}.self_attn.q_proj.weight": None,
        "model.layers.{}.self_attn.k_proj.weight": None,
        "model.layers.{}.self_attn.v_proj.weight": None,
        "model.layers.{}.self_attn.o_proj.weight": "transformer.h.{l}.attn.proj.weight",
        "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
        "model.layers.{}.post_attention_layernorm.weight": "transformer.h.{l}.norm_2.weight",
        "model.layers.{}.post_attention_layernorm.bias": "transformer.h.{l}.norm_2.bias",
        "model.norm.weight": "transformer.ln_f.weight",
        "model.norm.bias": "transformer.ln_f.bias",
        "lm_head.weight": "lm_head.weight",
    }
    if config.mlp_class_name == "LLaMAMoE":
        weight_map.update(
            {
                "model.layers.{}.block_sparse_moe.gate.weight": "transformer.h.{l}.mlp.gate.weight",
                "model.layers.{}.block_sparse_moe.experts.{}.w1.weight": "transformer.h.{l}.mlp.experts.{e}.fc_1.weight",
                "model.layers.{}.block_sparse_moe.experts.{}.w3.weight": "transformer.h.{l}.mlp.experts.{e}.fc_2.weight",
                "model.layers.{}.block_sparse_moe.experts.{}.w2.weight": "transformer.h.{l}.mlp.experts.{e}.proj.weight",
            }
        )
    elif config.mlp_class_name in ("LLaMAMLP", "GemmaMLP"):
        weight_map.update(
            {
                "model.layers.{}.mlp.gate_proj.weight": "transformer.h.{l}.mlp.fc_1.weight",
                "model.layers.{}.mlp.up_proj.weight": "transformer.h.{l}.mlp.fc_2.weight",
                "model.layers.{}.mlp.down_proj.weight": "transformer.h.{l}.mlp.proj.weight",
            }
        )
    else:
        raise NotImplementedError

    if progress_per_file is not None:
        progress_per_file = progress_per_file / max(1, len(hf_weights) + len(qkv_weights))

    for name, param in hf_weights.items():
        if "model.layers" in name:
            from_name, l = layer_template(name, 2)
            e = None
            if "block_sparse_moe.experts" in name:
                from_name, e = layer_template(from_name, 5)
            qkv = qkv_weights.setdefault(l, [None, None, None])
            if "q_proj" in name:
                qkv[0] = param
            elif "k_proj" in name:
                qkv[1] = param
            elif "v_proj" in name:
                qkv[2] = param
            to_name = weight_map[from_name]
            if to_name is None:
                continue
            to_name = to_name.format(l=l, e=e)
        else:
            to_name = weight_map[name]
        param = load_param(param, name, dtype, verbose=debug_mode)
        if saver is not None:
            param = saver.store_early(param)
        state_dict[to_name] = param

        if progress_per_file is not None:
            pbar.update(progress_per_file)

    if "lm_head.weight" not in state_dict:
        state_dict["lm_head.weight"] = state_dict["transformer.wte.weight"]

    # convert separate q, k, v matrices into an interleaved qkv
    for i, (q, k, v) in list(qkv_weights.items()):
        if q is None or k is None or v is None:
            # split across different .bin files
            continue
        q = load_param(q, f"layer {i} q", dtype, verbose=debug_mode)
        k = load_param(k, f"layer {i} k", dtype, verbose=debug_mode)
        v = load_param(v, f"layer {i} v", dtype, verbose=debug_mode)
        q_per_kv = config.n_head // config.n_query_groups
        qs = torch.split(q, config.head_size * q_per_kv)
        ks = torch.split(k, config.head_size)
        vs = torch.split(v, config.head_size)
        cycled = [t for group in zip(qs, ks, vs) for t in group]
        qkv = torch.cat(cycled)
        state_dict[f"transformer.h.{i}.attn.attn.weight"] = qkv
        del qkv_weights[i]
        if progress_per_file is not None:
            pbar.update(progress_per_file)



def layer_template(layer_name: str, idx: int) -> Tuple[str, int]:
    split = layer_name.split(".")
    number = int(split[idx])
    split[idx] = "{}"
    from_name = ".".join(split)
    return from_name, number


def load_param(param: torch.Tensor, name: str, dtype: Optional[torch.dtype], verbose=False) -> torch.Tensor:
    if hasattr(param, "_load_tensor"):
        # support tensors loaded via `lazy_load()`
        if verbose:
            print(f"Loading {name!r} into RAM")
        param = param._load_tensor()
    if dtype is not None and dtype != param.dtype:
        if verbose:
            print(f"Converting {name!r} from {param.dtype} to {dtype}")
        param = param.to(dtype)
    return param


@torch.inference_mode()
def convert_hf_checkpoint(
    checkpoint_dir: Path,
    *,
    model_name: Optional[str] = None,
    dtype: Optional[str] = None,
    debug_mode: Optional[bool] = False
) -> None:
    """
    Convert a Hugging Face Transformers checkpoint into a .pt.

    Arguments:
        checkpoint_dir: Where to save the downloaded files.
        model_name: The existing config name to load. This is useful to download alternative weights of existing
            architectures.
        dtype: The data type to convert the checkpoint files to. If not specified, the weights will remain in the
            dtype they are downloaded in.
        debug_mode: Prints the individual layers being loaded instead of a progress bar
    """
    checkpoint_dir = extend_checkpoint_dir(checkpoint_dir)
    #pprint(locals())

    if model_name is None:
        model_name = checkpoint_dir.name
    if dtype is not None:
        dtype = getattr(torch, dtype)

    config = Config.from_name(model_name)
    save_config(config, checkpoint_dir)

    print(model_name)
    if model_name.lower().startswith("llama"):
        qkv_weights = {}
        copy_fn = partial(copy_weights_hf_llama, config, qkv_weights)
    else:
        raise ValueError(f"Unknown model_name not supported yet passed model_name {model_name}")


    # initialize a new empty state dict to hold our new weights
    sd = {}

    # Load the json file containing weight mapping
    pytorch_bin_map_json_path = checkpoint_dir / "pytorch_model.bin.index.json"
    model_safetensor_map_json_path = checkpoint_dir / "model.safetensors.index.json"
    if pytorch_bin_map_json_path.is_file():  # not all checkpoints have this file
        with open(pytorch_bin_map_json_path, encoding="utf-8") as json_map:
            bin_index = json.load(json_map)
        bin_files = {checkpoint_dir / bin for bin in bin_index["weight_map"].values()}
    elif model_safetensor_map_json_path.is_file():
        with open(model_safetensor_map_json_path, encoding="utf-8") as json_map:
            bin_index = json.load(json_map)
        bin_files = {checkpoint_dir / Path(bin).with_suffix(".bin") for bin in bin_index["weight_map"].values()}
    else:
        bin_files = set(checkpoint_dir.glob("*.bin"))
        # some checkpoints serialize the training arguments
        bin_files = {f for f in bin_files if f.name != "training_args.bin"}
    if not bin_files:
        raise ValueError(f"Expected {str(checkpoint_dir)!r} to contain .bin files")

    saved_model_name = model_name + ".pt"
    with incremental_save(checkpoint_dir / saved_model_name) as saver:
        # for checkpoints that split the QKV across several files, we need to keep all the bin files
        # open, so we use `ExitStack` to close them all together at the end

        if not debug_mode:
            # Using tqdm progress bar when not in debug mode

            total_size = max(1, sum(os.path.getsize(bin_file) for bin_file in bin_files))
            total_progress = 100

            with tqdm(total=total_progress, desc="Initializing", bar_format="{desc}{percentage:3.0f}%|{bar}| {elapsed}<{remaining}, {rate_fmt}") as pbar:
                for bin_file in sorted(bin_files):
                    pbar.set_description(f"Loading weights: {bin_file.name}")
                    current_file_size = os.path.getsize(bin_file)
                    progress_per_file = (current_file_size / total_size) * total_progress

                    #hf_weights = lazy_load(bin_file)
                    hf_weights = torch.load(bin_file)
                    copy_fn(sd, hf_weights, saver=None, dtype=dtype, pbar=pbar, progress_per_file=progress_per_file, debug_mode=debug_mode)
                gc.collect()

                if pbar.n < total_progress:
                    pbar.update(total_progress - pbar.n)
                pbar.close()
        else:
            # Handling files without progress bar in debug mode
            for bin_file in sorted(bin_files):
                hf_weights = lazy_load(bin_file)
                copy_fn(sd, hf_weights, saver=saver, dtype=dtype, debug_mode=debug_mode)

        print(f"Saving converted checkpoint to {checkpoint_dir}")
        #saver.save(sd)
        filename = os.path.join(checkpoint_dir,saved_model_name)
        torch.save(sd,filename)
