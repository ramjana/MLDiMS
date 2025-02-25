from collections import OrderedDict
import math
import os
import sys

import jax.numpy as jnp
from typing import List, Tuple, Any, Union
import flax
import jax
import yaml
from dataclasses import dataclass
from pprint import pprint
from configs import common
from configs.common import model_dict


def lists_to_tuples(l: list[Any]) -> Union[tuple[Any], list[Any]]:
  return tuple(lists_to_tuples(x) for x in l) if isinstance(l, list) else l

def get_num_target_devices():
    return len(jax.devices())


def get_num_slices():
    """Calculate num_slices based on number of devices."""
    devices = get_num_target_devices()
    try:
      return 1 + max([d.slice_index for d in devices])
    except:
      return 1


@dataclass
class algoConfig:
    """class for algorithm configuration"""

    Algo: str
    weight_l2_hit: jnp.float32
    weight_mall_hit: jnp.float32
    act_l2rd_hit: jnp.float32
    act_mallrd_hit: jnp.float32
    act_l2wr_hit: jnp.float32
    act_mallwr_hit: jnp.float32
    dpm_mode: int
    clk_eff: jnp.float32
    act_func:str = None
    tileSetup_time: int  = 500 #cycles
    gemm_eff_cap: jnp.float32 = 0.98

@dataclass
class attnConfig:

    Algo: str
    qtile_l2rd_hit : jnp.float32
    ktile_l2rd_hit : jnp.float32
    vtile_l2rd_hit : jnp.float32

    qtile_mallrd_hit : jnp.float32
    ktile_mallrd_hit : jnp.float32
    vtile_mallrd_hit : jnp.float32
    otile_l2wr_hit: jnp.float32
    otile_mallwr_hit: jnp.float32
    vlayout: str
    causal_mask_eff : float
    dpm_mode: int
    clk_eff: jnp.float32
    tileSetup_time: int  = 500 #cycles
    act_func:str = None
    weight_l2_hit: jnp.float32 = 0.0
    weight_mall_hit: jnp.float32 = 0.0
    gemm_eff_cap: jnp.float32 = 0.98

#_llmconfig = None
#llmconfig = None

#model_dict = {
#        "llama2_7b" : "models/llama2_7b.yml",
#        "base_model": "models/base_transformer.yml",
#        }

class _mlconfig:

    def update_config_command_line(self, config_keys, yaml_data, cmndlnArgs: list[str], **kwargs):
        argsDict = dict(arg.split("=",1) for arg in cmndlnArgs[2:])
        argsDict.update(kwargs)

        for k in yaml_data:
            if k in argsDict:
                config_keys[k]  = argsDict[k]
            if not k in argsDict:
                config_keys[k]  = yaml_data[k]

        return

    def __init__(self, userArgs: list[str], **kwargs):

        config_name = userArgs[1].split("=",1)
        file_name = model_dict[str(config_name[1])]
        config_file = os.path.join(os.path.dirname(__file__),file_name)
        with open(config_file,"r", encoding="utf-8") as file:
            config_from_yaml = yaml.safe_load(file)

        config_keys = OrderedDict()

        self.update_config_command_line(config_keys,config_from_yaml, userArgs, **kwargs)

        #print(config_keys)
        if config_keys["learning_rate_schedule_steps"] == -1:
            config_keys["learning_rate_schedule_steps"] = config_keys["steps"]
        if config_keys["steps"] == -1:
            config_keys["steps"] = config_keys["learning_rate_schedule_steps"]

        config_keys["num_slices"] = get_num_slices()
  

        if "tensor_parallelism" in config_keys and config_keys["model_dim"] % config_keys["tensor_parallelism"] != 0:
            raise valueError(f"model_dim is not divisible by tensor_parallelism setting { config_keys['tensor_parallelism'] }")

        if "data_parallelism" in config_keys:
            num_devices = config_keys["data_parallelism"] * config_keys["tensor_parallelism"]
   
        config_keys["dtype"] = jax.numpy.dtype(config_keys["dtype"])
        config_keys["weight_dtype"] = jax.numpy.dtype(config_keys["weight_dtype"])

        config_keys["logical_axis_rules"] = lists_to_tuples(config_keys["logical_axis_rules"])
        config_keys["data_sharding"] = lists_to_tuples(config_keys["data_sharding"])          

        config_keys["max_prefill_predict_len"] = int(config_keys["max_prefill_predict_len"])
        config_keys["max_seq_length"] = int(config_keys["max_seq_length"])

        config_keys["normalization_epsilon"]  = float(config_keys["normalization_epsilon"])

        config_keys["fsdp_modules"] = lists_to_tuples(config_keys["fsdp_modules"])

        config_keys["model_axis_size"] = int(config_keys["model_axis_size"])
        config_keys["num_devices"] = int(config_keys["num_devices"])
        config_keys["data_parallelism"] = int(config_keys["data_parallelism"])
        config_keys["device_batch_size"] = int(config_keys["device_batch_size"])
        config_keys["max_batch_size"]  = config_keys["data_parallelism"]*config_keys["device_batch_size"]

        if not os.path.isfile(config_keys['tokenizer_path']):
            dir_path = os.path.dirname(os.path.realpath(__file__))
            tokenizer_path= os.path.join(
                    dir_path, config_keys['tokenizer_path'])
         
            if os.path.isfile(tokenizer_path):
                config_keys['tokenizer_path'] = tokenizer_path
 
        self.configKeys = config_keys
        #for key,value in config_keys.items():
        #   print(f"Config param {key} : {value}")

                    

class llmConfig(object):

  def __init__(self):
    pass

  def __new__(cls):
    if not hasattr(cls,"instance"):
       cls.instance = super(llmConfig,cls).__new__(cls)
       return cls.instance

  def __getattr__(self, attr):
    if attr not in _llmconfig.configKeys:
      raise ValueError(f"Requested key {attr}, not in config")
    return _llmconfig.configKeys[attr]

  def __setattr__(self, attr, value):
      _llmconfig.configKeys[attr] = value

  def get_keys(self):
    return _llmconfig.configKeys

def ml_initialize(argv, **kwargs):
  global _llmconfig, llmconfig
  _llmconfig = _mlconfig(argv, **kwargs)
  llmconfig = llmConfig()
  return llmconfig

if __name__ == "__main__":
    ###--model=nameof the model
  llmCfg = ml_initialize(sys.argv)
  r = range(llmCfg.steps)
  print(llmCfg.get_keys())
