from collections import OrderedDict
import math
import os
import sys

from typing import List, Tuple, Any, Union
import yaml
import numpy as jnp

from dataclasses import dataclass


def lists_to_tuples(l: list[Any]) -> Union[tuple[Any], list[Any]]:
  return tuple(lists_to_tuples(x) for x in l) if isinstance(l, list) else l

def get_num_target_devices():
    return len(jax.devices())


_hw_config = None
hw_config = None

@dataclass
class ClockConfig:

    mode : int = 0
    gfxClk: jnp.float32 = 0.0
    mClk: jnp.float32 = 0.0
    fClk: jnp.float32 = 0.0

    def __post_init__(self):
        self.set_clk(self.mode)

    def set_mode0(self):
        self.gfxClk = 1100.0
        self.mClk = 900.0
        self.fClk = 1300.0

    def set_mode1(self):
        self.gfxClk = 1500.0
        self.mClk = 1100.0
        self.fClk = 1500.0

    def set_mode2(self):
        self.gfxClk = 2100.0
        self.mClk = 1300.0
        self.fClk = 1800.0

    def set_gfxclk(self,freq):
        self.gfxClk = freq

    def get_gfxclk(self):
        return self.gfxClk

    def get_mclk(self):
        return self.mClk

    def get_fclk(self):
        return self.fClk

    def set_clk(self,mode:int):
        if mode == 0:
            self.set_mode0()
        elif mode == 1:
            self.set_mode1()
        else:
            self.set_mode2()

class _hwconfig:

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

        config_file = userArgs[1]
        with open(config_file,"r", encoding="utf-8") as file:
            config_from_yaml = yaml.safe_load(file)

        config_keys = OrderedDict()

        self.update_config_command_line(config_keys,config_from_yaml, userArgs, **kwargs)

        config_keys["num_cus"] = config_keys["num_aids"] * config_keys["num_xcc_aid"] * config_keys["num_cu_xcc"]

        bit_rate = config_keys["hbm_strobe_freq"]*2*config_keys["num_bits_channel"]
        total_channels_rate = bit_rate * config_keys["num_hbm_channels"]
        config_keys["hbm_membw_acheivable"] = round(total_channels_rate/8 * config_keys["hbm_efficiency"],2)
        config_keys["hbm_bw_cu"] = round(config_keys["hbm_membw_acheivable"] / config_keys["num_cus"])

        config_keys["num_xccs"] = config_keys["num_xcc_aid"] * config_keys["num_aids"]
        config_keys["total_l2_channels"] = config_keys["num_l2channels_xcc"] * config_keys["num_xccs"]
        config_keys["total_mall_channels"] = config_keys["num_mallchannels_aids"] * config_keys["num_aids"]
        config_keys["l2rd_bandwidth"] = config_keys["l2_read_bw"] * config_keys["total_mall_channels"]
        config_keys["l2wr_bandwidth"] = config_keys["l2_write_bw"] * config_keys["total_mall_channels"]
        config_keys["mallrd_bandwidth"] = config_keys["mall_read_bw"] * config_keys["total_mall_channels"]
        config_keys["mallwr_bandwidth"] = config_keys["mall_write_bw"] * config_keys["total_mall_channels"]

        config_keys["l2rd_bw_cu"] = round(config_keys["l2rd_bandwidth"]/config_keys["num_cus"])
        config_keys["l2wr_bw_cu"] = round(config_keys["l2wr_bandwidth"]/config_keys["num_cus"])
        config_keys["mallrd_bw_cu"] = round(config_keys["mallrd_bandwidth"]/config_keys["num_cus"])
        config_keys["mallwr_bw_cu"] = round(config_keys["mallwr_bandwidth"]/config_keys["num_cus"])

        clkConfig = ClockConfig(mode=int(config_keys["dpm_mode"]))

        config_keys["clkConfig"] = clkConfig

        self.configKeys = config_keys
        #for key,value in config_keys.items():
        #   print(f"Config param {key} : {value}")

class HWConfig:

  def __init__(self):
    pass

  def __getattr__(self, attr):
    if attr not in _hw_config.configKeys:
      raise ValueError(f"Requested key {attr}, not in config")
    return _hw_config.configKeys[attr]

  def __setattr__(self, attr, value):
    _hw_config.configKeys[attr] = value

  def get_keys(self):
    return _hw_config.configKeys


def initialize(argv, **kwargs):
  global _hw_config, hw_config
  _hw_config = _hwconfig(argv, **kwargs)
  hw_config = HWConfig()

if __name__ == "__main__":
  initialize(sys.argv)
