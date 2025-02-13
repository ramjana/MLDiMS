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
from configs import hwconfig
from configs import mlconfig
from configs.mlconfig import algoConfig
from configs import common
from configs.common import ModelCfg

class llamaConfig(ModelCfg):

    def __new__(cls,*args,**kwargs):
        if not hasattr(cls,"instance"):
            cls.instance = super(llamaConfig,cls).__new__(cls)
            return cls.instance

    def __init__(self,modelArgs):
        self.userArgs = modelArgs
        if not ("model" in modelArgs[1]):
           ValueError(f" first argument must be --model ")
        self.llmCfg = mlconfig.ml_initialize(self.userArgs)
        self.hwCfg  = hwconfig.hw_initialize(self.userArgs[1:])
        super().__init__(name="llama2_7b")

    def get_llmcfg(self):
        return self.llmCfg

    def get_hwcfg(self):
        return self.hwCfg

    def get_gemmcfg(self):
        return algoConfig(Algo="default",weight_l2_hit= 0.0, weight_mall_hit=0.25, act_l2rd_hit=0.75, act_mallrd_hit = 0.75, act_l2wr_hit=1.0, act_mallwr_hit=1.0, tileSetup_time = 800,dpm_mode=2,clk_eff=0.65)

    def get_attncfg(self):
        return attnConfig(Algo="default",qtile_l2rd_hit= 0.0, qtile_mallrd_hit=0.0, ktile_l2rd_hit=0.75, ktile_mallrd_hit=0.25,vtile_mallrd_hit=0.25, vtile_l2rd_hit=0.75,tileSetup_time = 500,vlayout="r",dpm_mode=0, clk_eff=0.9,otile_l2wr_hit=0.0, otile_mallwr_hit=0.0,causal_mask_eff=1.7)

    def get_normcfg(self):
        return algoConfig(Algo="default",weight_l2_hit= 0, weight_mall_hit=0, act_l2rd_hit=0.0, act_mallrd_hit = 0.0, act_l2wr_hit=0.0, act_mallwr_hit = 0.0,dpm_mode=2,clk_eff=0.65)

    def get_actcfg(self):
        return algoConfig(Algo="default",weight_l2_hit= 0,
                          weight_mall_hit=0, act_l2rd_hit=0.0,
                          act_mallrd_hit = 0.0, act_l2wr_hit=0.0,
                          act_mallwr_hit = 0.0,
                          dpm_mode=2,clk_eff=0.65,
                          act_func="selu")
    def get_ropecfg(self):
        return algoConfig(Algo="default",weight_l2_hit= 0, weight_mall_hit=0, act_l2rd_hit=0.0, act_mallrd_hit = 0.0, act_l2wr_hit=0.0, act_mallwr_hit = 0.0,dpm_mode=2,clk_eff=0.65)

    def gemm_configs(self):
        return [ configs({'BLOCK_M' : 256, 'BLOCK_N' : 128, 'BLOCK_K' : 64, 'WAVEM': 2, 'WAVEN': 2}, BlockSize=256, LSU=1, GSU=1),
                 configs({'BLOCK_M' : 256, 'BLOCK_N' : 256, 'BLOCK_K' : 64, 'WAVEM' : 2, 'WAVEN' : 2}, BlockSize=256, LSU=1, GSU=1),
                 configs({'BLOCK_M' : 128, 'BLOCK_N' : 128, 'BLOCK_K' : 128, 'WAVEM' : 2, 'WAVEN' : 2}, BlockSize=256, LSU=1, GSU=1),
                 configs({'BLOCK_M' : 64, 'BLOCK_N' : 64, 'BLOCK_K' : 256, 'WAVEM' : 2, 'WAVEN' : 2}, BlockSize=256, LSU=1, GSU=1),
                 configs({'BLOCK_M' : 16, 'BLOCK_N' : 64, 'BLOCK_K' : 512, 'WAVEM' : 2, 'WAVEN' : 2}, BlockSize=256, LSU=1, GSU=1),
                 configs({'BLOCK_M' : 16, 'BLOCK_N' : 128, 'BLOCK_K' : 256, 'WAVEM' : 2, 'WAVEN' : 2}, BlockSize=256, LSU=1, GSU=1),
           ]
    def rope_configs(self):
        return [ configs({'BLOCK_M' : 1, 'BLOCK_K' : 4096, 'cols_per_thread':32, 'num_rows':1,'vec_bitwidth':16}, BlockSize=512,)]

    def rms_configs(self):
        return [ configs({'BLOCK_M' : 1, 'BLOCK_K' : 4096, 'cols_per_thread':32, 'num_rows':1,'vec_bitwidth':16}, BlockSize=512,)]

    def ln_configs(self):
        return [ configs({'BLOCK_M' : 1, 'BLOCK_K' : 4096, 'cols_per_thread':32, 'num_rows':1,'vec_bitwidth':16}, BlockSize=512,)]


    def act_configs(self):
        return [ configs({'BLOCK_M' : 1, 'BLOCK_K' : 4096, 'cols_per_thread':32, 'num_rows':1,'vec_bitwidth':16}, BlockSize=512,)]

    def fa_configs(self):
        return [ configs({'BLOCK_Q' : 128, 'BLOCK_K' : 128, 'BLOCK_O':128, 'KV_SPLIT':1}, BlockSize=512,)]
        return [ configs({'BLOCK_Q' : 16, 'BLOCK_K' : 128, 'BLOCK_O':128, 'KV_SPLIT':1}, BlockSize=512,)]


if __name__ == "__main__":
    llam2_7b = llamaConfig(sys.argv)
    common.set_modelCfg(llam2_7b)
    ropeCfg = common.get_modelCfg().get_ropecfg()
