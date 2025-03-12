from collections import OrderedDict
import math
import os
import sys
from dataclasses import dataclass
from kernels.autotuner import configs

_llmconfig = None
llmconfig = None
_hw_config = None
hw_config = None


model_dict = {
        "llama2_7b" : "models/llama2_7b.yml",
        "base_model": "models/base_transformer.yml",
        }
arch_dict = {
        "mi300x" : "hardware/mi300x.yml"
       }

modelCfg = None

class ModelCfg(type):
    _instance = {}

    def __call__(cls,*args, **kwargs):
        if cls not in cls._instance:
            cls._instance[cls] = super().__call__(*args, **kwargs)
        return cls._instance[cls]

def set_modelCfg(_Cfg:ModelCfg):
    global modelCfg
    assert(_Cfg!=None)
    modelCfg = _Cfg
    return modelCfg

def get_modelCfg():
    global modelCfg
    assert(modelCfg!=None)
    return modelCfg

def default_gemmconfigs():
    return [ configs({'BLOCK_M' : 256, 'BLOCK_N' : 128, 'BLOCK_K' : 64, 'WAVEM': 2, 'WAVEN': 2}, BlockSize=256, LSU=1, GSU=1),
            configs({'BLOCK_M' : 256, 'BLOCK_N' : 256, 'BLOCK_K' : 64, 'WAVEM' : 2, 'WAVEN' : 2}, BlockSize=256, LSU=1, GSU=1)
           ]

def default_ropeconfigs():
        return [ configs({'BLOCK_M' : 1, 'BLOCK_K' : 4096, 'cols_per_thread':32, 'num_rows':1,'vec_bitwidth':16}, BlockSize=512,)]

def default_rmsconfigs():
        return [ configs({'BLOCK_M' : 1, 'BLOCK_K' : 4096, 'cols_per_thread':32, 'num_rows':1,'vec_bitwidth':16}, BlockSize=512,)]

def default_lnconfigs():
        return [ configs({'BLOCK_M' : 1, 'BLOCK_K' : 4096, 'cols_per_thread':32, 'num_rows':1,'vec_bitwidth':16}, BlockSize=512,)]

def default_actconfigs():
        return [ configs({'BLOCK_M' : 1, 'BLOCK_K' : 4096, 'cols_per_thread':32, 'num_rows':1,'vec_bitwidth':16}, BlockSize=512,)]

def default_faconfigs():
        return [ configs({'BLOCK_Q' : 128, 'BLOCK_K' : 128, 'BLOCK_O':128, 'KV_SPLIT':1}, BlockSize=512,)]


def rope_configs():
    global modelCfg
    if modelCfg == None:
        return default_ropeconfigs()
    else:
        return modelCfg.rope_configs()

def rms_configs():
    global modelCfg
    if modelCfg == None:
        return default_rmsconfigs()
    else:
        return modelCfg.rms_configs()

def ln_configs():
    global modelCfg
    if modelCfg == None:
        return default_lnconfigs()
    else:
        return modelCfg.ln_configs()


def act_configs():
    global modelCfg
    if modelCfg == None:
        return default_actconfigs()
    else:
        return modelCfg.act_configs()

def fa_configs():
    global modelCfg
    if modelCfg == None:
        return default_gemmconfigs()
    else:
        return modelCfg.fa_configs()

def gemm_configs():
    global modelCfg
    if modelCfg == None:
        return default_gemmconfigs()
    else:
        return modelCfg.gemm_configs()
