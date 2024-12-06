"""MIT License.

Copyright (c) 2024 Phillip Lippe

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import os
import subprocess
import sys

import jax
from jax.experimental.shard_map import shard_map
import functools
from pprint import pprint
import flax.linen as nn

def set_XLA_flags_gpu():
    flags = os.environ.get("XLA_FLAGS", "")
    flags += (
        "--xla_gpu_enable_triton_softmax_fusion=true "
        "--xla_gpu_triton_gemm_any=false "
        "--xla_gpu_enable_async_collectives=true "
        "--xla_gpu_enable_latency_hiding_scheduler=true "
        "--xla_gpu_enable_highest_priority_async_stream=true "
    )
    os.environ["XLA_FLAGS"] = flags


def simulate_CPU_devices(device_count: int = 8):

    USE_CPU_ONLY = True
    # Set XLA flags to simulate a CPU with a given number of devices
    flags = os.environ.get("XLA_FLAGS", "")
    flags += f" --xla_force_host_platform_device_count={device_count}"
    os.environ["XLA_FLAGS"] = flags
    # Disable CUDA to force XLA to use the CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # Check for packages to be installed if needed. On Colab, the following packages are not installed by default:
    # - ml_collections
    try:
        import ml_collections
    except ImportError:
        install_package("ml_collections")


def install_package(package: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])


def singleton(cls):
    """make a class singleton"""
    @functools.wraps(cls)
    def wrapper_singleton(*args, **kwargs):
        if wrapper_singleton.instance is None:
            wrapper_singleton.instance = cls(*args,**kwargs)
        return wrapper_singleton.instance
    wrapper_singleton.instance = None
    return wrapper_singleton

def print_shapes(opName,get_attrvalue):
    def decorator_func(func):
        @functools.wraps(func)
        def wrapper_func(self,*args,**kwargs):
            if not (os.environ.get('PRINT_TENSOR_SHAPES',0)):
                return func(self,*args, **kwargs)
            Ops = ops()
            Opdict = {}
            numArgs = Ops.argsize(opName)
            Opdict["layerName"] = get_attrvalue(self).lower()
            Opdict["Op"]        = self.__class__.__name__
            if (numArgs <= 1):
               Opdict["input"] = args[0].shape
            if (numArgs <= 2):
               Opdict["weight"] = args[1].shape if opName is ["fadd","fmul"] else (args[0].shape[-1],self.features)
            if (numArgs <=3 ):
                if opName == "fa":
                    Opdict["Qtensor"] = args[0].shape
                    Opdict["Ktensor"] = args[1].shape
                    Opdict["Vtensor"] = args[2].shape

            pprint(Opdict)
            return func(self,*args, **kwargs)
        return wrapper_func
    return decorator_func


@singleton
class ops:
    def __init__(self):
        self.unaryOps = ["relu","silu","gelu",
                         "RMSNorm","LayerNorm","softmax",
                         "argmax"]
        self.binaryOps = ["fmul","fadd","fma","Dense","DenseGeneral","Linear","Embed"]
        self.fusedOps = ["fa"]
    def argsize(self,OpName):
        if (OpName in self.unaryOps):
            return 1
        elif (OpName in self.binaryOps):
            return 2
        elif (OpName in self.fusedOps):
            return 3
        else:
            raise ValueError(f"Unknown ops given ops_name= {OpName}")
