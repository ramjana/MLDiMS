### utilities for tuner for various LLM tensor ops

from pprint import pprint
import inspect
import builtins
import time
import os
from typing import Dict
import functools

class configs:
    """
    params:
       kwargs : dictionary of tuning parameters for GEMM tile-sizes
       num_warps: =1 single warp pipeline, =2 ping-pong pipeline
    """
    def __init__(self,kwargs, BlockSize:int, GSU:int = None, LSU:int = None):
        self.kwargs = kwargs
        self.BlockSize= BlockSize 
        self.LSU = LSU
        self.GSU = GSU

    def _all_args(self):
        return {
            **self.kwargs, **{
                k: v for (k,v) in (
                    ("BlockSize" , self.BlockSize),
                    ("LSU" , self.LSU),
                    ("GSU" , self.GSU),
                ) if v is not None
           }
        }

    def __repr__(self):
        
        configStr = {} 
        for k,v in self.kwargs.items():
            configStr[k] = v
        configStr["BlockSize"] = self.BlockSize
        configStr["localSplit"] = self.LSU
        configStr["globalSplit"] = self.GSU
        pprint(configStr)


class PerfTune:
    def __init__(self, configs, problemDesc, perf_fn, argnames):
        self.configs = configs
        self.problemDesc = problemDesc
        self.perf_fn = perf_fn
        self.argnames = argnames
        self.cache = {}


    def run(self,*args,**kwargs):
        self.fnArgs = dict(zip(self.argnames, args))

        if len(self.configs) > 1:
            allArgs = {**self.fnArgs, **kwargs}
            Args = {k:v for (k,v) in allArgs.items() if k in self.argnames}
            key = [Args[key] for key in self.problemDesc if key in Args]
            key = tuple(key)
            timings = {config: self.do_bench(config, *args, **kwargs) for config in self.configs}
            #cache timing of kernel
            self.cache[key] = builtins.min(timings.values())
            #self.cache[key] = builtins.min(timings,key=timings.get)
            run_time = self.cache[key] 
        else:
            config = self.configs[0]
            run_time = self.do_bench(config, *args, **kwargs)
        #return only timings of the kernel function
        return run_time

    def do_bench(self,config, *args, **kwargs) -> float:
        return self.perf_fn(*args,**kwargs,**config._all_args())


def perftune(configs, keys):
    def decorator_func(func):
        @functools.wraps(func)
        def wrapper_func(*args,**kwargs):
            arg_names = inspect.getfullargspec(func)[0]
            autotuner = PerfTune(configs, keys, func, arg_names)
            cycles  = autotuner.run(*args, **kwargs)
            return cycles
        return wrapper_func
    return decorator_func 
