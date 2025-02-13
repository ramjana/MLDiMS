import sys
import math
import functools
from pprint import pprint
from operator import attrgetter
from kernels.autotuner import perftune
from kernels.autotuner import configs
from configs import common
from configs.common import (
       gemm_configs,
       rope_configs,
       ln_configs,
       rms_configs,
       act_configs,
       fa_configs
     )
from configs.mlconfig import algoConfig, attnConfig
from configs import hwconfig
from kernels.rope import Rope
from kernels.gemm import gemm
from kernels.rmsnorm import rmsnorm
from kernels.layernorm import layernorm
from kernels.attention import attention
from kernels.activations import Activation
import jax.numpy as jnp


@perftune(configs=gemm_configs(),
          keys = ['M','N','K'])
def matmul_kernel(
        M,N,K,
        BLOCK_M: int, 
        BLOCK_N: int,
        BLOCK_K: int,
        BlockSize: int,
        GSU: int,
        LSU: int,
        WAVEM: int,
        WAVEN: int,
        Wdtype: jnp.dtype,
        Xdtype: jnp.dtype,
        Odtype: jnp.dtype,
        hwCfg: hwconfig.HWConfig,
        algoCfg: algoConfig,
        fusedOps: tuple = None
        ):
   if (M < BLOCK_M or  N < BLOCK_N or K < BLOCK_K):
      kernel_time = sys.maxsize
   else:
      kernel_time = gemm(M=M,N=N,K=K,BLOCK_M=BLOCK_M,BLOCK_N=BLOCK_N,BLOCK_K=BLOCK_K,WAVEM=WAVEM,WAVEN=WAVEN,BlockSize=BlockSize,LSU=LSU,GSU=GSU,Wdtype=Wdtype,Xdtype=Xdtype,Odtype=Odtype,hwCfg=hwCfg,algoCfg=algoCfg)
   return kernel_time

def gemm_call(*gemm_args, **gemm_kwargs):
   def decorator(func):
     @functools.wraps(func)
     def decorated(*args, **kwargs):
        #N = args[0].shape[0]
        N = math.prod(args[0].features) if isinstance(args[0].features,tuple) else args[0].features
        assert(len(args[1].shape) <=4)
        if (len(args[1].shape) == 3):
           M = args[1].shape[0] * args[1].shape[1]
           K = args[1].shape[2]
        elif (len(args[1].shape) == 4):
           M = args[1].shape[0] * args[1].shape[1]
           K = args[1].shape[2] * args[1].shape[3]
        else:
           M = args[1].shape[0]
           K = args[1].shape[1]
        #print(f"Calling GEMM kernel for M={M} N={N} K={K}")
        Wdtype=args[0].dtype
        Xdtype=args[1].dtype
        Odtype= args[0].dtype
        llmCfg = common.get_modelCfg()
        hwCfg = llmCfg.get_hwcfg()
        algoCfg = llmCfg.get_gemmcfg()
        #N1 = getattr(args[0],gemm_args[0])
        #assert(N==N1)
        ktime = matmul_kernel(M=M,N=N,K=K,Wdtype=Wdtype,Xdtype=Xdtype,Odtype=Odtype,hwCfg=hwCfg,algoCfg=algoCfg)
        #print(f"matmul-kernel time {ktime}")
        return func(*args,**kwargs)
     return decorated
   return decorator

@perftune(configs = rope_configs(),
           keys = ['M', 'K'],)   
def rope_kernel(
        M:int, K:int,
        BLOCK_M:int,
        BLOCK_K:int,
        BlockSize: int,
        cols_per_thread:int,
        num_rows:int,
        Xdtype: jnp.dtype,
        hwCfg: hwconfig.HWConfig,
        algoCfg: algoConfig,
        vec_bitwidth:int = 16,
        count_check: bool = False,
        persistent_kernel: bool = True,
        fusedOps: tuple=None):
   kernel_time = Rope(M=M,
                      K=K,
                      BLOCK_M=BLOCK_M,
                      BLOCK_K=BLOCK_K,
                      BlockSize=BlockSize,
                      cols_per_thread=cols_per_thread,
                      vec_bitwidth=vec_bitwidth,
                      num_rows=num_rows,
                      hwCfg=hwCfg,
                      algoCfg=algoCfg,
                      count_check=count_check,
                      persistent_kernel=persistent_kernel,
                      fusedOps=fusedOps)
   return kernel_time


def rope_call(*rope_args, **rope_kwargs):
   def decorator(func):
     @functools.wraps(func)
     def decorated(*args, **kwargs):
        #N = args[0].shape[0]
        M = args[1].shape[0]
        K = args[1].shape[1]
        assert(len(args[1].shape) <=4)
        if (len(args[1].shape) == 3):
           M = args[1].shape[0] * args[1].shape[1]
           K = args[1].shape[2]
        elif (len(args[1].shape) == 4):
           M = args[1].shape[0] * args[1].shape[1]
           K = args[1].shape[2] * args[1].shape[3]
        else:
           M = args[1].shape[0]
           K = args[1].shape[1]
        Xdtype=args[1].dtype
        llmCfg = common.get_modelCfg()
        hwCfg = llmCfg.get_hwcfg()
        algoCfg = llmCfg.get_ropecfg()
        ktime = rope_kernel(M=M,K=K,Xdtype=Xdtype,hwCfg=hwCfg,algoCfg=algoCfg)
        print(f"Rope-kernel time {ktime}")
        return func(*args,**kwargs)
     return decorated
   return decorator

@perftune(configs = act_configs(),
           keys = ['M', 'K'],)   
def act_kernel(
        M:int, K:int,
        BLOCK_M:int,
        BLOCK_K:int,
        BlockSize: int,
        cols_per_thread:int,
        num_rows:int,
        Xdtype: jnp.dtype,
        hwCfg: hwconfig.HWConfig,
        algoCfg: algoConfig,
        vec_bitwidth:int = 16,
        count_check: bool = False,
        persistent_kernel: bool = True,
        fusedOps: tuple=None):
   kernel_time = Activation(M=M,
                      K=K,
                      BLOCK_M=BLOCK_M,
                      BLOCK_K=BLOCK_K,
                      BlockSize=BlockSize,
                      cols_per_thread=cols_per_thread,
                      vec_bitwidth=vec_bitwidth,
                      num_rows=num_rows,
                      hwCfg=hwCfg,
                      algoCfg=algoCfg,
                      count_check=count_check,
                      persistent_kernel=persistent_kernel,
                      fusedOps=fusedOps)
   return kernel_time 

def activation_call(*act_args, **act_kwargs):
   def decorator(func):
     @functools.wraps(func)
     def decorated(*args, **kwargs):
        #N = args[0].shape[0]
        M = args[1].shape[0]
        K = args[1].shape[1]
        assert(len(args[1].shape) <=4)
        if (len(args[1].shape) == 3):
           M = args[1].shape[0] * args[1].shape[1]
           K = args[1].shape[2]
        elif (len(args[1].shape) == 4):
           M = args[1].shape[0] * args[1].shape[1]
           K = args[1].shape[2] * args[1].shape[3]
        else:
           M = args[1].shape[0]
           K = args[1].shape[1]
        Xdtype=args[1].dtype
        llmCfg = common.get_modelCfg()
        hwCfg = llmCfg.get_hwcfg()
        algoCfg = llmCfg.get_actcfg()
        ktime = act_kernel(M=M,K=K,Xdtype=Xdtype,hwCfg=hwCfg,algoCfg=algoCfg)
        print(f"Activation-kernel time {ktime}")
        return func(*args,**kwargs)
     return decorated
   return decorator

@perftune(configs = ln_configs(),
           keys = ['M', 'K'],)   
def ln_kernel(
        M:int, K:int,
        BLOCK_M:int,
        BLOCK_K:int,
        BlockSize: int,
        cols_per_thread:int,
        num_rows:int,
        Xdtype: jnp.dtype,
        hwCfg: hwconfig.HWConfig,
        algoCfg: algoConfig,
        vec_bitwidth:int = 16,
        count_check: bool = False,
        persistent_kernel: bool = True,
        fusedOps: tuple=None):
   kernel_time = layernorm(M=M,
                      K=K,
                      BLOCK_M=BLOCK_M,
                      BLOCK_K=BLOCK_K,
                      BlockSize=BlockSize,
                      cols_per_thread=cols_per_thread,
                      vec_bitwidth=vec_bitwidth,
                      num_rows=num_rows,
                      hwCfg=hwCfg,
                      algoCfg=algoCfg,
                      count_check=count_check,
                      persistent_kernel=persistent_kernel,
                      fusedOps=fusedOps)
   return kernel_time 

def ln_call(*ln_args, **ln_kwargs):
   def decorator(func):
     @functools.wraps(func)
     def decorated(*args, **kwargs):
        #N = args[0].shape[0]
        M = args[1].shape[0]
        K = args[1].shape[1]
        assert(len(args[1].shape) <=4)
        if (len(args[1].shape) == 3):
           M = args[1].shape[0] * args[1].shape[1]
           K = args[1].shape[2]
        elif (len(args[1].shape) == 4):
           M = args[1].shape[0] * args[1].shape[1]
           K = args[1].shape[2] * args[1].shape[3]
        else:
           M = args[1].shape[0]
           K = args[1].shape[1]
        Xdtype=args[1].dtype
        llmCfg = common.get_modelCfg()
        hwCfg = llmCfg.get_hwcfg()
        algoCfg = llmCfg.get_normcfg()
        ktime = ln_kernel(M=M,K=K,Xdtype=Xdtype,hwCfg=hwCfg,algoCfg=algoCfg)
        print(f"layernorm-kernel time {ktime}")
        return func(*args,**kwargs)
     return decorated
   return decorator

@perftune(configs = rms_configs(),
           keys = ['M', 'K'],)   
def rms_kernel(
        M:int, K:int,
        BLOCK_M:int,
        BLOCK_K:int,
        BlockSize: int,
        cols_per_thread:int,
        num_rows:int,
        Xdtype: jnp.dtype,
        hwCfg: hwconfig.HWConfig,
        algoCfg: algoConfig,
        vec_bitwidth:int = 16,
        count_check: bool = False,
        persistent_kernel: bool = True,
        fusedOps: tuple=None):
   kernel_time = rmsnorm(M=M,
                      K=K,
                      BLOCK_M=BLOCK_M,
                      BLOCK_K=BLOCK_K,
                      BlockSize=BlockSize,
                      cols_per_thread=cols_per_thread,
                      vec_bitwidth=vec_bitwidth,
                      num_rows=num_rows,
                      hwCfg=hwCfg,
                      algoCfg=algoCfg,
                      count_check=count_check,
                      persistent_kernel=persistent_kernel,
                      fusedOps=fusedOps)
   return kernel_time 

def rms_call(*ln_args, **ln_kwargs):
   def decorator(func):
     @functools.wraps(func)
     def decorated(*args, **kwargs):
        #N = args[0].shape[0]
        M = args[1].shape[0]
        K = args[1].shape[1]
        assert(len(args[1].shape) <=4)
        if (len(args[1].shape) == 3):
           M = args[1].shape[0] * args[1].shape[1]
           K = args[1].shape[2]
        elif (len(args[1].shape) == 4):
           M = args[1].shape[0] * args[1].shape[1]
           K = args[1].shape[2] * args[1].shape[3]
        else:
           M = args[1].shape[0]
           K = args[1].shape[1]
        Xdtype=args[1].dtype
        llmCfg = common.get_modelCfg()
        hwCfg = llmCfg.get_hwcfg()
        algoCfg = llmCfg.get_normcfg()
        ktime = rms_kernel(M=M,K=K,Xdtype=Xdtype,hwCfg=hwCfg,algoCfg=algoCfg)
        print(f"rmsnorm-kernel time {ktime}")
        return func(*args,**kwargs)
     return decorated
   return decorator

@perftune(configs = fa_configs(),
           keys = ['batch_size', 'attn_dim', 'num_heads','kvheads_div','seqlen_q','seqlen_kv'],)   
def fa_kernel(
        batch_size:int, attn_dim:int,
        num_heads:int,
        kvheads_div:int,
        seqlen_q:int,
        seqlen_kv:int,
        BLOCK_Q:int,
        BLOCK_K:int,
        BLOCK_O:int,
        BlockSize: int,
        KV_SPLIT:int,
        q_dtype: jnp.dtype,
        k_dtype: jnp.dtype,
        v_dtype: jnp.dtype,
        o_dtype: jnp.dtype,
        causal_mask: bool,
        vlayout: str,
        hwCfg: hwconfig.HWConfig,
        algoCfg: attnConfig,
        label: str,
        persistent_kernel: bool = True,
        fusedOps: tuple=None):
   kernel_time = attention(batch_size=batch_size,
                      attn_dim=attn_dim,
                      num_heads=num_heads,
                      seqlen_q = seqlen_q,
                      seqlen_kv = seqlen_kv,
                      BLOCK_Q=BLOCK_M,
                      BLOCK_K=BLOCK_K,
                      BLOCK_O=BLOCK_O,
                      BlockSize=BlockSize,
                      vlayout=vlayout,
                      q_dtype=q_dtype,
                      k_dtype=k_dtype,
                      v_dtype=v_dtype,
                      o_dtype=o_dtype,
                      hwCfg=hwCfg,
                      algoCfg=algoCfg,
                      label=label,
                      persistent_kernel=persistent_kernel,
                      fusedOps=fusedOps)
   return kernel_time 

def attention_call(*fa_args, **fa_kwargs):
   def decorator(func):
     @functools.wraps(func)
     def decorated(*args, **kwargs):
        #N = args[0].shape[0]
        batch_size = args[1].shape[0]
        seqlen_q = args[1].shape[1]
        attn_dim = args[1].shape[-1]
        num_heads = args[1].shape[2]
        seqlen_kv = args[2].shape[1]
        kvheads_div = num_heads/args[2].shape[2]
        q_dtype=args[1].dtype
        k_dtype=args[2].dtype
        v_dtype=args[3].dtype
        o_dtype=args[3].dtype
        causal_mask = (args[5] != None)
        llmCfg = common.get_modelCfg()
        hwCfg = llmCfg.get_hwcfg()
        algoCfg = llmCfg.get_attncfg()
        label = getattr(args[0],fa_args[0])
        ktime = fa_kernel(batch_size=batch_size,
                          seqlen_q=seqlen_q,
                          attn_dim=attn_dim,
                          num_heads=num_heads,
                          kvheads_div=kvheads_div,
                          q_dtype=q_dtype,
                          k_dtype=k_dtype,
                          v_dtype=v_dtype,
                          o_dtype=o_dtype,
                          causal_mask=causal_mask,
                          label=label,
                          hwCfg=hwCfg,algoCfg=algoCfg)
        print(f"fa-kernel time {ktime}")
        return func(*args,**kwargs)
     return decorated
   return decorator
