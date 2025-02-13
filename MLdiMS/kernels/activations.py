import sys
import builtins
import math
import jax.numpy as jnp
from configs import hwconfig
from configs.hwconfig import ClockConfig
from configs.mlconfig import algoConfig
import jax.numpy as jnp

from functools import partial 


def Activation(M: int, K: int,
        Xdtype: jnp.dtype,
        hwCfg: hwconfig.HWConfig,
        algoCfg: algoConfig,
        BLOCK_M: int, BLOCK_K: int, BlockSize:int,
        cols_per_thread: int, num_rows: int, vec_bitwidth: int = 16,
        rounding_mode_rne_sw: bool = False,
        shuffle_reduction: bool = True,
        ds_permute_reduction: bool = False,
        persistent_kernel: bool = True,
        fusedOps: tuple=None) -> int:


    def shuffle_wave_reduction(lanes_per_row:int, num_rows:int, bpe):
        ## use  butterfly-shuffle to shift one element at a time and do welford_op
        num_steps = 5 if lanes_per_row > 32 else math.log(lanes_per_row)
        #+1 for broadcast
        if bpe < 4:
            sdwa_step = 1
        else:
            sdwa_step = 0
        latency = 0
        blocksize = (num_rows * lanes_per_row // 256)
        for iter in range(int(num_steps + sdwa_step)):
           payload = (blocksize//((iter+1)*2)) 
           #move shuffle instruction to read out mean,m2,count
           latency += 3 * (blocksize//(256*(4/bpe)))
           latency += payload//hwCfg.mul32_rate
        latency += (blocksize//256)*4
        return latency
    
    def dspermute_wave_reduction(lanes_per_row:int, num_rows:int, bpe):
        ## use  butterfly-shuffle to shift one element at a time and do welford_op
        num_steps = 5 if lanes_per_row > 32 else math.log(lanes_per_row)
        # 3 ds_bpermute mean,count,m2
        #issue 3 ds_bpermute blocksize * bpe // 128 
        latency = 0
        if bpe < 4:
            sdwa_step = 1
        else:
            sdwa_step = 0
        latency = 0
        blocksize = (num_rows * lanes_per_row // 256)
        for iter in range(int(num_steps+sdwa_step)):
           payload = (blocksize//((iter+1)*2)) 
           data_payload = payload * bpe
           latency += hwCfg.lds_latency + data_payload//hwCfg.lds_bandwidth 
           latency += (blocksize//256)*4
        ##broadcast
        latency += (blocksize//256)*4
        return latency

    def wg_reduction(num_wave32_per_row:int, num_rows:int,blocksize:int):
        writepayload_lat = num_wave32_per_row*num_rows*12    #mean, m2, count
        write_latency = (48+12)  if writepayload_lat < 48 else (12+writepayload_lat)
        read_latency = 48 + (blocksize//64)*num_rows*num_wave32_per_row*4  # 4 upper limit on wave data
        lds_latency = hwCfg.barrier_latency  + write_latency + read_latency
        #All waves read-out mean/m2/count and do wave-level reduction
        lds_latency += shuffle_wave_reduction(num_wave32_per_row,num_rows)
        return lds_latency

    def relu(payload:int):
        ## A(x) = max(x,0)
        return payload//hwCfg.mul32_rate
    
    def sigmoid(payload:int, rounding_mode:bool=False, rounding_insts:int =4):
        ## A(x) = 1/(1+e(-x))
        
        _cycles = 2*payload//hwCfg.transcendental_rate   ##e(x) , 1/(ex)
        _cycles += payload//hwCfg.mul32_rate
        if rounding_mode:
            _cycles += rounding_insts* payload//hwCfg.mul32_rate
        return _cycles
    
    def selu(payload:int, rounding_mode:bool=False, rounding_insts:int =4):
        ## A(x) = scale*(max(x,0) + min(0, alpha*(e(x)-1)))
        #1. transcendental
        #2 alpha*(e(x)-1) = fma (alpha*e(x)- alpha)
        #3. min
        #4. max
        #5. fadd
        #6. fmul
        
        _cycles = payload//hwCfg.transcendental_rate
        _cycles += payload//hwCfg.fma32_rate
        _cycles += 4*payload//hwCfg.mul32_rate
        if rounding_mode:
            _cycles += rounding_insts* payload//hwCfg.mul32_rate
        return _cycles

    def silu(payload:int):
        ### A(x) = x*sigmoid(x)
        _cycles = sigmoid(payload)
        _cycles += payload//hwCfg.mul32_rate
        
    def tanh(payload:int):
        ## A(x) = exp(x) - exp(-x) / (exp(x) + exp(-x))
        ## 2 conditional check to make sure we are dividing by zero or numerator 0
        _cycles = payload//hwCfg.transcendental_rate    #exp(x) 
        _cycles += (payload//hwCfg.mul32_rate)    #zero check
        _cycles += (payload//hwCfg.mul32_rate)  #(exp(x)+1.0)
        _cycles += payload//hwCfg.transcendental_rate   #1/(exp(x)+1.0)
        _cycles += (payload//hwCfg.mul32_rate)  # exp(x) - 1.0
        _cycles += (payload//hwCfg.mul32_rate)  #final mul
        return _cycles
    
    
    def softmax(payload:int,BlockSize:int):
        ##multi-pass algorithm if dimension > 65K
        ##A(x) = e(x)/ sum(e(x))
        _cycles = payload//hwCfg.transcendental_rate
        _cycles += (payload//BlockSize)//hwCfg.mul32_rate   if payload > BlockSize  else _cycles
        return _cycles
    
    def gelu(payload:int):
        ###A(x)=0.5∗x∗(1+Tanh(2/π∗(x+0.044715∗x**3)))
        _cycles = 2*payload//hwCfg.mul32_rate  #x*x
        _cycles += payload//hwCfg.mul32_rate  #0.044*x
        _cycles += payload//hwCfg.fma32_rate  # x+0.044*x**3
        _cycles += payload//hwCfg.mul32_rate  # 2/3.14 * ()
        _cycles += 2*payload//hwCfg.mul32_rate # conditional nan/inf check
        _cycles += tanh(payload)
        _cycles += 2*payload//hwCfg.mul32_rate 
        return _cycles
        

    op_funcs = {"gelu" : gelu, 
                "relu" : relu,
                "selu" : selu,
                "softmax" : softmax,
                "tanh" : tanh,
                "gelu" : gelu,
                "silu" : silu,
                }

    if hwCfg == None:
        raise ValueError("Missing hw config object")

    clkCfg = ClockConfig(algoCfg.dpm_mode)
    gclk = clkCfg.get_gfxclk() * algoCfg.clk_eff
    #fclk = clkCfg.get_fclk()
    #mclk = clkCfg.get_mclk()
    gclk_us = 1/(gclk)
    #latency in us
    kernel_launch_cycles = math.ceil(hwCfg.launch_latency/gclk_us)
    kernarg_cycles = math.ceil(hwCfg.kernarg_latency/gclk_us)
    
    def calc_mem_rd_cycles(payload :int, 
                           l2_rd_hit : jnp.float32,
                           mall_rd_hit : jnp.float32,
                           gclk: jnp.float32,
                           hbm_bw:jnp.float32 = 0):
        
        l2_rd_bw_per_cu = min(hwCfg.l2rd_bw_cu, hwCfg.l1_read_bw*hwCfg.l1_efficiency)
        l2_rd_bw_per_cu = l2_rd_bw_per_cu * hwCfg.l2_efficiency
        mall_rd_bw_per_cu = min(hwCfg.mallrd_bw_cu * hwCfg.mallrd_efficiency, l2_rd_bw_per_cu)
        hbw_bw_achievable = hwCfg.hbm_bw_cu if hbm_bw == 0.0 else math.ceil(hbm_bw/hwCfg.num_cus)
        hbm_bw_cc_cu = round(hbw_bw_achievable/(gclk/1000),2)
        hbm_rd_bw_per_cu = min(hbm_bw_cc_cu, mall_rd_bw_per_cu)
        mall_hit_latency = hwCfg.l2_miss_latency
        hbm_latency = hwCfg.mall_miss_latency + hwCfg.l2_miss_latency
        
        avgmem_latency = l2_rd_hit * hwCfg.l2_hit_latency + (1 - l2_rd_hit) * mall_rd_hit * mall_hit_latency + (1 - l2_rd_hit) * (1 - mall_rd_hit) * hbm_latency
        avgmem_rd_bw = l2_rd_bw_per_cu * l2_rd_hit + mall_rd_bw_per_cu * mall_rd_hit * (1-l2_rd_hit) + (1 - l2_rd_hit) * (1 - mall_rd_hit) * hbm_rd_bw_per_cu 
        
        rd_mem_cc = math.ceil(payload/avgmem_rd_bw)
        #print(f"memory read KPI {payload} {rd_mem_cc} {avgmem_rd_bw} {avgmem_latency}")
        return rd_mem_cc, avgmem_latency
        
    def calc_mem_wr_cycles(payload :int, 
                           l2_wr_hit : jnp.float32,
                           mall_wr_hit : jnp.float32,
                           gclk: jnp.float32,
                           hbm_bw:jnp.float32 = 0):
        
        l2_wr_bw_per_cu = min(hwCfg.l2wr_bw_cu, hwCfg.l1_write_bw*hwCfg.l1_efficiency)
        l2_wr_bw_per_cu = l2_wr_bw_per_cu * hwCfg.l2wr_efficiency
        mall_wr_bw_per_cu = min(hwCfg.mallwr_bw_cu * hwCfg.mallwr_efficiency, l2_wr_bw_per_cu)
        hbw_bw_achievable = hwCfg.hbm_bw_cu if hbm_bw == 0.0 else math.ceil(hbm_bw/hwCfg.num_cus)
        hbm_bw_cc_cu = round(hbw_bw_achievable/(gclk/1000),2)
        hbm_wr_bw_per_cu = min(hbm_bw_cc_cu, mall_wr_bw_per_cu)
        #calcualte average mem latency based on hit/misses on cache/memory hierarchy
        mall_latency = hwCfg.l2_miss_latency
        hbm_latency = hwCfg.mall_miss_latency + hwCfg.l2_miss_latency
        avgmem_latency = l2_wr_hit * hwCfg.l2_wr_latency + (1 - l2_wr_hit) * mall_wr_hit * mall_latency + (1 - l2_wr_hit) * (1 - mall_wr_hit) * hbm_latency
        avgmem_wr_bw = l2_wr_bw_per_cu * l2_wr_hit + mall_wr_bw_per_cu * mall_wr_hit * (1-l2_wr_hit) + (1 - l2_wr_hit) * (1 - mall_wr_hit) * hbm_wr_bw_per_cu 
        
        wr_mem_cc = math.ceil(payload/avgmem_wr_bw)
        #print(f"memory write KPI {payload} {wr_mem_cc} {avgmem_wr_bw} {avgmem_latency}")
        return wr_mem_cc, avgmem_latency    
        
    assert BLOCK_K <= K, ValueError(f"{BLOCK_K} must be  less than or equal to reduction dimension size {K}")
    assert num_rows <= M, ValueError(f"number of rows {num_rows} per BlockSize cannot be greater than tensor size(M)")

    
    ##Work Partition
    bpe = 2 if Xdtype == jnp.float16 else 4
    vec_bitwidth = 16 if cols_per_thread*bpe > 16 else vec_bitwidth
    
    num_threads_per_row = max(BLOCK_K // (vec_bitwidth//bpe) , BLOCK_K// cols_per_thread)
    num_threads_per_row = min(num_threads_per_row,BlockSize)
    num_packs_per_col = BLOCK_K//(num_threads_per_row * (vec_bitwidth//bpe))
    num_rows_per_access = BlockSize//num_threads_per_row
    num_row_packs = num_rows//num_rows_per_access
    num_tiles_per_cu = math.ceil(math.ceil(M/BLOCK_M)/hwCfg.num_cus)
    print(f"number of tiles_per cu {num_tiles_per_cu}")
    
    
    #calcualte payload and determine l1_capacity < payload
    #use l1_capacity as preftch size for alu cycles
    #not this changes anythin to the performance it just representative of algo
    payload = num_row_packs*num_packs_per_col  * BlockSize * vec_bitwidth // bpe
    col_loop_iter =1
    row_loop_iter =1
    if ((bpe*payload) > hwCfg.l1_capacity*1024):
        col_loop_iter = math.ceil((num_packs_per_col * BlockSize * vec_bitwidth)/(hwCfg.l1_capacity * 1024))
        row_loop_iter = num_row_packs if col_loop_iter > 1 else math.ceil((num_row_packs*num_packs_per_col * BlockSize * vec_bitwidth)/(hwCfg.l1_capacity * 1024))
    

    print(num_rows_per_access,num_threads_per_row,col_loop_iter,row_loop_iter)
    payload_lat,avg_mem_lat = calc_mem_rd_cycles(payload*bpe,
                                   algoCfg.act_l2rd_hit,
                                   algoCfg.act_mallrd_hit,
                                   gclk)
    
    # start doing math once half of the elements available if alu_cc < half of payload
    
    memrd_cycles = payload_lat + avg_mem_lat
    
    func_name = algoCfg.act_func
    
    _fncall = partial(op_funcs[func_name],payload)
    
    ##1. vector reduction for num_packs_per_col * vec_bitwidth/4
    ##2. shuffle reduction for BlockSize
    
    if (func_name == "softmax"):
        ln_alu_cycles = _fncall(BlockSize)
    else:
        ln_alu_cycles = _fncall()
    reduction_cycles = 0
    if func_name == "softmax":
        reduction_cycles = num_row_packs * 16   #reciprocal
        reduction_cycles += shuffle_reduction(num_threads_per_row,num_rows) if shuffle_reduction == True  else ds_permute_reduction(num_threads_per_row,num_rows)
        reduction_cycles += wg_reduction(num_threads_per_row//32,num_rows,BlockSize)   
    
    print(f"ln-KPI payload = {payload} memrd_cycles {memrd_cycles}, alu_op_cycles = {ln_alu_cycles}")
        
    if (ln_alu_cycles > memrd_cycles): 
        #not optimized version
        ln_cycles = ((memrd_cycles + ln_alu_cycles) *col_loop_iter * row_loop_iter)
        ln_cycles += reduction_cycles * row_loop_iter
        #divide the M2/count
    else: 
        ln_cycles,mem_latency =  calc_mem_rd_cycles(payload*bpe,
                                   algoCfg.act_l2rd_hit,
                                   algoCfg.act_mallrd_hit,
                                   gclk)
        ln_cycles = (mem_latency + ln_cycles)  * col_loop_iter * row_loop_iter
        ln_cycles += reduction_cycles * row_loop_iter
        
    print(f"ln-KPI info::Mean::Variance payload = {payload} ln_cycles = {ln_cycles}")    
       
    assert(col_loop_iter*row_loop_iter*payload*bpe == BLOCK_K*num_rows*bpe)
    
    #ln_cycles = ln_cycles + kernarg_cycles + kernel_launch_cycles + GSU_cycles
    
    if (fusedOps != None):
        if (fusedOps[0] == "matmul"):
            #write out mean and variance for each row
            write_cc = hwCfg.l2_hit_latency + math.ceil(num_rows*2*4/hwCfg.l2wr_bw_cu)
    else:
        write_cc,wr_latency = calc_mem_wr_cycles(num_rows*BLOCK_K*bpe,
                                      algoCfg.act_l2wr_hit,
                                      algoCfg.act_mallwr_hit,
                                      gclk)
        write_cc = (write_cc + wr_latency)
        print(f"ln-KPI write-cycles payload = {num_rows*BLOCK_K*bpe} write_cycles = {write_cc}")
        
        
    ln_cycles += write_cc
    assert(col_loop_iter*row_loop_iter*payload*bpe == BLOCK_K*num_rows*bpe)
    print(f"ln-KPI ln_cycles per num_rows/iter  = {ln_cycles}  {num_rows}") 
    
    ln_cycles = ln_cycles * math.ceil(BLOCK_M//num_rows)
    print(f"ln-KPI ln_cycles per BLOCK_M/iter  = {ln_cycles}  {BLOCK_M}") 
    
    if (persistent_kernel):
       ln_cycles = kernarg_cycles + kernel_launch_cycles + (ln_cycles + algoCfg.tileSetup_time) * num_tiles_per_cu
    else:
       ln_cycles +=  kernarg_cycles + kernel_launch_cycles + algoCfg.tileSetup_time
       ln_cycles = ln_cycles * num_tiles_per_cu
    

    return ln_cycles

if __name__ == "__main__":

    actConfig = algoConfig(Algo="default",weight_l2_hit= 0,
                          weight_mall_hit=0, act_l2rd_hit=0.0,
                          act_mallrd_hit = 0.0, act_l2wr_hit=0.0, 
                          act_mallwr_hit = 0.0,
                          dpm_mode=2,clk_eff=0.65,
                          act_func="gelu")
    hwconfig.initialize(sys.argv)
    hwConfig = hwconfig.hw_config
    
    cycles = Activation(4096,4096,jnp.float16,hwCfg=hwConfig,algoCfg=actConfig,
                       BLOCK_M=4, BLOCK_K=4096, BlockSize=512,
                       cols_per_thread=32, num_rows=4,vec_bitwidth=16,
                       rounding_mode_rne_sw=False,
                       shuffle_reduction=False,
                       ds_permute_reduction=True)
    gclk = 2100*actConfig.clk_eff
    time_us = cycles  * (1/gclk)
    print(f"kernel time (in micro-seconds) = {time_us}")
