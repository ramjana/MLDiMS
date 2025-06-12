
import sys
import builtins
import math
import jax.numpy as jnp
from configs import hwconfig
from configs.hwconfig import ClockConfig
from configs.mlconfig import algoConfig
import jax.numpy as jnp


def Rope(M: int, K: int,
        Xdtype: jnp.dtype,
        hwCfg: hwconfig.HWConfig,
        algoCfg: algoConfig,
        BLOCK_M: int, BLOCK_K: int, BlockSize:int,
        cols_per_thread: int, num_rows: int, vec_bitwidth: int = 16,
        count_check: bool = False,
        persistent_kernel: bool = True,
        fusedOps: tuple=None) -> int:

    #algo
    # token_idx = index % K  (index = threadIdx.x + blockIdx.x*blockdimX.x)
    # Theta = 10000**(idx/K)
    # position = X[i][]  i => 0,1,...M 
    # tok_pos = index / N
    # freq = 1/theta
    # pos_theta = position*freq
    # [x1,x2] = input[][2*i+1: 2*i]  for i = 0,1,... K/2
    # rope([x1,x2]) = cos(pos_theta)(x1) - sin(pos)(x2), cos(pos_theta)(x1) + sin(pos_theta)(x2)
    # additional SW ops
    #    fp16->fp32 
    #    
    
    

    if hwCfg == None:
        raise ValueError("Missing hw config object")

    clkCfg = ClockConfig(algoCfg.dpm_mode)
    gclk = clkCfg.get_gfxclk() * algoCfg.clk_eff
    gclk_us = 1/(gclk)
    #fclk = clkCfg.get_fclk()
    #mclk = clkCfg.get_mclk()
    #latency in us
    kernel_launch_cycles = math.ceil(hwCfg.launch_latency/gclk_us)
    kernarg_cycles = math.ceil(hwCfg.kernarg_latency/gclk_us)

    def  rope_op(num_elements:int, bpe:int=2):
        #0  f16->f32 if f16
        #1  pow(10000, token_idx/dim)
        #2  theta = 1/pow(1000,token_idx/dim)
        #3  m_theta = token_pos * theta
        #4  cos(m_theta) = cos(m_theta)
        #5  sin(m_theta) = sin(m_theta)
        #6  x1 = cos(m_theta)*x1 - sin(m_theta)(x2)
        #7  x2 = cos(m_theta)*x1 + sin(m_theta)(x2)
        
        #5 ops using mul/fma rate and f_rcp transcedental ops
        num_trans = 4
        num_mul  = 2
        num_fma = 2 
        if bpe == 2:
            num_mul += 1
        cycles = num_mul * math.ceil(num_elements/hwCfg.mul32_rate) + num_trans * math.ceil(num_elements/hwCfg.transcendental_rate) + num_fma*math.ceil(num_elements/hwCfg.fma32_rate)
        #print(f"rope_op_step {payload} {cycles}")
        return cycles

    #FIXME  count needs to be in float 
    #       check denormal for M2 and 
    def cnt_check(num_elements:int):
        #1. count(2)  conditional check (M2,mean)
        num_ops = 2  #u32->f32->u32
        return  math.ceil(num_elements//hwCfg.mul32_rate)*num_ops 

    def countin_int(num_elements:int):
        #1. uint32 -> f32
        #2. f32-> int32
        num_ops = 2  #u32->f32->u32
        cycles = math.ceil(num_elements//hwCfg.mul32_rate)* num_ops 
        return cycles

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
    
    memrd_cycles = payload_lat + avg_mem_lat
    
    ##1. vector reduction for num_packs_per_col * vec_bitwidth/4
    ##2. shuffle reduction for BlockSize
    vec_payload = vec_bitwidth//bpe * num_packs_per_col  * num_row_packs * BlockSize
    vec_alu_cycles = rope_op(vec_payload//2,bpe)
    ln_alu_cycles = rope_op(payload//2,bpe)
    print(f"ln-KPI payload = {payload} memrd_cycles {memrd_cycles}, alu_op_cycles = {ln_alu_cycles} vec_alu_cycles {vec_alu_cycles}")
        
    if (ln_alu_cycles > memrd_cycles): 
        #not optimized version
        ln_cycles = ((memrd_cycles + rope_op(vec_payload,bpe)) *col_loop_iter * row_loop_iter) 
        sw_alu_cycles += 0 if count_check == False else cnt_check(BlockSize*num_row_packs)
        ln_cycles += sw_alu_cycles
        #divide the M2/count
    else: 
        ln_cycles,mem_latency =  calc_mem_rd_cycles(payload*bpe,
                                   algoCfg.act_l2rd_hit,
                                   algoCfg.act_mallrd_hit,
                                   gclk)
        ln_cycles = (mem_latency + ln_cycles)  * col_loop_iter * row_loop_iter
        ln_cycles += 0 if count_check == False else cnt_check(BlockSize*num_row_packs)
    
    if (fusedOps != None):
        if (fusedOps[0] == "matmul"):
            #write out mean and variance for each row
            write_cc = hwCfg.l2_hit_latency + math.ceil(num_rows*2*4/hwCfg.l2wr_bw_cu)
    else:
        #re-enable if payload > greater than .75*register size
        #read_cc,rd_latency = calc_mem_rd_cycles(num_rows*BLOCK_K*bpe,
        #                           algoCfg.act_l2rd_hit,
        #                           algoCfg.act_mallrd_hit,
        #                          gclk)
        #read_cc = (rd_latency + read_cc)*(BLOCK_M//num_rows)
        
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
       ln_cycles = kernarg_cycles + kernel_launch_cycles + (ln_cycles * num_tiles_per_cu)
    else:
       ln_cycles +=  kernarg_cycles + kernel_launch_cycles
       ln_cycles = ln_cycles * num_tiles_per_cu
    

    return ln_cycles


if __name__ == "__main__":
    
    lnConfig = algoConfig(Algo="default",weight_l2_hit= 0, weight_mall_hit=0, act_l2rd_hit=0.0, act_mallrd_hit = 0.0, act_l2wr_hit=0.0, act_mallwr_hit = 0.0,dpm_mode=2,clk_eff=0.65)
    hwconfig.hw_initialize(sys.argv)
    hwConfig = hwconfig.hw_config

    cycles = Rope(4096,4096,jnp.float16,hwCfg=hwConfig,algoCfg=lnConfig,
                       BLOCK_M=4, BLOCK_K=4096, BlockSize=512,
                       cols_per_thread=32, num_rows=4,vec_bitwidth=16)
    gclk = 2100*.65
    time_us = cycles  * (1/gclk)
    print(f"kernel time (in micro-seconds) = {time_us}")
