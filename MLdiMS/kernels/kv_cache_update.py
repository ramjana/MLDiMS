import sys
import builtins
import math
import jax.numpy as jnp
from configs import hwconfig
from configs.hwconfig import ClockConfig
from configs.mlconfig import algoConfig
import jax.numpy as jnp


def transpose(bs:int, M:int, K:int,
        Xdtype: jnp.dtype,
        hwCfg: hwconfig.HWConfig,
        algoCfg: algoConfig,
        BLOCK_M: int, BLOCK_K: int, BlockSize:int,
        persistent_kernel: bool = True,
        fusedOps: tuple=None) -> int:

    clkCfg = ClockConfig(algoCfg.dpm_mode)
    gclk = clkCfg.get_gfxclk() * algoCfg.clk_eff
    gclk_us = 1/(gclk)
    #fclk = clkCfg.get_fclk()
    #mclk = clkCfg.get_mclk()

    def itemsize(inpdType):
        if inpdType == jnp.float16  or inpdType == jnp.bfloat16:
            return 2
        elif inpdType == jnp.float32:
            return 4
        else:
            return 1    

    
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
    assert BLOCK_M <= M, ValueError(f"number of rows {BLOCK_M} per BlockSize cannot be greater than tensor size(M)")
    
    m_tiles = math.ceil(M/BLOCK_M)
    k_tiles = math.ceil(K/BLOCK_K)
    num_tiles_per_cu = bs*math.ceil(m_tiles*k_tiles)/hwCfg.num_cus)
    print(f"number of tiles_per cu {num_tiles_per_cu}")
    
    read_cc,rd_latency = calc_mem_rd_cycles(BLOCK_M*BLOCK_K*bpe,
                                      algoCfg.act_l2rd_hit,
                                      algoCfg.act_mallrd_hit,
                                      gclk)
    lds_cycles = BLOCK_M*BLOCK_K*bpe//hwCfg.lds_bandwidth

    if num_tiles_per_cu > 1:
        transpose_lat = read_cc + rd_latency
    else:
        transpose_lat = read_cc + rd_latency + lds_cycles

    write_cc,wr_latency = calc_mem_wr_cycles(BLOCK_M*BLOCK_K*bpe,
                                      algoCfg.act_l2wr_hit,
                                      algoCfg.act_mallwr_hit,
                                      gclk)

    transpose_lat += transpose_lat + write_cc + wr_latency

    if (persistent_kernel):
       transpose_lat = kernarg_cycles + kernel_launch_cycles + (transpose_lat * num_tiles_per_cu)
    else:
       transpose_lat +=  kernarg_cycles + kernel_launch_cycles
       transpose_lat = transpose_lat * num_tiles_per_cu
    
    return transpose_lat


def kv_read(bs: int, M:int, K: int,
        Xdtype: jnp.dtype,
        kvdtype: jnp.dtype, 
        hwCfg: hwconfig.HWConfig,
        algoCfg: algoConfig,
        BLOCK_M: int, BLOCK_K: int, BlockSize:int,
        persistent_kernel: bool = True,
        fusedOps: tuple=None) -> int:


    if hwCfg == None:
        raise ValueError("Missing hw config object")

    clkCfg = ClockConfig(algoCfg.dpm_mode)
    gclk = clkCfg.get_gfxclk() * algoCfg.clk_eff
    gclk_us = 1/(gclk)
    #fclk = clkCfg.get_fclk()
    #mclk = clkCfg.get_mclk()

    def itemsize(inpdType):
        if inpdType == jnp.float16  or inpdType == jnp.bfloat16:
            return 2
        elif inpdType == jnp.float32:
            return 4
        else:
            return 1    

    
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
        
        
    assert BLOCK_K <= K, ValueError(f"{BLOCK_K} must be  less than or equal to reduction dimension size {K}")
    assert BLOCK_M <= M, ValueError(f"number of rows {BLOCK_M} per BlockSize cannot be greater than tensor size(M)")
    
    m_tiles = math.ceil(M/BLOCK_M)
    k_tiles = math.ceil(K/BLOCK_K)
    num_tiles_per_cu = math.ceil(m_tiles*k_tiles)/hwCfg.num_cus)
    print(f"number of tiles_per cu {num_tiles_per_cu}")
    
    bpe = itemsize(kvdtype) 

    read_cc,read_latency = calc_mem_rd_cycles(BLOCK_M*BLOCK_K*bpe,
                                      algoCfg.act_l2rd_hit,
                                      algoCfg.act_mallrd_hit,
                                      gclk)
    read_cc = (read_cc + wr_latency)
    print(f"ln-KPI kvappend-cycles payload = {num_rows*BLOCK_K*bpe} write_cycles = {write_cc}")
        

    //conversion cycles if kvdata precision is lower than seqlen data type
    if itemsize(kvdtype) < itemsize(Xdtype):
       conv_cycles = 2*BLOCK_M*BLOCK_K // hwCfg.mul32_rate   //if kvdatatype(fp8) and XDatatype(fp16) 

    if conv_cycles > read_cc:
        read_cc = conv_cycles
        
    kv_cycles = 2*read_cc * num_tiles_per_cu

    return kv_cycles

def kv_append(bs: int, M:int, K: int,
        Xdtype: jnp.dtype,
        kvdtype: jnp.dtype, 
        hwCfg: hwconfig.HWConfig,
        algoCfg: algoConfig,
        BLOCK_M: int, BLOCK_K: int, BlockSize:int,
        persistent_kernel: bool = True,
        fusedOps: tuple=None) -> int:


    if hwCfg == None:
        raise ValueError("Missing hw config object")

    clkCfg = ClockConfig(algoCfg.dpm_mode)
    gclk = clkCfg.get_gfxclk() * algoCfg.clk_eff
    gclk_us = 1/(gclk)
    #fclk = clkCfg.get_fclk()
    #mclk = clkCfg.get_mclk()

    def itemsize(inpdType):
        if inpdType == jnp.float16  or inpdType == jnp.bfloat16:
            return 2
        elif inpdType == jnp.float32:
            return 4
        else:
            return 1    

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
    assert BLOCK_M <= M, ValueError(f"number of rows {BLOCK_M} per BlockSize cannot be greater than tensor size(M)")
    
    m_tiles = math.ceil(M/BLOCK_M)
    k_tiles = math.ceil(K/BLOCK_K)
    num_tiles_per_cu = math.ceil(m_tiles*k_tiles)/hwCfg.num_cus)
    print(f"number of tiles_per cu {num_tiles_per_cu}")
    
    bpe = itemsize(kvdtype) 

    write_cc,wr_latency = calc_mem_wr_cycles(BLOCK_M*BLOCK_K*bpe,
                                      algoCfg.act_l2wr_hit,
                                      algoCfg.act_mallwr_hit,
                                      gclk)
    write_cc = (write_cc + wr_latency)
    print(f"ln-KPI kvappend-cycles payload = {num_rows*BLOCK_K*bpe} write_cycles = {write_cc}")
        
        
    kv_cycles = 2*write_cc * num_tiles_per_cu

    return kv_cycles

if __name__ == "__main__":

    lnConfig = algoConfig(Algo="default",weight_l2_hit= 0, weight_mall_hit=0, act_l2rd_hit=0.0, act_mallrd_hit = 0.0, act_l2wr_hit=0.0, act_mallwr_hit = 0.0,dpm_mode=2,clk_eff=0.65)
    hwconfig.initialize(sys.argv)
    hwConfig = hwconfig.hw_config

    cycles = kv_append(16,2048,4096,jnp.float16,hwCfg=hwConfig,algoCfg=lnConfig,
                       BLOCK_M=128, BLOCK_K=128, BlockSize=256)
