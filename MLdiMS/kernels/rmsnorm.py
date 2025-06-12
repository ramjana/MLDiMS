import sys
import builtins
import math
import jax.numpy as jnp
from configs import hwconfig
from configs.hwconfig import ClockConfig
from configs.mlconfig import algoConfig
import jax.numpy as jnp


def rmsnorm(M: int, K: int,
        Xdtype: jnp.dtype,
        hwCfg: hwconfig.HWConfig,
        algoCfg: algoConfig,
        BLOCK_M: int, BLOCK_K: int, BlockSize:int, GSU: int,
        cols_per_thread: int, num_rows: int, vec_bitwidth: int = 16,
        shuffle_reduction: bool = False,
        ds_permute_reduction: bool = False,
        count_check: bool = False,
        persistent_kernel: bool = True,
        fusedOps: tuple=None) -> int:

    #algo
    #for (tileIdx =0 tileIdx<num_tiles +tileIdx)
    #  num_rows_per_iter = BlockSize//cols_per_thread
    #  for (row_idx =0 row_idx<BLOCK_M; rowIdx+=num_rows_per_iter)
    #   numPacks = (BlockSize * cols_per_thread // L1_capacity)
    #   payload = (BLockSize * cols_per_thread) == L1_capacity ? Blocksize * cols_per_thread : l1_capacity
    #   for ( packIdx =0 packIdx < numPacks packIdx++)
    #      prefetch payload[packIdx[]]
    #      wait_for_data
    #      rmsnorm_op(payload)  & prefetch payload[packIdx++]
    #      do wave_level_reduction & wg_level_reduction
    #      payload = payload - mean / sqrt(variance)
    

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

    def  rmsnorm_op(num_elements:int, bpe:int=2):
        #0 f16 -> f32
        #1. sum += pow(x,2) dot2c_f32_f16 (256 eleemnts/cycle if f16
        cycles = 0
        #conversion
        if bpe == 2:
            cycles += num_elements//hwCfg.mul32_rate

        cycles += num_elements // hwCfg.mul32_rate  ## add is necessary
        return cycles

    ###SW / compiler inefficient count checking dimension count
    ###check for every iteration of block_K elements
    ### 
    def cnt_check(num_elements: int): 
       cycles = num_elements//hwCfg.mul32_rate
       return cycles

    def shuffle_wave_reduction(lanes_per_row:int, blocksize:int, bpe):
        ## use  butterfly-shuffle to shift one element at a time and do rmsnorm_op
        num_steps = 5 if lanes_per_row > 32 else math.log(lanes_per_row)
        #+1 for broadcast
        if bpe < 4:
            sdwa_step = 1
        else:
            sdwa_step = 0
        latency = 0
        for iter in range(int(num_steps + sdwa_step)):
           payload = (blocksize//((iter+1)*2)) 
           #move shuffle instruction to read out sum of power
           latency += (blocksize//(256*(4/bpe)))
           latency += rmsnorm_op(payload)
        ##broadcast
    
        latency += (blocksize//256)*4
        return latency
    
    def dspermute_wave_reduction(lanes_per_row:int, blocksize:int, bpe:int):
        ## use  butterfly-shuffle to shift one element at a time and do rmsnorm_op
        num_steps = 5 if lanes_per_row > 32 else math.log(lanes_per_row)
        # 1 ds_bpermute sum of power(elements)
        #issue 1 ds_bpermute blocksize * bpe // 128 
        latency = 0
        for iter in range(int(num_steps)):
           payload = (blocksize//((iter+1)*2)) 
           data_payload = payload * bpe
           latency += hwCfg.lds_latency + data_payload//hwCfg.lds_bandwidth 
           latency += rmsnorm_op(payload)
        ##broadcast
        latency += (blocksize//256)*4
        return latency
    
    
    def wg_reduction(num_wave32_per_row:int,bpe:int):
        writepayload_lat = ((BlockSize//64))*12    #mean, m2, count
        write_latency = (48+12)  if writepayload_lat < 48 else (12+writepayload_lat)
        read_latency = 48 + (BlockSize//64)*4  # 4 upper limit on wave data
        lds_latency = hwCfg.barrier_latency  + write_latency + read_latency
        #All waves read-out mean/m2/count and do wave-level reduction
        lds_latency += shuffle_wave_reduction(num_wave32_per_row,num_wave32_per_row*BlockSize//64,bpe)
        return lds_latency
        
    
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
    assert(shuffle_reduction != ds_permute_reduction)
    
    ##Work Partition
    apply_GSU_reduction = False
    if GSU > 1:
        #employ partition in reduction dimension and master WG reads means/m2 for all rows and do final reduction
        apply_GSU_reduction = True
        BLOCK_K = BLOCK_K//GSU
    
    
    bpe = 2 if Xdtype == jnp.float16 else 4
    vec_bitwidth = 16 if cols_per_thread*bpe > 16 else vec_bitwidth
    
    num_threads_per_row = max(BLOCK_K // (vec_bitwidth//bpe) , BLOCK_K// cols_per_thread)
    num_threads_per_row = min(num_threads_per_row,BlockSize)
    num_packs_per_col = BLOCK_K//(num_threads_per_row * (vec_bitwidth//bpe))
    num_rows_per_access = BlockSize//num_threads_per_row
    num_row_packs = num_rows//num_rows_per_access
    num_tiles_per_cu = math.ceil(math.ceil(M/BLOCK_M)*GSU/hwCfg.num_cus)
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
    
    ##1. vector reduction for num_packs_per_col * vec_bitwidth/4
    ##2. shuffle reduction for BlockSize
    vec_payload = vec_bitwidth//bpe * num_packs_per_col  * num_row_packs * BlockSize
    vec_alu_cycles = rmsnorm_op(vec_payload)
    vec_alu_cycles += 0 if count_check == False else cnt_check(vec_payload)
    ln_alu_cycles = rmsnorm_op(payload)
    print(f"ln-KPI payload = {payload} memrd_cycles {memrd_cycles}, alu_op_cycles = {ln_alu_cycles} vec_alu_cycles {vec_alu_cycles}")
    wave_reduction_cycles = shuffle_wave_reduction(num_threads_per_row, BlockSize*(4/bpe),bpe) if shuffle_reduction else dspermute_wave_reduction(num_threads_per_row, BlockSize*(4/bpe),bpe)
    wg_reduction_cycles = 0
    if (num_threads_per_row > 32):
        wg_reduction_cycles = wg_reduction(num_threads_per_row//32, bpe)
        
    print(f"ln wave_reduction {wave_reduction_cycles} wg reduction = {wg_reduction_cycles}")
        
    if (vec_alu_cycles > memrd_cycles): 
        #not optimized version
        ln_cycles = ((memrd_cycles + rmsnorm_op(vec_payload)) *col_loop_iter * row_loop_iter) 
        ln_cycles += (wave_reduction_cycles + wg_reduction_cycles)* row_loop_iter * num_row_packs
        #divide the M2/count
    else: 
        ln_cycles,mem_latency =  calc_mem_rd_cycles(payload*bpe,
                                   algoCfg.act_l2rd_hit,
                                   algoCfg.act_mallrd_hit,
                                   gclk)
        ln_cycles = (mem_latency + ln_cycles)  * col_loop_iter * row_loop_iter
        ln_cycles += (wave_reduction_cycles + wg_reduction_cycles) * row_loop_iter * num_row_packs
    
    print(f"ln-KPI info::Mean::Variance payload = {payload} ln_cycles = {ln_cycles}")    
    GSU_cycles = 0
    if (apply_GSU_reduction):
        ## All WG(s) write mean, m2 and count, semaphore_count 
        mall_write_cycles= hwCfg.l2_miss_latency + hwCfg.mall_hit_latency + math.ceil((num_rows * 4 * (M)* GSU)/(hwCfg.mallwr_bw_cu * hwCfg.mall_efficiency))
        atomic_latency = hwCfg.l2_miss_latency + hwCfg.mall_hit_latency + (M*GSU / (hwCfg.total_mall_channels * hwCfg.f32_atomic_rate * hwCfg.atomic_efficiency))
        mall_read_cycles =  hwCfg.l2_miss_latency + hwCfg.mall_hit_latency + math.ceil((num_rows * 4 * (M)* GSU)/(hwCfg.mallrd_bw_cu * hwCfg.mall_efficiency))
        GSU_cycles = mall_read_cycles + mall_write_cycles + atomic_latency + rmsnorm_op(GSU*M)
    
       
    assert(col_loop_iter*row_loop_iter*payload*bpe == BLOCK_K*num_rows*bpe)
    
    #ln_cycles = ln_cycles + kernarg_cycles + kernel_launch_cycles + GSU_cycles
    
    if (fusedOps != None):
        if (fusedOps[0] == "matmul"):
            #write out mean and variance for each row
            write_cc = hwCfg.l2_hit_latency + math.ceil(num_rows*2*4/hwCfg.l2wr_bw_cu)
    else:
        
        # v_rcp_f32   take inverse of #elements
        # f_rsq(sum of power(elements)) 
        # fma  f_sqrt(sum of power(elements))  
        count_inv  = (BlockSize // 16) * 4 * (num_rows//num_rows_per_access)
        frsq_cycles = (BlockSize // 16 ) * 4  * (num_rows//num_rows_per_access)  + count_inv
        #fmul count_inv * frsq
        frsq_cycles += (BlockSize // 64 ) * 4  * (num_rows//num_rows_per_access)  + count_inv
        # fmul
        ln_math_cycles = ((payload * col_loop_iter * row_loop_iter)//64) * 4
        
        ln_cycles += ln_math_cycles + frsq_cycles
        print(f"ln-KPI info::LN operator = {payload} ln_cycles = {ln_cycles}")
        #re-enable if K*4 > greater than .75*register size
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
        
        
    ln_cycles += write_cc + GSU_cycles
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

    cycles = rmsnorm(4096,4096,jnp.float16,hwCfg=hwConfig,algoCfg=lnConfig,
                       BLOCK_M=4, BLOCK_K=4096, BlockSize=512,GSU=1,
                       cols_per_thread=32, num_rows=4,vec_bitwidth=16,
                       shuffle_reduction=False,
                       count_check=True,
                       ds_permute_reduction=True,
                       persistent_kernel=True)
    gclk = 2100*.65
    time_us = cycles  * (1/gclk)
    print(f"kernel time (in micro-seconds) = {time_us}")
    cycles = rmsnorm(4096*200*64,64,jnp.float16,hwCfg=hwConfig,algoCfg=lnConfig,
                       BLOCK_M=32*256, BLOCK_K=64, BlockSize=512,GSU=1, 
                       cols_per_thread=32, num_rows=256,vec_bitwidth=16,
                       shuffle_reduction=True,
                       count_check=True,
                       ds_permute_reduction=False,
                       persistent_kernel=True)
    gclk = 2100*.65
    time_us = cycles  * (1/gclk)
    print(f"kernel time (in micro-seconds) = {time_us}")
    cycles = rmsnorm(4096*4096,4096,jnp.float16,hwCfg=hwConfig,algoCfg=lnConfig,
                       BLOCK_M=32*4, BLOCK_K=4096, BlockSize=512,GSU=1, 
                       cols_per_thread=8, num_rows=4,vec_bitwidth=16,
                       shuffle_reduction=True,
                       ds_permute_reduction=False,
                       count_check=True,
                       persistent_kernel=True)
    gclk = 2100*.65
    time_us = cycles  * (1/gclk)
    print(f"kernel time (in micro-seconds) = {time_us}")
