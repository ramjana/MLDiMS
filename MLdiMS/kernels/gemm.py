import sys
import builtins
import math
import jax.numpy as jnp
from configs import hwconfig
from configs.mlconfig import algoConfig

def is_integer(a,b):
    return (a%b == 0)

def cdiv(x:int, div:int):
    return (x + div -1 ) // div
       
def tostr(dtype:jnp.dtype):
    if dtype == jnp.float16:
        return "f16"
    elif dtype == jnp.bfloat16:
        return "bf16"
    else:   
        return "f8"

def gemm(
         M: int, N: int, K: int,
         BLOCK_M:int, BLOCK_N: int, BLOCK_K: int,
         LSU:int, GSU: int, WAVEM: int, WAVEN: int, BlockSize:int,
         #different precision types for weight,activation
         Wdtype: jnp.dtype, Xdtype: jnp.dtype, Odtype: jnp.dtype,
         ALayout: str='T', BLayout: str = 'N', CLayout: str = 'N',
         hwCfg: hwconfig.HWConfig = None,
         algoCfg: algoConfig = None,
         fusedOps: tuple=None,
        ) -> int:

    #check validity of parameters

    if hwCfg == None:
        raise ValueError("Missing hw config object")
    #BlockSize == WaveM*WaveN*WaveFront
    assert WAVEM * WAVEN * hwCfg.wavefront == BlockSize, f"illegal GEMM configuration parameters BlockSize do not match with tile-partition BlockSize={BlockSize}, WaveM::WaveN:WaveFront {WaveM},{WaveN},{WaveFront}"

    if algoCfg.Algo==None:
        Algo = "default"
    else:
        Algo = algoCfg.Algo

    if (LSU*GSU > 1):
        #check K//LSU*GSU should be integer
        assert is_integer(K,LSU*GSU) == True, "K//LSU*GSU must be integer"

    gemm_k = K if LSU*GSU == 1 else K//(LSU*GSU)
    #determine if there is any tail loop
    tail_loop = True if (gemm_k%BLOCK_K) != 0 else False

    #print(f" M={M} N={N} BLOCK_M={BLOCK_M} BLOCK_N={BLOCK_N} BLOCK_K={BLOCK_K}")
    mtiles = cdiv(M,BLOCK_M)
    ntiles = cdiv(N,BLOCK_N)
    ktiles = GSU


    #GPU granularity factor
    num_tiles_per_cu = cdiv(mtiles*ntiles*ktiles,hwCfg.num_cus)
    tail_tiles = (mtiles*ntiles*ktiles % hwCfg.num_cus)
    num_tiles_per_cu += 1 if tail_tiles > 0 else 0    # update with with different Algo approach
    

    def itemsize(inpdType):
        if inpdType == jnp.float16  or inpdType == jnp.bfloat16:
            return 2
        elif inpdType == jnp.float32:
            return 4
        else:
            return 1    
    
    #prologue cycles calculation
    #loadA&B load time (cannot be hidden)
    Wbpe =  itemsize(Wdtype)
    Xbpe =  itemsize(Xdtype)
    WPayload = BLOCK_N*BLOCK_K*Wbpe
    XPayload = BLOCK_M*BLOCK_K*Xbpe
    l2_pipe_lat = hwCfg.l2_hit_latency
    mall_pipe_lat = hwCfg.mall_hit_latency
    hbm_pipe_lat = hwCfg.hbm_latency

    def assert_check():
        assert BLOCK_M<=M, ValueError(f" given GEMM_M size = {M} BLOCK_M = {BLOCK_M}")
        assert BLOCK_N<=N, ValueError(f" given GEMM_N size = {N} BLOCK_M = {BLOCK_N}")
        assert(BLOCK_K<=K), ValueError(f" given GEMM_K size = {K} BLOCK_M = {BLOCK_K}")
        assert BLOCK_M*BLOCK_N*4//256 <= 1024, ValueError(f"output tile size too big(register spills) {BLOCK_M} {BLOCK_K}")
        ldsReq = BLOCK_M*BLOCK_K*Xbpe+BLOCK_N*BLOCK_K*Wbpe
        assert ldsReq <= hwCfg.lds_capacity*1024, ValueError(f"tile size larger than LDS size give {BLOCK_M}::{BLOCK_N}::{BLOCK_K}")

    def xdlrate():
        dtype = max(Wbpe,Xbpe)
        if dtype == 8:
           return hwCfg.xdl_f64_rate
        elif dtype == 4:
           return hwCfg.xdl_f32_rate
        elif dtype == 2:
           return hwCfg.xdl_f16_rate
        elif dtype == 1:
           return hwCfg.xdl_f8_rate
        elif dtype == 0.5:
           return hwCfg.xdl_f8_rate
        else:
           raise ValueError(f"Incorrect datatype given={ Wdtype, Xdtype }")

    
    def _loadtime(payload,bw):
        return cdiv(payload,bw)

    def _bufferLoad(l2hit,mallhit,payload,PipeLatency):
        bw_per_cu = hwCfg.l2rd_bw_cu
        loadTime_l2 = _loadtime(payload*l2hit,bw_per_cu)
        bw_per_cu = hwCfg.mallrd_bw_cu
        loadTime_mall = _loadtime(payload*(1-l2hit)*mallhit,bw_per_cu)
        bw_per_cu = hwCfg.hbm_bw_cu
        loadTime_hbm = _loadtime(payload*(1-l2hit)*(1-mallhit),bw_per_cu)
        payload_time = loadTime_l2 + loadTime_mall + loadTime_hbm
        tileLoad_time = PipeLatency + payload_time
        return tileLoad_time

    def _mempipelat():
        mem_lat = mall_pipe_lat + hbm_pipe_lat
        PipeLatency  = l2_pipe_lat + mem_lat
        return PipeLatency

    def _f82fp16(payload,inpType:str = "weight", layout:str = "N"):
        #nof8->f16 conversion available in mi300
        #2 ops / element op1 fp8->fp32 2.fp32->fp16
        num_ops = 2
        need_perm = False
        if (inpType == "weight" and layout == "N"):
           #layout summation dimension (gemm_k) is strided
           #need extra permute op (0.5 op/element)
           need_perm = True 

        conversion_cycles = payload*num_ops // hwCfg.f32_conv_rate
        if need_perm:
           conversion_cycles += payload // hwCfg.f32_mul_rate
        return conversion_cycles

    def lds_write():
        write_time =  cdiv((WPayload+XPayload),hwCfg.lds_bandwidth)
        #print(f" Write time = {write_time}")
        return write_time

    def lds_read():
        payload = WPayload*WAVEM + XPayload*WAVEN
        read_time =  cdiv(payload,hwCfg.lds_bandwidth)
        #print(f" read time = {read_time}")
        return read_time
    

    def math_time():
        #print(f"math-ops = {2*gemm_k*BLOCK_M*BLOCK_N}")
        #print(f"xdl-rate = {xdlrate()}")
        math_cycles = 2*gemm_k*BLOCK_M*BLOCK_N//xdlrate()
        #print(math_cycles)
        return math_cycles



    assert_check()
    #get tensor gemm input type
    dtype = tostr(Xdtype)
    xdl_inst = hwCfg.gemm_xdl_inst
    inst = xdl_inst[dtype]
    inst_k = inst['K']
    inst_m = inst['M'] if BLOCK_M//WAVEM > inst['M'] else 16
    inst_n = inst_m
    inst_cycles = inst['cycles']


    #tile_setup and first tile load time
    w_l2_hit = algoCfg.weight_l2_hit   
    w_mall_hit = algoCfg.weight_mall_hit   
    memLatency = _mempipelat()
    x_l2_hit = algoCfg.act_l2rd_hit
    x_mall_hit = algoCfg.act_mallrd_hit
    weightLoad_cycles = _bufferLoad(w_l2_hit,w_mall_hit,WPayload,memLatency)
    if (WPayload + XPayload) > (hwCfg.l1_capacity*1024):
        #weights payload is greater than l1 capacity; activation 
        # requests stalled at l1 till first requests comeback
        actLoad_cycles = _bufferLoad(x_l2_hit,x_mall_hit,XPayload,l2_pipe_lat)
    else:
        actLoad_cycles = _bufferLoad(x_l2_hit,x_mall_hit,XPayload,0)


    num_buffer_loads = math.floor(BLOCK_K * BLOCK_N * Wbpe // 4096) 
    num_buffer_loads += math.floor(BLOCK_K * BLOCK_M * Xbpe // 4096) 
    num_ds_writes = num_buffer_loads * 2
    if (BLOCK_K*BLOCK_M*Wbpe%4096 != 0):
        num_buffer_loads += 1
        num_ds_writes +=1
    if (BLOCK_K*BLOCK_N*Wbpe%4096 != 0):
        num_buffer_loads += 1
        num_ds_writes +=1


    x_tile = BLOCK_M/WAVEM
    w_tile = BLOCK_N/WAVEN

    x_inst_tiles = cdiv(x_tile,inst_m)
    w_inst_tiles = cdiv(w_tile,inst_n)

    TileLoad_cycles = weightLoad_cycles + actLoad_cycles
    prologue_time = algoCfg.tileSetup_time + TileLoad_cycles

    #print(TileLoad_cycles)
    #accum init overlapped with globalload time
    accum_init_cycles = (BLOCK_M*BLOCK_N//(4*64)) * 4  ## 64 bytes/per/cycle/SIMD
    if TileLoad_cycles > accum_init_cycles:
        accum_init_cycles = 0

    prologue_time += accum_init_cycles

    #Ds write cycles
    prologue_time += lds_write() + hwCfg.lds_latency

    
    #sync cycles
    #last ds_write throughput time + return ack interface time
    # 1 (index offset) + 4 (data)  * 2 + 48 
    prologue_time += 64

    ## LDS prefetch issue pipe latency  + 
    #gemm_precision  = max(Wbpe,Xbpe)
    bytes_per_weight =  inst_m * inst_k * Wbpe
    bytes_per_act    =  inst_n * inst_k * Xbpe # src_a,src_b
    ldscycles_per_xdl = cdiv(bytes_per_weight+bytes_per_act,hwCfg.lds_bandwidth)

    w_ldsinsts =  w_inst_tiles if Wbpe == Xbpe else w_inst_tiles//2
    x_ldsinsts =  x_inst_tiles
    total_lds_insts_per_iter = w_ldsinsts + x_ldsinsts

    #LDS issue cycles
    prefetch_issue_time = total_lds_insts_per_iter * hwCfg.lds_issue_latency
    #minimum number(2 xdl instructions) of instructions required to start math
     
    weight_bytes = 2*bytes_per_weight if w_ldsinsts > 1 else bytes_per_weight
    act_bytes = bytes_per_act if w_ldsinsts > 1 else 2*bytes_per_act

    ldscycles = cdiv(weight_bytes+act_bytes, hwCfg.lds_bandwidth)
    
    if (ldscycles + hwCfg.lds_latency) > prefetch_issue_time:
        prologue_time += ldscycles + hwCfg.lds_latency 
    else:
        prologue_time += prefetch_issue_time

    num_loop_iter = gemm_k//BLOCK_K
    #lds cycles (read + write)
    ## global->register->lds->register
    lds_cycles =  lds_read() + lds_write()
    math_cycles =  math_time()
    

    #caclculate inst scheduling conflict cycles
    #each MFMA_32x32_x can overlap 1 buffer+1 lds+ or 3 lds_read or 1 ds_write or 7 VALU/SALU instructions
    #number_of buffer_loads
    #//2 using wider bit_width
    #20 (ptr increament)
    total_non_mfma_cycles = 20*4 + num_buffer_loads*16 + num_ds_writes*24 + 8*(w_inst_tiles + x_inst_tiles)*BLOCK_K//inst_k//2

    s_wait_barrier_branch_cycles = 64 + 24 + (BLOCK_K//inst_k)*4  # waitcnt+barrier+s_branch

    total_non_mfma_cycles += s_wait_barrier_branch_cycles

    num_mfma_insts = (w_inst_tiles * x_inst_tiles) * (BLOCK_K//inst_k)
    schedule_conflict_cycles = 0  
    if total_non_mfma_cycles > (math_cycles - (num_mfma_insts*(inst_cycles-4))):
        schedule_conflict_cycles = total_non_mfma_cycles - (math_cycles - (num_mfma_insts*(inst_cycles-4)))

    per_iter_math = (w_inst_tiles * x_inst_tiles)
    ds_write_cycles =  num_ds_writes * 24
    #issue cycles
    ds_read_cycles  = 8*(w_inst_tiles+x_inst_tiles)//2  #wider read 
    #print(f"{ds_write_cycles} {ds_read_cycles} {ds_read_cycles} {per_iter_math}")  
    
    blockk_iter_cnt = BLOCK_K//inst_k
    per_iter_math  = BLOCK_M*BLOCK_N*inst_k*2//xdlrate()
    
    prefetch_overlap = math.floor(TileLoad_cycles/per_iter_math)
    write_overlap_iter = abs(blockk_iter_cnt - prefetch_overlap) if prefetch_overlap < blockk_iter_cnt else 1
    #barrier + branch 
    if (schedule_conflict_cycles == 0):
      if (ds_write_cycles+ds_read_cycles+64+20) > write_overlap_iter*per_iter_math*(inst_cycles-4):
        schedule_conflict_cycles = (ds_write_cycles+ds_read_cycles+64+20) - (write_overlap_iter*per_iter_math*(inst_cycles-4))
      
    ##f32->half_precision conversion time
    ## 1.5 ops per element  64 lanes/per cycles
    conv_pack_cycles = 0
    if (Odtype == Xdtype):
      if (Xbpe == 2):
         ##f32->f16/bf16
         conv_pack_cycles =  math.ceil(BLOCK_M*BLOCK_N*1.5/hwCfg.f32_conv_rate)
         ##f32->f8/bf8
      elif (Xbpe == 1):
         conv_pack_cycles =  math.ceil(BLOCK_M*BLOCK_N//(2*hwCfg.f32_conv_rate))  #packed (2 elem) conversion
      else:
         raise ValueError("Unknow X datatype given {Xdtype}")
      ##overlap pack cycles
      per_iter_cycle = 2*w_inst_tiles * x_inst_tiles * inst_k//xdlrate()//4
      per_iter_cycle = per_iter_cycle - inst_cycles
      conv_pack_cycles = abs(conv_pack_cycles - (per_iter_cycle - (w_inst_tiles * x_inst_tiles)*4))

    ##store time
    store_cycles = hwCfg.l2_wr_latency + math.ceil(BLOCK_M*BLOCK_N*itemsize(Odtype)//hwCfg.l2wr_bw_cu)
    
    #kernel time
    math_cycles = max(math_cycles,(num_loop_iter*(lds_cycles))) + store_cycles + num_loop_iter*(schedule_conflict_cycles)
    mem_time = store_cycles + (num_loop_iter-1)*(TileLoad_cycles+ds_write_cycles+schedule_conflict_cycles)
    loop_time = max(math_cycles,mem_time)
    kernel_time = (num_tiles_per_cu*loop_time) + prologue_time
    #print(f"kerneltime = {kernel_time} {loop_time} {store_cycles} {mem_time} {math_cycles} {prologue_time} {schedule_conflict_cycles}")
    #print(f"number of tiles per cu {num_tiles_per_cu}")
    #print(f"kerneltime = {kernel_time}")
    return (kernel_time)

if __name__ == "__main__":

    gemmConfig = algoConfig(Algo="default",weight_l2_hit= 0.75, weight_mall_hit=0.75, act_l2rd_hit=0.75, act_mallrd_hit = 0.75, tileSetup_time = 800)
    hwconfig.initialize(sys.argv)
    hwConfig = hwconfig.hw_config

    cycles = gemm(4864,4096,4096,256,256,64,1,1,2,2,256,jnp.float16,jnp.float16,jnp.float16,hwCfg=hwConfig,algoCfg=gemmConfig)
    time_us = cycles  * (1/1300) * 1e-3
    print(f"kernel time (in milli-seconds) = {time_us}")
