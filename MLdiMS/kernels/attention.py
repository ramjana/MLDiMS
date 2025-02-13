import sys
import builtins
import math
import jax.numpy as jnp
from configs import hwconfig
from configs.mlconfig import attnConfig
from configs.hwconfig import ClockConfig

def is_integer(a,b):
    return (a%b == 0)

def cdiv(x:int,div:int):
    return (x + div -1 ) // div

def xdlrate(Wbpe,Xbpe):
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
       
def tostr(dtype:jnp.dtype):
    if dtype == jnp.float16:
        return "f16"
    elif dtype == jnp.bfloat16:
        return "bf16"
    elif dtype == jnp.float32:
        return "f32"  
    else:
        return "f8"
    
def attention(batch_size: int,
              attn_dim: int, 
              num_heads: int,
              kvheads_div: int,    #1 means kv heads = q_heads
              seqlen_q: int,
              seqlen_kv:int,   # -1 ?  =seqlen_q
              BLOCK_Q:int, BLOCK_K:int, BLOCK_O:int,
              BlockSize:int, KV_SPLIT: int,
              q_dtype: jnp.dtype,
              k_dtype: jnp.dtype,
              v_dtype: jnp.dtype,
              o_dtype: jnp.dtype,
              causal_mask: bool,
              vlayout: str,     # rowlayout seqlenxattndim attndim is fast contiguous ["r',"c"]  "r" = head-dim (output is contiguous)
              hwCfg: hwconfig.HWConfig = None,
              algoCfg: attnConfig = None,
              enable_print: bool  = False,
              label:str = "prefill",
              persistent_kernel: bool = True,
              fusedOps: tuple=None,
              ):
    
    if hwCfg == None:
        raise ValueError("Missing hw config object")
    #BlockSize == WaveM*WaveN*WaveFront
    
    #attention algorithm block or 
    if algoCfg.Algo==None:
        Algo = "default"
    else:
        Algo = algoCfg.Algo

    if (KV_SPLIT > 1):
        #check K//LSU*GSU should be integer
        assert is_integer(seqlen_kv,KV_SPLIT) == True, "K//KV_SPLIT must be integer"
    
    assert kvheads_div == 1, ValueError(f"kvheads_div (MQA,GQA) is not supported yet")
    
    if seqlen_kv == -1:
        seqlen_kv = seqlen_q
    
    #tiles grid calculation
    
    def xdlrate(Wbpe,Xbpe):
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
           raise ValueError(f"Incorrect datatype given={ Wbpe,Xbpe }")
    
    num_q_tiles = math.ceil(seqlen_q//BLOCK_Q)
    num_o_tiles = math.ceil(attn_dim/BLOCK_O)   
    num_tiles = batch_size * num_q_tiles * num_heads 
    #print(f"num_tiles = {num_tiles} qtile ={num_q_tiles} num_heads = {num_heads}")
    num_o_tiles = math.ceil(num_o_tiles/KV_SPLIT)   # split O tiles into KV_SPLIT tiles
    
    num_tiles_per_cu = num_tiles * num_o_tiles / hwCfg.num_cus
    
    #prologue cycles calculation
    #loadA&B load time (cannot be hidden)
    WAVEM = 8  if Algo == "default" else 4
    WAVEN = 1
    #get tensor gemm input type
    dtype = tostr(q_dtype)
    xdl_inst = hwCfg.gemm_xdl_inst
    inst = xdl_inst[dtype]
    inst_k = inst['K']
    inst_m = inst['M'] if BLOCK_Q//WAVEM >= inst['M'] else 16
    inst_n = inst_m
    inst_cycles = inst['cycles']


    clkCfg = ClockConfig(algoCfg.dpm_mode)
    gclk = clkCfg.get_gfxclk() * algoCfg.clk_eff
    gclk_us = 1/(gclk)
    kernel_launch_cycles = math.ceil(hwCfg.launch_latency/gclk_us)
    kernarg_cycles = math.ceil(hwCfg.kernarg_latency/gclk_us)
    
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
    
    def softmax_op(qtile: int = 128, ktile: int=128):
        #number of ops required for softmax 
        #max3 -> reduction(wave32 reduction) -> max,  sub(x-max) -> exp() -> rowsum -> fma   
        #print(f" qtile = {qtile} ktile= {ktile}")
        num_softmax_per_wave = qtile*ktile    #64 threads/wave 4= simds/CU
        #print(f"number_softmax_per_wave = {num_softmax_per_wave}")
        num_max3_wave =  (num_softmax_per_wave//2)    #max3 = max(x,max(y,z))  
        _max3_cycles = num_max3_wave//hwCfg.mul32_rate   #16 elements/cycle/simd  
        #print(f" max3 cycles = {_max3_cycles}")
        _max3_reduction = 52  # max3 reduction across 32 threds using ds_bpermute (latency)
        #sub(x-max) using fma 
        sub_cycles = num_softmax_per_wave // hwCfg.mul32_rate  # 32 ops per cycle/simd  
        #print(f" sub cycles = {sub_cycles}")
        exp_cycles = num_softmax_per_wave // hwCfg.transcendental_rate # exp2f32()  4 ops per cycle/simd
        #print(f" exp cycles = {exp_cycles}")
        rowsum_cycles = num_softmax_per_wave // hwCfg.mul32_rate  # 16 ops per cycle/simd
        #print(f" rowsum = {rowsum_cycles}")
        rowsum_reduction = 52  # ds_bpermute
        scaling_cycles = num_softmax_per_wave // hwCfg.mul32_rate  # 16 ops per cycle/simd #pk_mul
        #print(f" scaling  = {scaling_cycles}")
        total_cycles = _max3_cycles + _max3_reduction + sub_cycles + exp_cycles
        total_cycles += rowsum_cycles + rowsum_reduction + scaling_cycles
        print(f"softmax_cycles = {total_cycles}")
        return total_cycles
    
    
    def attn_vtile_pack(block_n:int, block_k:int):
        ### called when vlayout = "row-major", output dimension (free) is stored contigously in memory
        ### v_perm op to construct 2 elements / lane from 2 sources ; to build 4 elements (mfma_16) require 2 ops
        
        v_perm_rate = hwCfg.fma32_rate
        cycles = block_n*block_k // 2 // v_perm_rate
        return cycles

    def dtype_conversion(payload, src_dtype: jnp.dtype, dst_dtype: jnp.dtype):
        ### type conversion
        ### native support fp32<-> fp16 (RNE support)  f32 ->bf16 thorough shifht right by 16 and apply rounding mode 
        ### support fp32 <-> fp8 , fp32 <-> bf8 packed version 
        ### support fp4 <-> fp16 , int4 <-> fp16 , int8 <-> fp16 , int8 <-> bf16 not supported natively, conversin happens either LUT or bit manipulation
        _cycles =0
        if tostr(src_dtype) == "f32" and tostr(dst_dtype)== "f16":
            # 3 ops / 2 el/lane (64/cu) 3rd op pack 
            _cycles = 1.5*(payload//2) // hwCfg.mul32_rate
        elif tostr(src_dtype) == "f32" and tostr(dst_dtype) == "bf16":
            # 5 ops / element + 1 packing total =11 ops/2 element
            _cycles = 11*(payload//2) // hwCfg.mul32_rate
        elif tostr(src_dtype) == "f16" and tostr(dst_dtype) == "f32":
            _cycles = (payload//2)*1.5 // hwCfg.mul32_rate
        elif tostr(src_dtype) == "bf16" and tostr(dst_dtype) == "f32":
            # 5 ops / element + 1 packing total =11 ops/2 element
            _cycles = 11*(payload//2) // hwCfg.mul32_rate
        elif tostr(src_dtype) == "f32" and tostr(dst_dtype) == "f8":
            _cycles = 2*(payload//4) // hwCfg.mul32_rate
        elif tostr(src_dtype) == "f32" and tostr(dst_dtype) == "bf8":
            _cycles = 2*(payload//4) // hwCfg.mul32_rate
        else:
            ValueError(f" unsupported src and dst types given src_type = {src_dtype} dst = {dst_dtype}")
        #print(f"conversion cycles src_type = f32 dst = {tostr(dtype)}  cycles= {_cycles}")
        return _cycles
        
    def qk_gemm(block_q: int,block_k: int,h_dim: int, dtype:jnp.dtype, blocksize:int = 256, algosw_gemm_eff:float=.95):
        ## qtile in register
        ## ktile in LDS
        ## number of LDS reads 
        ## number of XDL instructions
        q_inst_tiles = block_q//inst_m//(blocksize//64)
        k_inst_tiles  = block_k//inst_n//(blocksize//64)
        _xdl_insts =  q_inst_tiles * k_inst_tiles
        _lds_insts =  k_inst_tiles
        math_cycles = block_q*block_k*h_dim*2 // xdlrate(itemsize(dtype),itemsize(dtype))
        lds_payload = ((blocksize//256)*4*block_k)*h_dim*itemsize(dtype)
        bytes_per_cycle = math.ceil(lds_payload//math_cycles)
        num_iter = h_dim//inst_k
        math_cycles_per_iter = math_cycles//num_iter 
        slots_per_iter = math_cycles_per_iter - _xdl_insts*4
        _lds_cycles = k_inst_tiles*(num_iter-1)*8    
        lds_eff_loss = min(1,math.ceil(hwCfg.lds_bandwidth/bytes_per_cycle))
        lds_cycles_per_inst = inst_k*inst_n*itemsize(dtype)*4//hwCfg.lds_bandwidth
        sched_loss_eff = min(1.00, math.ceil(((slots_per_iter*num_iter)-8)/_lds_cycles))
        prefetch_cycles = (k_inst_tiles*8) + max((hwCfg.lds_latency+2*lds_cycles_per_inst - (k_inst_tiles*8)),0)
        prefetch_loss_eff = math.ceil(prefetch_cycles/math_cycles)
        cycles = math.ceil(math_cycles/(sched_loss_eff * lds_eff_loss * prefetch_loss_eff*algosw_gemm_eff))
        #print(f" qkgemm cycles = qtile = {block_q} ktile ={block_k}, attn_dim= {attn_dim} {cycles}")
        return cycles
    
    def kv_gemm(block_s: int,block_o: int,h_dim: int, dtype:jnp.dtype, blocksize:int = 256, algosw_gemm_eff:float = .95, layout:str = "r"):
        ## stile in register
        ## vtile in LDS
        ## number of LDS reads 
        ## number of XDL instructions
        ## for O_GEMM, we need f32->dtype and rowlayout conversion 
        
        s_inst_tiles = block_s//inst_m//(blocksize//64)
        o_inst_tiles  = block_o//inst_n
        _xdl_insts =  s_inst_tiles * o_inst_tiles
        _lds_insts =  o_inst_tiles
        math_cycles = block_s*block_o*h_dim*2 // xdlrate(itemsize(dtype),itemsize(dtype))
        lds_payload = ((blocksize//256)*4*block_o)*h_dim*itemsize(dtype)
        bytes_per_cycle = math.ceil(lds_payload//math_cycles)
        f32_f16_conversion_cycles = dtype_conversion(block_s*h_dim,jnp.float32,dtype)
        num_iter = h_dim//inst_k
        conv_per_iter = f32_f16_conversion_cycles//num_iter
        layout_conversion_per_iter = 0
        if layout == "r":
            layout_conversion_cycles = attn_vtile_pack(block_s,h_dim)
            layout_conversion_per_iter = layout_conversion_cycles//num_iter
        
        math_cycles_per_iter = math_cycles//num_iter 
        slots_per_iter = math_cycles_per_iter - _xdl_insts*4
        
        #prfetch lds for B matrices, 1 conversion f32->f16 , 2 layout 
        lds_cycles_per_inst = inst_k*inst_n*itemsize(dtype)*4//hwCfg.lds_bandwidth
        layout_conversion_per_inst = layout_conversion_per_iter//o_inst_tiles
        conversion_per_inst = conv_per_iter//s_inst_tiles
        
        ## scaling of SV accumulator post 'online softmax'
        scale_cycles = block_s*block_o//hwCfg.mul32_rate

        
        #prefetch of 1 iteration worth of B matrices 1 conversion of A matrices 2 insts worth of layout_conversion if required
        prefetch_cycles = (o_inst_tiles*8) + conversion_per_inst + max((hwCfg.lds_latency+2*lds_cycles_per_inst - ((o_inst_tiles*8) + conversion_per_inst)),0) + 2*layout_conversion_per_inst
        
        conversion_pack_lds_cycles = o_inst_tiles*(num_iter-1)*8 + (s_inst_tiles*num_iter - 1)*conversion_per_inst + layout_conversion_per_inst*(num_iter*o_inst_tiles - 2)
        conv_pack_lds_scale_cycles = conversion_pack_lds_cycles + scale_cycles
        sched_loss_eff = min(1.00, math.ceil(((slots_per_iter*num_iter)-8)/conv_pack_lds_scale_cycles))
        prefetch_loss_eff = math.ceil(prefetch_cycles/math_cycles)
        
        #print(f" layout_conversion_cycles {layout_conversion_cycles} f32->f16 {f32_f16_conversion_cycles}")
        cycles = math.ceil(math_cycles/(sched_loss_eff * prefetch_loss_eff * algosw_gemm_eff))
        #print(f" kvgemm cycles = stile = {block_s} ktile ={block_o}, attn_dim= {h_dim} {cycles}")
        return cycles
    
    
    #calcuate Q load cycles (should be amortized for when causal mask = 0 otherwise amortized depends upon number of K tiles calculated)
    #causal mask check consumes few ops in each iteration
    #barrier loss for consumer-producer pipeline
    #store lse, otile, load efficiency for diagnoal tiles when causal mask=1
    #

    ##check register and lds resources constraints
    ktile_payload = BLOCK_K*attn_dim*itemsize(k_dtype)
    vtile_payload = BLOCK_O*attn_dim*itemsize(v_dtype)
    single_buffer_kv = True if (ktile_payload + vtile_payload) > hwCfg.lds_capacity*1024 else False
    assert((ktile_payload + vtile_payload) <= hwCfg.lds_capacity*1024)
    
    num_waves_per_simd = BlockSize//hwCfg.wavefront//hwCfg.num_simds
    max_cap_registers_per_wave = hwCfg.vgprs_simd//num_waves_per_simd
    qwave_tile = BLOCK_Q//WAVEM
    Qregtile = qwave_tile*attn_dim*itemsize(q_dtype)//(hwCfg.wavefront*4)
    gemm1_acc = qwave_tile*BLOCK_K//(hwCfg.wavefront)
    gemm2_acc = qwave_tile*BLOCK_O//(hwCfg.wavefront) * 2  #second buffer to adjust softmax scaling
    swave_tile = qwave_tile
    kvtile_buf = BLOCK_K*attn_dim*itemsize(k_dtype)//(4*hwCfg.wavefront*4)


    #GEMM register buffers

    src_buf = (inst_n*inst_k*itemsize(k_dtype)) // (hwCfg.wavefront*4) * 4
    #number of prefetch buffers 
    total_reg = gemm1_acc+gemm2_acc+Qregtile
    #print(f"register total= {total_reg} break-up = {Qregtile} gemm1_acc = {gemm1_acc} gemm2_acc={gemm2_acc} src-buf = {src_buf} kvtile_buf = {kvtile_buf}")
    assert(total_reg <= max_cap_registers_per_wave)
    assert((src_buf+kvtile_buf)<64, f"temp resgisters for global/local/softmax should not be breater than 64 {src_buf+kvtile_buf}")

    #qload_time
    qlat, qpayload = calc_mem_rd_cycles(BLOCK_Q*attn_dim*itemsize(q_dtype),
                                        algoCfg.qtile_l2rd_hit,
                                        algoCfg.qtile_mallrd_hit,
                                        gclk)
  
    klat, kpayload = calc_mem_rd_cycles(BLOCK_K*attn_dim*itemsize(k_dtype),
                                        algoCfg.ktile_l2rd_hit,
                                        algoCfg.ktile_mallrd_hit,
                                        gclk)
    vlat, vpayload = calc_mem_rd_cycles(BLOCK_O*attn_dim*itemsize(v_dtype),
                                        algoCfg.vtile_l2rd_hit,
                                        algoCfg.vtile_mallrd_hit,
                                        gclk)

       
    qk_cycles = qk_gemm(qwave_tile*4, BLOCK_K, attn_dim, dtype=q_dtype, blocksize=256,algosw_gemm_eff=algoCfg.gemm_eff_cap)
    sv_cycles = kv_gemm(swave_tile*4, BLOCK_O, attn_dim, dtype=q_dtype, blocksize=256,algosw_gemm_eff=algoCfg.gemm_eff_cap,layout=vlayout)
    softmax_cycles = softmax_op(swave_tile*4, ktile=BLOCK_K)

    ##sync cycles
    if single_buffer_kv == True:
       ## wait for GEMM2 completion by other WG and sync
       if ((klat+kpayload) < sv_cycles):
           qk_cycles = max(qk_cycles, sv_cycles+hwCfg.barrier_latency)
       else:
           qk_cycles = max(qk_cycles, (klat+kpayload)+hwCfg.barrier_latency)

       if ((vlat+vpayload) < qk_cycles):
           sv_cycles = max(sv_cycles, qk_cycles+hwCfg.barrier_latency)
       else:
           sv_cycles = max(sv_cycles, (vlat+vpayload)+hwCfg.barrier_latency)

    ## scaling of SV accumulator post 'online softmax'
    ##scale_cycles = swave_tile*4*BLOCK_O//hwCfg.mul32_rate

    inner_iter_cnt = seqlen_kv//BLOCK_K
    if causal_mask == 1:
       inner_iter_cnt = inner_iter_cnt//2

    olat,opayload = calc_mem_wr_cycles(qwave_tile*4*BLOCK_O*itemsize(o_dtype),
                                      algoCfg.otile_l2wr_hit,
                                      algoCfg.otile_mallwr_hit,
                                      gclk)
    write_cycles = olat + opayload
    total_cycles_per_iter = qk_cycles+max(softmax_cycles,sv_cycles)
    print(f"attention-op per iter cycles {total_cycles_per_iter}")
    total_cycles_per_qtile = (total_cycles_per_iter * inner_iter_cnt)
    if BlockSize > 256:
        total_cycles_per_qtile = total_cycles_per_qtile * 2

    total_cycles_per_qtile += write_cycles   # other Qtile*block_0 is overlapped with second gemm
    print(f"attention-op per qtile cycles :: inner_loop_cnt = {inner_iter_cnt} write_cycles = {write_cycles} cycles_per_qtile={total_cycles_per_qtile}")

    tilesetup_cycles = algoCfg.tileSetup_time + (qlat+qpayload)
      
    clkCfg = ClockConfig(algoCfg.dpm_mode)
    gclk = clkCfg.get_gfxclk() * algoCfg.clk_eff
    gclk_us = 1/(gclk)
    kernel_launch_cycles = math.ceil(hwCfg.launch_latency/gclk_us)
    kernarg_cycles = math.ceil(hwCfg.kernarg_latency/gclk_us)
    if persistent_kernel == False:
       total_cycles = num_tiles_per_cu * (tilesetup_cycles + total_cycles_per_qtile + kernel_launch_cycles + kernarg_cycles)
    else:
        total_cycles = num_tiles_per_cu * (total_cycles_per_qtile)
    print(f"total cycles for num_tiles_per_cu = {num_tiles_per_cu} {total_cycles}")
    if enable_print:
        num_bytes = 0
        flops  = 0

        num_bytes +=  batch_size * num_heads * (seqlen_q * attn_dim * itemsize(q_dtype) + 
                                   seqlen_kv * attn_dim * itemsize(k_dtype) +
                                   seqlen_kv * attn_dim * itemsize(v_dtype) +
                                   seqlen_q * attn_dim * itemsize(o_dtype))
        flops += batch_size * num_heads * ((2*seqlen_q * attn_dim * seqlen_kv) + (2*seqlen_q * attn_dim * seqlen_kv))
        us = total_cycles * gclk_us
        ms = us/1000
        #print(f"{label}-{tostr(q_dtype)} {total_cycles}, {flops} {flops / us * 1e-6:.4f} TFLOPS")
        print(f"{label}-{tostr(q_dtype)}  TotalFlops={flops} time= {us/1000:.4f} ms,{flops / us * 1e-6:.4f} TFLOPS")
        print(f"{label}-{tostr(q_dtype)}, TotalBytes={num_bytes} {num_bytes / us   *1e-3:.4f} GBytes")
    return total_cycles

if __name__ == "__main__":

    attnCfg = attnConfig(Algo="default",qtile_l2rd_hit= 0.0, qtile_mallrd_hit=0.0, ktile_l2rd_hit=0.75, ktile_mallrd_hit=0.25,
                         vtile_mallrd_hit=0.25, vtile_l2rd_hit=0.75,tileSetup_time = 500,vlayout="r",dpm_mode=0, clk_eff=0.9,
                         otile_l2wr_hit=0.0, otile_mallwr_hit=0.0,causal_mask_eff=1.7)
    hwconfig.initialize(sys.argv)
    hwConfig = hwconfig.hw_config

    cycles = attention(batch_size=16,
                       attn_dim=128,
                       num_heads=16,
                       kvheads_div=1,
                       seqlen_q=1024,
                       seqlen_kv=1024,
                       BLOCK_Q=256,BLOCK_K=128, BLOCK_O=128,
                       BlockSize=512, KV_SPLIT=1,
                       q_dtype=jnp.float16,
                       k_dtype=jnp.float16,
                       v_dtype=jnp.float16,
                       o_dtype=jnp.float16,
                       causal_mask=False,
                       vlayout="r",
                       hwCfg=hwConfig,
                       algoCfg=attnCfg,
                       enable_print=True)
