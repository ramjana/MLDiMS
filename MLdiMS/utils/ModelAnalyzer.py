import os
import Path
import math
import jax.numpy as jnp

import yaml
import json
import functools

LAYER_PARAMETERS = [
        "OPS",
        "memory_bytes",
        "load_weight",
        "load_act",
        "store_act",
        "load_kvcache",
        "store_kvcache",
        "time",
        "rooflineTime",
]

def roofline(hbmBw,max_flops,aluOps,memBytes):
    """
    hbmBW: hardware HBM BW Tbytes/sec
    max_flops : hardware max flops Tflops/sec
    aluops : number of ops 
    memBytes : bytes 
    """

    y_max = max_flops
    slope_point = max_flops/hbmBw

    alu_intensity = aluOps/memBytes
    if alu_intensity < slope_point:
        perf = alu_intensity * hbmBw
        bound = "memory"
    else:
        bound = "compute"
        perf  =  y_max

    return alu_intensity,perf,bound


class ModelAnalyzer(Object):
    """
      model performance and stats data
    """

    def xdlrate(self,dtype):
        if dtype == 8:
           return self.hwCfg.xdl_f64_rate
        elif dtype == 4:
           return self.hwCfg.xdl_f32_rate
        elif dtype == 2:
           return self.hwCfg.xdl_f16_rate
        elif dtype == 1:
           return self.hwCfg.xdl_f8_rate
        elif dtype == 0.5:
           return self.hwCfg.xdl_f8_rate
        else:
           raise ValueError(f"Incorrect datatype given={ Wdtype, Xdtype }")

    def __new__(cls):
        if not hasattr(cls,"instance"):
            cls.instance = super(ModelAnalyzer,cls).__new__(cls)
        return cls.instance

    def __init__(self,hwCfg:HWConfig,mlCfg:llmConfig,model_name:str=None):
        """
           inputArguments:
           hwCfg: hardware architecural configurations
           mlCfg: model Configuration
           model_name: name of the model
        """

        if hwCfg is None:
            ValueError(f" passed None value of hardware config class")
        if mlCfg is None:
            ValueError(f" passed None value of model configuration class")

        self.perf_results = None
        self.hwCfg = hwCfg
        self.mlCfg = mlCfg
        self.modelname = model_name

    def perf_analyzer(self,
            phase: str = "prefill",
            layer_name: str = None,
            aluOps: int = 2,
            weight_bytes: int = 1,
            act_bytes: Tuple = (0,0),   # load and store
            kvcache_bytes: Tuple = (0,0), # load and store kvcache
            act_dtype: jnp.dtype,
            wt_dtype: jnp.dtype,
            kv_dtype: jnp.dtype,
            Optime: float,
            freq: float,
        ):

        hw_bandwidth = hwCfg.hbm_membw_acheivable  #inTBytes
        max_flops    = xdlrate(act_dtype)*freq*hwCfg.num_cus * 1e-6 #in TFlops
        total_mem_bytes = kvcache_bytes[0] + kvcacvhe_bytes[1] + act_bytes[0] + act_bytes[1] + weight_bytes

        alu_intensity, perf, bound = hw_roofline(hw_bandwidth,max_flops,aluOps,total_mem_bytes)

        roofline_time = aluOps/perf

        self.perf_results[phase][layer_name] = {
                "OPs" : aluOps,
                "memory_bytes" : total_mem_bytes,
                "alu_intensity" : alu_intensity,
                "performance" : perf,
                "bound": bound,
                "load_weight" : weight_bytes,
                "load_act"    : act_bytes[0],
                "store_act"   : act_bytes[1],
                "load_kvcache" : kvcache_bytes[0]
                "store_kvcache" : kvcache_bytes[1]
                "time" : Optime,
                "rooflineTime": roofline_time,
        }

    def modelPerformance(self):

        model_results = {"prefill" : {}, "generate" : {}}

        num_layers = self.mlCfg.num_layers if self.mlCfg.single_layer_sim == True else 1  # simulting single layer of whole layers

        for _items in LAYER_PARAMETERS:
            model_results["prefill"][_items] = 0
            model_results["generate"][_items] = 0
        for phase in ["prefill","generate"]:
            for layer,result in self.perf_results[phase].items():
                for _item in LAYER_PARAMETERS:
                    model_results[phase][_item] += result[_item] * num_layers

        #memory utilization of model

        kvcahe_size = model_results["prefill"]["store_kvcache"]
        weight_size = model_results["prefill"]["load_weight"]
        act_size = 0
        for layer,result in model_results["generate"].items()
            act_size += result["store_act"] 

        model_results['generate']['total_mem'] = kvcache_size + weight_size + act_size
        model_results['generate']['memory_weight'] = weight_size
        model_results['generate']['memory_activation'] = act_size
        model_results['generate']['memory_kvcache'] = kvcache_size


        kvcahe_size = model_results["generate"]["store_kvcache"]
        weight_size = model_results["generate"]["load_weight"]
        act_size = 0
        for layer, results in model_results['generate'].items():
            act_size += layer['store_act']

        model_results['prefill']['total_mem'] = kvcache_size + weight_size + act_size
        model_results['prefill']['memory_weight'] = weight_size
        model_results['prefill']['memory_activation'] = act_size
        model_results['prefill']['memory_kvcache'] = kvcache_size

        self.perf_results['ModelStats'] = model_results
        return self.perf_results

    def log_json(self, output_path: Path):
        
        assert(output_path != None)

        prefill_name = os.path.join(output_path,"prefill.json")
        generate_name = os.path.join(output_path,"generate.json")

        for filename,phase in [(prefill_name,"prefill"),(generate_name,"generate")]:

           with open(filename,"w") as f:
               f.write(
                   f"\n\n **** model={self.model_name} act_precision={self.mlCfg.dtype} weight_precision={self.mlCfg.weight_dtype} kvcache_precision={self.mlCfg.kvcache_dtype} tensor_parallelism = {self.mlCfg.tensor_parallelism} data_parallelsim = {self.mlCfg.data_parallelism} pipeline_parallelism = {self.mlCfg.pipeline_parallelism} num_devices = {self.mlCfg.num_devices}")
               f.write(
                   f"layer_name,AluOps,MemBytes,alu_intensity,performance,bound,load_weight,load_act,store_act,load_kvcache,store_kvcache,time,roofline_time\n"
               )
               data_dict = {}
               for layer, result in self.perf_results[phase].items():
                   data_dict['layer_name'] = layer
                   for k,v in result.items():
                       data_dict[k] = v
                   json.dump(data_dict,f)
                   data_dict.clear()
