## General Information

MlDims(Machine learning distributed performance modeling system) is a performance modeling tool for AMD GPU architecture to estimate performance of ML models supporting different model configurations, parallelsim strategies.


LLM is developed using jax framework and simulated using CPU for various parallelism strategies. 

kernels for ML operators are roofline performant kernels with SW/GPU compiler inefficiencies baked into to measure operator performance for AMD GPU (currrently supporting mi300x).


Currently respository supports llama2 with Tensor and data parallelism support.

```bash
git clone https://github.com/ramjana/MLDiMS.git  <name>
cd name
pip3 install -r requirements.txt
```

To run llama2 inference for mi300X hardware using CPU 

```bash
export XLA_FLAGS="--xla_force_host_platform_device_count=8
cd MLdiMS
python3 decode_llama2.py --model=llama2_7b --arch=mi300x
```

## Future updates
pipeline parallelism support</br>
llama3 model. 
