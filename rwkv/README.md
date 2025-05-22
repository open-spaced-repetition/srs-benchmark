For fast performance, a device that supports CUDA is required. The code in this directory has been tested on a 3090 RTX on Windows.

RWKV can still be run on non-CUDA devices for individual users. See the section [Run on a single user](#run-on-a-single-user).


## Setup
Compile the CUDA kernel.
```bash
pip install --no-build-isolation -e .
```

Create a helper db, used to precompute bins for the RMSE (bins) metric and to find which reviews are used in the benchmark. This requires 7 GB of storage.
```bash
python -m rwkv.find_equalize_test_reviews --config rwkv/find_equalize_test_reviews_config.toml
```

Preprocess the 10k dataset. With the default configuration, this will require ~400 GB of available storage.
```bash
python -m rwkv.data_processing --config rwkv/data_processing_config_train.toml
```
```bash
python -m rwkv.data_processing --config rwkv/data_processing_config_test.toml
```
## Training
RWKV uses the WSD lr scheduler. There are two distinct phases in training, a warmup + stable phase and then a decay phase.
To switch between the two phases, refer to the config file. RWKV was trained on approximately 10 epochs in the warmup + stable phase and 2 epochs in the decay phase.
```bash
python -m rwkv.train_rwkv --config rwkv/train_rwkv_config.toml
```
## Evaluation
### Fast evaluation with CUDA
```bash
python -m rwkv.get_result --config rwkv/get_result_config.toml
```

### Run on a single user
```bash
python -m rwkv.run_as_rnn --config rwkv/run_as_rnn_config.toml
```
