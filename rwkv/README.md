For fast performance, a GPU that supports CUDA is required. The code in this directory has only been tested on a single 3090 RTX on Windows.

Compile the CUDA kernel
```bash
pip install --no-build-isolation -e .
```

## Setup
Create a helper db, used to precompute bins for the RMSE (bins) metric and to find which reviews are used in the benchmark. This uses 7 GB of storage.
```bash
python -m rwkv.find_equalize_test_reviews --config rwkv\find_equalize_test_reviews_config.toml
```

Preprocess the 10k dataset. Note that with the default configuration, this will require ~400 GB of available storage.
```bash
python -m rwkv.data_processing --config rwkv\data_processing_config_train.toml
```
```bash
python -m rwkv.data_processing --config rwkv\data_processing_config_test.toml
```
## Training
RWKV uses the WSD lr scheduler. There are two distinct phases in training, the warmup + stable phase and then a decay phase.
To switch between the two phases, refer to the config file.
```bash
python -m rwkv.train_rwkv --config train_rwkv_config.toml
```
## Evaluation
```bash
python -m rwkv.get_result --config get_result_config.toml
```
