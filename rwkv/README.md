For fast performance, a device that supports CUDA is required. The code in this directory has been tested on a 3090 RTX on Windows. Training can fall back to CPU, but expect it to be dramatically slower.

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

### Training without CUDA (experimental)
You can run the trainer on CPU-only machines, but it is dramatically slower than CUDA. Edit `rwkv/train_rwkv_config.toml` and set `DEVICE = "cpu"` before launching the command above.
## Evaluation
### Fast evaluation with CUDA
```bash
python -m rwkv.get_result --config rwkv/get_result_config.toml
```

### Same-day (short-term) evaluation
Generate a separate benchmark keyset (includes same-day reviews) in the label-filter LMDB:
```bash
python -m rwkv.find_equalize_test_reviews --config rwkv/find_equalize_test_reviews_config_short_secs.toml --secs --short
```
Then run evaluation in two passes (each pass uses a model trained on the other half of users, and both append into the same output files):
```bash
python -m rwkv.get_result --config rwkv/get_result_config_short_secs_1_4999.toml
python -m rwkv.get_result --config rwkv/get_result_config_short_secs_5000_10000.toml
```
This produces `result/RWKV-short-secs.jsonl` and `result/RWKV-P-short-secs.jsonl`.

### Run on a single user
```bash
python -m rwkv.run_as_rnn --config rwkv/run_as_rnn_config.toml
```
