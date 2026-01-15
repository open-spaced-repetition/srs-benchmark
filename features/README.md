# Feature Engineering Architecture

This directory contains a refactored feature engineering system that replaces the monolithic `create_features_helper` function with a clean, modular architecture based on model-specific feature engineers.

## Architecture Overview

The new architecture follows the Strategy pattern with a factory for creating appropriate feature engineers:

```
BaseFeatureEngineer (Abstract Base Class)
├── FSRSFeatureEngineer      # FSRS family, RNN, GRU, Transformer, etc.
├── LSTMFeatureEngineer      # LSTM with additional features
├── DashFeatureEngineer      # DASH time window features
├── DashMCMFeatureEngineer   # DASH with decay
├── DashACTRFeatureEngineer  # DASH with ACT-R features
├── GRUPFeatureEngineer      # GRU-P with shifted intervals
├── HLRFeatureEngineer       # HLR with success/failure counts
├── ACTRFeatureEngineer      # ACT-R activation features
├── NN17FeatureEngineer      # NN-17 with lapse history
├── SM2FeatureEngineer       # SM2 rating sequences
├── EbisuFeatureEngineer     # Ebisu tuple sequences
├── AVGFeatureEngineer       # Average model
└── RMSEBinsExploitFeatureEngineer  # RMSE-BINS-EXPLOIT model
```

## Key Benefits

1. **Separation of Concerns**: Each model's feature engineering logic is isolated
2. **Maintainability**: Easy to modify or add new models without affecting others
3. **Testability**: Each feature engineer can be tested independently
4. **Readability**: Clear, focused classes instead of one large function
5. **Extensibility**: Simple to add new models by creating new engineer classes
6. **Simplified Interface**: All configuration managed through config object

## Usage

### Basic Usage

```python
from features import create_features

# Replace old code:
# processed_df = create_features_helper(df, model_name, secs_ivl)

# With new code:
processed_df = create_features(df, config=config)
```

### Using Feature Engineer Directly

```python
from features import create_feature_engineer

# Create specific feature engineer
engineer = create_feature_engineer(config)
processed_df = engineer.create_features(df)
```

### Supported Models

```python
from features import get_supported_models

models = get_supported_models()
print(models)
# ['FSRSv1', 'FSRSv2', 'FSRSv3', 'FSRSv4', 'FSRS-4.5', 'FSRS-5', 'FSRS-6',
#  'RNN', 'GRU', 'GRU-P', 'LSTM', 'Transformer', 'NN-17',
#  'SM2', 'SM2-trainable', 'Ebisu-v2', 'HLR', 'ACT-R', 'Anki',
#  'DASH', 'DASH[MCM]', 'DASH[ACT-R]', 'AVG', 'RMSE-BINS-EXPLOIT', '90%']
```

## File Structure

- `base.py`: Abstract base class with common preprocessing logic
- `fsrs_engineer.py`: FSRS family models
- `lstm_engineer.py`: LSTM model with additional features
- `dash_engineer.py`: DASH variants (DASH, DASH[MCM], DASH[ACT-R])
- `neural_engineer.py`: Other neural models (GRU-P, HLR, ACT-R, NN-17)
- `memory_engineer.py`: Memory models (SM2, Ebisu)
- `simple_engineer.py`: Simple models (AVG, RMSE-BINS-EXPLOIT)
- `factory.py`: Factory function for creating appropriate engineers
- `create_features.py`: New create_features function
- `usage_example.py`: Examples of how to use the new system
- `README.md`: This documentation

## Migration Guide

### Step 1: Update Imports

```python
# Old import
from other import create_features_helper

# New import
from features import create_features
```

### Step 2: Update Function Calls

```python
# Old call
processed_df = create_features_helper(df, model_name, secs_ivl)

# New call - simplified interface
processed_df = create_features(df, config=config)
```

### Step 3: Configuration Management

The new system manages all configuration through the config object:

```python
# Set model name
config.model_name = "FSRSv4"

# Set time interval type
config.use_secs_intervals = False  # Use days intervals
# config.use_secs_intervals = True  # Use seconds intervals

# Call feature engineering
dataset = create_features(df, config=config)
```

## Important Interface Simplifications

### Removed Parameters

1. **model_name parameter**: Now read from `config.model_name`
2. **secs_ivl parameter**: Now uses `config.use_secs_intervals`

### Before vs After

```python
# Before
create_features(df, model_name="FSRSv4", secs_ivl=False, config=config)
create_feature_engineer("FSRSv4", config, secs_ivl=False)

# After
config.model_name = "FSRSv4"
config.use_secs_intervals = False
create_features(df, config=config)
create_feature_engineer(config)
```

## Model-Specific Features

### FSRS Models
- Standard tensor format: `[time_intervals, ratings]`
- Used by: FSRSv1-6, RNN, GRU, Transformer, SM2-trainable, Anki, 90%

### LSTM Model
- Additional features: new card counts, review counts, daily statistics
- Tensor format: `[delta_t, rating]` history by default; add `duration` with `--duration`

### DASH Models
- **DASH**: Time window features without decay
- **DASH[MCM]**: Time window features with exponential decay
- **DASH[ACT-R]**: ACT-R style activation features

### Memory Models
- **SM2**: Rating history as string sequence
- **Ebisu**: (time_interval, rating) tuple sequences

### Other Neural Models
- **GRU-P**: Shifted time intervals and ratings
- **HLR**: Square root of success/failure counts
- **ACT-R**: Cumulative time intervals
- **NN-17**: Time intervals, ratings, and lapse history

### Simple Models
- **AVG**: Average success rate
- **RMSE-BINS-EXPLOIT**: Bin-based statistical exploitation

## Adding New Models

To add a new model:

1. Create a new feature engineer class inheriting from `BaseFeatureEngineer`
2. Implement the `_model_specific_features` method
3. Add the model to the factory function in `factory.py`
4. Update the supported models list

Example:

```python
class NewModelFeatureEngineer(BaseFeatureEngineer):
    def _model_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Implement model-specific feature creation
        # ...
        return df
```

## Testing

Each feature engineer can be tested independently:

```python
def test_fsrs_feature_engineer():
    config = create_test_config()
    config.model_name = "FSRSv4"
    engineer = create_feature_engineer(config)
    df = create_test_dataframe()
    result = engineer.create_features(df)
    assert 'tensor' in result.columns
    # Add more assertions
```

## Common Preprocessing

All models share common preprocessing steps implemented in `BaseFeatureEngineer`:

1. Add review sequence numbers
2. Filter invalid ratings
3. Handle two-button mode
4. Calculate review counts
5. Process time intervals
6. Handle short-term reviews
7. Compute history records
8. Set labels and post-process

Model-specific logic is only added in the `_model_specific_features` method.

## Configuration-Driven Architecture

The new architecture is completely based on the configuration object:

- `config.model_name`: Specifies which model to use
- `config.use_secs_intervals`: Controls time interval units (seconds vs days)
- `config.equalize_test_with_non_secs`: Controls test set equalization
- `config.n_splits`: Number of splits for time series cross-validation
- Other model-specific configuration parameters

This approach provides better encapsulation and a cleaner interface. 
