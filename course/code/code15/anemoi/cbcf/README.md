# Installation:

Four simple steps:
```
python3 -m venv venv
source venv/bin/activate
pip install uv
uv pip install .
```

## Alternatives

### DWD gpnl cluster: create venv:
```
module load python/3.10.11 # use this version as it has required python head files
module load proxy/gpu

python3.10 -m venv venv
source venv/bin/activate
pip install uv
uv pip --no-cache install -U torch torchvision "numpy<2" --index-url https://download.pytorch.org/whl/cu124
```

### On other systems
Just create a regular venv with python 3.10 to 3.12:
```
python3.12 -m venv venv
source venv/bin/activate
pip install uv
```

### Manually install local packages
```
uv pip install --no-cache -e "anemoi-core/training/[tests]"  -e "anemoi-core/graphs/[tests]" -e "anemoi-core/models/[tests]" -e "anemoi-datasets/[tests]" -e "anemoi-transform/[tests] cartopy matplotlib"
```

# Training:
```
cd training_config
ANEMOI_BASE_SEED=123 anemoi-training train --config-name cbcf_diagnose_R3B5
cd ..
```

Important settings in `cbcf_diagnose_R3B5.yaml`
```
model.processor.num_layers # number of layers. reduce this to simplify model
model.num_channels # Dimension of hidden space. reduce this to simplify model
system.hardware.accelerator # set this to "cpu" for Apple
system.input.dataset # Path to the training data
diagnostics.log.mlflow # Logging with mlflow
```

# Inference with latest checkpoint:
```
python main.py
```
