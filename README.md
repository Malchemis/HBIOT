# MEG Spike Detection Pipeline

A PyTorch Lightning-based deep learning framework for detecting interictal epileptic spikes in MEG recordings using transformer-based architectures.

## Features

- **Transformer-based Models**: Support for BIOT, Hierarchical BIOT, SFCN, and FAMED architectures
- **Efficient Data Processing**: HDF5-based preprocessing with online random windowing
- **Flexible Training**: PyTorch Lightning with DDP support for multi-GPU training
- **Comprehensive Evaluation**: Relaxed metrics with temporal tolerance for realistic spike detection
- **Configurable Pipeline**: YAML-based configuration with registry pattern for extensibility

## Installation

```bash
# Clone the repository
git clone https://github.com/Malchemis/HBIOT.git
cd HBIOT

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.12
- PyTorch 2.0+
- PyTorch Lightning 2.0+
- MNE-Python (for MEG data loading)
- See `requirements.txt` for complete list

## Quick Start

### 1. Preprocess Your Data

Convert raw MEG recordings to HDF5 format for efficient training:

```bash
python scripts/preprocess_recordings.py \
    --data-dir /path/to/meg/data \
    --output-dir /path/to/preprocessed \
    --n-workers 8
```

This step:
- Loads raw MEG files (.ds, .fif, or BTi formats)
- Applies filtering and normalization
- Extracts spike annotations
- Saves to HDF5 format for fast access

### 2. Generate Cross-Validation Splits

Create stratified K-fold splits ensuring balanced spike distribution:

```bash
python scripts/generate_splits.py --config configs/config-splits.yaml
```

This generates:
- `fold_1.json` through `fold_K.json`: Train/validation splits
- `test_files.json`: Holdout test set
- `patient_statistics.json`: Spike count statistics

### 3. Train a Model

Train using the main pipeline script:

```bash
python run_pipeline.py --config configs/default_config.yaml
```

**Training options:**

```bash
# Train with specific batch size
python run_pipeline.py --config configs/default_config.yaml --batch_size 32

# Test only mode (skip training)
python run_pipeline.py --config configs/default_config.yaml --test-only

# Custom number of windows
python run_pipeline.py --config configs/default_config.yaml --n_windows 20
```

### 4. Run Inference

Predict spikes on new recordings:

```bash
python scripts/predict.py \
    --checkpoint /path/to/model.ckpt \
    --config configs/default_config.yaml \
    --input /path/to/recording.ds \
    --output predictions.csv
```

## Project Structure

```
meg-spike-detection/
├── configs/
│   ├── config-splits.yaml       # Stratified K-fold splits configuration
│   └── default_config.yaml       # Main configuration file
├── scripts/
│   ├── generate_splits.py        # Generate train/val/test splits
│   ├── predict.py                # Run inference on new data
│   └── preprocess_recordings.py  # Preprocess MEG to HDF5
├── pipeline/
│   ├── data/                     # Data loading and preprocessing
│   │   ├── meg_datasets.py       # Dataset implementations
│   │   ├── meg_datamodules.py    # Lightning DataModules
│   │   └── preprocessing/        # Signal processing utilities
│   ├── models/                   # Model architectures
│   │   ├── biot.py              # BIOT transformer
│   │   ├── hbiot.py             # Hierarchical BIOT
│   │   ├── sfcn.py              # SFCN baseline
│   │   └── famed.py             # FAMED model
│   ├── training/                 # Training components
│   │   ├── lightning_module.py   # Lightning module
│   │   └── callback_registry.py  # Custom callbacks
│   ├── eval/                     # Evaluation metrics
│   ├── optim/                    # Optimizers and losses
│   └── utils/                    # Utilities
├── requirements.txt              # Python dependencies
└── run_pipeline.py              # Main training script
```

## Configuration

The pipeline uses YAML configuration files. Key sections:

### Data Configuration

```yaml
data:
  name: MEGOnTheFlyDataModule
  MEGOnTheFlyDataModule:
    dataset_name: OnlineWindowDataset
    preprocessed_dir: /path/to/preprocessed
    splits_dir: /path/to/splits

    dataset_config:
      sampling_rate: 200           # Target sampling rate (Hz)
      window_duration_s: 0.2       # Window duration (seconds)
      n_windows: 20                # Windows per chunk
      window_overlap: 0.5          # Overlap ratio (0.0-1.0)
```

### Model Configuration

```yaml
model:
  name: BIOTHierarchical
  BIOTHierarchical:
    window_encoder_depth: 2        # Local transformer depth
    inter_window_depth: 2          # Global transformer depth
    emb_size: 256                  # Model dimension
    heads: 4                       # Attention heads
    mode: "raw"                    # Input mode: "raw", "spec", "features"
```

### Training Configuration

```yaml
trainer:
  max_epochs: 100
  accelerator: "auto"
  devices: "auto"
  precision: 16-mixed              # Mixed precision training
  gradient_clip_val: null          # We use ZClip

optimizer:
  name: AdamW
  AdamW:
    lr: 0.0003
    weight_decay: 0.0001

loss:
  name: FocalLoss
  FocalLoss:
    alpha: 0.25
    gamma: 2.0
```

### Callbacks

The pipeline includes several monitoring callbacks:

- **ModelCheckpoint**: Save best models based on validation metrics
- **EarlyStopping**: Stop training when validation metrics plateau
- **LearningRateMonitor**: Track learning rate changes
- **MetricsEvaluationCallback**: Compute detailed evaluation metrics

## Advanced Usage

### Multi-GPU Training

The pipeline automatically detects multiple GPUs and uses DDP:

```bash
# Automatically uses all available GPUs
python run_pipeline.py --config configs/default_config.yaml
```

### Hierarchical BIOT Token Selection

For the Hierarchical BIOT model, you can configure token selection:

```bash
# Use CLS token
python run_pipeline.py --config configs/default_config.yaml --use_cls_token

# Use mean pooling
python run_pipeline.py --config configs/default_config.yaml --use_mean_pool 2
```

### Custom Evaluation Metrics

The pipeline computes relaxed metrics with temporal tolerance:

- **PR-AUC**: Precision-Recall Area Under Curve
- **ROC-AUC**: Receiver Operating Characteristic AUC
- **Relaxed F1**: F1 score with temporal tolerance window

Tolerance windows account for the inherent uncertainty in spike timing annotations.

## Data Format

### Input Data

The pipeline supports MEG data in the following formats:

- **CTF (.ds)**: CTF MEG systems
- **FIF (.fif)**: Neuromag/Elekta MEG systems
- **BTi**: 4D Neuroimaging MEG systems

Annotations should be embedded in the MEG file as MNE annotations.

### Preprocessed HDF5 Format

After preprocessing, data is stored in HDF5 files with the following structure:

```
recording.h5
├── data                 # MEG data array (n_channels, n_samples)
├── spike_labels         # Binary labels (n_samples,)
├── sampling_rate        # Sampling rate (float)
└── metadata            # Recording metadata (dict)
```

## Model Architectures

### BIOT (Biosignal Transformer)

A transformer-based architecture designed for biosignal processing with:
- Patch-based tokenization
- Positional encoding
- Multi-head self-attention

### Hierarchical BIOT (H-BIOT)

Extends BIOT with two-level hierarchy:
- **Local transformer**: Processes individual windows
- **Global transformer**: Aggregates information across windows
- **Token selection**: Flexible pooling strategies (CLS, extremals, central moments)

### SFCN (Shallow Fully Convolutional Network)

CNN baseline with:
- Temporal convolutions
- Batch normalization
- Global average pooling

## Citation and references

If you use this code in your research, please cite:

```bibtex
[to be added]
```

In this repository we used ZCLIP:
```bibtex
@misc{kumar2025zclipadaptivespikemitigation,
      title={ZClip: Adaptive Spike Mitigation for LLM Pre-Training}, 
      author={Abhay Kumar and Louis Owen and Nilabhra Roy Chowdhury and Fabian Güra},
      year={2025},
      eprint={2504.02507},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2504.02507}, 
}
```
And extend work from original BIOT:
```bibtex
@inproceedings{yang2023biot,
    title={BIOT: Biosignal Transformer for Cross-data Learning in the Wild},
    author={Yang, Chaoqi and Westover, M Brandon and Sun, Jimeng},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
    url={https://openreview.net/forum?id=c2LZyTyddi}
}
@article{yang2023biot,
  title={BIOT: Cross-data Biosignal Learning in the Wild},
  author={Yang, Chaoqi and Westover, M Brandon and Sun, Jimeng},
  journal={arXiv preprint arXiv:2305.10351},
  year={2023}
}
``
