# SRODecoderRing

> A machine learning approach to decode DICOM Spatial Registration Object (SRO) rotation order conventions in radiotherapy image-guidance

[![License](https://img.shields.io/badge/License-BSD-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1+-ee4c2c.svg)](https://pytorch.org/)

## Overview

SRODecoderRing uses deep learning to infer rotation order conventions from transformation matrices and Euler angles. This "sledgehammer solution" addresses the insidious problem of interpreting DICOM Spatial Registration Objects (SRO) in radiotherapy, where confusion about rotation conventions can lead to patient positioning errors.

### The Problem

In radiotherapy image-guidance, spatial transformations between coordinate systems are represented by SRO objects. However, the same transformation matrix can be represented with different rotation order conventions (e.g., XYZ vs ZYX), leading to ambiguity and potential errors in patient positioning. Given an SRO and an offset, how do we reliably determine the intended rotation order?

### The Solution

Instead of hand-crafted heuristics, SRODecoderRing uses an **Energy-Based Model (EBM)** to learn the relationship between transformation matrices and their rotation order interpretations. The model learns an energy landscape where correct interpretations have low energy values, enabling robust inference through MCMC sampling.

## Features

- **Deep Energy-Based Model**: PyTorch-based EBM for learning rotation order mappings
- **Multi-Language Architecture**: Python for ML training, F# for production inference
- **Production-Ready**: Flask REST API deployed on Azure
- **Decentralized Storage**: Model weights and frontend hosted on IPFS
- **Web & Desktop UI**: Elmish-based web frontend and WPF desktop application

## Quick Start

### Prerequisites

- **Python 3.11+** with Poetry
- **.NET SDK** (for F# and C# components)
- **Git**

### Installation

```bash
# Clone the repository
git clone https://github.com/DLaneAtElekta/SRODecoderRing.git
cd SRODecoderRing

# Install Python dependencies
poetry install

# Build .NET components
dotnet build SRODecoderRing.sln
```

### Training a Model

```bash
cd TrainRotationOrder
python sro_decoder_estimate.py
```

### Running Tests

```bash
dotnet test SRODecoderEngineTest/
```

## Architecture

SRODecoderRing is a multi-component system:

```
┌─────────────────┐
│   Training      │
│  (Python/PyTorch)│
└────────┬────────┘
         │ Exports weights
         ↓
    ┌────────┐
    │  IPFS  │ ← Decentralized weight storage
    └────┬───┘
         │
    ┌────┴─────────────────────┐
    │                          │
    ↓                          ↓
┌────────────┐          ┌──────────────┐
│ F# Engine  │          │ Flask API    │
│ (Inference)│          │ (Azure)      │
└─────┬──────┘          └──────┬───────┘
      │                        │
      └────────┬───────────────┘
               ↓
    ┌──────────────────┐
    │   UI Layer       │
    │ (Elmish + WPF)   │
    └──────────────────┘
```

### Tech Stack

**Python ML Stack**
- PyTorch 2.0.1+ (Deep learning framework)
- PyTorch Lightning (Training orchestration)
- NumPy 1.26.0+ & SciPy 1.11.3+ (Numerical computing)
- Rich (Terminal output formatting)

**.NET Stack**
- F# (Production inference engine)
- C# (Desktop UI and wrappers)
- WPF/XAML (Desktop interface)

**Web Stack**
- Flask (REST API)
- Fable (F# to JavaScript compiler)
- Elmish (Functional UI framework)
- IPFS (Decentralized hosting)

## Project Structure

```
SRODecoderRing/
├── DecoderRingEBM/           # Energy-Based Model implementation
│   ├── DeepEnergyModel.py    # Core EBM architecture
│   ├── torch_model.py        # PyTorch model definitions
│   ├── MCMCSampler.py        # MCMC sampling for inference
│   └── IGRTSyntheticDataset.py  # Synthetic dataset generation
│
├── TrainRotationOrder/       # Training scripts
│   ├── sro_decoder_estimate.py     # Model training
│   └── sro_decoder_optimize_input.py  # Input optimization
│
├── DomainModel/             # Domain logic and models
│   ├── TableDomainModel.py  # Coordinate system models
│   ├── service_layer.py     # Service layer abstraction
│   └── common.py            # Shared utilities
│
├── FsSRODecoderEngine/      # F# inference engine (production)
├── SRODecoderEngine/        # C# decoder engine
├── PythonWrapper/           # Python-to-.NET interop
├── DecoderUI/               # User interface components
├── SRODecoderEngineTest/    # Unit tests
└── docs/                    # Documentation
```

## Development Workflow

### 1. Training Phase
```bash
# Generate synthetic datasets
python DecoderRingEBM/IGRTSyntheticDataset.py

# Train the model
cd TrainRotationOrder
python sro_decoder_estimate.py

# Weights are exported to JSON/CSV format
# Deploy weights to IPFS
```

### 2. Production Deployment
- F# engine loads weights from IPFS
- Flask API provides inference endpoint
- UI sends requests to API and displays results

## Machine Learning Approach

The project uses an **Energy-Based Model (EBM)**:

1. **Input**: Transformation matrices and Euler angles from SRO
2. **Learning**: Model learns an energy function where correct rotation order interpretations have low energy
3. **Inference**: MCMC sampling explores the energy landscape to find the most likely rotation order
4. **Output**: Decoded rotation order convention

This approach is more robust than rule-based systems, as it learns from data rather than relying on hand-crafted heuristics.

## Domain Context

- **DICOM**: Digital Imaging and Communications in Medicine standard
- **SRO**: Spatial Registration Object describing coordinate transformations
- **Radiotherapy**: Medical treatment using radiation, requiring precise positioning
- **Rotation Order**: Convention for applying sequential rotations (e.g., XYZ, ZYX, XZY)
- **Patient Positioning**: HFS (Head First Supine), FFS (Feet First Supine), etc.

## Known Issues

- **Site association**: Multi-site treatment planning needs refinement
- **iView field association**: Applying shifts after first field with TPO that won't associate with second field

See [ROADMAP.md](ROADMAP.md) for planned improvements.

## Contributing

This is medical imaging software. All changes should be:
1. Carefully validated for patient safety
2. Tested across Python, F#, and C# components
3. Documented with domain context
4. Reviewed for impact on coordinate system calculations

## License

BSD License - see [LICENSE](LICENSE) for details.

## References

- [Keras Sequential Model Guide](https://keras.io/getting-started/sequential-model-guide/)
- [CLAUDE.md](CLAUDE.md) - Detailed project context for AI assistants
- [Learning Lie Groups](docs/LearningLieGroups.md) - Mathematical foundations

## Acknowledgments

Built to solve the insidious problem of SRO meaning mapping in radiotherapy image-guidance systems. Because patient safety matters, we brought a sledgehammer: machine learning.

---

**⚠️ Medical Device Notice**: This software is intended for research purposes. Any clinical use requires appropriate validation and regulatory compliance.
