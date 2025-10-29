# CLAUDE.md - Project Context for AI Assistants

## Project Overview

**SRODecoderRing** is a machine learning project that uses deep learning to infer rotation order from transformation matrices and Euler angles. The project specifically addresses the challenge of interpreting DICOM Spatial Registration Objects (SRO) in the context of radiotherapy image-guidance.

The core problem: Given an SRO (Spatial Registration Object) and an offset, the ML model learns to recognize the input-output relationship to decode the meaning of rotation orders, which is particularly important for patient positioning in medical imaging.

## Domain Context

- **DICOM**: Digital Imaging and Communications in Medicine - the standard for medical imaging data
- **SRO (Spatial Registration Object)**: DICOM objects that describe spatial transformations between coordinate systems
- **Radiotherapy Context**: Accurate spatial transformations are critical for patient positioning and treatment delivery
- **Rotation Order Problem**: Different conventions for applying rotations (e.g., XYZ vs ZYX) can lead to confusion and errors in patient positioning

## Architecture

This is a multi-language, multi-component project:

### Components

1. **Machine Learning Core** (Python)
   - Deep Energy-Based Models (EBM) for learning rotation mappings
   - PyTorch-based implementation with Lightning
   - Training notebooks and scripts
   - MCMC sampling for inference

2. **Inference Engine** (F#)
   - Production inference engine
   - Consumes trained weights and model parameters
   - Functional programming approach for reliability

3. **User Interface** (C#/XAML)
   - Desktop UI for interacting with the decoder
   - WPF/XAML-based interface

4. **Web Service** (Flask/Python)
   - RESTful API for serving predictions
   - Deployed on Azure
   - Integration with web frontend

5. **Frontend** (Fable/Elmish)
   - Functional web UI using F# compiled to JavaScript
   - Elmish architecture (Elm-inspired)
   - Hosted on IPFS

## Tech Stack

### Python
- **PyTorch** (2.0.1+): Deep learning framework
- **PyTorch Lightning**: Training framework
- **NumPy** (1.26.0+): Numerical computations
- **SciPy** (1.11.3+): Scientific computing
- **Rich**: Terminal output formatting
- **Python 3.11+**: Language version

### .NET
- **F#**: Functional programming for inference engine
- **C#**: UI and wrapper components
- **WPF/XAML**: Desktop UI framework

### Web
- **Flask**: Python web framework
- **Fable**: F# to JavaScript compiler
- **Elmish**: Functional UI framework
- **IPFS**: Decentralized storage for weights and frontend

## Project Structure

```
SRODecoderRing/
├── DecoderRingEBM/          # Energy-Based Model implementation
│   ├── DeepEnergyModel.py   # Core EBM architecture
│   ├── torch_model.py       # PyTorch model definitions
│   ├── MCMCSampler.py       # MCMC sampling for inference
│   └── IGRTSyntheticDataset.py  # Dataset generation
│
├── TrainRotationOrder/      # Training scripts
│   ├── sro_decoder_estimate.py    # Model training/estimation
│   └── sro_decoder_optimize_input.py  # Input optimization
│
├── DomainModel/            # Domain logic and models
│   ├── TableDomainModel.py  # Table/coordinate system models
│   ├── service_layer.py     # Service layer abstraction
│   └── common.py            # Shared utilities
│
├── FsSRODecoderEngine/     # F# inference engine (production)
│
├── SRODecoderEngine/       # C# decoder engine
│
├── PythonWrapper/          # Python-to-.NET interop
│   ├── PythonReverse/      # Reverse transformation
│   └── CommandLineReverse/ # CLI tool
│
├── DecoderUI/              # User interface components
│   ├── SRODecoderWebApp/   # Web application
│   └── Elmish.DecoderUI.Views/  # Elmish views
│
├── SRODecoderEngineTest/   # Unit tests
│
└── docs/                   # Documentation
    ├── LearningLieGroups.md
    └── WorklistTriageModel.md
```

## Key Files

- **README.md**: Basic project description
- **pyproject.toml**: Python dependencies and project metadata
- **SRODecoderRing.sln**: Visual Studio solution file
- **LICENSE**: BSD license
- **.gitignore**: Excludes `__pycache__` and `saved_models/`

## Development Workflow

### Training Phase
1. Generate synthetic datasets using `IGRTSyntheticDataset.py`
2. Train EBM model using scripts in `TrainRotationOrder/`
3. Export trained weights to JSON/CSV format
4. Store weights on IPFS for decentralized access

### Production Phase
1. F# engine loads weights from IPFS
2. Flask API provides inference endpoint
3. Web/desktop UI sends requests to API
4. Engine returns decoded rotation order

## Machine Learning Approach

The project uses an **Energy-Based Model (EBM)** approach:
- Maps rotation matrices and Euler angles to rotation order conventions
- Learns the relationship between input transformations and output interpretations
- Uses MCMC sampling for robust inference
- Treats the problem as learning an energy landscape where correct interpretations have low energy

## Common Tasks

### Training a New Model
```bash
cd TrainRotationOrder
python sro_decoder_estimate.py
```

### Running Tests
```bash
dotnet test SRODecoderEngineTest/
```

### Building the Solution
```bash
dotnet build SRODecoderRing.sln
```

### Installing Python Dependencies
```bash
poetry install
```

## Important Conventions

1. **Coordinate Systems**: Multiple coordinate systems are involved (patient, table, gantry)
2. **Rotation Order**: The core problem - different systems may use different rotation conventions
3. **Patient Positioning**: HFS (Head First Supine), FFS (Feet First Supine), etc.
4. **Field Association**: Special handling for multi-field treatments

## Known Issues & TODOs

From README.md:
- Site association needs work
- iView: Applying shift after first field - how to enter with TPO that won't associate with second field?

## References

- Keras Sequential Model Guide: https://keras.io/getting-started/sequential-model-guide/
- Project described as "sledgehammer solution" to the insidious problem of SRO meaning mapping

## Notes for AI Assistants

1. **Safety**: This is medical imaging software. Changes should be carefully validated.
2. **Multi-language**: Be prepared to work across Python, F#, and C# codebases
3. **Domain Knowledge**: Understanding radiotherapy coordinate systems is helpful
4. **Testing**: Always consider the impact on patient safety when making changes
5. **Dependencies**: PyTorch models are large; be mindful of model loading and inference performance
