# SRO Decoder Ring - SWI-Prolog Inference Engine

This directory contains a SWI-Prolog implementation of the MLP inference engine using predicate-based matrix operations.

## Core Predicates

The implementation is built on three fundamental predicates:

```prolog
% Matrix-vector multiplication: V_out = V_in @ M
mat_vec(Matrix, Vector, Result).

% Vector addition: C = A + B
vec_add(A, B, C).

% Activation function: V_out = f(V_in)
activation(Type, V_in, V_out).
```

These compose into the dense layer predicate:

```prolog
dense_layer(W, B, Act, V_in, V_out) :-
    mat_vec(W, V_in, Z),
    vec_add(Z, B, A),
    activation(Act, A, V_out).
```

## Usage

### Quick Start

```prolog
% Load the engine
?- consult('sro_decoder_mlp.pl').

% Load weights
?- consult('example_weights.pl').

% Run inference
?- mlp([0.5], Energy).
Energy = [0.00234567].

% With trace for debugging
?- mlp_with_trace([0.5], Energy, Trace).
```

### Running Tests

```prolog
?- consult('sro_decoder_mlp.pl').
?- run_tests.
```

### Using Random Weights (for testing)

```prolog
?- init_random_weights.
?- print_architecture.
?- mlp([0.5], Energy).
```

## Architecture

The SRO Decoder Ring MLP architecture:

```
Input (1) → Dense(6, swish) → Dense(12, swish) → Dense(24, swish) → Dense(6, swish) → Dense(1, linear) → Output
```

## Supported Activations

- `linear` - Identity function
- `relu` - Rectified Linear Unit: max(0, x)
- `swish` - Swish/SiLU: x * sigmoid(x)
- `sigmoid` - Logistic sigmoid: 1 / (1 + exp(-x))
- `softmax` - Softmax (normalized exponential)

## Exporting Weights from PyTorch

Use the provided Python utility:

```bash
# From a trained model
python export_weights_to_prolog.py --model checkpoint.pt --output weights.pl

# Generate random weights for testing
python export_weights_to_prolog.py --random --output weights.pl
```

## Weight File Format

Weights are stored as Prolog facts:

```prolog
% weight(LayerId, RowIndex, WeightRow)
weight(0, 0, [0.123, -0.456, 0.789]).
weight(0, 1, [-0.321, 0.654, -0.987]).

% bias(LayerId, BiasVector)
bias(0, [0.0, 0.0]).

% layer_activation(LayerId, ActivationType)
layer_activation(0, swish).

% layer_config(LayerId, InFeatures, OutFeatures)
layer_config(0, 3, 2).
```

## API Reference

### Predicates

| Predicate | Description |
|-----------|-------------|
| `mat_vec(+M, +V, -R)` | Matrix-vector multiplication |
| `vec_add(+A, +B, -C)` | Element-wise vector addition |
| `activation(+Type, +In, -Out)` | Apply activation function |
| `dense_layer(+W, +B, +Act, +In, -Out)` | Full dense layer |
| `mlp(+Input, -Output)` | Full MLP forward pass |
| `mlp_with_trace(+In, -Out, -Trace)` | Forward pass with trace |
| `load_weights(+File)` | Load weights from file |
| `init_random_weights` | Initialize with random weights |
| `print_architecture` | Display model architecture |
| `run_tests` | Run self-tests |

## Why Prolog?

The predicate-based approach offers several advantages:

1. **Declarative semantics** - Operations are defined relationally
2. **Bidirectional reasoning** - Potential for inverse inference
3. **Compositional** - Predicates chain naturally
4. **Interpretable** - Structure matches mathematical definitions
5. **Symbolic integration** - Easy to combine with symbolic AI

## Requirements

- SWI-Prolog 8.0 or later
- Python 3.8+ (for weight export utility)
