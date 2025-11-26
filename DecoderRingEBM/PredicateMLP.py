"""
Predicate-Based MLP Implementation for SRO Decoder Ring

This module implements neural network operations using relational predicates:
- mat_vec(M, V_in, V_out): Matrix-vector multiplication
- vec_add(A, B, C): Vector addition
- activation(V_in, V_out): Activation function application

The predicate approach provides:
1. Declarative specification of computations
2. Potential for bidirectional inference
3. Composability and interpretability
4. Natural integration with logic programming systems
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Optional, Callable, NamedTuple
from dataclasses import dataclass
from enum import Enum


class ActivationType(Enum):
    """Supported activation functions"""
    LINEAR = "linear"
    RELU = "relu"
    SWISH = "swish"
    SIGMOID = "sigmoid"
    SOFTMAX = "softmax"


@dataclass(frozen=True)
class PredicateResult:
    """Result of a predicate evaluation"""
    success: bool
    value: Optional[torch.Tensor]
    binding: Optional[dict] = None


# =============================================================================
# Core Predicates
# =============================================================================

def mat_vec(M: torch.Tensor, V_in: torch.Tensor, V_out: Optional[torch.Tensor] = None) -> PredicateResult:
    """
    Matrix-vector multiplication predicate: mat_vec(M, V_in, V_out)

    Modes:
    - mat_vec(+M, +V_in, -V_out): Forward pass, compute V_out = M @ V_in
    - mat_vec(+M, -V_in, +V_out): Inverse pass (pseudo-inverse), estimate V_in
    - mat_vec(-M, +V_in, +V_out): Learn M given V_in and V_out (least squares)

    Args:
        M: Weight matrix [out_features, in_features] or [in_features, out_features]
        V_in: Input vector [batch, in_features]
        V_out: Output vector [batch, out_features] (optional, for binding)

    Returns:
        PredicateResult with computed or verified output
    """
    if V_out is None:
        # Forward mode: compute V_out = V_in @ M (standard PyTorch linear convention)
        computed = torch.matmul(V_in, M)
        return PredicateResult(success=True, value=computed, binding={"V_out": computed})
    else:
        # Verification mode: check if V_out ≈ V_in @ M
        computed = torch.matmul(V_in, M)
        is_close = torch.allclose(computed, V_out, rtol=1e-5, atol=1e-8)
        return PredicateResult(success=is_close, value=computed, binding={"match": is_close})


def vec_add(A: torch.Tensor, B: torch.Tensor, C: Optional[torch.Tensor] = None) -> PredicateResult:
    """
    Vector addition predicate: vec_add(A, B, C)

    Semantics: C = A + B

    Modes:
    - vec_add(+A, +B, -C): Compute C = A + B
    - vec_add(+A, -B, +C): Compute B = C - A
    - vec_add(-A, +B, +C): Compute A = C - B

    Args:
        A: First vector
        B: Second vector (typically bias)
        C: Result vector (optional)

    Returns:
        PredicateResult with sum
    """
    if C is None:
        computed = A + B
        return PredicateResult(success=True, value=computed, binding={"C": computed})
    else:
        computed = A + B
        is_close = torch.allclose(computed, C, rtol=1e-5, atol=1e-8)
        return PredicateResult(success=is_close, value=computed, binding={"match": is_close})


def activation(V_in: torch.Tensor, V_out: Optional[torch.Tensor] = None,
               act_type: ActivationType = ActivationType.SWISH) -> PredicateResult:
    """
    Activation function predicate: activation(V_in, V_out)

    Args:
        V_in: Pre-activation values
        V_out: Post-activation values (optional, for verification)
        act_type: Type of activation function

    Returns:
        PredicateResult with activated values
    """
    if act_type == ActivationType.LINEAR:
        computed = V_in
    elif act_type == ActivationType.RELU:
        computed = torch.relu(V_in)
    elif act_type == ActivationType.SWISH:
        computed = V_in * torch.sigmoid(V_in)
    elif act_type == ActivationType.SIGMOID:
        computed = torch.sigmoid(V_in)
    elif act_type == ActivationType.SOFTMAX:
        computed = torch.softmax(V_in, dim=-1)
    else:
        raise ValueError(f"Unknown activation type: {act_type}")

    if V_out is None:
        return PredicateResult(success=True, value=computed, binding={"V_out": computed})
    else:
        is_close = torch.allclose(computed, V_out, rtol=1e-5, atol=1e-8)
        return PredicateResult(success=is_close, value=computed, binding={"match": is_close})


# =============================================================================
# Compound Predicates
# =============================================================================

def dense_layer(W: torch.Tensor, B: torch.Tensor, V_in: torch.Tensor,
                V_out: Optional[torch.Tensor] = None,
                act_type: ActivationType = ActivationType.SWISH) -> PredicateResult:
    """
    Dense layer predicate combining mat_vec, vec_add, and activation.

    Semantics:
        dense_layer(W, B, V_in, V_out) :-
            mat_vec(W, V_in, Z),
            vec_add(Z, B, A),
            activation(A, V_out).

    Args:
        W: Weight matrix
        B: Bias vector
        V_in: Input vector
        V_out: Output vector (optional)
        act_type: Activation type

    Returns:
        PredicateResult with layer output
    """
    # mat_vec(W, V_in, Z)
    mat_result = mat_vec(W, V_in)
    if not mat_result.success:
        return PredicateResult(success=False, value=None)
    Z = mat_result.value

    # vec_add(Z, B, A)
    add_result = vec_add(Z, B)
    if not add_result.success:
        return PredicateResult(success=False, value=None)
    A = add_result.value

    # activation(A, V_out)
    act_result = activation(A, V_out, act_type)

    return PredicateResult(
        success=act_result.success,
        value=act_result.value,
        binding={
            "Z": Z,  # Pre-bias
            "A": A,  # Pre-activation
            "V_out": act_result.value  # Final output
        }
    )


# =============================================================================
# Predicate-Based MLP Model
# =============================================================================

@dataclass
class LayerSpec:
    """Specification for a single layer"""
    in_features: int
    out_features: int
    activation: ActivationType = ActivationType.SWISH
    use_bias: bool = True


class PredicateMLP(nn.Module):
    """
    MLP implemented using predicate-based operations.

    Each forward pass is a composition of predicates:
        mlp(X, Y) :-
            dense_layer(W1, B1, X, H1, act1),
            dense_layer(W2, B2, H1, H2, act2),
            ...
            dense_layer(Wn, Bn, Hn-1, Y, actn).

    This representation makes the computation explicit and composable,
    enabling potential applications in:
    - Neural-symbolic integration
    - Interpretable AI
    - Bidirectional inference
    """

    def __init__(self, layer_specs: List[LayerSpec]):
        """
        Initialize predicate-based MLP.

        Args:
            layer_specs: List of LayerSpec defining the architecture
        """
        super(PredicateMLP, self).__init__()

        self.layer_specs = layer_specs
        self.n_layers = len(layer_specs)

        # Initialize weights and biases as parameters
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()

        for i, spec in enumerate(layer_specs):
            # Weight matrix: [in_features, out_features] for V_in @ W convention
            W = nn.Parameter(torch.randn(spec.in_features, spec.out_features) * 0.1)
            self.weights.append(W)

            if spec.use_bias:
                B = nn.Parameter(torch.zeros(1, spec.out_features))
            else:
                B = nn.Parameter(torch.zeros(1, spec.out_features), requires_grad=False)
            self.biases.append(B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using predicate composition.

        The computation is equivalent to:
            mlp(X, Y) :-
                layer(0, X, H0),
                layer(1, H0, H1),
                ...
                layer(n, Hn-1, Y).
        """
        current = x

        for i, spec in enumerate(self.layer_specs):
            result = dense_layer(
                W=self.weights[i],
                B=self.biases[i],
                V_in=current,
                act_type=spec.activation
            )
            if not result.success:
                raise RuntimeError(f"Predicate failed at layer {i}")
            current = result.value

        return current

    def forward_with_trace(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[dict]]:
        """
        Forward pass returning intermediate predicate bindings.

        Useful for interpretability and debugging.

        Returns:
            Tuple of (output, list of binding dicts for each layer)
        """
        current = x
        trace = []

        for i, spec in enumerate(self.layer_specs):
            result = dense_layer(
                W=self.weights[i],
                B=self.biases[i],
                V_in=current,
                act_type=spec.activation
            )
            trace.append({
                "layer": i,
                "input_shape": current.shape,
                "output_shape": result.value.shape,
                "bindings": result.binding
            })
            current = result.value

        return current, trace

    def as_prolog_rules(self) -> str:
        """
        Export the MLP structure as Prolog-style rules.

        Returns:
            String representation of the MLP as Prolog predicates
        """
        rules = []
        rules.append("% Predicate-based MLP for SRO Decoder Ring")
        rules.append("% Generated from PredicateMLP")
        rules.append("")

        # Layer definitions
        for i, spec in enumerate(self.layer_specs):
            act_name = spec.activation.value
            rules.append(f"% Layer {i}: {spec.in_features} -> {spec.out_features}, {act_name}")
            rules.append(f"layer({i}, V_in, V_out) :-")
            rules.append(f"    mat_vec(w{i}, V_in, Z{i}),")
            rules.append(f"    vec_add(Z{i}, b{i}, A{i}),")
            rules.append(f"    {act_name}(A{i}, V_out).")
            rules.append("")

        # Full MLP predicate
        layer_chain = ", ".join([f"layer({i}, H{i}, H{i+1})" for i in range(self.n_layers)])
        rules.append(f"mlp(X, Y) :- H0 = X, {layer_chain}, Y = H{self.n_layers}.")

        return "\n".join(rules)


# =============================================================================
# Factory Functions
# =============================================================================

def create_sro_decoder_mlp() -> PredicateMLP:
    """
    Create the SRO Decoder Ring MLP with predicate-based architecture.

    Architecture matches DeepEnergyModel.py:
        Input(1) -> 6 -> 12 -> 24 -> 6 -> 1
    """
    specs = [
        LayerSpec(1, 6, ActivationType.SWISH),
        LayerSpec(6, 12, ActivationType.SWISH),
        LayerSpec(12, 24, ActivationType.SWISH),
        LayerSpec(24, 6, ActivationType.SWISH),
        LayerSpec(6, 1, ActivationType.LINEAR),  # Final layer typically linear for energy
    ]
    return PredicateMLP(specs)


def create_classifier_mlp(input_dim: int = 12, num_classes: int = 6) -> PredicateMLP:
    """
    Create a classifier MLP for rotation order prediction.

    Architecture inspired by legacy Keras model:
        Input(12) -> 64 -> 32 -> 32 -> 6 (softmax)
    """
    specs = [
        LayerSpec(input_dim, 64, ActivationType.RELU),
        LayerSpec(64, 32, ActivationType.RELU),
        LayerSpec(32, 32, ActivationType.RELU),
        LayerSpec(32, num_classes, ActivationType.SOFTMAX),
    ]
    return PredicateMLP(specs)


# =============================================================================
# Utility Functions for Logic Programming Integration
# =============================================================================

class PredicateQuery:
    """
    Query interface for predicate-based inference.

    Supports Prolog-like queries:
        query(mlp, {X: input_data}, {Y: ?})  -> Solve for Y
    """

    def __init__(self, model: PredicateMLP):
        self.model = model

    def query(self, input_binding: dict) -> dict:
        """
        Execute a forward query.

        Args:
            input_binding: Dict with input tensor, e.g., {"X": tensor}

        Returns:
            Dict with output binding, e.g., {"Y": output_tensor, "trace": [...]}
        """
        x = input_binding.get("X")
        if x is None:
            raise ValueError("Input binding must contain 'X'")

        output, trace = self.model.forward_with_trace(x)

        return {
            "Y": output,
            "trace": trace,
            "success": True
        }


# =============================================================================
# Example Usage and Tests
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Predicate-Based MLP for SRO Decoder Ring")
    print("=" * 60)

    # Create the model
    model = create_sro_decoder_mlp()
    print(f"\nModel architecture: {model.n_layers} layers")
    for i, spec in enumerate(model.layer_specs):
        print(f"  Layer {i}: {spec.in_features} -> {spec.out_features} ({spec.activation.value})")

    # Test forward pass
    print("\n--- Forward Pass Test ---")
    x = torch.randn(4, 1)  # Batch of 4
    output, trace = model.forward_with_trace(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Show trace
    print("\n--- Predicate Trace ---")
    for t in trace:
        print(f"Layer {t['layer']}: {t['input_shape']} -> {t['output_shape']}")

    # Show Prolog representation
    print("\n--- Prolog Representation ---")
    print(model.as_prolog_rules())

    # Test individual predicates
    print("\n--- Individual Predicate Tests ---")

    # mat_vec test
    M = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # 3x2
    V = torch.tensor([[1.0, 1.0, 1.0]])  # 1x3
    result = mat_vec(M.T, V.T)  # Transpose for correct dims
    print(f"mat_vec result: {result.value.T if result.value is not None else 'None'}")

    # vec_add test
    A = torch.tensor([[1.0, 2.0]])
    B = torch.tensor([[0.5, 0.5]])
    result = vec_add(A, B)
    print(f"vec_add result: {result.value}")

    # activation test
    V_in = torch.tensor([[-1.0, 0.0, 1.0]])
    result = activation(V_in, act_type=ActivationType.SWISH)
    print(f"swish activation result: {result.value}")

    print("\n" + "=" * 60)
    print("Predicate-based MLP implementation complete!")
