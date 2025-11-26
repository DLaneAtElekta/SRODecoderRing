"""
Tests for Predicate-Based MLP Implementation

Run with: python -m pytest DecoderRingEBM/test_predicate_mlp.py -v
Or:       python DecoderRingEBM/test_predicate_mlp.py
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from DecoderRingEBM.PredicateMLP import (
    mat_vec,
    vec_add,
    activation,
    dense_layer,
    PredicateMLP,
    LayerSpec,
    ActivationType,
    create_sro_decoder_mlp,
    create_classifier_mlp,
    PredicateQuery,
)


class TestCorePredates:
    """Tests for the core predicate functions"""

    def test_mat_vec_forward(self):
        """Test matrix-vector multiplication predicate"""
        # Simple 2x3 matrix, batch of 2
        M = torch.tensor([[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0]])
        V_in = torch.tensor([[1.0, 1.0],
                             [2.0, 0.0]])  # batch=2, in_features=2

        result = mat_vec(M, V_in)

        assert result.success
        assert result.value is not None
        # Row 0: [1,1] @ [[1,2,3],[4,5,6]] = [5, 7, 9]
        # Row 1: [2,0] @ [[1,2,3],[4,5,6]] = [2, 4, 6]
        expected = torch.tensor([[5.0, 7.0, 9.0],
                                 [2.0, 4.0, 6.0]])
        assert torch.allclose(result.value, expected)

    def test_vec_add_forward(self):
        """Test vector addition predicate"""
        A = torch.tensor([[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0]])
        B = torch.tensor([[0.5, 0.5, 0.5]])

        result = vec_add(A, B)

        assert result.success
        expected = torch.tensor([[1.5, 2.5, 3.5],
                                 [4.5, 5.5, 6.5]])
        assert torch.allclose(result.value, expected)

    def test_activation_relu(self):
        """Test ReLU activation predicate"""
        V_in = torch.tensor([[-1.0, 0.0, 1.0, 2.0]])

        result = activation(V_in, act_type=ActivationType.RELU)

        assert result.success
        expected = torch.tensor([[0.0, 0.0, 1.0, 2.0]])
        assert torch.allclose(result.value, expected)

    def test_activation_swish(self):
        """Test Swish activation predicate"""
        V_in = torch.tensor([[0.0, 1.0, -1.0]])

        result = activation(V_in, act_type=ActivationType.SWISH)

        assert result.success
        # swish(x) = x * sigmoid(x)
        # swish(0) = 0, swish(1) ≈ 0.731, swish(-1) ≈ -0.269
        assert torch.allclose(result.value[:, 0], torch.tensor([0.0]), atol=1e-5)
        assert result.value[0, 1] > 0.7 and result.value[0, 1] < 0.8
        assert result.value[0, 2] > -0.3 and result.value[0, 2] < -0.2

    def test_activation_softmax(self):
        """Test Softmax activation predicate"""
        V_in = torch.tensor([[1.0, 2.0, 3.0]])

        result = activation(V_in, act_type=ActivationType.SOFTMAX)

        assert result.success
        # Softmax outputs should sum to 1
        assert torch.allclose(result.value.sum(dim=-1), torch.tensor([1.0]))
        # Values should be in increasing order
        assert result.value[0, 0] < result.value[0, 1] < result.value[0, 2]


class TestDenseLayerPredicate:
    """Tests for the compound dense layer predicate"""

    def test_dense_layer_forward(self):
        """Test dense layer predicate"""
        W = torch.tensor([[1.0, 0.0],
                          [0.0, 1.0]])  # Identity-like
        B = torch.tensor([[0.5, 0.5]])
        V_in = torch.tensor([[1.0, 2.0]])

        result = dense_layer(W, B, V_in, act_type=ActivationType.LINEAR)

        assert result.success
        expected = torch.tensor([[1.5, 2.5]])
        assert torch.allclose(result.value, expected)

    def test_dense_layer_with_swish(self):
        """Test dense layer with Swish activation"""
        W = torch.eye(2)
        B = torch.zeros(1, 2)
        V_in = torch.tensor([[0.0, 1.0]])

        result = dense_layer(W, B, V_in, act_type=ActivationType.SWISH)

        assert result.success
        # Check bindings contain intermediate values
        assert "Z" in result.binding
        assert "A" in result.binding
        assert "V_out" in result.binding


class TestPredicateMLP:
    """Tests for the full PredicateMLP model"""

    def test_model_creation(self):
        """Test creating a PredicateMLP"""
        specs = [
            LayerSpec(4, 8, ActivationType.RELU),
            LayerSpec(8, 2, ActivationType.SOFTMAX),
        ]
        model = PredicateMLP(specs)

        assert model.n_layers == 2
        assert len(model.weights) == 2
        assert len(model.biases) == 2

    def test_model_forward(self):
        """Test forward pass through model"""
        specs = [
            LayerSpec(3, 4, ActivationType.RELU),
            LayerSpec(4, 2, ActivationType.LINEAR),
        ]
        model = PredicateMLP(specs)

        x = torch.randn(5, 3)  # batch=5
        y = model(x)

        assert y.shape == (5, 2)

    def test_model_with_trace(self):
        """Test forward pass with trace"""
        specs = [
            LayerSpec(2, 4, ActivationType.SWISH),
            LayerSpec(4, 3, ActivationType.LINEAR),
        ]
        model = PredicateMLP(specs)

        x = torch.randn(3, 2)
        y, trace = model.forward_with_trace(x)

        assert y.shape == (3, 3)
        assert len(trace) == 2
        assert trace[0]["layer"] == 0
        assert trace[1]["layer"] == 1

    def test_sro_decoder_mlp(self):
        """Test creating the SRO decoder model"""
        model = create_sro_decoder_mlp()

        assert model.n_layers == 5
        # Architecture: 1 -> 6 -> 12 -> 24 -> 6 -> 1
        assert model.layer_specs[0].in_features == 1
        assert model.layer_specs[0].out_features == 6
        assert model.layer_specs[-1].out_features == 1

        x = torch.randn(10, 1)
        y = model(x)
        assert y.shape == (10, 1)

    def test_classifier_mlp(self):
        """Test creating the classifier model"""
        model = create_classifier_mlp()

        assert model.n_layers == 4
        x = torch.randn(5, 12)
        y = model(x)
        assert y.shape == (5, 6)
        # Softmax output should sum to 1
        assert torch.allclose(y.sum(dim=-1), torch.ones(5), atol=1e-5)


class TestPrologExport:
    """Tests for Prolog representation export"""

    def test_prolog_rules(self):
        """Test generating Prolog rules"""
        model = create_sro_decoder_mlp()
        prolog = model.as_prolog_rules()

        assert "mat_vec" in prolog
        assert "vec_add" in prolog
        assert "swish" in prolog
        assert "layer(0, V_in, V_out)" in prolog
        assert "mlp(X, Y)" in prolog


class TestPredicateQuery:
    """Tests for the query interface"""

    def test_query_forward(self):
        """Test query interface"""
        model = create_sro_decoder_mlp()
        query = PredicateQuery(model)

        x = torch.randn(4, 1)
        result = query.query({"X": x})

        assert result["success"]
        assert result["Y"].shape == (4, 1)
        assert len(result["trace"]) == 5


class TestGradients:
    """Tests that gradients flow correctly"""

    def test_gradient_flow(self):
        """Test that gradients propagate through predicates"""
        model = create_sro_decoder_mlp()

        x = torch.randn(4, 1)
        target = torch.randn(4, 1)

        y = model(x)
        loss = (y - target).pow(2).mean()
        loss.backward()

        # Check gradients exist
        for i, w in enumerate(model.weights):
            assert w.grad is not None, f"Weight {i} has no gradient"
            assert not torch.all(w.grad == 0), f"Weight {i} has zero gradient"


def run_tests():
    """Run all tests manually"""
    print("=" * 60)
    print("Running Predicate MLP Tests")
    print("=" * 60)

    test_classes = [
        TestCorePredates(),
        TestDenseLayerPredicate(),
        TestPredicateMLP(),
        TestPrologExport(),
        TestPredicateQuery(),
        TestGradients(),
    ]

    total = 0
    passed = 0

    for test_class in test_classes:
        class_name = test_class.__class__.__name__
        print(f"\n{class_name}:")

        for method_name in dir(test_class):
            if method_name.startswith("test_"):
                total += 1
                try:
                    getattr(test_class, method_name)()
                    print(f"  ✓ {method_name}")
                    passed += 1
                except Exception as e:
                    print(f"  ✗ {method_name}: {e}")

    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60)

    return passed == total


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
