"""
Integration Tests for CNN
=========================

End-to-end tests for the CNN class.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.cnn import CNN, benchmark_inference


class TestCNNConstruction:
    """Tests for CNN construction."""

    def test_default_construction(self):
        """Test default CNN construction."""
        model = CNN(input_shape=(1, 28, 28), num_classes=10)

        assert model.input_shape == (1, 28, 28)
        assert model.num_classes == 10
        assert len(model.layers) > 0

    def test_without_batchnorm(self):
        """Test CNN without batch normalization."""
        model = CNN(input_shape=(1, 28, 28), num_classes=10, use_batchnorm=False)

        # Should have fewer layers without batchnorm
        model_bn = CNN(input_shape=(1, 28, 28), num_classes=10, use_batchnorm=True)
        assert len(model.layers) < len(model_bn.layers)


class TestCNNForward:
    """Tests for CNN forward pass."""

    def test_forward_shape(self):
        """Test output shape."""
        model = CNN(input_shape=(1, 28, 28), num_classes=10)
        x = np.random.randn(4, 1, 28, 28)

        output = model.forward(x, training=False)

        assert output.shape == (4, 10)

    def test_forward_probabilities(self):
        """Test that output sums to 1 (softmax)."""
        model = CNN(input_shape=(1, 28, 28), num_classes=10)
        x = np.random.randn(4, 1, 28, 28)

        output = model.forward(x, training=False)

        # Each sample's probabilities should sum to 1
        for i in range(4):
            assert abs(np.sum(output[i]) - 1.0) < 1e-5

    def test_predict(self):
        """Test predict method."""
        model = CNN(input_shape=(1, 28, 28), num_classes=10)
        x = np.random.randn(4, 1, 28, 28)

        probs = model.predict(x)

        assert probs.shape == (4, 10)
        assert np.all(probs >= 0) and np.all(probs <= 1)

    def test_predict_classes(self):
        """Test predict_classes method."""
        model = CNN(input_shape=(1, 28, 28), num_classes=10)
        x = np.random.randn(4, 1, 28, 28)

        classes = model.predict_classes(x)

        assert classes.shape == (4,)
        assert np.all(classes >= 0) and np.all(classes < 10)


class TestCNNTraining:
    """Tests for CNN training."""

    def test_fit_reduces_loss(self):
        """Test that training reduces loss."""
        np.random.seed(42)

        model = CNN(input_shape=(1, 8, 8), num_classes=5)

        # Generate simple synthetic data
        X = np.random.randn(50, 1, 8, 8)
        y = np.random.randint(0, 5, 50)

        # Train for a few epochs
        history = model.fit(X, y, epochs=5, batch_size=16,
                           learning_rate=0.01, verbose=False)

        # Loss should generally decrease
        assert history['loss'][-1] < history['loss'][0], \
            "Loss should decrease during training"

    def test_fit_with_validation(self):
        """Test training with validation data."""
        np.random.seed(42)

        model = CNN(input_shape=(1, 8, 8), num_classes=5)

        X_train = np.random.randn(40, 1, 8, 8)
        y_train = np.random.randint(0, 5, 40)
        X_val = np.random.randn(10, 1, 8, 8)
        y_val = np.random.randint(0, 5, 10)

        history = model.fit(X_train, y_train, epochs=3, batch_size=16,
                           validation_data=(X_val, y_val), verbose=False)

        assert 'val_loss' in history
        assert 'val_accuracy' in history
        assert len(history['val_loss']) == 3


class TestCNNEvaluation:
    """Tests for CNN evaluation."""

    def test_evaluate(self):
        """Test evaluate method."""
        model = CNN(input_shape=(1, 28, 28), num_classes=10)

        X = np.random.randn(20, 1, 28, 28)
        y = np.random.randint(0, 10, 20)

        loss, accuracy = model.evaluate(X, y)

        assert loss >= 0
        assert 0 <= accuracy <= 1

    def test_score(self):
        """Test score method."""
        model = CNN(input_shape=(1, 28, 28), num_classes=10)

        X = np.random.randn(20, 1, 28, 28)
        y = np.random.randint(0, 10, 20)

        accuracy = model.score(X, y)

        assert 0 <= accuracy <= 1


class TestCNNFeatures:
    """Tests for feature extraction."""

    def test_get_feature_maps(self):
        """Test getting feature maps."""
        model = CNN(input_shape=(1, 28, 28), num_classes=10)

        x = np.random.randn(1, 1, 28, 28)
        feature_maps = model.get_feature_maps(x)

        assert len(feature_maps) > 0

        # Check first feature map
        fm = feature_maps[0]
        assert 'layer_index' in fm
        assert 'feature_map' in fm
        assert fm['feature_map'].shape[0] == 1  # batch size

    def test_get_filters(self):
        """Test getting filter weights."""
        model = CNN(input_shape=(1, 28, 28), num_classes=10)

        filters = model.get_filters()

        assert len(filters) > 0

        # Check first filter
        f = filters[0]
        assert 'layer_index' in f
        assert 'weights' in f
        assert f['weights'].ndim == 4  # (out, in, h, w)


class TestCNNSaveLoad:
    """Tests for model saving and loading."""

    def test_save_load(self, tmp_path):
        """Test saving and loading model."""
        model = CNN(input_shape=(1, 28, 28), num_classes=10)

        # Get prediction before save
        x = np.random.randn(2, 1, 28, 28)
        pred_before = model.predict(x)

        # Save
        save_path = str(tmp_path / "model.npz")
        model.save(save_path)

        # Create new model and load
        model2 = CNN(input_shape=(1, 28, 28), num_classes=10)
        model2.load(save_path)

        # Predictions should match
        pred_after = model2.predict(x)
        np.testing.assert_array_almost_equal(pred_before, pred_after)


class TestCNNBenchmark:
    """Tests for benchmarking."""

    def test_benchmark_inference(self):
        """Test inference benchmarking."""
        model = CNN(input_shape=(1, 28, 28), num_classes=10)
        x = np.random.randn(1, 1, 28, 28)

        results = benchmark_inference(model, x, n_runs=10)

        assert 'mean_ms' in results
        assert 'std_ms' in results
        assert results['mean_ms'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
