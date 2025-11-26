"""Tests for embedding utilities."""

import pytest
import numpy as np
from fashion_visual_search.embeddings import (
    compute_cosine_similarity,
    aggregate_embeddings
)


class TestCosineSimilarity:
    """Test cosine similarity computation."""

    def test_identical_vectors(self):
        """Test that identical vectors have similarity of 1.0."""
        vec = np.array([1.0, 2.0, 3.0])
        similarity = compute_cosine_similarity(vec, vec)
        assert abs(similarity - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        """Test that orthogonal vectors have similarity of 0.0."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        similarity = compute_cosine_similarity(vec1, vec2)
        assert abs(similarity - 0.0) < 1e-6

    def test_opposite_vectors(self):
        """Test that opposite vectors have similarity of -1.0."""
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([-1.0, -2.0, -3.0])
        similarity = compute_cosine_similarity(vec1, vec2)
        assert abs(similarity - (-1.0)) < 1e-6

    def test_zero_vector(self):
        """Test that zero vectors return 0.0 similarity."""
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([0.0, 0.0, 0.0])
        similarity = compute_cosine_similarity(vec1, vec2)
        assert similarity == 0.0


class TestAggregateEmbeddings:
    """Test embedding aggregation."""

    def test_mean_aggregation(self):
        """Test mean aggregation of embeddings."""
        embeddings = [
            np.array([1.0, 2.0, 3.0]),
            np.array([2.0, 4.0, 6.0]),
            np.array([3.0, 6.0, 9.0])
        ]
        result = aggregate_embeddings(embeddings, method="mean")
        expected = np.array([2.0, 4.0, 6.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_max_aggregation(self):
        """Test max aggregation of embeddings."""
        embeddings = [
            np.array([1.0, 5.0, 3.0]),
            np.array([4.0, 2.0, 6.0]),
            np.array([2.0, 3.0, 1.0])
        ]
        result = aggregate_embeddings(embeddings, method="max")
        expected = np.array([4.0, 5.0, 6.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_empty_list(self):
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError):
            aggregate_embeddings([])

    def test_invalid_method(self):
        """Test that invalid method raises ValueError."""
        embeddings = [np.array([1.0, 2.0])]
        with pytest.raises(ValueError):
            aggregate_embeddings(embeddings, method="invalid")
