"""Tests for recommendation scoring."""

import pytest
import numpy as np
from fashion_visual_search.recommendation import (
    RecommendationScorer,
    ProductCandidate,
    UserProfile,
    ScoringWeights,
    diversify_recommendations
)


class TestScoringWeights:
    """Test scoring weights configuration."""

    def test_default_weights(self):
        """Test default weight values."""
        weights = ScoringWeights()
        assert weights.visual == 0.5
        assert weights.user == 0.3
        assert weights.attribute == 0.2

    def test_normalize_weights(self):
        """Test weight normalization."""
        weights = ScoringWeights(visual=2.0, user=1.0, attribute=1.0)
        normalized = weights.normalize()

        assert abs(normalized.visual - 0.5) < 1e-6
        assert abs(normalized.user - 0.25) < 1e-6
        assert abs(normalized.attribute - 0.25) < 1e-6
        assert abs(normalized.visual + normalized.user + normalized.attribute - 1.0) < 1e-6


class TestRecommendationScorer:
    """Test recommendation scoring functionality."""

    @pytest.fixture
    def scorer(self):
        """Create a scorer instance."""
        return RecommendationScorer()

    @pytest.fixture
    def sample_product(self):
        """Create a sample product."""
        return ProductCandidate(
            product_id="prod_001",
            image_embedding=np.random.randn(512),
            category="Topwear",
            brand="TestBrand",
            color="Blue",
            price=50.0
        )

    @pytest.fixture
    def sample_user_profile(self):
        """Create a sample user profile."""
        return UserProfile(
            user_id="user_001",
            user_embedding=np.random.randn(512),
            category_prefs={"Topwear": 0.7, "Bottomwear": 0.3},
            brand_prefs={"TestBrand": 0.8, "OtherBrand": 0.2},
            color_prefs={"Blue": 0.6, "Red": 0.4},
            price_range=(30.0, 70.0)
        )

    def test_compute_visual_similarity(self, scorer, sample_product):
        """Test visual similarity computation."""
        query_embedding = sample_product.image_embedding.copy()
        similarity = scorer.compute_visual_similarity(query_embedding, sample_product)

        assert 0.99 <= similarity <= 1.01  # Should be ~1.0 for identical embeddings
        assert isinstance(similarity, float)

    def test_compute_user_similarity(self, scorer, sample_user_profile, sample_product):
        """Test user similarity computation."""
        similarity = scorer.compute_user_similarity(sample_user_profile, sample_product)

        assert -1.0 <= similarity <= 1.0
        assert isinstance(similarity, float)

    def test_compute_user_similarity_no_embedding(self, scorer, sample_product):
        """Test user similarity with no user embedding."""
        user_profile = UserProfile(user_id="user_001", user_embedding=None)
        similarity = scorer.compute_user_similarity(user_profile, sample_product)

        assert similarity == 0.0

    def test_compute_attribute_score(self, scorer, sample_user_profile, sample_product):
        """Test attribute-based scoring."""
        score = scorer.compute_attribute_score(sample_user_profile, sample_product)

        assert 0.0 <= score <= 1.0
        assert isinstance(score, float)

    def test_compute_attribute_score_with_budget(self, scorer, sample_user_profile):
        """Test attribute scoring with budget constraint."""
        # Product within budget
        product_affordable = ProductCandidate(
            product_id="prod_001",
            image_embedding=np.random.randn(512),
            category="Topwear",
            brand="TestBrand",
            price=40.0
        )

        score_affordable = scorer.compute_attribute_score(
            sample_user_profile, product_affordable, budget=50.0
        )

        # Product over budget
        product_expensive = ProductCandidate(
            product_id="prod_002",
            image_embedding=np.random.randn(512),
            category="Topwear",
            brand="TestBrand",
            price=60.0
        )

        score_expensive = scorer.compute_attribute_score(
            sample_user_profile, product_expensive, budget=50.0
        )

        # Hard budget constraint should penalize expensive product
        assert score_expensive < score_affordable

    def test_score_product(self, scorer, sample_product, sample_user_profile):
        """Test overall product scoring."""
        query_embedding = np.random.randn(512)

        scored = scorer.score_product(
            sample_product,
            query_embedding,
            sample_user_profile
        )

        assert scored.visual_sim >= 0.0
        assert scored.user_sim >= 0.0
        assert scored.attr_score >= 0.0
        assert 0.0 <= scored.final_score <= 1.0

    def test_rank_products(self, scorer, sample_user_profile):
        """Test product ranking."""
        # Create multiple products
        products = [
            ProductCandidate(
                product_id=f"prod_{i:03d}",
                image_embedding=np.random.randn(512),
                category="Topwear",
                brand="TestBrand",
                price=float(30 + i * 10)
            )
            for i in range(10)
        ]

        query_embedding = np.random.randn(512)

        ranked = scorer.rank_products(
            products, query_embedding, sample_user_profile, top_k=5
        )

        assert len(ranked) == 5
        # Check descending order
        for i in range(len(ranked) - 1):
            assert ranked[i].final_score >= ranked[i + 1].final_score


class TestDiversifyRecommendations:
    """Test recommendation diversification."""

    def test_diversify_by_category(self):
        """Test diversification limits products per category."""
        products = [
            ProductCandidate(
                product_id=f"prod_{i:03d}",
                image_embedding=np.random.randn(512),
                category="Topwear" if i < 7 else "Bottomwear",
                brand="TestBrand",
                final_score=1.0 - (i * 0.1)  # Descending scores
            )
            for i in range(10)
        ]

        diversified = diversify_recommendations(products, max_per_category=3)

        # Count categories
        topwear_count = sum(1 for p in diversified if p.category == "Topwear")
        bottomwear_count = sum(1 for p in diversified if p.category == "Bottomwear")

        assert topwear_count <= 3
        assert bottomwear_count <= 3

    def test_diversify_preserves_order(self):
        """Test that diversification maintains score ordering within categories."""
        products = [
            ProductCandidate(
                product_id=f"prod_{i:03d}",
                image_embedding=np.random.randn(512),
                category="Topwear",
                brand="TestBrand",
                final_score=1.0 - (i * 0.1)
            )
            for i in range(5)
        ]

        diversified = diversify_recommendations(products, max_per_category=3)

        # First 3 should be preserved (highest scores)
        assert len(diversified) == 3
        assert diversified[0].product_id == "prod_000"
        assert diversified[1].product_id == "prod_001"
        assert diversified[2].product_id == "prod_002"
