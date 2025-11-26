"""
Recommendation scoring and ranking logic combining visual similarity,
user preferences, and attribute-based heuristics.
"""

from typing import Dict, List, Optional, Any
import numpy as np
from dataclasses import dataclass


@dataclass
class ScoringWeights:
    """Configuration for recommendation scoring weights."""
    visual: float = 0.5
    user: float = 0.3
    attribute: float = 0.2

    def normalize(self) -> "ScoringWeights":
        """Normalize weights to sum to 1.0."""
        total = self.visual + self.user + self.attribute
        return ScoringWeights(
            visual=self.visual / total,
            user=self.user / total,
            attribute=self.attribute / total
        )


@dataclass
class ProductCandidate:
    """Product candidate with associated features and scores."""
    product_id: str
    image_embedding: np.ndarray
    category: str
    brand: str
    color: Optional[str] = None
    price: Optional[float] = None
    visual_sim: float = 0.0
    user_sim: float = 0.0
    attr_score: float = 0.0
    final_score: float = 0.0


@dataclass
class UserProfile:
    """User style profile with preferences and embedding."""
    user_id: str
    user_embedding: Optional[np.ndarray] = None
    category_prefs: Optional[Dict[str, float]] = None
    brand_prefs: Optional[Dict[str, float]] = None
    color_prefs: Optional[Dict[str, float]] = None
    price_range: Optional[tuple[float, float]] = None


class RecommendationScorer:
    """Scores and ranks product recommendations using multiple signals."""

    def __init__(self, weights: Optional[ScoringWeights] = None):
        """
        Initialize scorer with optional custom weights.

        Args:
            weights: ScoringWeights instance, defaults to balanced weights
        """
        self.weights = weights or ScoringWeights()
        self.weights = self.weights.normalize()

    def compute_visual_similarity(
        self,
        query_embedding: np.ndarray,
        product: ProductCandidate
    ) -> float:
        """
        Compute visual similarity between query and product.

        Args:
            query_embedding: Query image embedding
            product: Product candidate with embedding

        Returns:
            Cosine similarity score (0-1)
        """
        dot_product = np.dot(query_embedding, product.image_embedding)
        norm_query = np.linalg.norm(query_embedding)
        norm_product = np.linalg.norm(product.image_embedding)

        if norm_query == 0 or norm_product == 0:
            return 0.0

        return float(dot_product / (norm_query * norm_product))

    def compute_user_similarity(
        self,
        user_profile: UserProfile,
        product: ProductCandidate
    ) -> float:
        """
        Compute similarity between user profile and product.

        Args:
            user_profile: User's style profile
            product: Product candidate

        Returns:
            Similarity score (0-1)
        """
        if user_profile.user_embedding is None:
            return 0.0

        dot_product = np.dot(user_profile.user_embedding, product.image_embedding)
        norm_user = np.linalg.norm(user_profile.user_embedding)
        norm_product = np.linalg.norm(product.image_embedding)

        if norm_user == 0 or norm_product == 0:
            return 0.0

        return float(dot_product / (norm_user * norm_product))

    def compute_attribute_score(
        self,
        user_profile: UserProfile,
        product: ProductCandidate,
        budget: Optional[float] = None
    ) -> float:
        """
        Compute attribute-based score using user preferences.

        Args:
            user_profile: User's style profile with preferences
            product: Product candidate
            budget: Optional budget constraint

        Returns:
            Attribute score (0-1)
        """
        scores = []

        # Category match
        if user_profile.category_prefs:
            cat_score = user_profile.category_prefs.get(product.category, 0.0)
            scores.append(cat_score)

        # Brand match
        if user_profile.brand_prefs:
            brand_score = user_profile.brand_prefs.get(product.brand, 0.0)
            scores.append(brand_score)

        # Color match
        if user_profile.color_prefs and product.color:
            color_score = user_profile.color_prefs.get(product.color, 0.0)
            scores.append(color_score)

        # Price compatibility
        if product.price is not None:
            price_score = 1.0

            # Budget constraint (hard constraint)
            if budget and product.price > budget:
                price_score = 0.0
            # Preference range (soft preference)
            elif user_profile.price_range:
                min_price, max_price = user_profile.price_range
                if min_price <= product.price <= max_price:
                    price_score = 1.0
                elif product.price < min_price:
                    # Slightly prefer items in range
                    price_score = 0.7
                else:
                    # Penalize items above preferred range
                    overage_ratio = (product.price - max_price) / max_price
                    price_score = max(0.3, 1.0 - overage_ratio)

            scores.append(price_score)

        return float(np.mean(scores)) if scores else 0.5

    def score_product(
        self,
        product: ProductCandidate,
        query_embedding: np.ndarray,
        user_profile: Optional[UserProfile] = None,
        budget: Optional[float] = None
    ) -> ProductCandidate:
        """
        Compute final score for a product candidate.

        Args:
            product: Product to score
            query_embedding: Query image embedding
            user_profile: Optional user profile for personalization
            budget: Optional budget constraint

        Returns:
            Product with computed scores
        """
        # Visual similarity
        product.visual_sim = self.compute_visual_similarity(query_embedding, product)

        # User similarity (if profile provided)
        if user_profile:
            product.user_sim = self.compute_user_similarity(user_profile, product)
            product.attr_score = self.compute_attribute_score(user_profile, product, budget)
        else:
            product.user_sim = 0.0
            product.attr_score = 0.5  # Neutral score when no user context

        # Final weighted score
        product.final_score = (
            self.weights.visual * product.visual_sim +
            self.weights.user * product.user_sim +
            self.weights.attribute * product.attr_score
        )

        return product

    def rank_products(
        self,
        products: List[ProductCandidate],
        query_embedding: np.ndarray,
        user_profile: Optional[UserProfile] = None,
        budget: Optional[float] = None,
        top_k: Optional[int] = None
    ) -> List[ProductCandidate]:
        """
        Score and rank a list of product candidates.

        Args:
            products: List of product candidates
            query_embedding: Query image embedding
            user_profile: Optional user profile
            budget: Optional budget constraint
            top_k: Optional limit on number of results

        Returns:
            Ranked list of products with scores
        """
        # Score all products
        scored_products = [
            self.score_product(p, query_embedding, user_profile, budget)
            for p in products
        ]

        # Sort by final score (descending)
        ranked = sorted(scored_products, key=lambda x: x.final_score, reverse=True)

        # Apply top_k limit
        if top_k:
            ranked = ranked[:top_k]

        return ranked


def diversify_recommendations(
    ranked_products: List[ProductCandidate],
    max_per_category: int = 3
) -> List[ProductCandidate]:
    """
    Apply diversity constraints to avoid showing too many similar items.

    Args:
        ranked_products: Pre-ranked list of products
        max_per_category: Maximum products per category to include

    Returns:
        Diversified list of products
    """
    category_counts: Dict[str, int] = {}
    diversified = []

    for product in ranked_products:
        category = product.category
        count = category_counts.get(category, 0)

        if count < max_per_category:
            diversified.append(product)
            category_counts[category] = count + 1

    return diversified
