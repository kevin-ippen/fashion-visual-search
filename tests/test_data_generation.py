"""Tests for synthetic data generation."""

import pytest
from fashion_visual_search.data_generation import (
    SyntheticDataGenerator,
    compute_user_statistics
)


class TestSyntheticDataGenerator:
    """Test synthetic data generation."""

    @pytest.fixture
    def generator(self):
        """Create a generator with fixed seed."""
        return SyntheticDataGenerator(seed=42)

    def test_generate_users(self, generator):
        """Test user generation."""
        users = generator.generate_users(num_users=100)

        assert len(users) == 100
        assert all("user_id" in u for u in users)
        assert all("segment" in u for u in users)
        assert all("created_date" in u for u in users)

    def test_user_segments(self, generator):
        """Test that users have valid segments."""
        users = generator.generate_users(num_users=100)

        segments = {u["segment"] for u in users}
        expected_segments = {
            "casual", "formal", "athletic", "trendy",
            "vintage", "minimalist", "luxury", "budget"
        }

        assert segments.issubset(expected_segments)

    def test_generate_transactions(self, generator):
        """Test transaction generation."""
        users = generator.generate_users(num_users=10)
        products = [
            {
                "product_id": f"prod_{i:03d}",
                "category": "Topwear",
                "brand": "TestBrand",
                "price": 50.0
            }
            for i in range(20)
        ]

        transactions = generator.generate_transactions(
            users=users,
            products=products,
            transactions_per_user_range=(5, 10)
        )

        assert len(transactions) >= 50  # At least 5 per user
        assert len(transactions) <= 100  # At most 10 per user

        # Check transaction structure
        for txn in transactions[:5]:
            assert "transaction_id" in txn
            assert "user_id" in txn
            assert "product_id" in txn
            assert "event_type" in txn
            assert "timestamp" in txn

    def test_transaction_event_types(self, generator):
        """Test that transactions have valid event types."""
        users = generator.generate_users(num_users=5)
        products = [{"product_id": f"prod_{i:03d}", "category": "Topwear", "price": 50.0} for i in range(10)]

        transactions = generator.generate_transactions(users, products)

        event_types = {txn["event_type"] for txn in transactions}
        expected_types = {"view", "add_to_cart", "purchase"}

        assert event_types.issubset(expected_types)

    def test_reproducibility(self):
        """Test that same seed produces same results."""
        gen1 = SyntheticDataGenerator(seed=42)
        gen2 = SyntheticDataGenerator(seed=42)

        users1 = gen1.generate_users(num_users=10)
        users2 = gen2.generate_users(num_users=10)

        assert users1[0]["user_id"] == users2[0]["user_id"]
        assert users1[0]["segment"] == users2[0]["segment"]


class TestComputeUserStatistics:
    """Test user statistics computation."""

    def test_compute_statistics(self):
        """Test statistics computation from transactions."""
        transactions = [
            {"event_type": "view", "purchase_amount": 0, "quantity": 1},
            {"event_type": "view", "purchase_amount": 0, "quantity": 1},
            {"event_type": "add_to_cart", "purchase_amount": 0, "quantity": 1},
            {"event_type": "purchase", "purchase_amount": 50.0, "quantity": 1},
            {"event_type": "purchase", "purchase_amount": 75.0, "quantity": 2},
        ]

        stats = compute_user_statistics(transactions)

        assert stats["total_transactions"] == 5
        assert stats["total_purchases"] == 2
        assert stats["total_revenue"] == 50.0 + (75.0 * 2)
        assert abs(stats["conversion_rate"] - 0.4) < 1e-6

    def test_empty_transactions(self):
        """Test statistics with no transactions."""
        stats = compute_user_statistics([])

        assert stats["total_transactions"] == 0
        assert stats["total_purchases"] == 0
        assert stats["total_revenue"] == 0
        assert stats["conversion_rate"] == 0
