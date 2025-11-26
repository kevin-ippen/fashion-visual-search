"""
Synthetic user and transaction data generation for MVP testing.
"""

from typing import List, Dict, Any
import random
from datetime import datetime, timedelta
import uuid


class SyntheticDataGenerator:
    """Generates synthetic users and transactions for fashion recommendations."""

    def __init__(self, seed: int = 42):
        """
        Initialize the data generator.

        Args:
            seed: Random seed for reproducibility
        """
        random.seed(seed)
        self.seed = seed

        # Define style segments
        self.segments = [
            "casual", "formal", "athletic", "trendy",
            "vintage", "minimalist", "luxury", "budget"
        ]

        # Common categories in fashion
        self.categories = [
            "Topwear", "Bottomwear", "Dress", "Shoes",
            "Accessories", "Outerwear", "Innerwear"
        ]

        # Event types
        self.event_types = ["view", "add_to_cart", "purchase"]

    def generate_users(self, num_users: int = 10000) -> List[Dict[str, Any]]:
        """
        Generate synthetic user data.

        Args:
            num_users: Number of users to generate

        Returns:
            List of user dictionaries
        """
        users = []

        for i in range(num_users):
            user_id = f"user_{i:06d}"
            segment = random.choice(self.segments)

            # Create user with realistic attributes
            user = {
                "user_id": user_id,
                "segment": segment,
                "created_date": self._random_date(
                    datetime(2022, 1, 1),
                    datetime(2024, 1, 1)
                ).isoformat(),
                "preferred_categories": random.sample(
                    self.categories,
                    k=random.randint(2, 4)
                ),
                "avg_price_point": self._get_price_point_for_segment(segment)
            }

            users.append(user)

        return users

    def generate_transactions(
        self,
        users: List[Dict[str, Any]],
        products: List[Dict[str, Any]],
        transactions_per_user_range: tuple[int, int] = (5, 50)
    ) -> List[Dict[str, Any]]:
        """
        Generate synthetic transaction data based on user segments and product attributes.

        Args:
            users: List of user dictionaries
            products: List of product dictionaries
            transactions_per_user_range: Min and max transactions per user

        Returns:
            List of transaction dictionaries
        """
        transactions = []
        transaction_id = 0

        for user in users:
            num_transactions = random.randint(*transactions_per_user_range)
            user_segment = user["segment"]
            preferred_categories = user.get("preferred_categories", [])

            # Generate transactions with realistic patterns
            for _ in range(num_transactions):
                # Select product based on user preferences
                product = self._select_product_for_user(
                    products, user_segment, preferred_categories
                )

                if not product:
                    continue

                # Generate event sequence (view -> add_to_cart -> purchase)
                event_type = self._generate_event_type()
                timestamp = self._random_recent_date()

                transaction = {
                    "transaction_id": f"txn_{transaction_id:08d}",
                    "user_id": user["user_id"],
                    "product_id": product["product_id"],
                    "event_type": event_type,
                    "timestamp": timestamp.isoformat(),
                    "session_id": str(uuid.uuid4())
                }

                # Add purchase-specific fields
                if event_type == "purchase":
                    transaction["purchase_amount"] = product.get("price", 0.0)
                    transaction["quantity"] = random.randint(1, 3)

                transactions.append(transaction)
                transaction_id += 1

        return transactions

    def _select_product_for_user(
        self,
        products: List[Dict[str, Any]],
        segment: str,
        preferred_categories: List[str]
    ) -> Dict[str, Any]:
        """Select a product based on user preferences."""
        # Filter products by preferred categories (80% of the time)
        if random.random() < 0.8 and preferred_categories:
            filtered = [
                p for p in products
                if p.get("category") in preferred_categories
            ]
            if filtered:
                products = filtered

        # Further filter by price compatibility with segment (70% of the time)
        if random.random() < 0.7:
            price_range = self._get_price_range_for_segment(segment)
            filtered = [
                p for p in products
                if price_range[0] <= p.get("price", 0) <= price_range[1]
            ]
            if filtered:
                products = filtered

        return random.choice(products) if products else None

    def _generate_event_type(self) -> str:
        """Generate event type with realistic distribution."""
        rand = random.random()
        if rand < 0.6:
            return "view"
        elif rand < 0.85:
            return "add_to_cart"
        else:
            return "purchase"

    def _get_price_point_for_segment(self, segment: str) -> float:
        """Get average price point for a user segment."""
        price_points = {
            "luxury": 200.0,
            "formal": 120.0,
            "trendy": 80.0,
            "casual": 50.0,
            "athletic": 60.0,
            "minimalist": 90.0,
            "vintage": 70.0,
            "budget": 30.0
        }
        return price_points.get(segment, 60.0)

    def _get_price_range_for_segment(self, segment: str) -> tuple[float, float]:
        """Get price range for a user segment."""
        avg_price = self._get_price_point_for_segment(segment)
        # Range is roughly +/- 50% of average
        return (avg_price * 0.5, avg_price * 1.5)

    def _random_date(self, start_date: datetime, end_date: datetime) -> datetime:
        """Generate random date between start and end."""
        delta = end_date - start_date
        random_days = random.randint(0, delta.days)
        return start_date + timedelta(days=random_days)

    def _random_recent_date(self, days_back: int = 180) -> datetime:
        """Generate random recent date."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        return self._random_date(start_date, end_date)


def compute_user_statistics(transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute aggregate statistics from transaction data.

    Args:
        transactions: List of transaction dictionaries

    Returns:
        Dictionary of statistics
    """
    total_transactions = len(transactions)
    purchases = [t for t in transactions if t["event_type"] == "purchase"]
    total_purchases = len(purchases)

    total_revenue = sum(
        t.get("purchase_amount", 0) * t.get("quantity", 1)
        for t in purchases
    )

    return {
        "total_transactions": total_transactions,
        "total_purchases": total_purchases,
        "total_revenue": total_revenue,
        "conversion_rate": total_purchases / total_transactions if total_transactions > 0 else 0,
        "avg_purchase_value": total_revenue / total_purchases if total_purchases > 0 else 0
    }
