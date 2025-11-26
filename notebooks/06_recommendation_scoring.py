# Databricks notebook source
# MAGIC %md
# MAGIC # 06 - Recommendation Scoring
# MAGIC
# MAGIC This notebook implements and tests the recommendation scoring function that combines:
# MAGIC - Visual similarity (from Vector Search)
# MAGIC - User embedding similarity
# MAGIC - Attribute-based scoring (category, brand, color, price preferences)
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - Vector Search index created
# MAGIC - `main.fashion_demo.user_style_features` table
# MAGIC - All previous notebooks completed
# MAGIC
# MAGIC **Output:**
# MAGIC - Recommendation scoring function registered as UC function (optional)
# MAGIC - Test results and evaluation metrics

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Configuration
CATALOG = "main"
SCHEMA = "fashion_demo"
PRODUCTS_TABLE = "products"
EMBEDDINGS_TABLE = "product_image_embeddings"
USER_FEATURES_TABLE = "user_style_features"

# Vector Search
VECTOR_SEARCH_ENDPOINT = "fashion_vector_search"
INDEX_NAME = f"{CATALOG}.{SCHEMA}.product_embeddings_index"

# Scoring weights
DEFAULT_WEIGHTS = {
    "visual": 0.5,
    "user": 0.3,
    "attribute": 0.2
}

# Test parameters
TEST_USER_ID = None  # Will select a random user
TEST_PRODUCT_ID = None  # Will select a random product
TOP_K = 20

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from pyspark.sql import functions as F
import sys
import numpy as np

sys.path.append("/Workspace/Repos/.../fashion-visual-search/src")  # Update path

from fashion_visual_search.recommendation import (
    RecommendationScorer,
    ProductCandidate,
    UserProfile,
    ScoringWeights,
    diversify_recommendations
)

# COMMAND ----------

# Initialize Vector Search client
vsc = VectorSearchClient()
vs_index = vsc.get_index(
    endpoint_name=VECTOR_SEARCH_ENDPOINT,
    index_name=INDEX_NAME
)

print(f"Connected to Vector Search index: {INDEX_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

products_df = spark.table(f"{CATALOG}.{SCHEMA}.{PRODUCTS_TABLE}")
embeddings_df = spark.table(f"{CATALOG}.{SCHEMA}.{EMBEDDINGS_TABLE}")
user_features_df = spark.table(f"{CATALOG}.{SCHEMA}.{USER_FEATURES_TABLE}")

print(f"Products: {products_df.count():,}")
print(f"Embeddings: {embeddings_df.count():,}")
print(f"User features: {user_features_df.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helper Functions

# COMMAND ----------

def get_test_user():
    """Get a random user with features for testing."""
    test_user = (
        user_features_df
        .filter(F.col("user_embedding").isNotNull())
        .orderBy(F.rand())
        .limit(1)
        .first()
    )
    return test_user

def get_test_product():
    """Get a random product with embedding for testing."""
    test_product = (
        products_df
        .join(embeddings_df, "product_id")
        .orderBy(F.rand())
        .limit(1)
        .first()
    )
    return test_product

def create_user_profile(user_row):
    """Convert user features row to UserProfile object."""
    return UserProfile(
        user_id=user_row["user_id"],
        user_embedding=np.array(user_row["user_embedding"]) if user_row["user_embedding"] else None,
        category_prefs=user_row.get("category_prefs"),
        brand_prefs=user_row.get("brand_prefs"),
        color_prefs=user_row.get("color_prefs"),
        price_range=(user_row.get("p25_price"), user_row.get("p75_price")) if user_row.get("p25_price") else None
    )

def create_product_candidate(product_row, embedding):
    """Convert product row to ProductCandidate object."""
    return ProductCandidate(
        product_id=product_row["product_id"],
        image_embedding=np.array(embedding),
        category=product_row["category"],
        brand=product_row.get("brand", "Unknown"),
        color=product_row.get("color"),
        price=float(product_row["price"]) if product_row.get("price") else None
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test: Visual-Only Search

# COMMAND ----------

print("=" * 60)
print("TEST 1: Visual-Only Similarity Search")
print("=" * 60)

# Get test product
test_product = get_test_product()
print(f"\nQuery Product: {test_product['display_name']}")
print(f"  Category: {test_product['category']}")
print(f"  Color: {test_product['color']}")
print(f"  Price: ${test_product['price']:.2f}")

# COMMAND ----------

# Perform visual search
query_embedding = test_product["image_embedding"]

vs_results = vs_index.similarity_search(
    query_vector=query_embedding,
    columns=["product_id"],
    num_results=TOP_K
)

# COMMAND ----------

# Display results
if vs_results and "result" in vs_results:
    result_ids = [r["product_id"] for r in vs_results["result"]["data_array"]]
    result_scores = {r["product_id"]: r["score"] for r in vs_results["result"]["data_array"]}

    visual_results = (
        products_df
        .filter(F.col("product_id").isin(result_ids))
        .withColumn("visual_similarity", F.lit(None).cast("double"))
    )

    # Add scores
    for pid, score in result_scores.items():
        visual_results = visual_results.withColumn(
            "visual_similarity",
            F.when(F.col("product_id") == pid, score).otherwise(F.col("visual_similarity"))
        )

    visual_results = visual_results.orderBy(F.desc("visual_similarity"))

    print("\nTop 10 Visually Similar Products:")
    display(visual_results.select(
        "product_id", "display_name", "category", "color", "price", "visual_similarity"
    ).limit(10))
else:
    print("No results from visual search")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test: Personalized Recommendations

# COMMAND ----------

print("=" * 60)
print("TEST 2: Personalized Recommendations")
print("=" * 60)

# Get test user
test_user = get_test_user()
print(f"\nTest User: {test_user['user_id']}")
print(f"  Segment: {test_user['segment']}")
print(f"  Category prefs: {test_user.get('category_prefs', {})}")
print(f"  Price range: ${test_user.get('min_price', 0):.2f} - ${test_user.get('max_price', 0):.2f}")

# COMMAND ----------

# Create user profile
user_profile = create_user_profile(test_user)

# COMMAND ----------

# Get visual search candidates
vs_results = vs_index.similarity_search(
    query_vector=query_embedding,
    columns=["product_id"],
    num_results=50  # Get more candidates for re-ranking
)

candidate_ids = [r["product_id"] for r in vs_results["result"]["data_array"]]

# COMMAND ----------

# Load candidate products with embeddings
candidates_df = (
    products_df
    .filter(F.col("product_id").isin(candidate_ids))
    .join(embeddings_df, "product_id")
    .collect()
)

print(f"Loaded {len(candidates_df)} candidates for scoring")

# COMMAND ----------

# Create ProductCandidate objects
product_candidates = [
    create_product_candidate(row, row["image_embedding"])
    for row in candidates_df
]

# COMMAND ----------

# Initialize scorer with default weights
scorer = RecommendationScorer(weights=ScoringWeights(**DEFAULT_WEIGHTS))

# Score and rank candidates
ranked_products = scorer.rank_products(
    products=product_candidates,
    query_embedding=query_embedding,
    user_profile=user_profile,
    budget=None,  # No budget constraint
    top_k=TOP_K
)

print(f"\nRanked {len(ranked_products)} products")

# COMMAND ----------

# Display ranked results
results_data = []
for p in ranked_products[:10]:
    results_data.append({
        "product_id": p.product_id,
        "category": p.category,
        "brand": p.brand,
        "color": p.color,
        "price": p.price,
        "visual_sim": f"{p.visual_sim:.3f}",
        "user_sim": f"{p.user_sim:.3f}",
        "attr_score": f"{p.attr_score:.3f}",
        "final_score": f"{p.final_score:.3f}"
    })

results_df = spark.createDataFrame(results_data)

print("\nTop 10 Personalized Recommendations:")
display(
    results_df.join(
        products_df.select("product_id", "display_name"),
        "product_id"
    ).select(
        "product_id", "display_name", "category", "color", "price",
        "visual_sim", "user_sim", "attr_score", "final_score"
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test: Diversified Recommendations

# COMMAND ----------

print("=" * 60)
print("TEST 3: Diversified Recommendations")
print("=" * 60)

# Apply diversification
diversified_products = diversify_recommendations(
    ranked_products,
    max_per_category=3
)

print(f"\nDiversified to {len(diversified_products)} products")

# COMMAND ----------

# Display diversified results
div_results_data = []
for p in diversified_products[:10]:
    div_results_data.append({
        "product_id": p.product_id,
        "category": p.category,
        "final_score": f"{p.final_score:.3f}"
    })

div_results_df = spark.createDataFrame(div_results_data)

print("\nTop 10 Diversified Recommendations:")
display(
    div_results_df.join(
        products_df.select("product_id", "display_name"),
        "product_id"
    ).select("product_id", "display_name", "category", "final_score")
)

# COMMAND ----------

# Show category distribution
print("Category distribution in diversified results:")
div_df = spark.createDataFrame([{"product_id": p.product_id, "category": p.category} for p in diversified_products])
div_df.groupBy("category").count().orderBy(F.desc("count")).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test: Budget-Constrained Recommendations

# COMMAND ----------

print("=" * 60)
print("TEST 4: Budget-Constrained Recommendations")
print("=" * 60)

# Set budget constraint
BUDGET = 50.0
print(f"\nBudget: ${BUDGET:.2f}")

# COMMAND ----------

# Rank with budget constraint
budget_ranked = scorer.rank_products(
    products=product_candidates,
    query_embedding=query_embedding,
    user_profile=user_profile,
    budget=BUDGET,
    top_k=TOP_K
)

# COMMAND ----------

# Display budget-constrained results
budget_results_data = []
for p in budget_ranked[:10]:
    budget_results_data.append({
        "product_id": p.product_id,
        "price": p.price,
        "final_score": f"{p.final_score:.3f}",
        "within_budget": p.price <= BUDGET if p.price else False
    })

budget_results_df = spark.createDataFrame(budget_results_data)

print(f"\nTop 10 Recommendations within ${BUDGET:.2f} budget:")
display(
    budget_results_df.join(
        products_df.select("product_id", "display_name", "category"),
        "product_id"
    ).select("product_id", "display_name", "category", "price", "final_score", "within_budget")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compare Scoring Approaches

# COMMAND ----------

print("=" * 60)
print("COMPARISON: Different Scoring Weights")
print("=" * 60)

# Test different weight configurations
weight_configs = [
    {"name": "Visual Only", "weights": ScoringWeights(visual=1.0, user=0.0, attribute=0.0)},
    {"name": "Balanced", "weights": ScoringWeights(visual=0.5, user=0.3, attribute=0.2)},
    {"name": "User-Focused", "weights": ScoringWeights(visual=0.3, user=0.5, attribute=0.2)},
    {"name": "Attribute-Heavy", "weights": ScoringWeights(visual=0.3, user=0.3, attribute=0.4)}
]

# COMMAND ----------

comparison_results = []

for config in weight_configs:
    scorer = RecommendationScorer(weights=config["weights"])
    ranked = scorer.rank_products(
        products=product_candidates[:20],  # Use subset for speed
        query_embedding=query_embedding,
        user_profile=user_profile,
        top_k=5
    )

    for rank, p in enumerate(ranked, 1):
        comparison_results.append({
            "config": config["name"],
            "rank": rank,
            "product_id": p.product_id,
            "category": p.category,
            "final_score": f"{p.final_score:.3f}"
        })

# COMMAND ----------

# Display comparison
comparison_df = spark.createDataFrame(comparison_results).join(
    products_df.select("product_id", "display_name"),
    "product_id"
)

print("\nRecommendations with Different Scoring Weights:")
display(
    comparison_df.select("config", "rank", "display_name", "category", "final_score")
    .orderBy("config", "rank")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Performance Metrics

# COMMAND ----------

print("=" * 60)
print("PERFORMANCE METRICS")
print("=" * 60)

# Measure scoring time
import time

start = time.time()
test_ranked = scorer.rank_products(
    products=product_candidates,
    query_embedding=query_embedding,
    user_profile=user_profile,
    top_k=TOP_K
)
end = time.time()

print(f"\nScoring Performance:")
print(f"  Candidates scored: {len(product_candidates)}")
print(f"  Time taken: {(end - start) * 1000:.2f}ms")
print(f"  Avg time per product: {((end - start) / len(product_candidates)) * 1000:.2f}ms")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Scoring Configuration

# COMMAND ----------

# Save default scoring weights to config table
from fashion_visual_search.utils import save_config_to_table

save_config_to_table(
    config_key="scoring_weights",
    config_value=DEFAULT_WEIGHTS,
    config_table=f"{CATALOG}.{SCHEMA}.config"
)

print("✓ Saved default scoring weights to config table")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print("=" * 60)
print("RECOMMENDATION SCORING SUMMARY")
print("=" * 60)
print("\n✓ Visual-only search tested")
print("✓ Personalized recommendations tested")
print("✓ Diversification tested")
print("✓ Budget constraints tested")
print("✓ Multiple weight configurations compared")
print("\nScoring components working:")
print("  - Visual similarity (Vector Search)")
print("  - User embedding similarity")
print("  - Attribute-based scoring (category, brand, color, price)")
print("\n" + "=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC 1. Run notebook `07_claude_stylist_agent` to implement the AI agent
# MAGIC 2. Run notebook `08_app_ui` to build the Streamlit interface
# MAGIC 3. Consider implementing:
# MAGIC    - A/B testing framework for different scoring weights
# MAGIC    - Online evaluation metrics (click-through rate, conversion)
# MAGIC    - Model monitoring for embedding quality
