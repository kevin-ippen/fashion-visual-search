# Databricks notebook source
# MAGIC %md
# MAGIC # 05 - User Style Features
# MAGIC
# MAGIC This notebook computes user style features from transaction history for personalized recommendations.
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - `main.fashion_demo.users` table
# MAGIC - `main.fashion_demo.transactions` table
# MAGIC - `main.fashion_demo.products` table
# MAGIC - `main.fashion_demo.product_image_embeddings` table
# MAGIC
# MAGIC **Output:**
# MAGIC - `main.fashion_demo.user_style_features` Delta table with user preferences and embeddings

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Configuration
CATALOG = "main"
SCHEMA = "fashion_demo"
USERS_TABLE = "users"
TRANSACTIONS_TABLE = "transactions"
PRODUCTS_TABLE = "products"
EMBEDDINGS_TABLE = "product_image_embeddings"
USER_FEATURES_TABLE = "user_style_features"

# Feature configuration
MIN_INTERACTIONS = 3  # Minimum interactions to compute features
EMBEDDING_AGG_METHOD = "mean"  # mean, weighted_mean

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

from pyspark.sql import functions as F, Window
from pyspark.sql.types import *
import sys

sys.path.append("/Workspace/Repos/.../fashion-visual-search/src")  # Update path

from fashion_visual_search.utils import add_table_comment, optimize_table

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

users_df = spark.table(f"{CATALOG}.{SCHEMA}.{USERS_TABLE}")
transactions_df = spark.table(f"{CATALOG}.{SCHEMA}.{TRANSACTIONS_TABLE}")
products_df = spark.table(f"{CATALOG}.{SCHEMA}.{PRODUCTS_TABLE}")
embeddings_df = spark.table(f"{CATALOG}.{SCHEMA}.{EMBEDDINGS_TABLE}")

print(f"Users: {users_df.count():,}")
print(f"Transactions: {transactions_df.count():,}")
print(f"Products: {products_df.count():,}")
print(f"Embeddings: {embeddings_df.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute Category Preferences

# COMMAND ----------

# Join transactions with products to get category info
transactions_enriched = (
    transactions_df
    .join(products_df.select("product_id", "category", "brand", "color", "price"), "product_id")
)

# COMMAND ----------

# Compute category preferences
# Weight recent interactions more and purchases highest
event_weights = F.when(F.col("event_type") == "purchase", 3.0) \
                 .when(F.col("event_type") == "add_to_cart", 2.0) \
                 .otherwise(1.0)

category_prefs = (
    transactions_enriched
    .withColumn("event_weight", event_weights)
    .groupBy("user_id", "category")
    .agg(
        F.sum("event_weight").alias("category_score")
    )
)

# Normalize to get preference distribution per user
window_spec = Window.partitionBy("user_id")

category_prefs_normalized = (
    category_prefs
    .withColumn("total_score", F.sum("category_score").over(window_spec))
    .withColumn("preference_score", F.col("category_score") / F.col("total_score"))
    .select("user_id", "category", "preference_score")
)

display(category_prefs_normalized.limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute Brand Preferences

# COMMAND ----------

brand_prefs = (
    transactions_enriched
    .withColumn("event_weight", event_weights)
    .groupBy("user_id", "brand")
    .agg(
        F.sum("event_weight").alias("brand_score")
    )
)

brand_prefs_normalized = (
    brand_prefs
    .withColumn("total_score", F.sum("brand_score").over(window_spec))
    .withColumn("preference_score", F.col("brand_score") / F.col("total_score"))
    .select("user_id", "brand", "preference_score")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute Color Preferences

# COMMAND ----------

color_prefs = (
    transactions_enriched
    .filter(F.col("color").isNotNull())
    .withColumn("event_weight", event_weights)
    .groupBy("user_id", "color")
    .agg(
        F.sum("event_weight").alias("color_score")
    )
)

color_prefs_normalized = (
    color_prefs
    .withColumn("total_score", F.sum("color_score").over(window_spec))
    .withColumn("preference_score", F.col("color_score") / F.col("total_score"))
    .select("user_id", "color", "preference_score")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute Price Range

# COMMAND ----------

# Compute preferred price range from purchase history
price_stats = (
    transactions_enriched
    .filter(F.col("event_type") == "purchase")
    .groupBy("user_id")
    .agg(
        F.min("price").alias("min_price"),
        F.max("price").alias("max_price"),
        F.avg("price").alias("avg_price"),
        F.percentile_approx("price", 0.25).alias("p25_price"),
        F.percentile_approx("price", 0.75).alias("p75_price")
    )
)

display(price_stats.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute User Embeddings
# MAGIC
# MAGIC Aggregate product embeddings from user's interaction history

# COMMAND ----------

# Get user's interacted products with embeddings
user_product_embeddings = (
    transactions_enriched
    .filter(F.col("event_type").isin(["purchase", "add_to_cart"]))  # Focus on strong signals
    .select("user_id", "product_id", "timestamp")
    .join(embeddings_df.select("product_id", "image_embedding"), "product_id")
)

print(f"User-product pairs with embeddings: {user_product_embeddings.count():,}")

# COMMAND ----------

# Define UDF to aggregate embeddings (mean pooling)
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, DoubleType
import numpy as np

@udf(returnType=ArrayType(DoubleType()))
def aggregate_embeddings(embedding_list):
    """Aggregate multiple embeddings using mean pooling."""
    if not embedding_list or len(embedding_list) == 0:
        return None

    embeddings_array = np.array(embedding_list)
    mean_embedding = np.mean(embeddings_array, axis=0)

    return mean_embedding.tolist()

# COMMAND ----------

# Aggregate embeddings per user
user_embeddings = (
    user_product_embeddings
    .groupBy("user_id")
    .agg(
        F.collect_list("image_embedding").alias("embeddings_list"),
        F.count("product_id").alias("num_interactions")
    )
    .filter(F.col("num_interactions") >= MIN_INTERACTIONS)
    .withColumn("user_embedding", aggregate_embeddings(F.col("embeddings_list")))
    .select("user_id", "user_embedding", "num_interactions")
)

print(f"Users with embeddings: {user_embeddings.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Combine All Features

# COMMAND ----------

# Convert preferences to maps
category_prefs_map = (
    category_prefs_normalized
    .groupBy("user_id")
    .agg(F.map_from_entries(F.collect_list(F.struct("category", "preference_score"))).alias("category_prefs"))
)

brand_prefs_map = (
    brand_prefs_normalized
    .groupBy("user_id")
    .agg(F.map_from_entries(F.collect_list(F.struct("brand", "preference_score"))).alias("brand_prefs"))
)

color_prefs_map = (
    color_prefs_normalized
    .groupBy("user_id")
    .agg(F.map_from_entries(F.collect_list(F.struct("color", "preference_score"))).alias("color_prefs"))
)

# COMMAND ----------

# Combine all features
user_features = (
    users_df.select("user_id", "segment")
    .join(category_prefs_map, "user_id", "left")
    .join(brand_prefs_map, "user_id", "left")
    .join(color_prefs_map, "user_id", "left")
    .join(price_stats, "user_id", "left")
    .join(user_embeddings, "user_id", "left")
    .withColumn("created_at", F.current_timestamp())
)

# COMMAND ----------

# Display sample
display(user_features.limit(10))

# COMMAND ----------

print(f"Total users with features: {user_features.count():,}")
print(f"Users with embeddings: {user_features.filter(F.col('user_embedding').isNotNull()).count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Quality Checks

# COMMAND ----------

# Check for users without any preferences
users_no_prefs = (
    user_features
    .filter(
        F.col("category_prefs").isNull() &
        F.col("brand_prefs").isNull() &
        F.col("color_prefs").isNull()
    )
    .count()
)

print(f"Users without any preferences: {users_no_prefs:,}")

# COMMAND ----------

# Verify embedding dimensions
sample_with_embedding = user_features.filter(F.col("user_embedding").isNotNull()).first()
if sample_with_embedding:
    embedding_dim = len(sample_with_embedding["user_embedding"])
    print(f"User embedding dimension: {embedding_dim}")
else:
    print("No user embeddings generated")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write to Unity Catalog

# COMMAND ----------

user_features_table = f"{CATALOG}.{SCHEMA}.{USER_FEATURES_TABLE}"

user_features.write.format("delta").mode("overwrite").saveAsTable(user_features_table)

print(f"✓ Written {user_features.count():,} user features to {user_features_table}")

# COMMAND ----------

# Add table comment
add_table_comment(
    CATALOG,
    SCHEMA,
    USER_FEATURES_TABLE,
    "User style features including preference distributions and aggregated embeddings from interaction history"
)

# COMMAND ----------

# Optimize table
optimize_table(CATALOG, SCHEMA, USER_FEATURES_TABLE, zorder_cols=["user_id"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary Statistics

# COMMAND ----------

print("=" * 60)
print("USER STYLE FEATURES SUMMARY")
print("=" * 60)

features_table = spark.table(user_features_table)

print(f"\nTotal users: {features_table.count():,}")
print(f"Users with embeddings: {features_table.filter(F.col('user_embedding').isNotNull()).count():,}")
print(f"Users with category prefs: {features_table.filter(F.col('category_prefs').isNotNull()).count():,}")
print(f"Users with brand prefs: {features_table.filter(F.col('brand_prefs').isNotNull()).count():,}")
print(f"Users with color prefs: {features_table.filter(F.col('color_prefs').isNotNull()).count():,}")
print(f"Users with price stats: {features_table.filter(F.col('avg_price').isNotNull()).count():,}")

print("\n" + "=" * 60)

# COMMAND ----------

# Show sample user profile
print("Sample user profile:")
sample_user = features_table.filter(F.col("user_embedding").isNotNull()).first()

print(f"\nUser ID: {sample_user['user_id']}")
print(f"Segment: {sample_user['segment']}")
print(f"Category preferences: {sample_user['category_prefs']}")
print(f"Brand preferences: {sample_user['brand_prefs']}")
print(f"Color preferences: {sample_user['color_prefs']}")
print(f"Price range: ${sample_user['min_price']:.2f} - ${sample_user['max_price']:.2f} (avg: ${sample_user['avg_price']:.2f})")
print(f"Embedding: {len(sample_user['user_embedding'])}-dim vector")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC 1. Run notebook `06_recommendation_scoring` to test personalized recommendations
# MAGIC 2. Consider updating user features periodically (daily/weekly) as new transactions occur
# MAGIC 3. Experiment with different embedding aggregation methods (weighted by recency, etc.)
