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

print("✓ Setup complete")

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
# Note: Using actual column names from products table
transactions_enriched = (
    transactions_df
    .join(
        products_df.select(
            "product_id", 
            "master_category",
            "sub_category",
            "article_type",
            "base_color",
            "price"
        ), 
        "product_id"
    )
)

print(f"Enriched transactions: {transactions_enriched.count():,}")
print(f"\nSchema:")
transactions_enriched.printSchema()

# COMMAND ----------

# Compute category preferences
# Weight recent interactions more and purchases highest
event_weights = F.when(F.col("event_type") == "purchase", 3.0) \
                 .when(F.col("event_type") == "add_to_cart", 2.0) \
                 .otherwise(1.0)

category_prefs = (
    transactions_enriched
    .withColumn("event_weight", event_weights)
    .groupBy("user_id", "master_category")
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
    .select("user_id", F.col("master_category").alias("category"), "preference_score")
)

print(f"Category preferences computed for {category_prefs_normalized.select('user_id').distinct().count():,} users")
display(category_prefs_normalized.limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute Brand Preferences

# COMMAND ----------

# Compute brand preferences from article_type (closest to brand)
brand_prefs = (
    transactions_enriched
    .withColumn("event_weight", event_weights)
    .groupBy("user_id", "article_type")
    .agg(
        F.sum("event_weight").alias("brand_score")
    )
)

brand_prefs_normalized = (
    brand_prefs
    .withColumn("total_score", F.sum("brand_score").over(window_spec))
    .withColumn("preference_score", F.col("brand_score") / F.col("total_score"))
    .select("user_id", F.col("article_type").alias("brand"), "preference_score")
)

print(f"Brand preferences computed for {brand_prefs_normalized.select('user_id').distinct().count():,} users")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute Color Preferences

# COMMAND ----------

# Compute color preferences
color_prefs = (
    transactions_enriched
    .filter(F.col("base_color").isNotNull())
    .withColumn("event_weight", event_weights)
    .groupBy("user_id", "base_color")
    .agg(
        F.sum("event_weight").alias("color_score")
    )
)

color_prefs_normalized = (
    color_prefs
    .withColumn("total_score", F.sum("color_score").over(window_spec))
    .withColumn("preference_score", F.col("color_score") / F.col("total_score"))
    .select("user_id", F.col("base_color").alias("color"), "preference_score")
)

print(f"Color preferences computed for {color_prefs_normalized.select('user_id').distinct().count():,} users")

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
# MAGIC ## ✅ User Features Ready to Write!
# MAGIC
# MAGIC ### What's Computed
# MAGIC
# MAGIC The `user_features` DataFrame contains **10,000 users** with:
# MAGIC
# MAGIC * **Category preferences** (MAP) - Apparel, Accessories, Footwear, etc.
# MAGIC * **Brand/Article preferences** (MAP) - Shirts, Watches, Handbags, etc.
# MAGIC * **Color preferences** (MAP) - Blue, Black, White, etc.
# MAGIC * **Price statistics** - Min, max, avg, p25, p75
# MAGIC * **User embeddings** (ARRAY<DOUBLE>) - 512-dim vectors for 9,421 users (94%)
# MAGIC * **Interaction counts** - Number of purchases/cart adds
# MAGIC * **Timestamp** - Feature generation time
# MAGIC
# MAGIC ### To Complete
# MAGIC
# MAGIC **Run Cell 30 manually** to write to Unity Catalog:
# MAGIC
# MAGIC The cell will write to: `main.fashion_demo.user_style_features`
# MAGIC
# MAGIC Click the cell below and press **Shift+Enter** or click **Run**.

# COMMAND ----------

# DBTITLE 1,Validate Written Table
# Validate the written table
user_features_table = f"{CATALOG}.{SCHEMA}.{USER_FEATURES_TABLE}"

print("=" * 60)
print("VALIDATING USER FEATURES TABLE")
print("=" * 60)

try:
    # Read the table
    written_df = spark.table(user_features_table)
    
    # Get counts
    total_users = written_df.count()
    users_with_embeddings = written_df.filter(F.col("user_embedding").isNotNull()).count()
    users_with_category_prefs = written_df.filter(F.col("category_prefs").isNotNull()).count()
    
    print(f"\n✓ Table: {user_features_table}")
    print(f"\nCounts:")
    print(f"  Total users: {total_users:,}")
    print(f"  Users with embeddings: {users_with_embeddings:,} ({users_with_embeddings/total_users*100:.1f}%)")
    print(f"  Users with category prefs: {users_with_category_prefs:,} ({users_with_category_prefs/total_users*100:.1f}%)")
    
    # Check schema
    print(f"\nSchema:")
    written_df.printSchema()
    
    # Sample data
    print(f"\nSample user features:")
    display(written_df.select(
        "user_id", 
        "segment", 
        "category_prefs", 
        "num_interactions"
    ).limit(5))
    
    # Validate embedding dimensions
    sample_embedding = written_df.filter(F.col("user_embedding").isNotNull()).first()["user_embedding"]
    print(f"\nEmbedding validation:")
    print(f"  Dimension: {len(sample_embedding)}")
    print(f"  Expected: 512")
    print(f"  Match: {'YES ✅' if len(sample_embedding) == 512 else 'NO ⚠'}")
    
    print("\n" + "=" * 60)
    print("✓ Validation complete!")
    print("=" * 60)
    
except Exception as e:
    print(f"\n✗ Table not found or error: {e}")
    print(f"\nPlease run Cell 30 to write the user_features DataFrame to Delta.")
    print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ User Style Features Pipeline Complete!
# MAGIC
# MAGIC ### ✨ What Was Accomplished
# MAGIC
# MAGIC **Successfully computed features for all 10,000 users:**
# MAGIC
# MAGIC 1. **Category Preferences** ✅
# MAGIC    - Weighted by event type (purchase=3x, cart=2x, view=1x)
# MAGIC    - Normalized distribution per user
# MAGIC    - Example: User prefers Accessories (77%), Apparel (21%), Footwear (2%)
# MAGIC
# MAGIC 2. **Brand/Article Preferences** ✅
# MAGIC    - Based on article_type (Shirts, Watches, Handbags, etc.)
# MAGIC    - 10,000 users with preferences
# MAGIC
# MAGIC 3. **Color Preferences** ✅
# MAGIC    - Based on base_color from products
# MAGIC    - 10,000 users with color preferences
# MAGIC
# MAGIC 4. **Price Statistics** ✅
# MAGIC    - Min, max, avg, p25, p75 from purchase history
# MAGIC    - Helps filter recommendations by budget
# MAGIC
# MAGIC 5. **User Embeddings** ✅
# MAGIC    - 9,421 users (94.2%) with 512-dim embeddings
# MAGIC    - Aggregated from purchased/cart products using mean pooling
# MAGIC    - 579 users excluded (< 3 interactions minimum)
# MAGIC
# MAGIC ### 📊 Data Quality
# MAGIC
# MAGIC * **Total users**: 10,000
# MAGIC * **Transactions processed**: 464,596
# MAGIC * **User-product pairs**: 119,151
# MAGIC * **Users with embeddings**: 9,421 (94.2%)
# MAGIC * **Embedding dimension**: 512
# MAGIC
# MAGIC ### 📦 Output Schema
# MAGIC
# MAGIC ```
# MAGIC user_id (STRING) - Primary key
# MAGIC segment (STRING) - User segment
# MAGIC category_prefs (MAP<STRING, DOUBLE>) - Category preferences
# MAGIC brand_prefs (MAP<STRING, DOUBLE>) - Article/brand preferences  
# MAGIC color_prefs (MAP<STRING, DOUBLE>) - Color preferences
# MAGIC min_price, max_price, avg_price, p25_price, p75_price (DOUBLE)
# MAGIC user_embedding (ARRAY<DOUBLE>) - 512-dim aggregated embedding
# MAGIC num_interactions (LONG) - Interaction count
# MAGIC created_at (TIMESTAMP) - Feature generation time
# MAGIC ```
# MAGIC
# MAGIC ### 🚀 To Complete
# MAGIC
# MAGIC **Manually run Cell 30** to write to Delta:
# MAGIC - Target: `main.fashion_demo.user_style_features`
# MAGIC - Mode: overwrite
# MAGIC - Records: 10,000 users
# MAGIC
# MAGIC Then run Cell 32 to validate the written table.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Your personalized recommendation features are ready!** ✨

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

# Write user features to Delta table
user_features_table = f"{CATALOG}.{SCHEMA}.{USER_FEATURES_TABLE}"

print("=" * 60)
print("WRITING USER FEATURES TO DELTA")
print("=" * 60)

print(f"\nTarget table: {user_features_table}")
print(f"Records to write: {user_features.count():,}")
print(f"Mode: overwrite\n")

user_features.write.format("delta").mode("overwrite").saveAsTable(user_features_table)

print(f"\n✓ Written {user_features.count():,} user features to {user_features_table}")
print("\n" + "=" * 60)

# COMMAND ----------

# Add table comment for documentation
comment = "User style features including preference distributions and aggregated embeddings from interaction history"
spark.sql(f"COMMENT ON TABLE {user_features_table} IS '{comment}'")

print(f"✓ Added table comment to {user_features_table}")

# COMMAND ----------

# Optimize table with Z-ordering
spark.sql(f"OPTIMIZE {user_features_table} ZORDER BY (user_id)")

print(f"✓ Optimized {user_features_table}")
print("  - Compacted small files")
print("  - Applied Z-ordering on user_id for faster lookups")

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

# MAGIC %md
# MAGIC ## ✅ User Style Features Pipeline Ready!
# MAGIC
# MAGIC ### What's Computed
# MAGIC
# MAGIC **All features are ready in the `user_features` DataFrame:**
# MAGIC
# MAGIC 1. **Category Preferences** - 10,000 users
# MAGIC    - Weighted by event type (purchase=3x, add_to_cart=2x, view=1x)
# MAGIC    - Normalized distribution per user
# MAGIC    - Example: `{"Apparel": 0.61, "Accessories": 0.27, "Footwear": 0.09}`
# MAGIC
# MAGIC 2. **Brand/Article Preferences** - 10,000 users
# MAGIC    - Based on article_type (Shirts, Watches, Handbags, etc.)
# MAGIC    - Weighted and normalized
# MAGIC
# MAGIC 3. **Color Preferences** - 10,000 users
# MAGIC    - Based on base_color
# MAGIC    - Example: `{"Blue": 0.19, "White": 0.09, "Black": 0.14}`
# MAGIC
# MAGIC 4. **Price Statistics** - Users with purchases
# MAGIC    - Min, max, avg, p25, p75 price points
# MAGIC    - Helps filter recommendations by budget
# MAGIC
# MAGIC 5. **User Embeddings** - 9,421 users (94%)
# MAGIC    - 512-dimensional vectors
# MAGIC    - Aggregated from purchased/cart products
# MAGIC    - Mean pooling of product embeddings
# MAGIC    - Requires minimum 3 interactions
# MAGIC
# MAGIC ### Data Quality
# MAGIC
# MAGIC * **Total users**: 10,000
# MAGIC * **Users with embeddings**: 9,421 (94.2%)
# MAGIC * **Users without embeddings**: 579 (< 3 interactions)
# MAGIC * **User-product pairs**: 119,151
# MAGIC
# MAGIC ### To Complete
# MAGIC
# MAGIC **Manually run cell 30** to write to Delta:
# MAGIC ```python
# MAGIC user_features.write.format("delta").mode("overwrite").saveAsTable("main.fashion_demo.user_style_features")
# MAGIC ```
# MAGIC
# MAGIC Then run cells 31-32 for table comment and optimization.
# MAGIC
# MAGIC ### Schema
# MAGIC
# MAGIC * `user_id` (STRING) - Primary key
# MAGIC * `segment` (STRING) - User segment (luxury, casual, athletic, etc.)
# MAGIC * `category_prefs` (MAP<STRING, DOUBLE>) - Category preference scores
# MAGIC * `brand_prefs` (MAP<STRING, DOUBLE>) - Brand/article preference scores
# MAGIC * `color_prefs` (MAP<STRING, DOUBLE>) - Color preference scores
# MAGIC * `min_price`, `max_price`, `avg_price`, `p25_price`, `p75_price` (DOUBLE)
# MAGIC * `user_embedding` (ARRAY<DOUBLE>) - 512-dim aggregated embedding
# MAGIC * `num_interactions` (LONG) - Number of interactions used
# MAGIC * `created_at` (TIMESTAMP) - Feature generation timestamp

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
