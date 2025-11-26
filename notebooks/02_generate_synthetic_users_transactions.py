# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Generate Synthetic Users and Transactions
# MAGIC
# MAGIC This notebook generates synthetic user and transaction data for testing the recommendation system.
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - `main.fashion_demo.products` table exists
# MAGIC
# MAGIC **Output:**
# MAGIC - `main.fashion_demo.users` Delta table
# MAGIC - `main.fashion_demo.transactions` Delta table

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Configuration
CATALOG = "main"
SCHEMA = "fashion_demo"
PRODUCTS_TABLE = "products"
USERS_TABLE = "users"
TRANSACTIONS_TABLE = "transactions"

# Synthetic data parameters
NUM_USERS = 10000
TRANSACTIONS_PER_USER_MIN = 5
TRANSACTIONS_PER_USER_MAX = 50

RANDOM_SEED = 42

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import *
import sys

# Add src to path
sys.path.append("/Workspace/Repos/.../fashion-visual-search/src")  # Update path as needed

from fashion_visual_search.data_generation import SyntheticDataGenerator, compute_user_statistics
from fashion_visual_search.utils import add_table_comment

# COMMAND ----------

# Initialize data generator
generator = SyntheticDataGenerator(seed=RANDOM_SEED)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Users

# COMMAND ----------

# Generate synthetic users
users_list = generator.generate_users(num_users=NUM_USERS)

print(f"Generated {len(users_list)} synthetic users")

# COMMAND ----------

# Convert to DataFrame
users_df = spark.createDataFrame(users_list)

# Display sample
display(users_df.limit(10))

# COMMAND ----------

# Show segment distribution
print("User segment distribution:")
users_df.groupBy("segment").count().orderBy(F.desc("count")).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Transactions

# COMMAND ----------

# Load products for transaction generation
products_table = spark.table(f"{CATALOG}.{SCHEMA}.{PRODUCTS_TABLE}")
products_list = [row.asDict() for row in products_table.collect()]

print(f"Loaded {len(products_list)} products for transaction generation")

# COMMAND ----------

# Generate synthetic transactions
# Note: This may take a few minutes for 10k users
print("Generating synthetic transactions...")

transactions_list = generator.generate_transactions(
    users=users_list,
    products=products_list,
    transactions_per_user_range=(TRANSACTIONS_PER_USER_MIN, TRANSACTIONS_PER_USER_MAX)
)

print(f"Generated {len(transactions_list)} transactions")

# COMMAND ----------

# Compute statistics
stats = compute_user_statistics(transactions_list)

print("\nTransaction Statistics:")
print(f"  Total transactions: {stats['total_transactions']:,}")
print(f"  Total purchases: {stats['total_purchases']:,}")
print(f"  Total revenue: ${stats['total_revenue']:,.2f}")
print(f"  Conversion rate: {stats['conversion_rate']:.2%}")
print(f"  Avg purchase value: ${stats['avg_purchase_value']:.2f}")

# COMMAND ----------

# Convert to DataFrame
transactions_df = spark.createDataFrame(transactions_list)

# COMMAND ----------

# Display sample transactions
display(transactions_df.limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Quality Validation

# COMMAND ----------

# Verify all users have transactions
users_with_txns = transactions_df.select("user_id").distinct().count()
print(f"Users with transactions: {users_with_txns} / {NUM_USERS}")

# Verify transaction distribution
print("\nEvent type distribution:")
transactions_df.groupBy("event_type").count().orderBy(F.desc("count")).show()

# COMMAND ----------

# Verify products referenced in transactions exist
product_ids_in_txns = set([row["product_id"] for row in transactions_df.select("product_id").distinct().collect()])
product_ids_in_catalog = set([row["product_id"] for row in products_table.select("product_id").collect()])

orphaned_products = product_ids_in_txns - product_ids_in_catalog

if orphaned_products:
    print(f"WARNING: {len(orphaned_products)} product IDs in transactions not found in catalog")
else:
    print("✓ All transaction product IDs exist in catalog")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write to Unity Catalog

# COMMAND ----------

# Write users table
users_table_name = f"{CATALOG}.{SCHEMA}.{USERS_TABLE}"
users_df.write.format("delta").mode("overwrite").saveAsTable(users_table_name)

print(f"✓ Written {users_df.count()} users to {users_table_name}")

add_table_comment(
    CATALOG,
    SCHEMA,
    USERS_TABLE,
    "Synthetic user data for fashion recommendation testing"
)

# COMMAND ----------

# Write transactions table
transactions_table_name = f"{CATALOG}.{SCHEMA}.{TRANSACTIONS_TABLE}"
transactions_df.write.format("delta").mode("overwrite").saveAsTable(transactions_table_name)

print(f"✓ Written {transactions_df.count()} transactions to {transactions_table_name}")

add_table_comment(
    CATALOG,
    SCHEMA,
    TRANSACTIONS_TABLE,
    "Synthetic user-product interaction history (views, add-to-cart, purchases)"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary and Verification

# COMMAND ----------

print("=" * 60)
print("SYNTHETIC DATA GENERATION SUMMARY")
print("=" * 60)
print(f"\nUsers: {users_table_name}")
print(f"  Count: {spark.table(users_table_name).count():,}")
print(f"  Segments: {users_df.select('segment').distinct().count()}")

print(f"\nTransactions: {transactions_table_name}")
print(f"  Count: {spark.table(transactions_table_name).count():,}")
print(f"  Event types: {transactions_df.select('event_type').distinct().count()}")
print(f"  Date range: {transactions_df.agg(F.min('timestamp'), F.max('timestamp')).first()}")

print("\n" + "=" * 60)

# COMMAND ----------

# Show top active users
print("Top 10 most active users:")
spark.table(transactions_table_name).groupBy("user_id").count().orderBy(F.desc("count")).show(10)

# COMMAND ----------

# Show top purchased products
print("Top 10 most purchased products:")
(
    spark.table(transactions_table_name)
    .filter(F.col("event_type") == "purchase")
    .groupBy("product_id")
    .count()
    .orderBy(F.desc("count"))
    .join(products_table, "product_id")
    .select("product_id", "display_name", "category", F.col("count").alias("purchase_count"))
    .show(10, truncate=False)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC 1. Run notebook `03_image_embeddings_pipeline` to generate product embeddings
# MAGIC 2. Run notebook `05_user_style_features` to compute user preference features
