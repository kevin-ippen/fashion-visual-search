# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - Ingest Product Catalog
# MAGIC
# MAGIC This notebook ingests the Fashion Product Images dataset and creates the `products` table in Unity Catalog.
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - Download the Fashion Product Images dataset from Kaggle
# MAGIC - Upload the CSV and images to a volume or DBFS location
# MAGIC
# MAGIC **Output:**
# MAGIC - `main.fashion_demo.products` Delta table

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Configuration
CATALOG = "main"
SCHEMA = "fashion_demo"
PRODUCTS_TABLE = "products"

# Update these paths based on where you uploaded the dataset
DATA_PATH = "/Volumes/main/fashion_demo/raw_data/styles.csv"
IMAGES_PATH = "/Volumes/main/fashion_demo/raw_data/images/"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import *
import sys

# Add src to path for module imports
sys.path.append("/Workspace/Repos/.../fashion-visual-search/src")  # Update path as needed

from fashion_visual_search.utils import create_catalog_schema, table_exists, add_table_comment

# COMMAND ----------

# Create schema if it doesn't exist
create_catalog_schema(CATALOG, SCHEMA)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Product Data

# COMMAND ----------

# Define schema for products CSV
product_schema = StructType([
    StructField("id", IntegerType(), False),
    StructField("gender", StringType(), True),
    StructField("masterCategory", StringType(), True),
    StructField("subCategory", StringType(), True),
    StructField("articleType", StringType(), True),
    StructField("baseColour", StringType(), True),
    StructField("season", StringType(), True),
    StructField("year", IntegerType(), True),
    StructField("usage", StringType(), True),
    StructField("productDisplayName", StringType(), True)
])

# COMMAND ----------

# Read CSV data
raw_df = spark.read.csv(
    DATA_PATH,
    header=True,
    schema=product_schema
)

print(f"Loaded {raw_df.count()} products from CSV")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Transform and Enrich

# COMMAND ----------

# Transform to our standard schema
products_df = (
    raw_df
    .withColumnRenamed("id", "product_id")
    .withColumn("product_id", F.col("product_id").cast("string"))
    .withColumnRenamed("masterCategory", "category")
    .withColumnRenamed("subCategory", "subcategory")
    .withColumnRenamed("articleType", "article_type")
    .withColumnRenamed("baseColour", "color")
    .withColumnRenamed("productDisplayName", "display_name")
    .withColumn("brand", F.lit("Generic"))  # Dataset doesn't have brands, using placeholder
    .withColumn(
        "image_path",
        F.concat(F.lit(IMAGES_PATH), F.col("product_id"), F.lit(".jpg"))
    )
    .withColumn(
        "price",
        # Generate synthetic prices based on category
        F.when(F.col("category") == "Apparel", F.round(F.rand() * 100 + 30, 2))
        .when(F.col("category") == "Accessories", F.round(F.rand() * 80 + 20, 2))
        .when(F.col("category") == "Footwear", F.round(F.rand() * 150 + 40, 2))
        .otherwise(F.round(F.rand() * 60 + 25, 2))
    )
    .withColumn("created_at", F.current_timestamp())
    .select(
        "product_id",
        "display_name",
        "category",
        "subcategory",
        "article_type",
        "brand",
        "color",
        "gender",
        "season",
        "year",
        "usage",
        "price",
        "image_path",
        "created_at"
    )
)

# COMMAND ----------

# Display sample
display(products_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Quality Checks

# COMMAND ----------

from fashion_visual_search.utils import DataQualityChecker

# Check for nulls in critical columns
null_counts = DataQualityChecker.check_nulls(
    products_df,
    ["product_id", "display_name", "category", "image_path"]
)

if null_counts:
    print("WARNING: Null values found:")
    for col, count in null_counts.items():
        print(f"  {col}: {count} nulls")
else:
    print("✓ No null values in critical columns")

# COMMAND ----------

# Check for duplicates
duplicate_count = DataQualityChecker.check_duplicates(products_df, ["product_id"])

if duplicate_count > 0:
    print(f"WARNING: {duplicate_count} duplicate product_ids found")
    # Deduplicate
    products_df = products_df.dropDuplicates(["product_id"])
else:
    print("✓ No duplicate product_ids")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write to Unity Catalog

# COMMAND ----------

full_table_name = f"{CATALOG}.{SCHEMA}.{PRODUCTS_TABLE}"

# Write as Delta table
products_df.write.format("delta").mode("overwrite").saveAsTable(full_table_name)

print(f"✓ Written {products_df.count()} products to {full_table_name}")

# COMMAND ----------

# Add table comment for documentation
add_table_comment(
    CATALOG,
    SCHEMA,
    PRODUCTS_TABLE,
    "Fashion product catalog with metadata and image references"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify and Summary

# COMMAND ----------

# Verify table
products_table = spark.table(full_table_name)

print(f"Products table: {full_table_name}")
print(f"Total products: {products_table.count()}")
print(f"\nCategory distribution:")
products_table.groupBy("category").count().orderBy(F.desc("count")).show()

print(f"\nColor distribution (top 10):")
products_table.groupBy("color").count().orderBy(F.desc("count")).show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC 1. Verify images are accessible at the paths in `image_path` column
# MAGIC 2. Run notebook `02_generate_synthetic_users_transactions` to create user data
# MAGIC 3. Run notebook `03_image_embeddings_pipeline` to generate embeddings
