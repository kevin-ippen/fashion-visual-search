# Databricks notebook source
# MAGIC %md
# MAGIC # Fashion Product Images Dataset Ingestion
# MAGIC
# MAGIC This notebook downloads the Fashion Product Images Dataset from Kaggle and loads it into Unity Catalog.
# MAGIC
# MAGIC **Dataset**: [Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset)
# MAGIC
# MAGIC **Target Locations**:
# MAGIC - Metadata Table: `main.fashion_demo.products`
# MAGIC - Raw Images: `/Volumes/main/fashion_demo/raw_data/images/`

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration

# COMMAND ----------

# Configuration - Update these as needed
CATALOG = "main"
SCHEMA = "fashion_demo"
TABLE_NAME = "products_full"
VOLUME_NAME = "raw_data"

# Derived paths
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}"
TABLE_FULL_NAME = f"{CATALOG}.{SCHEMA}.{TABLE_NAME}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Setup Schema and Volume

# COMMAND ----------

# Create schema if not exists
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

# Create volume for raw data if not exists
spark.sql(f"""
    CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME_NAME}
    COMMENT 'Raw fashion product data including images'
""")

print(f"✓ Schema: {CATALOG}.{SCHEMA}")
print(f"✓ Volume: {VOLUME_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Install Dependencies and Configure Kaggle

# COMMAND ----------

# MAGIC %pip install kaggle --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# Re-import after restart
CATALOG = "main"
SCHEMA = "fashion_demo"
TABLE_NAME = "products"
VOLUME_NAME = "raw_data"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}"
TABLE_FULL_NAME = f"{CATALOG}.{SCHEMA}.{TABLE_NAME}"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Configure Kaggle Credentials
# MAGIC
# MAGIC **Option 1**: Store credentials in Databricks Secrets (Recommended)
# MAGIC ```
# MAGIC databricks secrets create-scope --scope kaggle
# MAGIC databricks secrets put --scope kaggle --key username
# MAGIC databricks secrets put --scope kaggle --key key
# MAGIC ```
# MAGIC
# MAGIC **Option 2**: Set credentials directly (for testing only)

# COMMAND ----------



# COMMAND ----------

import os

# Option 1: Using Databricks Secrets (recommended for production)
try:
    os.environ['KAGGLE_USERNAME'] = dbutils.secrets.get(scope="kaggle", key="scope")
    os.environ['KAGGLE_KEY'] = dbutils.secrets.get(scope="kaggle", key="key")
    print("✓ Loaded Kaggle credentials from Databricks Secrets")
except Exception as e:
    print(f"⚠ Could not load from secrets: {e}")
    print("Please set KAGGLE_USERNAME and KAGGLE_KEY manually or use secrets")
    
# Option 2: Manual override (uncomment and fill in for testing)
# os.environ['KAGGLE_USERNAME'] = "your_kaggle_username"
# os.environ['KAGGLE_KEY'] = "your_kaggle_api_key"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Download Dataset from Kaggle

# COMMAND ----------

import subprocess
import zipfile
import shutil
from pathlib import Path

# Working directory for download
WORK_DIR = "/tmp/fashion_data"
os.makedirs(WORK_DIR, exist_ok=True)

# Download dataset
print("Downloading dataset from Kaggle (this may take several minutes)...")
result = subprocess.run(
    [
        "kaggle", "datasets", "download",
        "-d", "paramaggarwal/fashion-product-images-dataset",
        "-p", WORK_DIR
    ],
    capture_output=True,
    text=True
)

if result.returncode != 0:
    print(f"Error: {result.stderr}")
    raise Exception("Failed to download dataset. Check Kaggle credentials.")
else:
    print("✓ Download complete")

# COMMAND ----------

# Extract the zip file
zip_path = f"{WORK_DIR}/fashion-product-images-dataset.zip"
extract_path = f"{WORK_DIR}/extracted"

print("Extracting dataset...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
print("✓ Extraction complete")

# List extracted contents
for item in os.listdir(extract_path):
    item_path = os.path.join(extract_path, item)
    if os.path.isdir(item_path):
        count = len(os.listdir(item_path))
        print(f"  📁 {item}/ ({count} items)")
    else:
        size = os.path.getsize(item_path) / (1024*1024)
        print(f"  📄 {item} ({size:.1f} MB)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Load and Process Metadata

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import *
import random

# Find the styles.csv file (may be in root or subdirectory)
csv_path = None
for root, dirs, files in os.walk(extract_path):
    for file in files:
        if file == "styles.csv":
            csv_path = os.path.join(root, file)
            break

if csv_path is None:
    raise FileNotFoundError("Could not find styles.csv in extracted data")

print(f"Found metadata file: {csv_path}")

# COMMAND ----------

from pyspark.sql.types import *
import pandas as pd

# Workaround for Shared cluster: Use pandas to read from /tmp, then convert to Spark
print(f"Reading CSV from: {csv_path}")

# Define schema including the productDescriptionText field
schema = StructType([
    StructField("id", IntegerType(), True),
    StructField("gender", StringType(), True),
    StructField("masterCategory", StringType(), True),
    StructField("subCategory", StringType(), True),
    StructField("articleType", StringType(), True),
    StructField("baseColour", StringType(), True),
    StructField("season", StringType(), True),
    StructField("year", IntegerType(), True),
    StructField("usage", StringType(), True),
    StructField("productDisplayName", StringType(), True),
    StructField("productDescriptionText", StringType(), True)  # The missing field!
])

# Read with pandas (can access /tmp on Shared clusters)
pandas_df = pd.read_csv(
    csv_path,
    encoding='utf-8',
    on_bad_lines='skip'
)

# Convert to Spark DataFrame with schema
raw_df = spark.createDataFrame(pandas_df.assign(productDescriptionText=''), schema=schema)

print(f"Loaded {raw_df.count():,} records")
raw_df.printSchema()

# COMMAND ----------

# Preview the data
display(raw_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Data Transformation and Enrichment

# COMMAND ----------

# Rename columns to snake_case standard format
transformed_df = (raw_df
    .withColumnRenamed("id", "product_id")
    .withColumnRenamed("masterCategory", "master_category")
    .withColumnRenamed("subCategory", "sub_category")
    .withColumnRenamed("articleType", "article_type")
    .withColumnRenamed("baseColour", "base_color")
    .withColumnRenamed("productDisplayName", "product_display_name")
)

# COMMAND ----------

# Generate synthetic prices based on category
# (Original dataset doesn't include prices)

# Price ranges by master category
price_ranges = {
    "Apparel": (19.99, 199.99),
    "Accessories": (9.99, 149.99),
    "Footwear": (29.99, 299.99),
    "Personal Care": (4.99, 49.99),
    "Free Items": (0.0, 0.0),
    "Sporting Goods": (14.99, 199.99),
    "Home": (9.99, 99.99),
}

# Create a UDF to generate prices
@F.udf(returnType=DoubleType())
def generate_price(category, product_id):
    random.seed(product_id)  # Reproducible prices
    min_price, max_price = price_ranges.get(category, (19.99, 99.99))
    if min_price == max_price:
        return min_price
    price = random.uniform(min_price, max_price)
    return round(price, 2)

# Add price column
transformed_df = transformed_df.withColumn(
    "price",
    generate_price(F.col("master_category"), F.col("product_id"))
)

# COMMAND ----------

# Add image path reference
transformed_df = transformed_df.withColumn(
    "image_path",
    F.concat(F.lit(f"{VOLUME_PATH}/images/"), F.col("product_id").cast("string"), F.lit(".jpg"))
)

# Add ingestion timestamp
transformed_df = transformed_df.withColumn(
    "ingested_at",
    F.current_timestamp()
)

# COMMAND ----------

# Preview transformed data
display(transformed_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Data Quality Checks

# COMMAND ----------

# Check for nulls in key columns
print("=== Null Value Analysis ===")
key_columns = ["product_id", "master_category", "article_type", "product_display_name"]
for col in key_columns:
    null_count = transformed_df.filter(F.col(col).isNull()).count()
    print(f"  {col}: {null_count:,} nulls")

# Check for duplicates
print("\n=== Duplicate Analysis ===")
total_count = transformed_df.count()
distinct_count = transformed_df.select("product_id").distinct().count()
duplicate_count = total_count - distinct_count
print(f"  Total records: {total_count:,}")
print(f"  Distinct product IDs: {distinct_count:,}")
print(f"  Duplicates: {duplicate_count:,}")

# Category distribution
print("\n=== Category Distribution ===")
display(
    transformed_df
    .groupBy("master_category")
    .agg(F.count("*").alias("count"))
    .orderBy(F.desc("count"))
)

# COMMAND ----------

# Remove duplicates (keep first occurrence) and filter nulls
clean_df = (transformed_df
    .filter(F.col("product_id").isNotNull())
    .dropDuplicates(["product_id"])
)

print(f"Records after cleaning: {clean_df.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Write to Unity Catalog Delta Table

# COMMAND ----------

# Write to Delta table
(clean_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TABLE_FULL_NAME)
)

print(f"✓ Data written to {TABLE_FULL_NAME}")

# COMMAND ----------

# Add table comment and column descriptions
spark.sql(f"""
    ALTER TABLE {TABLE_FULL_NAME}
    SET TBLPROPERTIES (
        'delta.minReaderVersion' = '1',
        'delta.minWriterVersion' = '2'
    )
""")

spark.sql(f"COMMENT ON TABLE {TABLE_FULL_NAME} IS 'Fashion product catalog with metadata and pricing'")

print("✓ Table properties set")

# COMMAND ----------

# Optimize table for query performance
spark.sql(f"OPTIMIZE {TABLE_FULL_NAME} ZORDER BY (master_category, article_type)")
print("✓ Table optimized with Z-ORDER on master_category, article_type")

# COMMAND ----------

# Verify the table
display(spark.sql(f"SELECT * FROM {TABLE_FULL_NAME} LIMIT 10"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Copy Images to Unity Catalog Volume

# COMMAND ----------

# Find images directory
images_source = None
for root, dirs, files in os.walk(extract_path):
    if "images" in dirs:
        images_source = os.path.join(root, "images")
        break

if images_source is None:
    # Try alternate structure
    for root, dirs, files in os.walk(extract_path):
        jpg_files = [f for f in files if f.endswith('.jpg')]
        if len(jpg_files) > 100:  # Found image directory
            images_source = root
            break

if images_source:
    print(f"Found images at: {images_source}")
    image_count = len([f for f in os.listdir(images_source) if f.endswith('.jpg')])
    print(f"Total images: {image_count:,}")
else:
    print("⚠ Could not locate images directory")

# COMMAND ----------

# Copy images to UC Volume
# Note: This can take a while for 44k+ images

if images_source:
    images_dest = f"{VOLUME_PATH}/images"
    
    # Create destination directory
    dbutils.fs.mkdirs(images_dest)
    
    print(f"Copying images to {images_dest}...")
    print("This may take 10-30 minutes depending on cluster size...")
    
    # Use dbutils to copy (handles the file:// to dbfs:// conversion)
    dbutils.fs.cp(f"file:{images_source}", images_dest, recurse=True)
    
    # Verify copy
    copied_files = dbutils.fs.ls(images_dest)
    print(f"✓ Copied {len(copied_files):,} files to Volume")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Validation and Summary

# COMMAND ----------

# Final validation
print("=" * 60)
print("INGESTION SUMMARY")
print("=" * 60)

# Table stats
table_count = spark.table(TABLE_FULL_NAME).count()
print(f"\n📊 Table: {TABLE_FULL_NAME}")
print(f"   Records: {table_count:,}")

# Category breakdown
print("\n📁 Category Breakdown:")
cat_counts = (spark.table(TABLE_FULL_NAME)
    .groupBy("master_category")
    .count()
    .orderBy(F.desc("count"))
    .collect()
)
for row in cat_counts:
    print(f"   {row['master_category']}: {row['count']:,}")

# Price statistics
print("\n💰 Price Statistics:")
price_stats = spark.table(TABLE_FULL_NAME).select(
    F.min("price").alias("min"),
    F.max("price").alias("max"),
    F.avg("price").alias("avg")
).collect()[0]
print(f"   Min: ${price_stats['min']:.2f}")
print(f"   Max: ${price_stats['max']:.2f}")
print(f"   Avg: ${price_stats['avg']:.2f}")

# Volume stats
print(f"\n🖼️  Images Volume: {VOLUME_PATH}/images/")
try:
    image_files = dbutils.fs.ls(f"{VOLUME_PATH}/images")
    print(f"   Image files: {len(image_files):,}")
except:
    print("   (Images not yet copied)")

print("\n" + "=" * 60)
print("✅ INGESTION COMPLETE")
print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Cleanup (Optional)

# COMMAND ----------

# Uncomment to clean up temporary files
# shutil.rmtree(WORK_DIR, ignore_errors=True)
# print("✓ Cleaned up temporary files")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC 1. **Create Vector Search Index**: Run the embedding generation notebook to create searchable vectors
# MAGIC 2. **Build User Features**: Generate user style profiles based on interactions
# MAGIC 3. **Deploy Model**: Set up the visual search endpoint
# MAGIC
# MAGIC ### Quick Queries
# MAGIC
# MAGIC ```sql
# MAGIC -- Browse products
# MAGIC SELECT * FROM main.fashion_demo.products LIMIT 100;
# MAGIC
# MAGIC -- Category summary
# MAGIC SELECT master_category, sub_category, COUNT(*) as count
# MAGIC FROM main.fashion_demo.products
# MAGIC GROUP BY 1, 2
# MAGIC ORDER BY 1, 3 DESC;
# MAGIC
# MAGIC -- Find products by color
# MAGIC SELECT * FROM main.fashion_demo.products
# MAGIC WHERE base_color = 'Navy Blue' AND master_category = 'Apparel';
# MAGIC ```

# COMMAND ----------

# Generate code for the other notebook
code_for_other_notebook = '''
# ============================================================
# ADD THESE CELLS TO YOUR OTHER NOTEBOOK (after cell 40)
# ============================================================

# CELL 1: Markdown Header
%md
## Save Complete Dataset to New Table

This section saves the complete Kaggle dataset (including `productDescriptionText`) to a new Delta table. We'll then join this with the original products table to add missing columns.

# CELL 2: Read complete CSV with all columns
import pandas as pd
from pyspark.sql.types import *
from pyspark.sql import functions as F

# Find the styles.csv file
csv_path = None
for root, dirs, files in os.walk(extract_path):
    for file in files:
        if file == "styles.csv":
            csv_path = os.path.join(root, file)
            break
    if csv_path:
        break

if not csv_path:
    raise FileNotFoundError("Could not find styles.csv")

print(f"Reading complete CSV from: {csv_path}")

# Read with pandas first to see all columns
pandas_df = pd.read_csv(csv_path, encoding='utf-8', on_bad_lines='skip')

print(f"\\nColumns in CSV:")
for i, col in enumerate(pandas_df.columns, 1):
    print(f"  {i}. {col}")

print(f"\\nTotal rows: {len(pandas_df):,}")

# CELL 3: Define complete schema with description field
# Define schema including the productDescriptionText field
complete_schema = StructType([
    StructField("id", IntegerType(), True),
    StructField("gender", StringType(), True),
    StructField("masterCategory", StringType(), True),
    StructField("subCategory", StringType(), True),
    StructField("articleType", StringType(), True),
    StructField("baseColour", StringType(), True),
    StructField("season", StringType(), True),
    StructField("year", IntegerType(), True),
    StructField("usage", StringType(), True),
    StructField("productDisplayName", StringType(), True),
    StructField("productDescriptionText", StringType(), True)  # The missing field!
])

# Convert to Spark DataFrame with complete schema
complete_df = spark.createDataFrame(pandas_df, schema=complete_schema)

print(f"Loaded {complete_df.count():,} records with complete schema")
complete_df.printSchema()

# CELL 4: Transform to match table naming conventions
# Rename columns to snake_case to match existing table
complete_transformed_df = (complete_df
    .withColumnRenamed("id", "product_id")
    .withColumnRenamed("masterCategory", "master_category")
    .withColumnRenamed("subCategory", "sub_category")
    .withColumnRenamed("articleType", "article_type")
    .withColumnRenamed("baseColour", "base_color")
    .withColumnRenamed("productDisplayName", "product_display_name")
    .withColumnRenamed("productDescriptionText", "product_description")
)

# Show sample with descriptions
print("\\nSample products with descriptions:")
display(
    complete_transformed_df
    .filter(F.col("product_description").isNotNull())
    .select("product_id", "product_display_name", "product_description")
    .limit(5)
)

# CELL 5: Save to new Delta table
# Save complete dataset to a new table
COMPLETE_TABLE = f"{CATALOG}.{SCHEMA}.products_complete"

print(f"Saving complete dataset to: {COMPLETE_TABLE}")

(complete_transformed_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(COMPLETE_TABLE)
)

print(f"✓ Saved {complete_transformed_df.count():,} products to {COMPLETE_TABLE}")

# Add table comment
spark.sql(f"""
    COMMENT ON TABLE {COMPLETE_TABLE} 
    IS 'Complete Kaggle Fashion Product dataset with all metadata including descriptions'
""")

print("✓ Table created successfully")

# CELL 6: Verify complete table
# Verify the new table
print(f"Table: {COMPLETE_TABLE}")
print(f"\\nSchema:")
spark.table(COMPLETE_TABLE).printSchema()

# Check description coverage
desc_stats = spark.sql(f"""
    SELECT 
        COUNT(*) as total_products,
        COUNT(product_description) as products_with_description,
        ROUND(COUNT(product_description) * 100.0 / COUNT(*), 2) as pct_with_description
    FROM {COMPLETE_TABLE}
""")

print("\\nDescription coverage:")
display(desc_stats)

print("\\n✓ Complete dataset ready for joining with original products table")
'''

print(code_for_other_notebook)
