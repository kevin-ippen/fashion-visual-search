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

dbutils.secrets.get(scope="redditscope", key="redditkey")

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Add Missing Metadata Columns
# MAGIC
# MAGIC This section surgically adds missing columns from the original Kaggle Fashion Product Images dataset to the existing products table. The original dataset includes a `productDescriptionText` field that wasn't captured during initial ingestion.

# COMMAND ----------

# DBTITLE 1,Check for complete CSV with all columns
import os

# Check if we can find the original CSV with complete columns
# This notebook may have downloaded it to /tmp/fashion_data
possible_csv_locations = [
    "/tmp/fashion_data/extracted",
    "/Volumes/main/fashion_demo/raw_data",
    "/dbfs/FileStore/fashion_demo"
]

csv_found = False
csv_path = None

for base_path in possible_csv_locations:
    try:
        # Walk through directory to find styles.csv
        if os.path.exists(base_path):
            for root, dirs, files in os.walk(base_path):
                if 'styles.csv' in files:
                    csv_path = os.path.join(root, 'styles.csv')
                    csv_found = True
                    print(f"✓ Found styles.csv at: {csv_path}")
                    break
        if csv_found:
            break
    except Exception as e:
        continue

if not csv_found:
    print("⚠ Original styles.csv not found in expected locations.")
    print("\nThe CSV may have been downloaded earlier in this notebook.")
    print("Check /tmp/fashion_data/extracted/ if you ran the download cells above.")

# COMMAND ----------

# DBTITLE 1,Inspect CSV schema to confirm missing columns
# If CSV was found, check what columns it actually contains
if csv_found and csv_path:
    import pandas as pd
    
    # Read just the header to see available columns
    sample_df = pd.read_csv(csv_path, nrows=5)
    
    print("Columns in the original CSV:")
    for i, col in enumerate(sample_df.columns, 1):
        print(f"  {i}. {col}")
    
    # Check for the missing column
    if 'productDescriptionText' in sample_df.columns:
        print("\n✓ Found 'productDescriptionText' column!")
        print("\nSample description:")
        sample_desc = sample_df[sample_df['productDescriptionText'].notna()]['productDescriptionText'].iloc[0] if 'productDescriptionText' in sample_df.columns else None
        if sample_desc:
            print(f"  {sample_desc[:200]}...")
    else:
        print("\n⚠ 'productDescriptionText' column not found in CSV")
else:
    print("Skipping CSV inspection - file not found")

# COMMAND ----------

# DBTITLE 1,Add product_description column to table
# Add the missing column to the existing products table
full_table_name = f"{CATALOG}.{SCHEMA}.{PRODUCTS_TABLE}"

try:
    # Add product_description column
    spark.sql(f"""
        ALTER TABLE {full_table_name}
        ADD COLUMN IF NOT EXISTS product_description STRING
        COMMENT 'Detailed product description from Kaggle dataset'
    """)
    
    print(f"✓ Added 'product_description' column to {full_table_name}")
    print("\nColumn added successfully. It will be NULL until populated with actual data.")
    
except Exception as e:
    print(f"Error adding column: {e}")

# COMMAND ----------

# DBTITLE 1,Merge missing data from CSV if available
# If we have the complete CSV, merge the missing data into the table
if csv_found and csv_path:
    print(f"Reading complete dataset from: {csv_path}")
    
    # Read the complete CSV
    import pandas as pd
    pandas_df = pd.read_csv(csv_path, encoding='utf-8', on_bad_lines='skip')
    
    # Check if productDescriptionText exists
    if 'productDescriptionText' in pandas_df.columns:
        # Convert to Spark DataFrame with only needed columns
        from pyspark.sql.types import StructType, StructField, IntegerType, StringType
        
        merge_schema = StructType([
            StructField("id", IntegerType(), True),
            StructField("productDescriptionText", StringType(), True)
        ])
        
        # Select only ID and description columns
        pandas_subset = pandas_df[['id', 'productDescriptionText']].copy()
        
        # Convert to Spark
        missing_data_df = spark.createDataFrame(pandas_subset, schema=merge_schema)
        
        # Rename columns to match table schema
        missing_data_df = (
            missing_data_df
            .withColumnRenamed("id", "product_id")
            .withColumnRenamed("productDescriptionText", "product_description")
            .filter(F.col("product_description").isNotNull())
        )
        
        print(f"Loaded {missing_data_df.count():,} products with descriptions")
        
        # Create temp view for merge
        missing_data_df.createOrReplaceTempView("missing_metadata_temp")
        
        # Merge the missing data into the products table
        spark.sql(f"""
            MERGE INTO {full_table_name} AS target
            USING missing_metadata_temp AS source
            ON target.product_id = source.product_id
            WHEN MATCHED THEN
                UPDATE SET target.product_description = source.product_description
        """)
        
        print(f"✓ Merged missing metadata into {full_table_name}")
        
        # Verify the update
        updated_count = spark.sql(f"""
            SELECT COUNT(*) as count
            FROM {full_table_name}
            WHERE product_description IS NOT NULL
        """).collect()[0]['count']
        
        print(f"✓ {updated_count:,} products now have descriptions")
    else:
        print("⚠ 'productDescriptionText' column not found in CSV")
        print("The CSV may not contain the complete dataset.")
else:
    print("⚠ Skipping merge - CSV file not found.")
    print("\nTo populate the product_description column:")
    print("1. Ensure you've run the Kaggle download cells above (cells 12-15)")
    print("2. Or manually download and upload styles.csv to /Volumes/main/fashion_demo/raw_data/")
    print("3. Re-run this cell to merge the data")

# COMMAND ----------

# DBTITLE 1,Verify updated table schema and data
# Verify the updated table
full_table_name = f"{CATALOG}.{SCHEMA}.{PRODUCTS_TABLE}"
updated_table = spark.table(full_table_name)

print(f"Updated table: {full_table_name}")
print(f"Total columns: {len(updated_table.columns)}")
print(f"\nColumn list:")
for col in updated_table.columns:
    print(f"  - {col}")

# Check coverage of descriptions
if 'product_description' in updated_table.columns:
    desc_stats = spark.sql(f"""
        SELECT 
            COUNT(*) as total_products,
            COUNT(product_description) as products_with_description,
            ROUND(COUNT(product_description) * 100.0 / COUNT(*), 2) as pct_with_description
        FROM {full_table_name}
    """)
    
    print("\nDescription coverage:")
    display(desc_stats)
    
    # Show sample with descriptions
    print("\nSample products with descriptions:")
    display(
        updated_table
        .filter(F.col("product_description").isNotNull())
        .select("product_id", "product_display_name", "product_description")
        .limit(5)
    )
else:
    print("\n⚠ product_description column not yet added")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary: Missing Metadata Resolution
# MAGIC
# MAGIC ### What Was Done
# MAGIC * Added `product_description` column to the products table structure
# MAGIC * Attempted to locate and merge data from the original Kaggle CSV
# MAGIC
# MAGIC ### If Descriptions Are Still Missing
# MAGIC
# MAGIC The original Kaggle Fashion Product Images dataset should contain a `productDescriptionText` column. If it's not populated:
# MAGIC
# MAGIC **Option 1: Use the download cells above**
# MAGIC * Cells 12-15 in this notebook download the complete dataset from Kaggle
# MAGIC * After running those cells, re-run the merge cell above
# MAGIC
# MAGIC **Option 2: Manual download and upload**
# MAGIC 1. Download from [Kaggle Fashion Product Images](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset)
# MAGIC 2. Extract and locate `styles.csv`
# MAGIC 3. Upload to `/Volumes/main/fashion_demo/raw_data/styles.csv`
# MAGIC 4. Re-run the merge cell above
# MAGIC
# MAGIC **Option 3: Check if descriptions exist in a different format**
# MAGIC * Some versions of this dataset may have descriptions in a separate file
# MAGIC * Check for `styles.json` or other metadata files in the download
# MAGIC
# MAGIC ### Other Potentially Missing Fields
# MAGIC The Kaggle dataset may also include:
# MAGIC * Product material/composition
# MAGIC * Care instructions  
# MAGIC * Fit information
# MAGIC * Brand details (currently using placeholder)
# MAGIC
# MAGIC Consider adding these columns if needed for your use case.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Add Missing Metadata Columns
# MAGIC
# MAGIC This section adds missing columns from the original Kaggle Fashion Product Images dataset to the products table. The original dataset includes additional metadata fields that weren't captured in the initial ingestion.

# COMMAND ----------

# DBTITLE 1,Define complete schema with missing fields
# Complete schema from Kaggle Fashion Product Images dataset
# The original dataset has these additional columns that we're missing:
complete_schema = StructType([
    StructField("id", IntegerType(), False),
    StructField("gender", StringType(), True),
    StructField("masterCategory", StringType(), True),
    StructField("subCategory", StringType(), True),
    StructField("articleType", StringType(), True),
    StructField("baseColour", StringType(), True),
    StructField("season", StringType(), True),
    StructField("year", IntegerType(), True),
    StructField("usage", StringType(), True),
    StructField("productDisplayName", StringType(), True),
    # Missing columns from original dataset:
    StructField("productDescriptionText", StringType(), True)  # Detailed product description
])

print("Complete schema defined with missing columns:")
print("  - productDescriptionText: Detailed product descriptions")

# COMMAND ----------

# DBTITLE 1,Check if CSV file with complete data exists
# First, let's check if we can find the original CSV with all columns
import os

# Possible locations for the complete dataset
possible_paths = [
    "/Volumes/main/fashion_demo/raw_data/styles.csv",
    "/Volumes/main/fashion_demo/styles.csv",
    "/dbfs/FileStore/fashion_demo/styles.csv",
    "/dbfs/tmp/fashion_demo/styles.csv"
]

csv_found = False
csv_path = None

for path in possible_paths:
    try:
        # Try to check if file exists
        files = dbutils.fs.ls(os.path.dirname(path))
        for f in files:
            if 'styles.csv' in f.name.lower():
                csv_path = f.path
                csv_found = True
                print(f"✓ Found CSV file at: {csv_path}")
                break
        if csv_found:
            break
    except Exception as e:
        continue

if not csv_found:
    print("⚠ Original CSV file not found in expected locations.")
    print("\nTo add missing metadata, you need to:")
    print("1. Re-download the complete Fashion Product Images dataset from Kaggle")
    print("2. Upload the styles.csv file to: /Volumes/main/fashion_demo/raw_data/")
    print("3. Ensure the CSV includes the 'productDescriptionText' column")
    print("\nAlternatively, we can add empty columns now and populate them later.")

# COMMAND ----------

# DBTITLE 1,Option 1: Add empty columns for later population
# Add missing columns to the existing table structure
# This allows you to populate them later when you have the complete dataset

full_table_name = f"{CATALOG}.{SCHEMA}.{PRODUCTS_TABLE}"

try:
    # Add productDescriptionText column
    spark.sql(f"""
        ALTER TABLE {full_table_name}
        ADD COLUMN IF NOT EXISTS product_description STRING
        COMMENT 'Detailed product description from original dataset'
    """)
    
    print(f"✓ Added 'product_description' column to {full_table_name}")
    print("\nColumn added successfully. It will be NULL until populated with actual data.")
    
    # Show updated schema
    print("\nUpdated table schema:")
    spark.table(full_table_name).printSchema()
    
except Exception as e:
    print(f"Error adding column: {e}")

# COMMAND ----------

# DBTITLE 1,Option 2: Load complete CSV and merge data (if CSV exists)
# If you have the complete CSV file, use this cell to merge the missing data

if csv_found and csv_path:
    print(f"Reading complete dataset from: {csv_path}")
    
    # Read the complete CSV with all columns
    complete_df = spark.read.csv(
        csv_path,
        header=True,
        schema=complete_schema
    )
    
    # Select only the ID and missing columns
    missing_data_df = (
        complete_df
        .select(
            F.col("id").cast("string").alias("product_id"),
            F.col("productDescriptionText").alias("product_description")
        )
        .filter(F.col("productDescriptionText").isNotNull())
    )
    
    print(f"Loaded {missing_data_df.count()} products with descriptions")
    
    # Create temp view for merge
    missing_data_df.createOrReplaceTempView("missing_metadata")
    
    # Merge the missing data into the products table
    spark.sql(f"""
        MERGE INTO {full_table_name} AS target
        USING missing_metadata AS source
        ON target.product_id = source.product_id
        WHEN MATCHED THEN
            UPDATE SET target.product_description = source.product_description
    """)
    
    print(f"✓ Merged missing metadata into {full_table_name}")
    
    # Verify the update
    updated_count = spark.sql(f"""
        SELECT COUNT(*) as count
        FROM {full_table_name}
        WHERE product_description IS NOT NULL
    """).collect()[0]['count']
    
    print(f"✓ {updated_count} products now have descriptions")
    
else:
    print("⚠ Skipping merge - CSV file not found.")
    print("Upload the complete styles.csv file and re-run this cell to populate the data.")

# COMMAND ----------

# DBTITLE 1,Verify updated table structure
# Verify the updated table
full_table_name = f"{CATALOG}.{SCHEMA}.{PRODUCTS_TABLE}"
updated_table = spark.table(full_table_name)

print(f"Updated table: {full_table_name}")
print(f"Total columns: {len(updated_table.columns)}")
print(f"\nColumn list:")
for col in updated_table.columns:
    print(f"  - {col}")

# Check how many products have descriptions
if 'product_description' in updated_table.columns:
    desc_stats = spark.sql(f"""
        SELECT 
            COUNT(*) as total_products,
            COUNT(product_description) as products_with_description,
            ROUND(COUNT(product_description) * 100.0 / COUNT(*), 2) as pct_with_description
        FROM {full_table_name}
    """)
    
    print("\nDescription coverage:")
    display(desc_stats)
    
    # Show sample with descriptions
    print("\nSample products with descriptions:")
    display(
        updated_table
        .filter(F.col("product_description").isNotNull())
        .select("product_id", "display_name", "product_description")
        .limit(5)
    )
else:
    print("\n⚠ product_description column not yet added")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps to Complete Missing Metadata
# MAGIC
# MAGIC The `product_description` column has been added to the table structure. To populate it with actual data:
# MAGIC
# MAGIC ### 1. Download Complete Dataset
# MAGIC * Go to [Kaggle Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset)
# MAGIC * Download the complete dataset (includes `styles.csv` with all columns)
# MAGIC * Ensure the CSV contains the `productDescriptionText` column
# MAGIC
# MAGIC ### 2. Upload to Volume
# MAGIC ```python
# MAGIC # Upload the styles.csv file to:
# MAGIC /Volumes/main/fashion_demo/raw_data/styles.csv
# MAGIC ```
# MAGIC
# MAGIC ### 3. Re-run Option 2 Cell
# MAGIC * Once the CSV is uploaded, re-run the "Option 2: Load complete CSV and merge data" cell above
# MAGIC * This will populate the `product_description` column with actual data
# MAGIC
# MAGIC ### 4. Verify Data Quality
# MAGIC * Check that descriptions are populated for most products
# MAGIC * Review sample descriptions to ensure they're meaningful
# MAGIC * Consider adding additional metadata columns if needed (material, care instructions, etc.)

# COMMAND ----------


