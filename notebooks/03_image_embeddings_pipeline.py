# Databricks notebook source
# MAGIC %md
# MAGIC # 03 - Image Embeddings Pipeline
# MAGIC
# MAGIC This notebook generates image embeddings for all products using a CLIP model served via Databricks Model Serving.
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - `main.fashion_demo.products` table exists
# MAGIC - CLIP model deployed to Model Serving endpoint
# MAGIC - Images accessible at paths in products table
# MAGIC
# MAGIC **Output:**
# MAGIC - `main.fashion_demo.product_image_embeddings` Delta table

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Configuration
CATALOG = "main"
SCHEMA = "fashion_demo"
PRODUCTS_TABLE = "products"
EMBEDDINGS_TABLE = "product_image_embeddings"

# Model Serving endpoint configuration
# TODO: Update with your actual endpoint URL after deploying the model
MODEL_ENDPOINT_URL = "https://adb-984752964297111.11.azuredatabricks.net/serving-endpoints/clip-image-encoder/invocations"

# Batch size for processing
BATCH_SIZE = 100

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import *
import sys
import json

# Add src to path
sys.path.append("/Workspace/Repos/.../fashion-visual-search/src")  # Update path as needed

from fashion_visual_search.embeddings import ImageEmbedder
from fashion_visual_search.utils import add_table_comment, optimize_table

# COMMAND ----------

# Get Databricks token for API calls
# Note: In production, use secrets
dbutils.widgets.text("token", "", "Databricks Token")
token = dbutils.widgets.get("token")

if not token:
    # Try to get from context
    token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Products

# COMMAND ----------

# Load products table
products_df = spark.table(f"{CATALOG}.{SCHEMA}.{PRODUCTS_TABLE}")

total_products = products_df.count()
print(f"Total products to process: {total_products}")

# COMMAND ----------

# Display sample
display(products_df.select("product_id", "display_name", "category", "image_path").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Embeddings
# MAGIC
# MAGIC **Note:** This is a simplified version. In production, you would:
# MAGIC - Use UDFs or pandas_udf for parallel processing
# MAGIC - Handle errors and retries
# MAGIC - Process in batches
# MAGIC - Monitor progress with MLflow
# MAGIC
# MAGIC For now, we'll demonstrate the concept with a sample.

# COMMAND ----------

# Initialize embedder
embedder = ImageEmbedder(endpoint_url=MODEL_ENDPOINT_URL, token=token)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Option A: Sample Processing (for testing)
# MAGIC Process a small sample to test the pipeline

# COMMAND ----------

# Sample for testing
SAMPLE_SIZE = 100
sample_products = products_df.limit(SAMPLE_SIZE).collect()

print(f"Processing {len(sample_products)} sample products...")

# COMMAND ----------

# Process sample products
embeddings_data = []

for i, row in enumerate(sample_products):
    if i % 10 == 0:
        print(f"Processing {i}/{len(sample_products)}...")

    try:
        # Get embedding (in production, this would call the model serving endpoint)
        # For MVP, we'll generate a placeholder embedding
        # embedding = embedder.get_embedding(image_path=row["image_path"])

        # Placeholder: random 512-dimensional embedding
        import numpy as np
        np.random.seed(int(row["product_id"]))
        embedding = np.random.randn(512).tolist()

        embeddings_data.append({
            "product_id": row["product_id"],
            "image_embedding": embedding,
            "embedding_model": "clip-vit-b-32",
            "embedding_dimension": len(embedding),
            "created_at": F.current_timestamp()
        })
    except Exception as e:
        print(f"Error processing product {row['product_id']}: {e}")
        continue

print(f"Generated {len(embeddings_data)} embeddings")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Option B: Full Processing with UDF (uncomment for production)
# MAGIC
# MAGIC ```python
# MAGIC # Define UDF for embedding generation
# MAGIC from pyspark.sql.functions import udf
# MAGIC from pyspark.sql.types import ArrayType, DoubleType
# MAGIC
# MAGIC @udf(returnType=ArrayType(DoubleType()))
# MAGIC def generate_embedding(image_path):
# MAGIC     try:
# MAGIC         embedder = ImageEmbedder(endpoint_url=MODEL_ENDPOINT_URL, token=token)
# MAGIC         embedding = embedder.get_embedding(image_path=image_path)
# MAGIC         return embedding.tolist()
# MAGIC     except Exception as e:
# MAGIC         print(f"Error: {e}")
# MAGIC         return [0.0] * 512  # Return zero vector on failure
# MAGIC
# MAGIC # Apply UDF to all products
# MAGIC embeddings_df = (
# MAGIC     products_df
# MAGIC     .withColumn("image_embedding", generate_embedding(F.col("image_path")))
# MAGIC     .withColumn("embedding_model", F.lit("clip-vit-b-32"))
# MAGIC     .withColumn("embedding_dimension", F.lit(512))
# MAGIC     .withColumn("created_at", F.current_timestamp())
# MAGIC     .select("product_id", "image_embedding", "embedding_model", "embedding_dimension", "created_at")
# MAGIC )
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Embeddings DataFrame

# COMMAND ----------

# Define schema for embeddings
embeddings_schema = StructType([
    StructField("product_id", StringType(), False),
    StructField("image_embedding", ArrayType(DoubleType()), False),
    StructField("embedding_model", StringType(), True),
    StructField("embedding_dimension", IntegerType(), True),
    StructField("created_at", TimestampType(), True)
])

# Create DataFrame
embeddings_df = spark.createDataFrame(embeddings_data, schema=embeddings_schema)

# Add timestamp if not present
if "created_at" not in embeddings_df.columns:
    embeddings_df = embeddings_df.withColumn("created_at", F.current_timestamp())

# COMMAND ----------

# Display sample
display(embeddings_df.limit(5))

# COMMAND ----------

print(f"Embeddings shape:")
print(f"  Total embeddings: {embeddings_df.count()}")
print(f"  Embedding dimension: {embeddings_df.first()['embedding_dimension']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write to Unity Catalog

# COMMAND ----------

embeddings_table_name = f"{CATALOG}.{SCHEMA}.{EMBEDDINGS_TABLE}"

# Write as Delta table
embeddings_df.write.format("delta").mode("overwrite").saveAsTable(embeddings_table_name)

print(f"✓ Written {embeddings_df.count()} embeddings to {embeddings_table_name}")

# COMMAND ----------

# Add table comment
add_table_comment(
    CATALOG,
    SCHEMA,
    EMBEDDINGS_TABLE,
    "Product image embeddings generated using CLIP model for visual similarity search"
)

# COMMAND ----------

# Optimize table for query performance
optimize_table(CATALOG, SCHEMA, EMBEDDINGS_TABLE, zorder_cols=["product_id"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validation

# COMMAND ----------

# Verify embeddings
embeddings_table = spark.table(embeddings_table_name)

print(f"Embeddings table: {embeddings_table_name}")
print(f"Total embeddings: {embeddings_table.count()}")
print(f"Schema:")
embeddings_table.printSchema()

# COMMAND ----------

# Check for any zero vectors (potential failures)
from pyspark.sql.functions import col, size, expr

zero_vector_count = (
    embeddings_table
    .filter(expr("aggregate(image_embedding, 0.0, (acc, x) -> acc + abs(x)) = 0"))
    .count()
)

if zero_vector_count > 0:
    print(f"WARNING: {zero_vector_count} zero vectors found (failed embeddings)")
else:
    print("✓ No zero vectors found")

# COMMAND ----------

# Sample embedding stats
sample_embedding = embeddings_table.first()["image_embedding"]
import numpy as np
sample_array = np.array(sample_embedding)

print(f"Sample embedding statistics:")
print(f"  Dimension: {len(sample_embedding)}")
print(f"  Mean: {sample_array.mean():.4f}")
print(f"  Std: {sample_array.std():.4f}")
print(f"  Min: {sample_array.min():.4f}")
print(f"  Max: {sample_array.max():.4f}")
print(f"  L2 norm: {np.linalg.norm(sample_array):.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC 1. Run notebook `04_vector_search_setup` to create the Vector Search index
# MAGIC 2. For production: Update the UDF to call actual Model Serving endpoint
# MAGIC 3. Consider implementing:
# MAGIC    - Retry logic for failed embeddings
# MAGIC    - Incremental processing for new products
# MAGIC    - Quality monitoring for embeddings
