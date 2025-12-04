# Databricks notebook source
# MAGIC %md
# MAGIC # 04 - Vector Search Setup
# MAGIC
# MAGIC This notebook creates and configures a Mosaic AI Vector Search index for product image embeddings.
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - `main.fashion_demo.product_image_embeddings` table exists with embeddings
# MAGIC - Vector Search enabled in your Databricks workspace
# MAGIC
# MAGIC **Output:**
# MAGIC - Vector Search endpoint (if not exists)
# MAGIC - Vector Search index: `main.fashion_demo.product_image_embeddings_index`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Configuration
CATALOG = "main"
SCHEMA = "fashion_demo"
EMBEDDINGS_TABLE = "product_image_embeddings"

# Vector Search configuration
VECTOR_SEARCH_ENDPOINT = "fashion_vector_search"
INDEX_NAME = f"{CATALOG}.{SCHEMA}.product_embeddings_index"
SOURCE_TABLE = f"{CATALOG}.{SCHEMA}.{EMBEDDINGS_TABLE}"

# Embedding configuration
PRIMARY_KEY = "product_id"
EMBEDDING_VECTOR_COLUMN = "image_embedding"
EMBEDDING_DIMENSION = 512  # CLIP ViT-B/32 dimension
DISTANCE_METRIC = "COSINE"  # COSINE, L2, or INNER_PRODUCT

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from pyspark.sql import functions as F
import time

# COMMAND ----------

# Initialize Vector Search client
vsc = VectorSearchClient()

print("Vector Search client initialized")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Vector Search Endpoint

# COMMAND ----------

# Check if endpoint exists
try:
    endpoint = vsc.get_endpoint(name=VECTOR_SEARCH_ENDPOINT)
    print(f"✓ Endpoint '{VECTOR_SEARCH_ENDPOINT}' already exists")
    print(f"  Status: {endpoint.get('endpoint_status', {}).get('state', 'UNKNOWN')}")
except Exception as e:
    print(f"Endpoint does not exist, creating...")
    # Create endpoint
    vsc.create_endpoint(
        name=VECTOR_SEARCH_ENDPOINT,
        endpoint_type="STANDARD"
    )
    print(f"✓ Created endpoint '{VECTOR_SEARCH_ENDPOINT}'")

# COMMAND ----------

# Wait for endpoint to be online
print("Waiting for endpoint to be ready...")
max_wait = 600  # 10 minutes
wait_interval = 30  # Check every 30 seconds
elapsed = 0

while elapsed < max_wait:
    try:
        endpoint = vsc.get_endpoint(name=VECTOR_SEARCH_ENDPOINT)
        state = endpoint.get("endpoint_status", {}).get("state", "UNKNOWN")

        if state == "ONLINE":
            print(f"✓ Endpoint is ONLINE after {elapsed} seconds")
            break
        else:
            print(f"  Current state: {state}, waiting...")
            time.sleep(wait_interval)
            elapsed += wait_interval
    except Exception as e:
        print(f"  Error checking endpoint: {e}")
        time.sleep(wait_interval)
        elapsed += wait_interval

if elapsed >= max_wait:
    print("WARNING: Endpoint did not come online within timeout")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Source Table

# COMMAND ----------

# Verify embeddings table exists and has data
embeddings_df = spark.table(SOURCE_TABLE)
embedding_count = embeddings_df.count()

print(f"Source table: {SOURCE_TABLE}")
print(f"  Total records: {embedding_count}")
print(f"  Schema:")
embeddings_df.printSchema()

# COMMAND ----------

# Verify embedding dimension
sample_embedding = embeddings_df.first()[EMBEDDING_VECTOR_COLUMN]
actual_dimension = len(sample_embedding)

print(f"\nEmbedding configuration:")
print(f"  Expected dimension: {EMBEDDING_DIMENSION}")
print(f"  Actual dimension: {actual_dimension}")

if actual_dimension != EMBEDDING_DIMENSION:
    print(f"WARNING: Dimension mismatch! Updating to {actual_dimension}")
    EMBEDDING_DIMENSION = actual_dimension

# COMMAND ----------

# DBTITLE 1,Enable Change Data Feed on Source Table


# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Vector Search Index

# COMMAND ----------

# Check if index already exists
try:
    existing_index = vsc.get_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT,
        index_name=INDEX_NAME
    )
    print(f"✓ Index '{INDEX_NAME}' already exists")
    print(f"  Status: {existing_index.describe().get('status', {}).get('detailed_state', 'UNKNOWN')}")

    # Optionally sync the index
    print("\nSyncing index with source table...")
    existing_index.sync()
    print("✓ Sync triggered")

except Exception as e:
    print(f"Index does not exist, creating...")

    # Create Delta Sync index
    index = vsc.create_delta_sync_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT,
        source_table_name=SOURCE_TABLE,
        index_name=INDEX_NAME,
        pipeline_type="TRIGGERED",  # or "CONTINUOUS" for real-time sync
        primary_key=PRIMARY_KEY,
        embedding_dimension=EMBEDDING_DIMENSION,
        embedding_vector_column=EMBEDDING_VECTOR_COLUMN
    )

    print(f"✓ Created index '{INDEX_NAME}'")
    print("  Index will begin syncing from source table")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Wait for Index to be Ready

# COMMAND ----------

print("Waiting for index to be ready...")
max_wait = 1800  # 30 minutes for initial indexing
wait_interval = 30
elapsed = 0

while elapsed < max_wait:
    try:
        index = vsc.get_index(
            endpoint_name=VECTOR_SEARCH_ENDPOINT,
            index_name=INDEX_NAME
        )
        status = index.describe()
        state = status.get("status", {}).get("detailed_state", "UNKNOWN")

        if state == "ONLINE_CONTINUOUS_UPDATE" or state == "ONLINE_TRIGGERED_UPDATE":
            print(f"✓ Index is ONLINE after {elapsed} seconds")
            print(f"  Indexed records: {status.get('status', {}).get('num_indexed_rows', 0)}")
            break
        else:
            print(f"  Current state: {state}, waiting...")
            time.sleep(wait_interval)
            elapsed += wait_interval
    except Exception as e:
        print(f"  Error checking index: {e}")
        time.sleep(wait_interval)
        elapsed += wait_interval

if elapsed >= max_wait:
    print("WARNING: Index did not come online within timeout")

# COMMAND ----------

# DBTITLE 1,Check Index Status


# COMMAND ----------

# DBTITLE 1,Trigger Index Sync


# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Vector Search

# COMMAND ----------

# Get index
index = vsc.get_index(
    endpoint_name=VECTOR_SEARCH_ENDPOINT,
    index_name=INDEX_NAME
)

# COMMAND ----------

# Test query with a random product's embedding
test_product = embeddings_df.limit(1).collect()[0]
test_product_id = test_product[PRIMARY_KEY]
test_embedding = test_product[EMBEDDING_VECTOR_COLUMN]

print(f"Test query with product: {test_product_id}")
print(f"Embedding dimension: {len(test_embedding)}")

# COMMAND ----------

# Perform similarity search
results = index.similarity_search(
    query_vector=test_embedding,
    columns=["product_id"],
    num_results=10
)

print(f"\nTop 10 similar products to {test_product_id}:")
print(results)

# COMMAND ----------

# Join with products table for human-readable results
from pyspark.sql.types import *

products_df = spark.table(f"{CATALOG}.{SCHEMA}.products")

# Convert results to DataFrame
if results and "result" in results and "data_array" in results["result"]:
    results_data = results["result"]["data_array"]

    results_schema = StructType([
        StructField("product_id", StringType(), True),
        StructField("score", FloatType(), True)
    ])

    results_df = spark.createDataFrame(
        [(r["product_id"], r["score"]) for r in results_data],
        schema=results_schema
    )

    # Join with products
    similar_products = (
        results_df
        .join(products_df, "product_id")
        .select(
            "product_id",
            "display_name",
            "category",
            "color",
            "price",
            "score"
        )
        .orderBy(F.desc("score"))
    )

    print("\nSimilar products with details:")
    display(similar_products)
else:
    print("No results returned")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Index Statistics

# COMMAND ----------

# Get index statistics
index_info = index.describe()

print("Vector Search Index Information:")
print("=" * 60)
print(f"Index name: {INDEX_NAME}")
print(f"Endpoint: {VECTOR_SEARCH_ENDPOINT}")
print(f"Status: {index_info.get('status', {}).get('detailed_state', 'UNKNOWN')}")
print(f"Indexed rows: {index_info.get('status', {}).get('num_indexed_rows', 0)}")
print(f"Source table: {SOURCE_TABLE}")
print(f"Primary key: {PRIMARY_KEY}")
print(f"Embedding dimension: {EMBEDDING_DIMENSION}")
print(f"Distance metric: {DISTANCE_METRIC}")
print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC 1. Run notebook `05_user_style_features` to compute user preferences
# MAGIC 2. Run notebook `06_recommendation_scoring` to test the full recommendation pipeline
# MAGIC 3. Monitor index sync status for production use
# MAGIC
# MAGIC **Note:** For production:
# MAGIC - Consider using CONTINUOUS pipeline_type for real-time updates
# MAGIC - Set up monitoring for index freshness
# MAGIC - Configure appropriate permissions for the index
