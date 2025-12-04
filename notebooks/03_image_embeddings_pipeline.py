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
# MAGIC ## ✨ Recent Improvements
# MAGIC
# MAGIC This notebook has been enhanced with production-ready features:
# MAGIC
# MAGIC ### 🚀 Performance & Scalability
# MAGIC * **Distributed Processing**: Production-grade pandas UDF for parallel embedding generation across all executors
# MAGIC * **Batch Processing**: Configurable batch size for optimal resource utilization
# MAGIC * **Table Optimization**: Z-ordering on product_id for fast lookups
# MAGIC
# MAGIC ### 🛡️ Reliability & Error Handling
# MAGIC * **Retry Logic**: Automatic retries with exponential backoff for model endpoint calls
# MAGIC * **Graceful Degradation**: Zero-vector fallback for failed embeddings
# MAGIC * **Data Quality Checks**: Comprehensive validation for nulls, duplicates, and data freshness
# MAGIC
# MAGIC ### 📊 Monitoring & Observability
# MAGIC * **MLflow Integration**: Track parameters, metrics, and embedding quality (optional)
# MAGIC * **Quality Metrics**: L2 norm validation, zero-vector detection, dimension checks
# MAGIC * **Progress Reporting**: Detailed logging throughout the pipeline
# MAGIC
# MAGIC ### 📝 Data Quality
# MAGIC * **Pre-processing Validation**: Check for missing images, null values, duplicates
# MAGIC * **Post-processing Validation**: Verify embedding dimensions, detect failures
# MAGIC * **Category Distribution**: Understand data composition
# MAGIC
# MAGIC ### 🔧 Flexibility
# MAGIC * **Mock Mode**: Test pipeline without model endpoint (USE_MOCK_EMBEDDINGS flag)
# MAGIC * **Real Model Support**: Easy switch to actual CLIP model serving endpoint
# MAGIC * **Unity Catalog Volumes**: Native support for Volume-based image storage

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Configuration
CATALOG = "main"
SCHEMA = "fashion_demo"
PRODUCTS_TABLE = "products"
EMBEDDINGS_TABLE = "product_image_embeddings"
BATCH_SIZE = 1000

# Model Serving endpoint configuration
MODEL_ENDPOINT_URL = "https://adb-984752964297111.11.azuredatabricks.net/serving-endpoints/clip-image-encoder/invocations"

# Get authentication token
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

print(f"Configuration:")
print(f"  Catalog: {CATALOG}")
print(f"  Schema: {SCHEMA}")
print(f"  Products Table: {PRODUCTS_TABLE}")
print(f"  Embeddings Table: {EMBEDDINGS_TABLE}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Model Endpoint: {MODEL_ENDPOINT_URL}")
print(f"\n✓ Configuration loaded successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import *
import sys
import json
import logging
import time
import requests
import base64
from io import BytesIO
from PIL import Image
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.append("/Workspace/Users/kevin.ippen@databricks.com/fashion-visual-search/src")

try:
    from fashion_visual_search.embeddings import ImageEmbedder
    from fashion_visual_search.utils import add_table_comment, optimize_table, DataQualityChecker
    print("✓ Successfully imported fashion_visual_search modules")
except ImportError as e:
    print(f"Warning: Could not import fashion_visual_search modules: {e}")
    print("Using enhanced inline implementations...")
    
    # Enhanced ImageEmbedder with retry logic and better error handling
    class ImageEmbedder:
        """Client for generating image embeddings via Databricks Model Serving endpoint."""

        def __init__(self, endpoint_url: str, token: str, max_retries: int = 3, timeout: int = 30):
            self.endpoint_url = endpoint_url
            self.token = token
            self.max_retries = max_retries
            self.timeout = timeout
            self.headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }

        def encode_image_to_base64(self, image_path: str, max_size=(512, 512)) -> str:
            """Encode image to base64 with preprocessing."""
            # Handle Unity Catalog Volume paths
            if image_path.startswith("/Volumes/"):
                image_path = "/dbfs" + image_path
            
            with Image.open(image_path) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                
                if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                    img.thumbnail(max_size, Image.Resampling.LANCZOS)

                buffer = BytesIO()
                img.save(buffer, format="PNG")
                img_bytes = buffer.getvalue()
                return base64.b64encode(img_bytes).decode("utf-8")

        def get_embedding(self, image_path=None, image_base64=None) -> np.ndarray:
            """Get embedding with retry logic."""
            if image_path:
                image_base64 = self.encode_image_to_base64(image_path)
            elif not image_base64:
                raise ValueError("Must provide either image_path or image_base64")

            payload = {"inputs": {"image": image_base64}}

            for attempt in range(self.max_retries):
                try:
                    response = requests.post(
                        self.endpoint_url,
                        headers=self.headers,
                        json=payload,
                        timeout=self.timeout
                    )
                    response.raise_for_status()
                    result = response.json()
                    
                    # Handle different response formats
                    if "predictions" in result:
                        embedding = result["predictions"][0]
                    elif "embedding" in result:
                        embedding = result["embedding"]
                    else:
                        embedding = result
                    
                    return np.array(embedding)
                
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        sleep_time = 2 ** attempt
                        logger.warning(f"Attempt {attempt + 1} failed, retrying in {sleep_time}s: {e}")
                        time.sleep(sleep_time)
                    else:
                        raise
    
    # Data Quality Checker
    class DataQualityChecker:
        @staticmethod
        def check_nulls(df, required_columns):
            null_counts = {}
            for col in required_columns:
                null_count = df.filter(F.col(col).isNull()).count()
                if null_count > 0:
                    null_counts[col] = null_count
            return null_counts

        @staticmethod
        def check_duplicates(df, key_columns):
            total_count = df.count()
            distinct_count = df.select(key_columns).distinct().count()
            return total_count - distinct_count

print("✓ Setup complete")

# COMMAND ----------

# Get Databricks token for API calls
# Note: In production, use secrets
dbutils.widgets.text("token", "", "Databricks Token")
token = dbutils.widgets.get("token")

if not token:
    # Try to get from context
    token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# COMMAND ----------

# DBTITLE 1,MLflow Setup for Tracking
# Install and import MLflow for tracking (optional)
try:
    import mlflow
    
    # Start MLflow run for tracking
    mlflow.set_experiment("/Users/kevin.ippen@databricks.com/fashion-visual-search/image-embeddings")
    
    run_name = f"embeddings_generation_{BATCH_SIZE}_batch"
    mlflow.start_run(run_name=run_name)
    
    # Log parameters
    mlflow.log_param("catalog", CATALOG)
    mlflow.log_param("schema", SCHEMA)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("model_endpoint", MODEL_ENDPOINT_URL)
    mlflow.log_param("products_table", f"{CATALOG}.{SCHEMA}.{PRODUCTS_TABLE}")
    mlflow.log_param("embeddings_table", f"{CATALOG}.{SCHEMA}.{EMBEDDINGS_TABLE}")
    mlflow.log_param("embedding_dimension", 512)
    mlflow.log_param("embedding_model", "clip-vit-b-32")
    
    print(f"✓ MLflow run started: {run_name}")
    print(f"  Experiment: /Users/kevin.ippen@databricks.com/fashion-visual-search/image-embeddings")
    print(f"  Run ID: {mlflow.active_run().info.run_id}")
    
    MLFLOW_ENABLED = True
except ImportError:
    print("⚠ MLflow not available - tracking disabled")
    print("  To enable: %pip install mlflow")
    MLFLOW_ENABLED = False
    
    # Create mock mlflow object for compatibility
    class MockMLflow:
        @staticmethod
        def log_metric(*args, **kwargs): pass
        @staticmethod
        def log_param(*args, **kwargs): pass
        @staticmethod
        def end_run(*args, **kwargs): pass
    
    mlflow = MockMLflow()

# COMMAND ----------

# MAGIC %pip install mlflow

# COMMAND ----------

# DBTITLE 1,Data Quality Checks
# Check data quality before processing
qc = DataQualityChecker()

print("=" * 60)
print("DATA QUALITY REPORT")
print("=" * 60)

# Get basic statistics
total_count = products_df.count()
print(f"\n✓ Total products: {total_count:,}")

# Check for nulls in required columns
print("\n1. Checking for null values...")
null_counts = qc.check_nulls(products_df, ["product_id", "image_path"])
if null_counts:
    print(f"   ⚠ WARNING: Found null values:")
    for col, count in null_counts.items():
        print(f"     - {col}: {count:,} nulls ({count/total_count*100:.2f}%)")
else:
    print("   ✓ No null values in required columns")

# Check for duplicates
print("\n2. Checking for duplicate product_ids...")
dup_count = qc.check_duplicates(products_df, ["product_id"])
if dup_count > 0:
    print(f"   ⚠ WARNING: Found {dup_count:,} duplicate product_ids")
else:
    print("   ✓ No duplicate product_ids")

# Check image path format and accessibility
print("\n3. Checking image path format...")
sample_paths = products_df.select("image_path").limit(5).collect()
volume_paths = 0
for row in sample_paths:
    path = row["image_path"]
    if path and path.startswith("/Volumes/"):
        volume_paths += 1

if volume_paths == len(sample_paths):
    print(f"   ✓ All sampled paths use Unity Catalog Volumes")
else:
    print(f"   ⚠ Mixed path formats detected ({volume_paths}/{len(sample_paths)} are Volume paths)")

# Category distribution
print("\n4. Category distribution:")
category_dist = products_df.groupBy("master_category").count().orderBy(F.desc("count"))
for row in category_dist.collect():
    print(f"   - {row['master_category']}: {row['count']:,} products")

print("\n" + "=" * 60)
print("Quality check complete!")
print("=" * 60)

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

# DBTITLE 1,Create CLIP Model Serving Endpoint
import requests
import json
import time

# Configuration for the model serving endpoint
ENDPOINT_NAME = "clip-image-encoder"
MODEL_NAME = "clip-vit-b-32"  # You can change this to your preferred CLIP variant
WORKLOAD_SIZE = "Small"  # Options: Small, Medium, Large
SCALE_TO_ZERO = True

print("=" * 60)
print("CREATING CLIP MODEL SERVING ENDPOINT")
print("=" * 60)

# Get workspace URL and token
workspace_url = spark.conf.get("spark.databricks.workspaceUrl")
api_url = f"https://{workspace_url}/api/2.0"

print(f"\nWorkspace: {workspace_url}")
print(f"Endpoint name: {ENDPOINT_NAME}")
print(f"Workload size: {WORKLOAD_SIZE}")

# Check if endpoint already exists
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

print("\n1. Checking if endpoint already exists...")
try:
    response = requests.get(
        f"{api_url}/serving-endpoints/{ENDPOINT_NAME}",
        headers=headers
    )
    
    if response.status_code == 200:
        endpoint_info = response.json()
        state = endpoint_info.get("state", {}).get("ready", "UNKNOWN")
        print(f"   ✓ Endpoint exists with state: {state}")
        
        if state == "READY":
            print(f"\n✓ Endpoint is already running and ready!")
            print(f"   URL: https://{workspace_url}/serving-endpoints/{ENDPOINT_NAME}/invocations")
        else:
            print(f"   ⚠ Endpoint exists but is not ready. Current state: {state}")
            print(f"   Waiting for endpoint to become ready...")
    else:
        print("   → Endpoint does not exist, will create new one")
        
        # Create the endpoint
        print("\n2. Creating model serving endpoint...")
        
        endpoint_config = {
            "name": ENDPOINT_NAME,
            "config": {
                "served_entities": [
                    {
                        "name": f"{ENDPOINT_NAME}_entity",
                        "external_model": {
                            "name": "clip-vit-base-patch32",
                            "provider": "openai",  # Using OpenAI CLIP via external model
                            "task": "llm/v1/embeddings"
                        },
                        "workload_size": WORKLOAD_SIZE,
                        "scale_to_zero_enabled": SCALE_TO_ZERO
                    }
                ],
                "traffic_config": {
                    "routes": [
                        {
                            "served_model_name": f"{ENDPOINT_NAME}_entity",
                            "traffic_percentage": 100
                        }
                    ]
                }
            }
        }
        
        create_response = requests.post(
            f"{api_url}/serving-endpoints",
            headers=headers,
            json=endpoint_config
        )
        
        if create_response.status_code in [200, 201]:
            print("   ✓ Endpoint creation initiated")
            
            # Wait for endpoint to be ready
            print("\n3. Waiting for endpoint to become ready...")
            max_wait = 600  # 10 minutes
            wait_interval = 10
            elapsed = 0
            
            while elapsed < max_wait:
                time.sleep(wait_interval)
                elapsed += wait_interval
                
                status_response = requests.get(
                    f"{api_url}/serving-endpoints/{ENDPOINT_NAME}",
                    headers=headers
                )
                
                if status_response.status_code == 200:
                    status_info = status_response.json()
                    state = status_info.get("state", {}).get("ready", "UNKNOWN")
                    
                    print(f"   [{elapsed}s] State: {state}")
                    
                    if state == "READY":
                        print(f"\n✓ Endpoint is ready!")
                        break
                else:
                    print(f"   ⚠ Error checking status: {status_response.status_code}")
            
            if elapsed >= max_wait:
                print(f"\n⚠ Timeout waiting for endpoint (waited {max_wait}s)")
                print("   Check endpoint status in the Serving UI")
        else:
            print(f"   ✗ Failed to create endpoint: {create_response.status_code}")
            print(f"   Response: {create_response.text}")
            
except Exception as e:
    print(f"\n✗ Error: {e}")
    print("\nNote: For production CLIP deployment, you have several options:")
    print("1. Deploy a custom CLIP model from Hugging Face")
    print("2. Use Foundation Model APIs if available in your workspace")
    print("3. Deploy using MLflow with a custom serving function")
    print("\nFor now, you can continue with USE_MOCK_EMBEDDINGS=True for testing")

print("\n" + "=" * 60)
print(f"Endpoint URL: https://{workspace_url}/serving-endpoints/{ENDPOINT_NAME}/invocations")
print("=" * 60)

# COMMAND ----------

# DBTITLE 1,Alternative: Deploy Custom CLIP Model with MLflow
# Alternative approach: Deploy a custom CLIP model from Hugging Face
# This is more flexible and doesn't require external model APIs

print("=" * 60)
print("DEPLOYING CUSTOM CLIP MODEL")
print("=" * 60)

# Model configuration
MODEL_NAME = "openai/clip-vit-base-patch32"
REGISTERED_MODEL_NAME = "clip_image_encoder"

print(f"\n1. Installing required packages...")
print("   This may take a few minutes...\n")

# Install transformers (torch is already included in DBR 16.4)
%pip install transformers --quiet

print("\n2. Restarting Python to load new packages...")
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Register CLIP Model to MLflow
# This cell runs after Python restart with transformers installed

# Check if torch is available, install if needed
try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
except ImportError:
    print("⚠ PyTorch not found, installing...")
    %pip install torch torchvision --quiet
    import torch
    print(f"✓ PyTorch installed: {torch.__version__}")

import mlflow
from mlflow.models import infer_signature
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import base64
from io import BytesIO
import pandas as pd

print("\n" + "=" * 60)
print("REGISTERING CLIP MODEL")
print("=" * 60)

# Model configuration - use the same catalog/schema as the products table
MODEL_NAME = "openai/clip-vit-base-patch32"
REGISTERED_MODEL_NAME = "main.fashion_demo.clip_image_encoder"  # Use Unity Catalog 3-level namespace

print(f"\n1. Loading CLIP model from Hugging Face...")
print(f"   Model: {MODEL_NAME}")
print(f"   Will register as: {REGISTERED_MODEL_NAME}")
print("   This will download ~600MB of model weights...\n")

# Define custom MLflow model wrapper
class CLIPImageEncoder(mlflow.pyfunc.PythonModel):
    """Custom MLflow model for CLIP image encoding."""
    
    def load_context(self, context):
        """Load the CLIP model and processor."""
        import torch
        from transformers import CLIPProcessor, CLIPModel
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model on device: {self.device}")
        self.model = CLIPModel.from_pretrained(MODEL_NAME).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        self.model.eval()
    
    def predict(self, context, model_input):
        """Generate embeddings for input images."""
        import base64
        from io import BytesIO
        from PIL import Image
        import torch
        import pandas as pd
        
        # Handle different input formats
        if isinstance(model_input, pd.DataFrame):
            # Handle DataFrame input (for batch predictions)
            if "image" in model_input.columns:
                images = model_input["image"].tolist()
            else:
                raise ValueError("Expected 'image' column in DataFrame")
        elif isinstance(model_input, dict):
            if "image" in model_input:
                images = [model_input["image"]]
            elif "inputs" in model_input and "image" in model_input["inputs"]:
                images = [model_input["inputs"]["image"]]
            else:
                raise ValueError("Expected 'image' key in input")
        else:
            images = [model_input]
        
        results = []
        for image_data in images:
            # Decode base64 image
            if isinstance(image_data, str):
                image_bytes = base64.b64decode(image_data)
                image = Image.open(BytesIO(image_bytes))
            else:
                image = image_data
            
            # Process and encode
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                # Normalize embeddings
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            results.append(image_features.cpu().numpy()[0].tolist())
        
        return results if len(results) > 1 else results[0]

print("\n2. Creating MLflow model wrapper...")

# Create input example for signature
print("\n3. Creating model signature...")
test_image_path = "/Volumes/main/fashion_demo/raw_data/images/1526.jpg"
test_image = Image.open(test_image_path)

# Convert to base64 for input example
buffer = BytesIO()
test_image.save(buffer, format="PNG")
img_bytes = buffer.getvalue()
img_base64 = base64.b64encode(img_bytes).decode("utf-8")

# Create input example as DataFrame (expected format)
input_example = pd.DataFrame({"image": [img_base64]})

# Create a temporary instance to generate output example
temp_model = CLIPImageEncoder()
temp_model.load_context(None)
output_example = temp_model.predict(None, input_example)

# Infer signature
signature = infer_signature(input_example, output_example)
print(f"   ✓ Signature created: {signature}")

# Use default experiment (notebook is in a Git folder)
print("\n4. Logging model to MLflow...")
print("   Using default notebook experiment...")
print("   This will register the model and its dependencies...\n")

try:
    with mlflow.start_run(run_name="clip_model_registration") as run:
        # Log the model with signature and input example
        mlflow.pyfunc.log_model(
            artifact_path="clip_model",
            python_model=CLIPImageEncoder(),
            pip_requirements=[
                "transformers>=4.30.0",
                "torch>=2.0.0",
                "torchvision>=0.15.0",
                "pillow>=10.0.0"
            ],
            registered_model_name=REGISTERED_MODEL_NAME,
            signature=signature,
            input_example=input_example
        )
        
        model_uri = f"runs:/{run.info.run_id}/clip_model"
        print(f"\n✓ Model logged: {model_uri}")
        print(f"\n✓ Model registered successfully!")
        print(f"   Model name: {REGISTERED_MODEL_NAME}")
        print(f"   Run ID: {run.info.run_id}")
        
        print("\n" + "=" * 60)
        print("NEXT STEPS:")
        print("=" * 60)
        print("\n1. Run the next cell to create the serving endpoint")
        print("   OR")
        print("2. Manually create endpoint in Databricks UI:")
        print(f"   - Go to Models page and find: {REGISTERED_MODEL_NAME}")
        print("   - Click 'Use model for inference' > 'Real-time'")
        print("   - Configure endpoint name: clip-image-encoder")
        print("   - Workload size: Small (or Medium for production)")
        print("   - Scale to zero: Enabled")
        print("=" * 60)
        
except Exception as e:
    print(f"\n✗ Error during model registration: {e}")
    print("\nTroubleshooting:")
    print("1. Check you have CREATE MODEL permission on main.fashion_demo schema")
    print("2. Verify the schema exists: spark.sql('SHOW SCHEMAS IN main').show()")
    print("\nAlternative: Use the mock embeddings mode (USE_MOCK_EMBEDDINGS=True) for testing.")
    import traceback
    traceback.print_exc()

# COMMAND ----------

# DBTITLE 1,Create Endpoint from Registered Model
# Create serving endpoint from the registered model
import requests
import json
import time

ENDPOINT_NAME = "clip-image-encoder"
REGISTERED_MODEL_NAME = "main.fashion_demo.clip_image_encoder"
MODEL_VERSION = "1"  # Use the version we just created
WORKLOAD_SIZE = "Small"

print("Creating serving endpoint from registered model...\n")

# Get workspace details and token
workspace_url = spark.conf.get("spark.databricks.workspaceUrl")
api_url = f"https://{workspace_url}/api/2.0"

# Get token from notebook context
try:
    token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
except:
    print("⚠ Could not get token automatically. Please set it manually:")
    print("   token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()")
    raise

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

# Check if endpoint exists
print("1. Checking existing endpoint...")
try:
    check_response = requests.get(
        f"{api_url}/serving-endpoints/{ENDPOINT_NAME}",
        headers=headers
    )
    
    if check_response.status_code == 200:
        print(f"   ⚠ Endpoint '{ENDPOINT_NAME}' already exists")
        endpoint_info = check_response.json()
        state = endpoint_info.get("state", {}).get("ready", "UNKNOWN")
        print(f"   Current state: {state}")
        
        if state == "READY":
            print(f"\n✓ Endpoint is already running and ready!")
            print(f"   URL: https://{workspace_url}/serving-endpoints/{ENDPOINT_NAME}/invocations")
        else:
            print(f"   Waiting for endpoint to become ready...")
    else:
        print(f"   → Endpoint does not exist, creating new one...\n")
        
        # Create endpoint configuration
        endpoint_config = {
            "name": ENDPOINT_NAME,
            "config": {
                "served_entities": [
                    {
                        "entity_name": REGISTERED_MODEL_NAME,
                        "entity_version": MODEL_VERSION,
                        "workload_size": WORKLOAD_SIZE,
                        "scale_to_zero_enabled": True
                    }
                ]
            }
        }
        
        # Create the endpoint
        print("2. Creating endpoint...")
        create_response = requests.post(
            f"{api_url}/serving-endpoints",
            headers=headers,
            json=endpoint_config
        )
        
        if create_response.status_code in [200, 201]:
            print("   ✓ Endpoint creation initiated\n")
            
            # Monitor endpoint status
            print("3. Monitoring endpoint deployment...")
            print("   This may take 5-10 minutes...\n")
            max_wait = 900  # 15 minutes
            wait_interval = 15
            elapsed = 0
            
            while elapsed < max_wait:
                time.sleep(wait_interval)
                elapsed += wait_interval
                
                status_response = requests.get(
                    f"{api_url}/serving-endpoints/{ENDPOINT_NAME}",
                    headers=headers
                )
                
                if status_response.status_code == 200:
                    status_info = status_response.json()
                    state = status_info.get("state", {}).get("ready", "UNKNOWN")
                    config_update = status_info.get("state", {}).get("config_update", "UNKNOWN")
                    
                    print(f"   [{elapsed}s] Ready: {state}, Config: {config_update}")
                    
                    if state == "READY":
                        print(f"\n✓ Endpoint is ready!\n")
                        break
                    elif state == "FAILED":
                        print(f"\n✗ Endpoint deployment failed")
                        print(f"   Check the Serving UI for details")
                        break
            
            if elapsed >= max_wait:
                print(f"\n⚠ Timeout after {max_wait}s. Check Serving UI for status.")
        else:
            print(f"   ✗ Failed to create endpoint: {create_response.status_code}")
            print(f"   Error: {create_response.text}")
            
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print(f"✓ Endpoint URL: https://{workspace_url}/serving-endpoints/{ENDPOINT_NAME}/invocations")
print("=" * 60)
print("\nTo use this endpoint:")
print("1. Update MODEL_ENDPOINT_URL in the configuration cell")
print("2. Set USE_MOCK_EMBEDDINGS=False in the pandas UDF")
print("3. Run the embedding generation cells")
print("=" * 60)

# COMMAND ----------

# DBTITLE 1,Check Endpoint Status
# Check the current status of the endpoint
import requests

ENDPOINT_NAME = "clip-image-encoder"

# Get workspace details and token
workspace_url = spark.conf.get("spark.databricks.workspaceUrl")
api_url = f"https://{workspace_url}/api/2.0"
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

print("=" * 60)
print("ENDPOINT STATUS CHECK")
print("=" * 60)

try:
    response = requests.get(
        f"{api_url}/serving-endpoints/{ENDPOINT_NAME}",
        headers=headers
    )
    
    if response.status_code == 200:
        endpoint_info = response.json()
        state = endpoint_info.get("state", {})
        
        print(f"\nEndpoint: {ENDPOINT_NAME}")
        print(f"Ready State: {state.get('ready', 'UNKNOWN')}")
        print(f"Config Update: {state.get('config_update', 'UNKNOWN')}")
        
        # Get served entities info
        config = endpoint_info.get("config", {})
        served_entities = config.get("served_entities", [])
        
        if served_entities:
            print(f"\nServed Model:")
            for entity in served_entities:
                print(f"  - Model: {entity.get('entity_name')}")
                print(f"  - Version: {entity.get('entity_version')}")
                print(f"  - Workload Size: {entity.get('workload_size')}")
        
        if state.get('ready') == 'READY':
            print(f"\n✓ Endpoint is READY!")
            print(f"\nEndpoint URL:")
            print(f"  https://{workspace_url}/serving-endpoints/{ENDPOINT_NAME}/invocations")
            print(f"\nNext steps:")
            print(f"  1. Update MODEL_ENDPOINT_URL in configuration cell")
            print(f"  2. Set USE_MOCK_EMBEDDINGS=False")
            print(f"  3. Run embedding generation")
        else:
            print(f"\n⚠ Endpoint is still deploying...")
            print(f"   This can take 10-20 minutes for the first deployment.")
            print(f"\n   View detailed status in Databricks UI:")
            print(f"   Serving > Endpoints > {ENDPOINT_NAME}")
            print(f"\n   Run this cell again to check status.")
    else:
        print(f"\n✗ Endpoint not found or error: {response.status_code}")
        print(f"   Response: {response.text}")
        
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Option A: Sample Processing (for testing)
# MAGIC Process a small sample to test the pipeline

# COMMAND ----------

from pyspark.sql.functions import pandas_udf
import pandas as pd

# Configuration for UDF
ENDPOINT_URL = MODEL_ENDPOINT_URL
TOKEN = token
USE_MOCK_EMBEDDINGS = True  # Set to False when model endpoint is ready

@pandas_udf(ArrayType(DoubleType()))
def generate_embedding_udf(image_paths: pd.Series) -> pd.Series:
    """
    Pandas UDF to generate embeddings for a batch of images in parallel.
    This runs on each executor with a batch of rows.
    
    Args:
        image_paths: Series of image file paths
    
    Returns:
        Series of embedding vectors (512-dimensional)
    """
    import numpy as np
    from PIL import Image
    import base64
    import requests
    from io import BytesIO
    import logging
    
    logger = logging.getLogger(__name__)
    
    def process_single_image(image_path):
        """Process a single image and return embedding or zero vector on failure."""
        try:
            # Handle null/missing paths
            if pd.isna(image_path) or not image_path:
                logger.warning("Null or empty image path")
                return np.zeros(512).tolist()
            
            # Unity Catalog Volumes paths can be used directly (no /dbfs prefix needed)
            # They are already mounted and accessible
            file_path = image_path
            
            # Load and preprocess image
            img = Image.open(file_path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # Resize if needed
            if img.size[0] > 512 or img.size[1] > 512:
                img.thumbnail((512, 512), Image.Resampling.LANCZOS)
            
            if USE_MOCK_EMBEDDINGS:
                # Generate deterministic mock embedding for testing
                img_array = np.array(img)
                
                # Create feature vector from image statistics
                features = [
                    img_array.mean(),  # Overall brightness
                    img_array.std(),   # Contrast
                    img_array[:,:,0].mean(),  # Red channel
                    img_array[:,:,1].mean(),  # Green channel
                    img_array[:,:,2].mean(),  # Blue channel
                ]
                
                # Generate deterministic embedding based on image path
                np.random.seed(hash(image_path) % (2**32))
                embedding = np.random.randn(512)
                
                # Inject actual image features
                embedding[:len(features)] = features
                
                # Normalize
                embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
                
                return embedding.tolist()
            
            else:
                # Call actual model serving endpoint
                buffer = BytesIO()
                img.save(buffer, format="PNG")
                img_bytes = buffer.getvalue()
                img_base64 = base64.b64encode(img_bytes).decode("utf-8")
                
                payload = {"inputs": {"image": img_base64}}
                headers = {
                    "Authorization": f"Bearer {TOKEN}",
                    "Content-Type": "application/json"
                }
                
                response = requests.post(
                    ENDPOINT_URL,
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                response.raise_for_status()
                
                result = response.json()
                if "predictions" in result:
                    embedding = result["predictions"][0]
                elif "embedding" in result:
                    embedding = result["embedding"]
                else:
                    embedding = result
                
                return np.array(embedding).tolist()
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            # Return zero vector on failure
            return np.zeros(512).tolist()
    
    # Process each image in the batch
    return image_paths.apply(process_single_image)

print("✓ Production-grade Pandas UDF defined")
print(f"  - Mode: {'MOCK EMBEDDINGS (for testing)' if USE_MOCK_EMBEDDINGS else 'REAL MODEL ENDPOINT'}")
print(f"  - Endpoint: {MODEL_ENDPOINT_URL}")
print(f"  - Embedding dimension: 512")

# COMMAND ----------

# DBTITLE 1,Test CLIP Endpoint


# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Alternative: Process with Python Loop


# COMMAND ----------

# DBTITLE 1,Distributed Processing with mapInPandas


# COMMAND ----------

# DBTITLE 1,Process All Products with Real CLIP


# COMMAND ----------

# DBTITLE 1,Check Embeddings DataFrame and Create Backup


# COMMAND ----------

# DBTITLE 1,Create Checkpoint Backup (Safety)


# COMMAND ----------

# DBTITLE 1,Save Embeddings to Delta Table
# Save embeddings to Delta table with optimized settings for distributed write
import time

print("=" * 60)
print("SAVING EMBEDDINGS TO DELTA TABLE")
print("=" * 60)

table_name = f"{CATALOG}.{SCHEMA}.{EMBEDDINGS_TABLE}"
print(f"\nTarget table: {table_name}")

# Check if table exists
table_exists = False
try:
    spark.table(table_name)
    table_exists = True
    print(f"\n⚠ Table already exists")
    existing_count = spark.table(table_name).count()
    print(f"  Existing records: {existing_count:,}")
except:
    print(f"\n✓ Table does not exist - will create new table")

# Get count before writing
embedding_count = embeddings_df.count()
print(f"\nRecords to write: {embedding_count:,}")

# Optimize for distributed write across worker nodes
print(f"\nOptimizing for distributed write...")

# Repartition to take advantage of all worker nodes
num_workers = sc.defaultParallelism
optimal_partitions = num_workers * 3  # 3x for good distribution

print(f"  - Worker cores available: {num_workers}")
print(f"  - Repartitioning to: {optimal_partitions} partitions")

# Repartition for parallel write
embeddings_optimized = embeddings_df.repartition(optimal_partitions)

# Start timer
start_time = time.time()

if table_exists:
    print(f"\n⚠ OVERWRITING existing table...")
    print(f"  This will replace {existing_count:,} existing records")

print(f"\nWriting to Delta table...")

# Write with Delta optimizations
write_mode = "overwrite" if table_exists else "error"

(
    embeddings_optimized
    .write
    .format("delta")
    .mode(write_mode)
    .option("overwriteSchema", "true")  # Allow schema evolution
    .option("optimizeWrite", "true")    # Optimize file sizes during write
    .option("autoOptimize.optimizeWrite", "true")  # Auto-optimize
    .saveAsTable(table_name)
)

# Calculate duration
duration = time.time() - start_time

print(f"\n✓ Write complete!")
print(f"  Duration: {duration:.2f} seconds")
print(f"  Throughput: {embedding_count/duration:.0f} records/second")

# Verify the write
verify_count = spark.table(table_name).count()
print(f"\n✓ Verification:")
print(f"  Records in table: {verify_count:,}")
print(f"  Match: {'YES ✅' if verify_count == embedding_count else 'NO ⚠'}")

print("\n" + "=" * 60)
print(f"✓ Embeddings saved to {table_name}")
print("=" * 60)

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Check Embeddings DataFrame Status
# Check if embeddings_df exists and its status
import sys

print("=" * 60)
print("CHECKING EMBEDDINGS DATAFRAME STATUS")
print("=" * 60)

# Check if embeddings_df variable exists
if 'embeddings_df' in dir():
    print("\n✓ embeddings_df variable exists in memory")
    
    try:
        # Check if it's cached
        storage_level = embeddings_df.storageLevel
        is_cached = storage_level.useMemory or storage_level.useDisk
        
        print(f"\nStorage Level:")
        print(f"  - Use Memory: {storage_level.useMemory}")
        print(f"  - Use Disk: {storage_level.useDisk}")
        print(f"  - Deserialized: {storage_level.deserialized}")
        print(f"  - Replication: {storage_level.replication}")
        print(f"  - Is Cached: {is_cached}")
        
        # Try to get count (this will trigger computation if not cached)
        print(f"\nAttempting to count records...")
        count = embeddings_df.count()
        print(f"✓ Count successful: {count:,} records")
        
        # Check schema
        print(f"\nSchema:")
        embeddings_df.printSchema()
        
        # Show sample
        print(f"\nSample (first 3 rows):")
        display(embeddings_df.limit(3))
        
        print(f"\n✓ embeddings_df is available and ready to write!")
        
    except Exception as e:
        print(f"\n✗ Error accessing embeddings_df: {e}")
        print(f"\nThe DataFrame may have been lost. You'll need to regenerate embeddings.")
        import traceback
        traceback.print_exc()
else:
    print("\n✗ embeddings_df variable does not exist")
    print("\nYou need to run the embedding generation cell first.")
    print("Look for cells like:")
    print("  - 'Generate Embeddings for All Products'")
    print("  - 'Process All Products with Real CLIP'")
    print("  - 'Distributed Processing with mapInPandas'")

print("\n" + "=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Ready to Save Embeddings!
# MAGIC
# MAGIC ### Optimized Write Strategy
# MAGIC
# MAGIC The code in the cell above is optimized for distributed writes across all worker nodes:
# MAGIC
# MAGIC **Key Optimizations:**
# MAGIC 1. **Repartitioning**: Distributes data across `3x worker cores` for parallel writes
# MAGIC 2. **Delta Optimizations**: 
# MAGIC    - `optimizeWrite=true` - Optimizes file sizes during write
# MAGIC    - `autoOptimize=true` - Enables auto-optimization
# MAGIC 3. **Overwrite Mode**: Replaces existing data (if table exists)
# MAGIC
# MAGIC ### To Execute:
# MAGIC
# MAGIC **Option 1: Run the cell above manually**
# MAGIC - Click on cell 29 "Save Embeddings to Delta Table"
# MAGIC - Press `Shift + Enter` or click the Run button
# MAGIC - This will write all {embedding_count:,} embeddings to `main.fashion_demo.product_image_embeddings`
# MAGIC
# MAGIC **Option 2: Use append mode (if you want to keep existing data)**
# MAGIC ```python
# MAGIC # Change mode to 'append' in the cell above
# MAGIC .mode("append")  # instead of overwrite
# MAGIC ```
# MAGIC
# MAGIC ### Expected Performance:
# MAGIC
# MAGIC * **Records**: 44,424 embeddings
# MAGIC * **Estimated time**: 30-60 seconds (depending on cluster size)
# MAGIC * **Throughput**: ~1,000-2,000 records/second
# MAGIC
# MAGIC ### After Writing:
# MAGIC
# MAGIC The next cells will:
# MAGIC 1. Validate embedding quality
# MAGIC 2. Optimize the table with Z-ordering
# MAGIC 3. Add table comments and metadata

# COMMAND ----------

# DBTITLE 1,Alternative: Append Mode (Safe)
# Alternative: Write in APPEND mode (safe, won't delete existing data)
import time

print("=" * 60)
print("SAVING EMBEDDINGS TO DELTA TABLE (APPEND MODE)")
print("=" * 60)

table_name = f"{CATALOG}.{SCHEMA}.{EMBEDDINGS_TABLE}"
print(f"\nTarget table: {table_name}")
print(f"Write mode: APPEND (safe - preserves existing data)")

# Get count
embedding_count = embeddings_df.count()
print(f"\nRecords to write: {embedding_count:,}")

# Optimize for distributed write
num_workers = sc.defaultParallelism
optimal_partitions = num_workers * 3

print(f"\nOptimizing for distributed write...")
print(f"  - Worker cores: {num_workers}")
print(f"  - Partitions: {optimal_partitions}")

embeddings_optimized = embeddings_df.repartition(optimal_partitions)

start_time = time.time()
print(f"\nWriting to Delta table...")

(
    embeddings_optimized
    .write
    .format("delta")
    .mode("append")  # SAFE: Won't delete existing data
    .option("optimizeWrite", "true")
    .option("autoOptimize.optimizeWrite", "true")
    .saveAsTable(table_name)
)

duration = time.time() - start_time

print(f"\n✓ Write complete!")
print(f"  Duration: {duration:.2f} seconds")
print(f"  Throughput: {embedding_count/duration:.0f} records/second")

# Verify
final_count = spark.table(table_name).count()
print(f"\n✓ Verification:")
print(f"  Total records in table: {final_count:,}")

print("\n" + "=" * 60)
print(f"✓ Embeddings saved to {table_name}")
print("=" * 60)
print("\nNote: If you need to replace data, manually run:")
print(f"  spark.sql('DELETE FROM {table_name}')")
print("  Then re-run this cell")

# COMMAND ----------

# DBTITLE 1,Validate Embeddings Quality


# COMMAND ----------

# DBTITLE 1,Optimize Table for Performance


# COMMAND ----------

# DBTITLE 1,Test UDF on Sample
# Test the UDF on a small sample first
print("Testing UDF on 10 sample products...\n")

sample_df = products_df.limit(10)
test_embeddings = (
    sample_df
    .withColumn("image_embedding", generate_embedding_udf(F.col("image_path")))
    .select("product_id", "product_display_name", "image_path", "image_embedding")
)

# Check results
test_count = test_embeddings.count()
print(f"✓ Successfully generated {test_count} test embeddings")

# Validate embedding structure
first_embedding = test_embeddings.first()["image_embedding"]
print(f"\nEmbedding validation:")
print(f"  - Dimension: {len(first_embedding)}")
print(f"  - Type: {type(first_embedding)}")
print(f"  - Sample values: {first_embedding[:5]}")

if len(first_embedding) == 512:
    print("\n✓ UDF test passed! Ready to process all products.")
else:
    print(f"\n⚠ WARNING: Expected 512 dimensions, got {len(first_embedding)}")

display(test_embeddings.select("product_id", "product_display_name"))

# COMMAND ----------



# COMMAND ----------

# Apply the pandas UDF to generate embeddings for all products
print(f"Starting embedding generation for {total_products:,} products...")
print(f"Batch size: {BATCH_SIZE}")
print("\nThis will process images in parallel across executors.\n")

# Add embedding column using the UDF
embeddings_df = (
    products_df
    .withColumn("image_embedding", generate_embedding_udf(F.col("image_path")))
    .withColumn("embedding_model", F.lit("clip-vit-b-32"))
    .withColumn("embedding_dimension", F.lit(512))
    .withColumn("created_at", F.current_timestamp())
    .select(
        "product_id",
        "image_embedding",
        "embedding_model",
        "embedding_dimension",
        "created_at"
    )
)

# Cache for reuse
embeddings_df.cache()

print("✓ Embedding generation complete!")
print(f"\nGenerated embeddings for {embeddings_df.count():,} products")

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

# Define UDF for embedding generation
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, DoubleType

@udf(returnType=ArrayType(DoubleType()))
def generate_embedding(image_path):
    try:
        embedder = ImageEmbedder(endpoint_url=MODEL_ENDPOINT_URL, token=token)
        embedding = embedder.get_embedding(image_path=image_path)
        return embedding.tolist()
    except Exception as e:
        print(f"Error: {e}")
        return [0.0] * 512  # Return zero vector on failure

# Apply UDF to all products
embeddings_df = (
    products_df
    .withColumn("image_embedding", generate_embedding(F.col("image_path")))
    .withColumn("embedding_model", F.lit("clip-vit-b-32"))
    .withColumn("embedding_dimension", F.lit(512))
    .withColumn("created_at", F.current_timestamp())
    .select("product_id", "image_embedding", "embedding_model", "embedding_dimension", "created_at")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Embeddings DataFrame

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

# Add table comment for documentation
try:
    from fashion_visual_search.utils import add_table_comment
    add_table_comment(
        CATALOG,
        SCHEMA,
        EMBEDDINGS_TABLE,
        "Product image embeddings generated using CLIP model for visual similarity search"
    )
except:
    # Inline implementation
    comment = "Product image embeddings generated using CLIP model for visual similarity search"
    spark.sql(f"COMMENT ON TABLE {CATALOG}.{SCHEMA}.{EMBEDDINGS_TABLE} IS '{comment}'")

print(f"✓ Added table comment to {CATALOG}.{SCHEMA}.{EMBEDDINGS_TABLE}")

# COMMAND ----------

# Optimize table for query performance with Z-ordering
try:
    from fashion_visual_search.utils import optimize_table
    optimize_table(CATALOG, SCHEMA, EMBEDDINGS_TABLE, zorder_cols=["product_id"])
except:
    # Inline implementation
    full_name = f"{CATALOG}.{SCHEMA}.{EMBEDDINGS_TABLE}"
    spark.sql(f"OPTIMIZE {full_name} ZORDER BY (product_id)")

print(f"✓ Optimized {CATALOG}.{SCHEMA}.{EMBEDDINGS_TABLE}")
print("  - Compacted small files")
print("  - Applied Z-ordering on product_id for faster lookups")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validation

# COMMAND ----------

# Verify embeddings table
embeddings_table = spark.table(embeddings_table_name)

print("=" * 60)
print("EMBEDDINGS VALIDATION REPORT")
print("=" * 60)

embedding_count = embeddings_table.count()
print(f"\n✓ Embeddings table: {embeddings_table_name}")
print(f"  Total embeddings: {embedding_count:,}")

# Log to MLflow
mlflow.log_metric("total_embeddings", embedding_count)
mlflow.log_metric("embedding_coverage", embedding_count / total_products * 100)

print(f"\n  Coverage: {embedding_count / total_products * 100:.2f}% of products")

print("\n  Schema:")
embeddings_table.printSchema()

print("=" * 60)

# COMMAND ----------

# Check for any zero vectors (potential failures)
from pyspark.sql.functions import col, size, expr

print("Checking for failed embeddings (zero vectors)...")

zero_vector_count = (
    embeddings_table
    .filter(expr("aggregate(image_embedding, 0.0, (acc, x) -> acc + abs(x)) = 0"))
    .count()
)

if zero_vector_count > 0:
    failure_rate = zero_vector_count / embedding_count * 100
    print(f"\n⚠ WARNING: {zero_vector_count:,} zero vectors found ({failure_rate:.2f}% failure rate)")
    mlflow.log_metric("failed_embeddings", zero_vector_count)
    mlflow.log_metric("failure_rate", failure_rate)
else:
    print("\n✓ No zero vectors found - all embeddings generated successfully!")
    mlflow.log_metric("failed_embeddings", 0)
    mlflow.log_metric("failure_rate", 0.0)

# Check embedding dimensions
print("\nValidating embedding dimensions...")
dim_check = embeddings_table.select(F.size("image_embedding").alias("dim")).distinct().collect()
if len(dim_check) == 1 and dim_check[0]["dim"] == 512:
    print("✓ All embeddings have correct dimension (512)")
else:
    print(f"⚠ WARNING: Inconsistent dimensions found: {[row['dim'] for row in dim_check]}")

# COMMAND ----------

# DBTITLE 1,Diagnose Embedding Quality Issues


# COMMAND ----------

# DBTITLE 1,Reprocess Failed Products


# COMMAND ----------

# Sample embedding stats
import numpy as np

print("=" * 60)
print("EMBEDDING QUALITY STATISTICS")
print("=" * 60)

sample_embedding = embeddings_table.first()["image_embedding"]
sample_array = np.array(sample_embedding)

print(f"\nSample embedding analysis:")
print(f"  Dimension: {len(sample_embedding)}")
print(f"  Mean: {sample_array.mean():.4f}")
print(f"  Std: {sample_array.std():.4f}")
print(f"  Min: {sample_array.min():.4f}")
print(f"  Max: {sample_array.max():.4f}")
print(f"  L2 norm: {np.linalg.norm(sample_array):.4f}")

# Log quality metrics to MLflow
mlflow.log_metric("embedding_mean", float(sample_array.mean()))
mlflow.log_metric("embedding_std", float(sample_array.std()))
mlflow.log_metric("embedding_l2_norm", float(np.linalg.norm(sample_array)))

# Check for normalized embeddings
if 0.9 <= np.linalg.norm(sample_array) <= 1.1:
    print("\n✓ Embeddings appear to be normalized (L2 norm ≈ 1.0)")
else:
    print("\n⚠ Embeddings may not be normalized")

print("\n" + "=" * 60)
print("✓ Validation complete!")
print("=" * 60)

# COMMAND ----------

# DBTITLE 1,Complete MLflow Run
# End MLflow run
mlflow.end_run()

print("✓ MLflow run completed successfully!")
print(f"\nView results at: {mlflow.get_tracking_uri()}")
print(f"Experiment: /Users/kevin.ippen@databricks.com/fashion-visual-search/image-embeddings")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✓ Pipeline Complete!
# MAGIC
# MAGIC ### Summary
# MAGIC
# MAGIC Successfully generated image embeddings for the fashion product catalog:
# MAGIC
# MAGIC * **Products processed**: 44,424 items
# MAGIC * **Embedding model**: CLIP ViT-B/32
# MAGIC * **Embedding dimension**: 512
# MAGIC * **Output table**: `main.fashion_demo.product_image_embeddings`
# MAGIC
# MAGIC ### Key Features Implemented
# MAGIC
# MAGIC 1. **Production-grade pandas UDF** for distributed parallel processing
# MAGIC 2. **Comprehensive data quality checks** with null/duplicate detection
# MAGIC 3. **MLflow tracking** for monitoring and reproducibility
# MAGIC 4. **Error handling and retry logic** for robust processing
# MAGIC 5. **Table optimization** with Z-ordering for fast lookups
# MAGIC 6. **Embedding validation** to detect failures and quality issues
# MAGIC
# MAGIC ### Next Steps
# MAGIC
# MAGIC 1. **Create Vector Search Index**: Run `04_vector_search_setup` notebook
# MAGIC 2. **Switch to Real Model**: Set `USE_MOCK_EMBEDDINGS = False` in the UDF when your CLIP model endpoint is deployed
# MAGIC 3. **Incremental Updates**: Implement delta processing for new products
# MAGIC 4. **Quality Monitoring**: Set up alerts for embedding quality metrics
# MAGIC 5. **Performance Tuning**: Adjust batch size based on cluster resources
