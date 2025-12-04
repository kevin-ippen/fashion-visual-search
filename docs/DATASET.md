%md
# Kaggle Fashion Dataset to Unity Catalog
# This notebook downloads the Fashion Product Images dataset from Kaggle and uploads it to Unity Catalog

# Step 1: Install required packages
%pip install kaggle opendatasets --quiet
dbutils.library.restartPython()

# Step 2: Configure Kaggle credentials
# You need to download your kaggle.json from https://www.kaggle.com/settings -> API -> Create New Token
# Then upload it to Databricks or set environment variables

import os
import json

# Option A: Upload kaggle.json to /Workspace/Users/<your-email>/ and uncomment:
# with open('/Workspace/Users/kevin.ippen@databricks.com/kaggle.json', 'r') as f:
#     kaggle_creds = json.load(f)
#     os.environ['KAGGLE_USERNAME'] = kaggle_creds['username']
#     os.environ['KAGGLE_KEY'] = kaggle_creds['key']

# Option B: Set credentials directly (not recommended for production)
# os.environ['KAGGLE_USERNAME'] = 'your_username'
# os.environ['KAGGLE_KEY'] = 'your_api_key'

# Option C: Use Databricks secrets (recommended)
try:
    os.environ['KAGGLE_USERNAME'] = dbutils.secrets.get(scope='kaggle', key='username')
    os.environ['KAGGLE_KEY'] = dbutils.secrets.get(scope='kaggle', key='api_key')
    print("✓ Kaggle credentials loaded from secrets")
except:
    print("⚠ Kaggle secrets not found. Please configure credentials using Option A or B above.")

# Step 3: Download dataset from Kaggle
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

print("Downloading Fashion Product Images dataset...")
dataset_name = 'paramaggarwal/fashion-product-images-dataset'
download_path = '/tmp/fashion-data'

api.dataset_download_files(dataset_name, path=download_path, unzip=True)
print(f"✓ Dataset downloaded to {download_path}")

# Step 4: Create Unity Catalog structure
spark.sql("CREATE CATALOG IF NOT EXISTS main")
spark.sql("CREATE SCHEMA IF NOT EXISTS main.fashion_demo")
spark.sql("""
    CREATE VOLUME IF NOT EXISTS main.fashion_demo.raw_data
""")
print("✓ Unity Catalog structure created: main.fashion_demo.raw_data")

# Step 5: Upload CSV to Unity Catalog Volume
import shutil

volume_path = '/Volumes/main/fashion_demo/raw_data'
csv_source = f'{download_path}/styles.csv'
csv_dest = f'{volume_path}/styles.csv'

dbutils.fs.cp(f'file:{csv_source}', csv_dest)
print(f"✓ CSV uploaded to {csv_dest}")

# Step 6: Upload images to Unity Catalog Volume
images_source = f'{download_path}/images'
images_dest = f'{volume_path}/images'

print("Uploading images (this may take several minutes for ~44k images)...")

# Copy images directory
import os
from pathlib import Path

image_files = list(Path(images_source).glob('*.jpg'))
print(f"Found {len(image_files)} images to upload")

# Create images directory in volume
dbutils.fs.mkdirs(images_dest)

# Upload in batches for better progress tracking
batch_size = 1000
for i in range(0, len(image_files), batch_size):
    batch = image_files[i:i+batch_size]
    for img_file in batch:
        src = f'file:{str(img_file)}'
        dst = f'{images_dest}/{img_file.name}'
        dbutils.fs.cp(src, dst)
    print(f"  Uploaded {min(i+batch_size, len(image_files))}/{len(image_files)} images")

print(f"✓ All images uploaded to {images_dest}")

# Step 7: Verify upload and create Delta table
print("\n=== Verification ===")

# Check CSV
df_styles = spark.read.csv(csv_dest, header=True, inferSchema=True)
print(f"✓ CSV loaded: {df_styles.count()} products")
print("\nSample data:")
display(df_styles.limit(5))

# Check images
image_count = len(dbutils.fs.ls(images_dest))
print(f"✓ Images in volume: {image_count} files")

# Create Delta table from CSV
df_styles.write.format('delta').mode('overwrite').saveAsTable('main.fashion_demo.products_raw')
print("\n✓ Delta table created: main.fashion_demo.products_raw")

print("\n=== Setup Complete ===")
print(f"CSV location: {csv_dest}")
print(f"Images location: {images_dest}")
print(f"Delta table: main.fashion_demo.products_raw")
print("\nNext steps: Run the preprocessing notebook to clean and enrich the data.")