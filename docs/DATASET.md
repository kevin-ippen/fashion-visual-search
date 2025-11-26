# Dataset Guide

## Fashion Product Images Dataset

### Overview

This project uses the **Fashion Product Images Dataset** from Kaggle, which contains approximately 44,000 high-resolution product images with rich metadata.

- **Source**: [Kaggle - Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset)
- **Size**: ~15 GB (images) + CSV metadata
- **License**: Check Kaggle for current license terms

### Dataset Structure

```
fashion-product-images/
├── styles.csv          # Product metadata
└── images/             # Product images
    ├── 1163.jpg
    ├── 1164.jpg
    └── ...
```

### Metadata Schema

The `styles.csv` file contains the following columns:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `id` | Integer | Unique product ID | 15970 |
| `gender` | String | Target gender | "Men", "Women", "Boys", "Girls", "Unisex" |
| `masterCategory` | String | Top-level category | "Apparel", "Accessories", "Footwear" |
| `subCategory` | String | Sub-category | "Topwear", "Bottomwear", "Watches" |
| `articleType` | String | Specific article type | "Shirts", "Jeans", "Watches" |
| `baseColour` | String | Primary color | "Navy Blue", "Black", "White" |
| `season` | String | Season | "Fall", "Summer", "Winter", "Spring" |
| `year` | Integer | Year of production | 2011-2017 |
| `usage` | String | Usage type | "Casual", "Formal", "Sports" |
| `productDisplayName` | String | Full product name | "Turtle Check Men Navy Blue Shirt" |

### Download Instructions

#### Method 1: Kaggle Website

1. Go to https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset
2. Click "Download" (requires Kaggle account)
3. Extract the ZIP file
4. You'll have `styles.csv` and an `images/` directory

#### Method 2: Kaggle API

```bash
# Install Kaggle CLI
pip install kaggle

# Set up Kaggle credentials (~/.kaggle/kaggle.json)
# Download from: https://www.kaggle.com/settings -> API -> Create New Token

# Download dataset
kaggle datasets download -d paramaggarwal/fashion-product-images-dataset

# Extract
unzip fashion-product-images-dataset.zip -d fashion-data/
```

### Upload to Databricks

#### Option A: Databricks UI

1. Navigate to Data → Volumes in Databricks UI
2. Create volume: `main.fashion_demo.raw_data`
3. Upload `styles.csv` and `images/` folder

#### Option B: Databricks CLI

```bash
# Upload CSV
databricks fs cp styles.csv dbfs:/Volumes/main/fashion_demo/raw_data/styles.csv

# Upload images (this may take a while)
databricks fs cp -r images/ dbfs:/Volumes/main/fashion_demo/raw_data/images/
```

#### Option C: Cloud Storage + External Location

For large datasets, consider using cloud storage:

```sql
-- Azure
CREATE EXTERNAL LOCATION fashion_data_location
    URL 'abfss://container@storageaccount.dfs.core.windows.net/fashion-data'
    WITH (STORAGE CREDENTIAL azure_credential);

-- Then create volume pointing to external location
```

### Data Quality Notes

1. **Missing Images**: Some product IDs in CSV may not have corresponding images
2. **Duplicates**: Some products may have multiple color variants
3. **Inconsistent Naming**: Category names may have variations in capitalization
4. **Price Information**: Original dataset doesn't include prices (we generate synthetic prices)

### Preprocessing Steps

The notebook `01_ingest_products.py` handles:

1. Loading CSV with proper schema
2. Renaming columns to standard format
3. Generating synthetic prices based on category
4. Creating image path references
5. Data quality checks (nulls, duplicates)
6. Writing to Delta table

### Sample Data

For quick testing without downloading the full dataset, we provide sample data:

```python
# Generate sample data for testing
from fashion_visual_search.data_generation import SyntheticDataGenerator

generator = SyntheticDataGenerator(seed=42)
users = generator.generate_users(num_users=100)
# ... etc
```

## Alternative Datasets

If the Kaggle dataset is unavailable, consider:

1. **DeepFashion**: http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html
   - Larger dataset with more annotations
   - Requires academic access

2. **Fashion-MNIST**: https://github.com/zalandoresearch/fashion-mnist
   - Smaller, simpler dataset
   - Good for prototyping

3. **Your Own Data**:
   - Adapt `01_ingest_products.py` for your schema
   - Ensure you have: product_id, category, image_path

## Data Storage Best Practices

1. **Use Delta Lake**: All tables stored as Delta for ACID compliance
2. **Unity Catalog**: Leverage three-level namespace (catalog.schema.table)
3. **Optimize Tables**: Run OPTIMIZE regularly with Z-ORDER on key columns
4. **Volume Storage**: Use UC Volumes for image files
5. **Partitioning**: For very large catalogs, consider partitioning by category

## Privacy & Compliance

- This is public dataset from Kaggle - check license terms
- For production with real customer data:
  - Implement data governance policies
  - Tag PII columns appropriately
  - Use Unity Catalog access controls
  - Comply with GDPR/CCPA requirements
  - Sanitize logs to remove PII

## Data Updates

For production systems:

1. **Incremental Ingestion**: Use Delta MERGE for updates
2. **Change Data Capture**: Enable Delta CDC if needed
3. **Vector Index Sync**: Ensure Vector Search index stays in sync
4. **User Features Refresh**: Schedule regular updates of user style features

```python
# Example incremental update
new_products_df = spark.read.csv("new_products.csv")

(spark.table("main.fashion_demo.products")
    .merge(new_products_df, "product_id")
    .whenMatchedUpdateAll()
    .whenNotMatchedInsertAll()
    .execute())

# Trigger vector index sync
vsc.get_index(index_name="...").sync()
```
