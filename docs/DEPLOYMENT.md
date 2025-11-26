# Deployment Guide

## Pre-Deployment Checklist

### Infrastructure

- [ ] Azure Databricks workspace with Unity Catalog enabled
- [ ] Mosaic AI Vector Search enabled
- [ ] AI Gateway configured with Claude route
- [ ] Shared compute cluster available OR Serverless enabled
- [ ] Unity Catalog schema created: `main.fashion_demo`

### Data

- [ ] Fashion Product Images dataset downloaded from Kaggle
- [ ] CSV uploaded to Unity Catalog Volume
- [ ] Images uploaded to Volume or external storage
- [ ] File paths updated in notebook 01

### Configuration

- [ ] Repo cloned/synced to Databricks Repos
- [ ] Python package paths updated in notebooks
- [ ] Model Serving endpoint URL configured (notebook 03)
- [ ] AI Gateway route name set (notebook 07)
- [ ] Cluster ID updated in `databricks.yml`

## Deployment Steps

### 1. Initial Setup (One-time)

#### Create Unity Catalog Schema

```sql
-- Run in Databricks SQL or notebook
CREATE SCHEMA IF NOT EXISTS main.fashion_demo
COMMENT 'Fashion visual search and recommendation system';

-- Create config table for scoring weights
CREATE TABLE IF NOT EXISTS main.fashion_demo.config (
    config_key STRING NOT NULL,
    config_value STRING,
    updated_at TIMESTAMP,
    CONSTRAINT config_pk PRIMARY KEY (config_key)
) USING DELTA;

-- Create volume for raw data
CREATE VOLUME IF NOT EXISTS main.fashion_demo.raw_data
COMMENT 'Raw fashion product data and images';
```

#### Upload Dataset

See [DATASET.md](DATASET.md) for detailed instructions.

```bash
# Using Databricks CLI
databricks fs cp styles.csv dbfs:/Volumes/main/fashion_demo/raw_data/styles.csv
databricks fs cp -r images/ dbfs:/Volumes/main/fashion_demo/raw_data/images/
```

#### Configure Secrets

```bash
# Create secret scope
databricks secrets create-scope fashion-demo

# Add secrets
databricks secrets put-secret fashion-demo databricks-token
databricks secrets put-secret fashion-demo anthropic-api-key
```

### 2. Data Pipeline Execution

Run notebooks in sequence:

#### Notebook 01: Ingest Products
```bash
databricks runs submit --json '{
    "run_name": "Ingest Products",
    "notebook_task": {
        "notebook_path": "/Repos/<user>/fashion-visual-search/notebooks/01_ingest_products"
    },
    "existing_cluster_id": "0304-162117-qgsi1x04"
}'
```

**Validation**:
```sql
SELECT COUNT(*) FROM main.fashion_demo.products;
-- Expected: ~44,000 products
```

#### Notebook 02: Generate Synthetic Data
```bash
databricks runs submit --json '{
    "run_name": "Generate Synthetic Data",
    "notebook_task": {
        "notebook_path": "/Repos/<user>/fashion-visual-search/notebooks/02_generate_synthetic_users_transactions"
    },
    "existing_cluster_id": "0304-162117-qgsi1x04"
}'
```

**Validation**:
```sql
SELECT COUNT(*) FROM main.fashion_demo.users;      -- Expected: 10,000
SELECT COUNT(*) FROM main.fashion_demo.transactions; -- Expected: 100,000+
```

#### Notebook 03: Generate Embeddings

**Option A: With Model Serving (Production)**

1. Deploy CLIP model to Model Serving:
   ```python
   # Use Databricks Foundation Model APIs
   # Or deploy custom CLIP model
   ```

2. Update endpoint URL in notebook

3. Run notebook to generate embeddings

**Option B: With Placeholder Embeddings (MVP Testing)**

Run notebook as-is to generate synthetic embeddings for testing.

**Validation**:
```sql
SELECT COUNT(*) FROM main.fashion_demo.product_image_embeddings;
-- Should match products count
```

#### Notebook 04: Create Vector Search Index

**Important**: This step may take 10-30 minutes for initial indexing.

```python
# Run notebook interactively to monitor progress
# Index status can be checked with:
index.describe()
```

**Validation**:
- Index state: ONLINE_TRIGGERED_UPDATE or ONLINE_CONTINUOUS_UPDATE
- Indexed rows: Should match embeddings count

#### Notebook 05: User Style Features

```bash
databricks runs submit --json '{
    "run_name": "Compute User Features",
    "notebook_task": {
        "notebook_path": "/Repos/<user>/fashion-visual-search/notebooks/05_user_style_features"
    },
    "existing_cluster_id": "0304-162117-qgsi1x04"
}'
```

**Validation**:
```sql
SELECT COUNT(*) FROM main.fashion_demo.user_style_features;
SELECT COUNT(*) FROM main.fashion_demo.user_style_features WHERE user_embedding IS NOT NULL;
```

#### Notebook 06-07: Testing

Run notebooks 06 and 07 interactively to validate:
- Recommendation scoring works
- Claude agent integration works
- All tools function correctly

### 3. Application Deployment

#### Option A: Databricks App (Recommended)

1. Create app directory structure:
   ```
   fashion-visual-search/
   └── app/
       ├── app.py          # Streamlit app code
       └── app.yaml        # App configuration
   ```

2. Create `app/app.yaml`:
   ```yaml
   command: ["streamlit", "run", "app.py", "--server.port", "8080"]
   env:
     - name: CATALOG
       value: "main"
     - name: SCHEMA
       value: "fashion_demo"
     - name: VECTOR_SEARCH_ENDPOINT
       value: "fashion_vector_search"
   ```

3. Deploy via UI:
   - Navigate to Apps in Databricks workspace
   - Click "Create App"
   - Select the app directory
   - Click "Deploy"

4. Or deploy via CLI:
   ```bash
   databricks apps create fashion-visual-search --source-code-path ./app
   ```

#### Option B: External Hosting

1. Build Docker image:
   ```dockerfile
   FROM python:3.10-slim
   WORKDIR /app
   COPY . .
   RUN pip install -e .
   CMD ["streamlit", "run", "app/app.py"]
   ```

2. Deploy to cloud platform (Azure Container Instances, AWS ECS, etc.)

### 4. Production Configuration

#### Enable Continuous Vector Search Sync

Update Vector Search index to continuous mode:

```python
# In notebook 04 or via API
vsc.create_delta_sync_index(
    ...
    pipeline_type="CONTINUOUS"  # Changed from TRIGGERED
)
```

#### Set Up Scheduled Jobs

Using Databricks Asset Bundles:

```yaml
# databricks.yml
resources:
  jobs:
    daily_user_features_refresh:
      name: "[PROD] Daily User Features Refresh"
      schedule:
        quartz_cron_expression: "0 0 2 * * ?"  # 2 AM daily
        timezone_id: "America/Los_Angeles"
      tasks:
        - task_key: refresh_user_features
          notebook_task:
            notebook_path: notebooks/05_user_style_features
          existing_cluster_id: "0304-162117-qgsi1x04"
```

Deploy:
```bash
databricks bundle deploy --target prod
```

#### Configure Monitoring

```python
# Add to notebooks
from fashion_visual_search.utils import log_job_metrics

log_job_metrics(
    job_name="embedding_generation",
    metrics={
        "products_processed": count,
        "duration_seconds": duration,
        "success_rate": success_rate
    }
)
```

Query metrics:
```sql
SELECT *
FROM main.fashion_demo.job_metrics
WHERE job_name = 'embedding_generation'
ORDER BY timestamp DESC
LIMIT 10;
```

#### Set Up Alerts

Create SQL alerts in Databricks SQL:

```sql
-- Alert: Low Vector Search index freshness
SELECT CURRENT_TIMESTAMP() as check_time,
       DATEDIFF(hour, MAX(created_at), CURRENT_TIMESTAMP()) as hours_since_last_update
FROM main.fashion_demo.product_image_embeddings;
-- Alert if > 24 hours

-- Alert: High error rate
SELECT COUNT(*) as failed_embeddings
FROM main.fashion_demo.product_image_embeddings
WHERE array_size(image_embedding) = 0 OR image_embedding IS NULL;
-- Alert if > 100
```

### 5. Performance Optimization

#### Optimize Delta Tables

Run regularly (weekly):

```sql
-- Optimize and Z-ORDER
OPTIMIZE main.fashion_demo.products ZORDER BY (category, brand);
OPTIMIZE main.fashion_demo.product_image_embeddings ZORDER BY (product_id);
OPTIMIZE main.fashion_demo.transactions ZORDER BY (user_id, timestamp);
OPTIMIZE main.fashion_demo.user_style_features ZORDER BY (user_id);

-- Vacuum old files (careful in production!)
VACUUM main.fashion_demo.products RETAIN 168 HOURS;
```

#### Tune Vector Search

- Monitor query latency
- Adjust endpoint size if needed
- Consider partitioning for very large indexes

#### Application Performance

- Enable caching in Streamlit app
- Use connection pooling
- Implement request throttling
- Add CDN for image serving

### 6. Security Hardening

#### Access Controls

```sql
-- Grant read access to app service principal
GRANT SELECT ON SCHEMA main.fashion_demo TO `app-service-principal`;

-- Grant write access to ETL jobs only
GRANT ALL PRIVILEGES ON SCHEMA main.fashion_demo TO `etl-service-principal`;

-- Deny access to sensitive tables
REVOKE ALL PRIVILEGES ON main.fashion_demo.users FROM `public`;
```

#### Audit Logging

Enable Unity Catalog audit logs:
- Monitor all data access
- Track model serving usage
- Log Vector Search queries

#### Secrets Rotation

```bash
# Rotate API keys regularly
databricks secrets put-secret fashion-demo anthropic-api-key --string-value "new-key"

# Rotate Databricks tokens
databricks tokens create --comment "fashion-app-prod" --lifetime-seconds 7776000  # 90 days
```

## Rollback Procedures

### Rollback Data Pipeline

```sql
-- Restore previous version using Delta time travel
RESTORE TABLE main.fashion_demo.products TO VERSION AS OF 10;
RESTORE TABLE main.fashion_demo.user_style_features TO TIMESTAMP AS OF '2024-01-01';
```

### Rollback Vector Search Index

```python
# Trigger sync to rebuild from source table
vsc.get_index(index_name="...").sync()
```

### Rollback Application

```bash
# Revert to previous app version
databricks apps update fashion-visual-search --source-code-path ./app-v1.0
```

## Troubleshooting

### Vector Search Index Stuck

```python
# Check status
index = vsc.get_index(endpoint_name="...", index_name="...")
print(index.describe())

# Force sync
index.sync()

# If still stuck, recreate index (last resort)
# Note: This will cause downtime
```

### Model Serving Timeout

- Check endpoint status in Model Serving UI
- Increase timeout in client code
- Scale up endpoint if under high load
- Check model logs for errors

### App Performance Issues

- Check Streamlit logs
- Monitor database connection pool
- Review query performance
- Scale up app resources

## Post-Deployment

- [ ] Smoke test all features
- [ ] Verify metrics are being logged
- [ ] Set up alerts and dashboards
- [ ] Document any customizations
- [ ] Train users on the system
- [ ] Schedule regular maintenance windows

## Support

For issues during deployment:
1. Check logs in Databricks workspace
2. Review [ARCHITECTURE.md](ARCHITECTURE.md) for system details
3. Open GitHub issue with details
