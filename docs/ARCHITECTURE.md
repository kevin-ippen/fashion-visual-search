# Architecture Documentation

## System Overview

The Fashion Visual Search & Recommender system is built on Databricks with three core components:

1. **Data Layer**: Unity Catalog with Delta Lake tables
2. **ML Layer**: CLIP embeddings + Mosaic AI Vector Search
3. **Application Layer**: Recommendation engine + Claude AI agent

## Component Architecture

### 1. Data Layer (Unity Catalog)

All data stored in Unity Catalog using Delta Lake format for ACID compliance and time travel.

#### Tables

**`main.fashion_demo.products`**
```sql
CREATE TABLE main.fashion_demo.products (
    product_id STRING NOT NULL,
    display_name STRING,
    category STRING,
    subcategory STRING,
    article_type STRING,
    brand STRING,
    color STRING,
    gender STRING,
    season STRING,
    year INT,
    usage STRING,
    price DOUBLE,
    image_path STRING,
    created_at TIMESTAMP
) USING DELTA;
```

**`main.fashion_demo.users`**
- Synthetic user profiles for testing
- Fields: user_id, segment, created_date, preferred_categories, avg_price_point

**`main.fashion_demo.transactions`**
- User-product interactions (views, add-to-cart, purchases)
- Fields: transaction_id, user_id, product_id, event_type, timestamp, purchase_amount

**`main.fashion_demo.product_image_embeddings`**
- CLIP embeddings for all products
- Fields: product_id, image_embedding (ARRAY<DOUBLE>), embedding_model, embedding_dimension, created_at

**`main.fashion_demo.user_style_features`**
- Aggregated user preferences and embeddings
- Fields: user_id, segment, category_prefs (MAP), brand_prefs (MAP), color_prefs (MAP), price stats, user_embedding

### 2. ML Layer

#### Image Embeddings (CLIP)

- **Model**: OpenAI CLIP (ViT-B/32 or ViT-L/14)
- **Embedding Dimension**: 512 (ViT-B/32) or 768 (ViT-L/14)
- **Deployment**: Databricks Model Serving (GPU endpoint)
- **Processing**: Batch processing via PySpark UDFs

```python
# Embedding generation
image → CLIP encoder → 512-dim vector → normalize → store in Delta
```

#### Vector Search Index

**Configuration:**
```python
{
    "endpoint_name": "fashion_vector_search",
    "index_name": "main.fashion_demo.product_embeddings_index",
    "source_table": "main.fashion_demo.product_image_embeddings",
    "primary_key": "product_id",
    "embedding_dimension": 512,
    "embedding_vector_column": "image_embedding",
    "distance_metric": "COSINE",
    "pipeline_type": "TRIGGERED"  # or CONTINUOUS for real-time
}
```

**Query Flow:**
```
Query image → Embedding → Vector Search (ANN) → Top-K similar products
```

**Performance:**
- Query latency: 50-100ms for top-10 results
- Index build time: ~10-30 minutes for 50k products
- Scalability: Millions of vectors

#### User Embedding Generation

User embeddings are computed from interaction history:

```python
user_embedding = mean([
    product_embedding_1,
    product_embedding_2,
    ...
]) weighted by event_type
```

Weighting scheme:
- purchase: 3.0
- add_to_cart: 2.0
- view: 1.0

### 3. Recommendation Engine

#### Scoring Function

Final score is a weighted combination of three signals:

```python
final_score = w_visual * sim_visual + w_user * sim_user + w_attr * attr_score

where:
- sim_visual: cosine similarity between query and product embeddings
- sim_user: cosine similarity between user and product embeddings
- attr_score: heuristic score from user preferences
- w_visual, w_user, w_attr: configurable weights (sum to 1.0)
```

#### Default Weights

```python
{
    "visual": 0.5,      # Visual similarity
    "user": 0.3,        # User preference
    "attribute": 0.2    # Attribute matching
}
```

#### Attribute Scoring

Considers:
1. Category preference (from purchase history)
2. Brand preference
3. Color preference
4. Price compatibility
   - Hard constraint: budget limit
   - Soft preference: user's historical price range

#### Re-ranking Pipeline

```
1. Vector Search → Top-N candidates (N=50-100)
2. Load product metadata
3. Score each candidate:
   - Visual similarity (from Vector Search)
   - User similarity (if user context available)
   - Attribute score (preferences + constraints)
4. Compute final weighted score
5. Sort by final_score descending
6. Apply diversification (optional)
7. Return Top-K (K=10-20)
```

### 4. Claude AI Stylist Agent

#### Architecture

```
User Query → Claude → Tool Selection → Tool Execution → Claude → Response
```

#### Available Tools

1. **search_similar_by_image**
   - Input: product_id, num_results
   - Output: Visually similar products
   - Use case: "Show me items like this dress"

2. **get_personalized_recommendations**
   - Input: user_id, query_product_id, budget (optional), num_results
   - Output: Personalized ranked recommendations
   - Use case: "Recommend shirts for my style under $50"

3. **complete_the_look**
   - Input: product_id, user_id (optional), num_results
   - Output: Complementary products
   - Use case: "What shoes go with these pants?"

#### Integration

- **AI Gateway**: Claude accessed via Databricks AI Gateway
- **Route Configuration**: Anthropic Claude Sonnet
- **Token Budget**: ~2000 tokens per response
- **Conversation History**: Stored in session state (UI) or database (production)

### 5. Application Layer

#### Streamlit App

**Pages:**
1. **Search Tab**: Visual similarity search with filters
2. **AI Stylist Tab**: Chat interface with Claude agent
3. **Analytics Tab**: System metrics and insights

**State Management:**
- Session state for user selections
- Cache for expensive operations (data loads, embeddings)
- Real-time updates from Vector Search

#### Deployment Options

1. **Databricks Apps**: Native deployment on Databricks
2. **External Hosting**: Docker container with Databricks SDK
3. **Notebook Integration**: Gradio embedded in notebooks

## Data Flow Diagrams

### Ingestion Pipeline

```
Kaggle Dataset
    ↓
  CSV + Images
    ↓
Notebook 01: Ingest → products table (Delta)
    ↓
Notebook 02: Generate → users + transactions tables
    ↓
Notebook 03: Embeddings → product_image_embeddings table
    ↓
Notebook 04: Vector Search → Index creation
    ↓
Notebook 05: Features → user_style_features table
```

### Recommendation Request Flow

```
User Query (product_id + user_id)
    ↓
Get product embedding → Vector Search query
    ↓
Top-N candidates retrieved
    ↓
Load user profile → user_style_features
    ↓
Recommendation Scorer
    ├─ Visual similarity (from Vector Search)
    ├─ User embedding similarity
    └─ Attribute score (preferences + constraints)
    ↓
Weighted scoring → final_score
    ↓
Sort + Diversify
    ↓
Top-K Results → User
```

## Scalability Considerations

### Horizontal Scaling

- **Vector Search**: Auto-scales with index size
- **Delta Tables**: Distributed storage via cloud object store
- **Spark Processing**: Distributed computation for embeddings
- **Model Serving**: Auto-scaling GPU endpoints

### Vertical Scaling

- **Embedding Generation**: Use larger GPU instances for faster processing
- **Vector Search**: Increase endpoint size for higher QPS
- **Scoring**: In-memory computation, scales with cluster size

### Performance Optimization

1. **Z-ORDER**: Optimize Delta tables by key columns
2. **Caching**: Cache frequently accessed data
3. **Partitioning**: Partition large tables by category/date
4. **Batch Processing**: Process embeddings in batches
5. **Connection Pooling**: Reuse database connections

## Security & Governance

### Unity Catalog Governance

- **Three-level namespace**: catalog.schema.table
- **Access Control**: Row/column level security
- **Data Lineage**: Automatic tracking
- **Audit Logs**: All access logged

### Secrets Management

```python
# Using Databricks secrets
token = dbutils.secrets.get(scope="production", key="databricks-token")
claude_key = dbutils.secrets.get(scope="production", key="anthropic-key")
```

### Data Privacy

- **PII Handling**: Tag sensitive columns
- **Sanitization**: Remove PII from logs
- **Encryption**: At-rest and in-transit
- **Access Controls**: Principle of least privilege

## Monitoring & Observability

### Metrics to Track

1. **System Metrics**:
   - Vector Search query latency
   - Recommendation scoring time
   - Model serving response time
   - End-to-end latency

2. **Business Metrics**:
   - Click-through rate (CTR)
   - Conversion rate
   - Average order value (AOV)
   - Recommendation diversity

3. **Data Quality**:
   - Null rates in critical columns
   - Duplicate rates
   - Data freshness
   - Embedding quality

### Logging

```python
from fashion_visual_search.utils import log_job_metrics

log_job_metrics(
    job_name="embedding_pipeline",
    metrics={
        "products_processed": 44000,
        "embeddings_generated": 43950,
        "failures": 50,
        "duration_minutes": 45
    }
)
```

## Future Enhancements

1. **Multi-modal Search**: Text + image queries
2. **Real-time Updates**: Continuous Vector Search sync
3. **A/B Testing**: Experiment framework for scoring weights
4. **Feedback Loop**: Incorporate user clicks/purchases
5. **Advanced Features**:
   - Outfit embeddings (multi-item)
   - Seasonal trend detection
   - Inventory-aware recommendations
   - Dynamic pricing integration
