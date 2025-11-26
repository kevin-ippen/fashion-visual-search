# Mosaic Fashion Visual Search & Recommender MVP

> **Status**: ✅ MVP Complete - Code ready for dataset integration and deployment

## 1. Objective

Build a **production-ready, Databricks-native** solution for fashion e-commerce that delivers:

1. **Visual similarity search** using CLIP embeddings and Mosaic AI Vector Search
2. **Personalized recommendations** combining visual similarity, user preferences, and business constraints
3. **AI stylist agent** powered by Claude for natural language outfit suggestions and product discovery

The system provides modular, reusable components that can be adapted to any fashion retailer's catalog and customer data.

### What's Implemented

- ✅ Complete data pipeline (8 production-ready notebooks)
- ✅ Python package with embedding, recommendation, and Vector Search utilities
- ✅ Comprehensive unit test suite
- ✅ Synthetic data generation for testing
- ✅ Claude AI agent with tool integration
- ✅ Streamlit + Gradio UI options
- ✅ Documentation (Architecture, Dataset Guide, Deployment)
- ✅ Databricks Asset Bundles configuration

---

## 2. High-Level Architecture

### Data Layer (Unity Catalog)

Schema example: `main.fashion_demo`

Core tables:

- `products`
  - Product metadata and image paths.
- `users`
  - User identities and optional segments.
- `transactions`
  - User–product interaction history (views, add-to-cart, purchases).
- `user_style_features` (derived)
  - Aggregated preferences and optional user embeddings.
- `product_image_embeddings` (derived)
  - Embeddings for each product image.

All tables stored as **Delta**; governed in **Unity Catalog**.

### Embeddings & Vector Search

- Image encoder model (CLIP-like) served via **Databricks Model Serving (GPU)**.
- Batch/streaming pipeline to:
  - Read `products`
  - Generate `image_embedding`
  - Write to `product_image_embeddings`.

- **Mosaic AI Vector Search** index:
  - Primary key: `product_id`
  - Embedding column: `image_embedding`
  - Metric: cosine / L2
  - Auto-sync from Delta.

### Recommendation & Personalization

- Feature engineering from `transactions`:
  - Product popularity, co-purchase stats.
  - User preferences by category/brand/color/price band.
  - Optional `user_embedding` from historically interacted product embeddings.

- Scoring function components:
  - `sim_visual`: similarity between query image and product embeddings.
  - `sim_user`: similarity between `user_embedding` and product embeddings.
  - `attr_score`: heuristic score based on attribute matches and constraints.

- Final score:
  - `score = w_visual * sim_visual + w_user * sim_user + w_attr * attr_score`
  - Weights configurable via a parameter table.

### Orchestration & UX

- **Claude-based Stylist Agent** (Claude Code / Claude 3.x) with tools:
  - `search_similar_by_image(image)`
  - `get_personalized_recs(user_id, candidate_products, budget, style_constraints)`
  - `complete_the_look(product_id, user_id)`

- **Databricks App / Streamlit UI**:
  - Upload an image or pick a product.
  - Optional user selector (for demo).
  - Display:
    - Visually similar products
    - Personalized ranking for a given user
    - Outfit completion suggestions.

---

## 3. Datasets

### 3.1 Product Catalog + Images

For the MVP, we’ll use an open fashion dataset such as:

- **Fashion Product Images Dataset** (approx. 44k images)
  - Professionally shot, high-resolution product images.
  - Metadata: brand, category, age group, season, usage, text description.
  - Source: Kaggle / Labelbox.

We’ll load:

- Product-level CSV/JSON into `products`.
- Images into a UC Volume or external object store; store URIs in `products.image_path`.

### 3.2 User and Transaction Data

Option A: Ingest an open fashion user dataset with interactions.

Option B (default for MVP): **Generate synthetic user and transaction data**:

- `users`: e.g. 10k synthetic users with basic attributes or segments.
- `transactions`: sampled from `products` following simple behavior rules:
  - Users prefer some categories, brands, colors, and price ranges.
  - Label events as `view`, `add_to_cart`, or `purchase`.

Synthetic data should be reproducible via a Databricks notebook.

---

## 4. Components & Notebooks

Recommended notebook set (can be converted into jobs or DLT pipelines):

1. **01_ingest_products**
   - Download / mount the product dataset.
   - Create `products` Delta table in UC.

2. **02_generate_synthetic_users_transactions**
   - Create `users` and `transactions`.
   - Document synthetic behavior assumptions.

3. **03_image_embeddings_pipeline**
   - Set up image encoder model endpoint (Model Serving).
   - Generate embeddings for each `product_id`.
   - Write to `product_image_embeddings`.

4. **04_vector_search_setup**
   - Create Mosaic AI Vector Search endpoint.
   - Create index on `product_image_embeddings`.
   - Test nearest-neighbor queries from a sample product image.

5. **05_user_style_features**
   - Build `user_style_features` with:
     - Preference distributions (category, brand, color, price band).
     - Optional user embedding as mean of historically interacted product embeddings.

6. **06_recommendation_scoring**
   - Implement the scoring function:
     - Visual similarity + user embedding + attribute-based heuristics.
   - Return ranked recommendation list for:
     - (a) purely visual search
     - (b) visual + personalization.

7. **07_claude_stylist_agent**
   - Configure Claude via AI Gateway or external call.
   - Define tools:
     - `search_similar_by_image`
     - `get_personalized_recs`
     - `complete_the_look`
   - Implement an agent loop that:
     - Parses user natural-language requests.
     - Calls the appropriate tools.
     - Returns curated recommendations and explanations.

8. **08_app_ui (Databricks App / Streamlit)**
   - Build a simple interface:
     - Image uploader.
     - User selector dropdown (for demo).
     - Display of results in panels:
       - Visually similar items
       - Personalized ranking
       - “Complete the look” suggestions.

---

## 5. Setup Steps (Initial Checklist)

1. **Create UC catalog & schema**
   - Example: `CREATE CATALOG main` (if not present).
   - `CREATE SCHEMA main.fashion_demo;`

2. **Provision Databricks compute**
   - One GPU cluster or Serverless GPU endpoint for model serving.
   - A standard warehouse or SQL endpoint for interactive queries.

3. **Configure Model Serving**
   - Deploy an image encoder (CLIP-like) as a serving endpoint.
   - Confirm it returns fixed-length embedding vectors from an image URL or bytes.

4. **Enable Mosaic AI Vector Search**
   - Create a vector search endpoint.
   - Configure index over `product_image_embeddings`.

5. **Configure Claude access**
   - Set up Anthropic/Claude endpoint via AI Gateway or workspace secrets.
   - Confirm you can call Claude from a notebook (for code/agent orchestration).

6. **Run notebooks 01–08 in sequence**
   - Ingest data.
   - Generate synthetic users/transactions.
   - Generate embeddings.
   - Build vector index.
   - Compute user style features.
   - Test recommendation scoring.
   - Deploy the stylist agent and UI.

---

## 6. Next Extensions

Once the MVP is working:

- Add **multi-item “shop the look” embedding** for outfits.
- Introduce **multi-modal retrieval** (image + text query).
- Capture **online feedback** (clicks in the app) and feed back into `transactions`.
- Integrate a **GenAI layer** to generate styling tips and product descriptions.
- Explore **A/B testing** of visual-only vs visual+personalized ranking.

---
