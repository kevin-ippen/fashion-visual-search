# Databricks notebook source
# MAGIC %md
# MAGIC # 07 - Claude Stylist Agent
# MAGIC
# MAGIC This notebook implements a Claude-powered AI stylist agent that can:
# MAGIC - Search for visually similar products
# MAGIC - Provide personalized recommendations
# MAGIC - Complete outfit suggestions
# MAGIC - Answer natural language queries about products
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - All previous notebooks completed
# MAGIC - Claude API access via AI Gateway configured
# MAGIC
# MAGIC **Output:**
# MAGIC - Reusable stylist agent functions
# MAGIC - Example conversations

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Configuration
CATALOG = "main"
SCHEMA = "fashion_demo"
PRODUCTS_TABLE = "products"
EMBEDDINGS_TABLE = "product_image_embeddings"
USER_FEATURES_TABLE = "user_style_features"

# Vector Search
VECTOR_SEARCH_ENDPOINT = "fashion_vector_search"
INDEX_NAME = f"{CATALOG}.{SCHEMA}.product_embeddings_index"

# Claude Configuration (via AI Gateway)
# Get AI Gateway route - Update with your actual route
CLAUDE_ROUTE = "anthropic-claude-sonnet"  # Your AI Gateway route name

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from pyspark.sql import functions as F
import sys
import json
import numpy as np
from typing import List, Dict, Any, Optional

sys.path.append("/Workspace/Repos/.../fashion-visual-search/src")  # Update path

from fashion_visual_search.recommendation import (
    RecommendationScorer,
    ProductCandidate,
    UserProfile,
    ScoringWeights
)

# COMMAND ----------

# Initialize clients
vsc = VectorSearchClient()
vs_index = vsc.get_index(
    endpoint_name=VECTOR_SEARCH_ENDPOINT,
    index_name=INDEX_NAME
)

# COMMAND ----------

# Load tables
products_df = spark.table(f"{CATALOG}.{SCHEMA}.{PRODUCTS_TABLE}")
embeddings_df = spark.table(f"{CATALOG}.{SCHEMA}.{EMBEDDINGS_TABLE}")
user_features_df = spark.table(f"{CATALOG}.{SCHEMA}.{USER_FEATURES_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tool Definitions

# COMMAND ----------

def search_similar_by_image(product_id: str, num_results: int = 10) -> List[Dict[str, Any]]:
    """
    Search for visually similar products given a product ID.

    Args:
        product_id: ID of the query product
        num_results: Number of similar products to return

    Returns:
        List of similar products with metadata
    """
    # Get product embedding
    product_emb = embeddings_df.filter(F.col("product_id") == product_id).first()

    if not product_emb:
        return []

    query_embedding = product_emb["image_embedding"]

    # Search
    vs_results = vs_index.similarity_search(
        query_vector=query_embedding,
        columns=["product_id"],
        num_results=num_results
    )

    if not vs_results or "result" not in vs_results:
        return []

    # Enrich with product details
    result_ids = [r["product_id"] for r in vs_results["result"]["data_array"]]
    scores = {r["product_id"]: r["score"] for r in vs_results["result"]["data_array"]}

    results_df = products_df.filter(F.col("product_id").isin(result_ids)).collect()

    results = []
    for row in results_df:
        results.append({
            "product_id": row["product_id"],
            "name": row["display_name"],
            "category": row["category"],
            "brand": row["brand"],
            "color": row["color"],
            "price": float(row["price"]),
            "similarity_score": float(scores.get(row["product_id"], 0.0))
        })

    # Sort by similarity
    results.sort(key=lambda x: x["similarity_score"], reverse=True)

    return results


def get_personalized_recommendations(
    user_id: str,
    query_product_id: str,
    budget: Optional[float] = None,
    num_results: int = 10
) -> List[Dict[str, Any]]:
    """
    Get personalized product recommendations for a user.

    Args:
        user_id: User ID
        query_product_id: Product ID to find similar items
        budget: Optional budget constraint
        num_results: Number of recommendations

    Returns:
        List of personalized recommendations
    """
    # Get user profile
    user_row = user_features_df.filter(F.col("user_id") == user_id).first()

    if not user_row:
        return []

    user_profile = UserProfile(
        user_id=user_row["user_id"],
        user_embedding=np.array(user_row["user_embedding"]) if user_row["user_embedding"] else None,
        category_prefs=user_row.get("category_prefs"),
        brand_prefs=user_row.get("brand_prefs"),
        color_prefs=user_row.get("color_prefs"),
        price_range=(user_row.get("p25_price"), user_row.get("p75_price")) if user_row.get("p25_price") else None
    )

    # Get query embedding
    product_emb = embeddings_df.filter(F.col("product_id") == query_product_id).first()
    if not product_emb:
        return []

    query_embedding = np.array(product_emb["image_embedding"])

    # Get candidates from vector search
    vs_results = vs_index.similarity_search(
        query_vector=query_embedding,
        columns=["product_id"],
        num_results=50
    )

    if not vs_results or "result" not in vs_results:
        return []

    candidate_ids = [r["product_id"] for r in vs_results["result"]["data_array"]]

    # Load candidates
    candidates_df = (
        products_df
        .filter(F.col("product_id").isin(candidate_ids))
        .join(embeddings_df, "product_id")
        .collect()
    )

    # Create ProductCandidate objects
    product_candidates = [
        ProductCandidate(
            product_id=row["product_id"],
            image_embedding=np.array(row["image_embedding"]),
            category=row["category"],
            brand=row["brand"],
            color=row["color"],
            price=float(row["price"])
        )
        for row in candidates_df
    ]

    # Score and rank
    scorer = RecommendationScorer()
    ranked = scorer.rank_products(
        products=product_candidates,
        query_embedding=query_embedding,
        user_profile=user_profile,
        budget=budget,
        top_k=num_results
    )

    # Format results
    results = []
    for p in ranked:
        # Get product details
        prod_row = products_df.filter(F.col("product_id") == p.product_id).first()
        results.append({
            "product_id": p.product_id,
            "name": prod_row["display_name"],
            "category": p.category,
            "brand": p.brand,
            "color": p.color,
            "price": p.price,
            "recommendation_score": float(p.final_score),
            "visual_score": float(p.visual_sim),
            "personalization_score": float(p.user_sim),
            "attribute_score": float(p.attr_score)
        })

    return results


def complete_the_look(
    product_id: str,
    user_id: Optional[str] = None,
    num_results: int = 5
) -> List[Dict[str, Any]]:
    """
    Suggest complementary products to complete an outfit.

    Args:
        product_id: Base product ID
        user_id: Optional user ID for personalization
        num_results: Number of suggestions

    Returns:
        List of complementary products
    """
    # Get base product
    base_product = products_df.filter(F.col("product_id") == product_id).first()

    if not base_product:
        return []

    base_category = base_product["category"]

    # Define complementary categories
    complements = {
        "Topwear": ["Bottomwear", "Shoes", "Accessories"],
        "Bottomwear": ["Topwear", "Shoes", "Accessories"],
        "Dress": ["Shoes", "Accessories", "Outerwear"],
        "Shoes": ["Topwear", "Bottomwear", "Accessories"],
        "Accessories": ["Topwear", "Dress", "Shoes"]
    }

    target_categories = complements.get(base_category, [])

    if not target_categories:
        # Fallback to different category
        target_categories = [c for c in complements.keys() if c != base_category]

    # Get candidates from complementary categories
    candidates_df = (
        products_df
        .filter(F.col("category").isin(target_categories))
        .join(embeddings_df, "product_id")
    )

    if user_id:
        # Personalized completion
        return get_personalized_recommendations(
            user_id=user_id,
            query_product_id=product_id,
            num_results=num_results
        )
    else:
        # Visual similarity based completion
        return search_similar_by_image(product_id, num_results=num_results)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Claude Agent Implementation

# COMMAND ----------

def call_claude(messages: List[Dict], tools: List[Dict] = None) -> Dict:
    """
    Call Claude via Databricks AI Gateway.

    Args:
        messages: List of message dicts
        tools: Optional list of tool definitions

    Returns:
        Claude's response
    """
    import mlflow.deployments

    client = mlflow.deployments.get_deploy_client("databricks")

    request_payload = {
        "messages": messages,
        "max_tokens": 2000,
        "temperature": 0.7
    }

    if tools:
        request_payload["tools"] = tools

    response = client.predict(
        endpoint=CLAUDE_ROUTE,
        inputs=request_payload
    )

    return response


# COMMAND ----------

# Define tools for Claude
TOOLS = [
    {
        "name": "search_similar_by_image",
        "description": "Search for visually similar fashion products given a product ID. Returns products that look similar in style, color, and design.",
        "input_schema": {
            "type": "object",
            "properties": {
                "product_id": {
                    "type": "string",
                    "description": "The ID of the product to find similar items for"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of similar products to return (default 10)"
                }
            },
            "required": ["product_id"]
        }
    },
    {
        "name": "get_personalized_recommendations",
        "description": "Get personalized product recommendations based on a user's style preferences, purchase history, and budget. Takes into account the user's preferred categories, brands, colors, and price range.",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "The user ID to get recommendations for"
                },
                "query_product_id": {
                    "type": "string",
                    "description": "Product ID to base recommendations on"
                },
                "budget": {
                    "type": "number",
                    "description": "Optional budget constraint in dollars"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of recommendations (default 10)"
                }
            },
            "required": ["user_id", "query_product_id"]
        }
    },
    {
        "name": "complete_the_look",
        "description": "Suggest complementary products to complete an outfit. For example, if given a dress, suggest matching shoes and accessories.",
        "input_schema": {
            "type": "object",
            "properties": {
                "product_id": {
                    "type": "string",
                    "description": "The base product ID to complete the look for"
                },
                "user_id": {
                    "type": "string",
                    "description": "Optional user ID for personalized suggestions"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of suggestions (default 5)"
                }
            },
            "required": ["product_id"]
        }
    }
]

# COMMAND ----------

def execute_tool(tool_name: str, tool_input: Dict) -> str:
    """Execute a tool and return formatted results."""
    if tool_name == "search_similar_by_image":
        results = search_similar_by_image(**tool_input)
    elif tool_name == "get_personalized_recommendations":
        results = get_personalized_recommendations(**tool_input)
    elif tool_name == "complete_the_look":
        results = complete_the_look(**tool_input)
    else:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    return json.dumps(results, indent=2)

# COMMAND ----------

def stylist_agent(user_message: str, max_turns: int = 5) -> str:
    """
    Main agent loop that processes user requests.

    Args:
        user_message: User's natural language request
        max_turns: Maximum conversation turns

    Returns:
        Agent's final response
    """
    messages = [{"role": "user", "content": user_message}]

    for turn in range(max_turns):
        response = call_claude(messages, tools=TOOLS)

        # Handle tool use
        if response.get("stop_reason") == "tool_use":
            tool_uses = [block for block in response.get("content", []) if block.get("type") == "tool_use"]

            # Execute tools
            tool_results = []
            for tool_use in tool_uses:
                tool_name = tool_use["name"]
                tool_input = tool_use["input"]

                print(f"Executing tool: {tool_name}")
                result = execute_tool(tool_name, tool_input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use["id"],
                    "content": result
                })

            # Continue conversation with tool results
            messages.append({"role": "assistant", "content": response["content"]})
            messages.append({"role": "user", "content": tool_results})
        else:
            # Final response
            text_content = [block["text"] for block in response.get("content", []) if block.get("type") == "text"]
            return "\n".join(text_content)

    return "Agent reached maximum turns"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the Agent

# COMMAND ----------

# Test 1: Visual similarity search
print("=" * 60)
print("TEST 1: Visual Similarity Search")
print("=" * 60)

test_product = products_df.orderBy(F.rand()).first()
print(f"\nTest product: {test_product['display_name']} (ID: {test_product['product_id']})")

query = f"Show me products similar to product ID {test_product['product_id']}"
print(f"\nUser: {query}")
print("\nAgent:")
response = stylist_agent(query)
print(response)

# COMMAND ----------

# Test 2: Personalized recommendations
print("\n" + "=" * 60)
print("TEST 2: Personalized Recommendations")
print("=" * 60)

test_user = user_features_df.filter(F.col("user_embedding").isNotNull()).orderBy(F.rand()).first()
print(f"\nTest user: {test_user['user_id']} (Segment: {test_user['segment']})")

query = f"I'm user {test_user['user_id']}. Can you recommend products similar to {test_product['product_id']} that match my style?"
print(f"\nUser: {query}")
print("\nAgent:")
response = stylist_agent(query)
print(response)

# COMMAND ----------

# Test 3: Complete the look
print("\n" + "=" * 60)
print("TEST 3: Complete the Look")
print("=" * 60)

query = f"I'm buying product {test_product['product_id']}. What else should I get to complete the outfit?"
print(f"\nUser: {query}")
print("\nAgent:")
response = stylist_agent(query)
print(response)

# COMMAND ----------

# Test 4: Budget-constrained recommendations
print("\n" + "=" * 60)
print("TEST 4: Budget-Constrained Search")
print("=" * 60)

query = f"As user {test_user['user_id']}, show me products similar to {test_product['product_id']} but under $50"
print(f"\nUser: {query}")
print("\nAgent:")
response = stylist_agent(query)
print(response)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print("=" * 60)
print("CLAUDE STYLIST AGENT SUMMARY")
print("=" * 60)
print("\n✓ Agent tools implemented:")
print("  - search_similar_by_image")
print("  - get_personalized_recommendations")
print("  - complete_the_look")
print("\n✓ Claude integration via AI Gateway")
print("✓ Natural language interface")
print("✓ Multi-turn conversations")
print("\n" + "=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC 1. Run notebook `08_app_ui` to build the Streamlit interface
# MAGIC 2. Consider adding more tools:
# MAGIC    - search_by_attributes (filter by category, color, price)
# MAGIC    - get_trending_products
# MAGIC    - get_product_details
# MAGIC 3. Implement conversation history storage
# MAGIC 4. Add evaluation metrics for agent quality
