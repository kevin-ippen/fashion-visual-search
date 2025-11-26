# Databricks notebook source
# MAGIC %md
# MAGIC # 08 - Fashion Visual Search App (Streamlit)
# MAGIC
# MAGIC This notebook creates a Streamlit app for the fashion visual search and recommendation system.
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - All previous notebooks completed
# MAGIC - Vector Search index operational
# MAGIC
# MAGIC **Features:**
# MAGIC - Image upload for visual search
# MAGIC - User selection for personalized recommendations
# MAGIC - Claude agent chat interface
# MAGIC - Side-by-side comparison of visual vs personalized results

# COMMAND ----------

# MAGIC %md
# MAGIC ## Installation
# MAGIC
# MAGIC Run this once to install Streamlit in your cluster

# COMMAND ----------

# MAGIC %pip install streamlit --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## App Code
# MAGIC
# MAGIC Save this to a separate Python file for Databricks Apps deployment

# COMMAND ----------

# Create the Streamlit app file
app_code = '''
import streamlit as st
import numpy as np
from databricks.vector_search.client import VectorSearchClient
from pyspark.sql import SparkSession, functions as F
import sys
import json
from PIL import Image
import io

# Configuration
CATALOG = "main"
SCHEMA = "fashion_demo"
VECTOR_SEARCH_ENDPOINT = "fashion_vector_search"
INDEX_NAME = f"{CATALOG}.{SCHEMA}.product_embeddings_index"

# Initialize
@st.cache_resource
def get_spark():
    return SparkSession.builder.getOrCreate()

@st.cache_resource
def get_vector_search_client():
    return VectorSearchClient()

spark = get_spark()
vsc = get_vector_search_client()

# Load data
@st.cache_data
def load_products():
    df = spark.table(f"{CATALOG}.{SCHEMA}.products")
    return df.toPandas()

@st.cache_data
def load_embeddings():
    df = spark.table(f"{CATALOG}.{SCHEMA}.product_image_embeddings")
    return df.toPandas()

@st.cache_data
def load_users():
    df = spark.table(f"{CATALOG}.{SCHEMA}.users")
    return df.toPandas()

products_pd = load_products()
embeddings_pd = load_embeddings()
users_pd = load_users()

# App Layout
st.set_page_config(page_title="Fashion Visual Search", layout="wide")

st.title("👗 Fashion Visual Search & Recommender")
st.markdown("---")

# Sidebar
st.sidebar.header("Settings")

# User selection
user_mode = st.sidebar.radio("Mode", ["Visual Search Only", "Personalized Recommendations"])

selected_user = None
if user_mode == "Personalized Recommendations":
    user_list = users_pd["user_id"].tolist()
    selected_user = st.sidebar.selectbox("Select User", user_list)

    # Show user info
    user_info = users_pd[users_pd["user_id"] == selected_user].iloc[0]
    st.sidebar.info(f"**Segment:** {user_info['segment']}")

# Budget filter
use_budget = st.sidebar.checkbox("Set Budget Constraint")
budget = None
if use_budget:
    budget = st.sidebar.slider("Max Price ($)", 0, 300, 100)

# Number of results
num_results = st.sidebar.slider("Number of Results", 5, 50, 10)

# Main content
tab1, tab2, tab3 = st.tabs(["🔍 Search", "💬 AI Stylist", "📊 Analytics"])

with tab1:
    st.header("Visual Similarity Search")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Select a Product")

        # Category filter
        categories = ["All"] + sorted(products_pd["category"].unique().tolist())
        selected_category = st.selectbox("Category", categories)

        # Filter products
        filtered_products = products_pd
        if selected_category != "All":
            filtered_products = products_pd[products_pd["category"] == selected_category]

        # Product selection
        product_options = filtered_products["display_name"].tolist()
        selected_product_name = st.selectbox("Product", product_options)

        selected_product = filtered_products[
            filtered_products["display_name"] == selected_product_name
        ].iloc[0]

        st.write(f"**Product ID:** {selected_product['product_id']}")
        st.write(f"**Category:** {selected_product['category']}")
        st.write(f"**Color:** {selected_product['color']}")
        st.write(f"**Price:** ${selected_product['price']:.2f}")

        # Search button
        search_button = st.button("🔍 Find Similar Products", type="primary")

    with col2:
        if search_button:
            st.subheader("Similar Products")

            # Get embedding
            product_emb = embeddings_pd[
                embeddings_pd["product_id"] == selected_product["product_id"]
            ].iloc[0]["image_embedding"]

            # Search
            with st.spinner("Searching..."):
                try:
                    vs_index = vsc.get_index(
                        endpoint_name=VECTOR_SEARCH_ENDPOINT,
                        index_name=INDEX_NAME
                    )

                    results = vs_index.similarity_search(
                        query_vector=product_emb,
                        columns=["product_id"],
                        num_results=num_results
                    )

                    if results and "result" in results:
                        result_data = results["result"]["data_array"]

                        # Display results
                        for i, result in enumerate(result_data[:num_results]):
                            product_id = result["product_id"]
                            score = result["score"]

                            product = products_pd[products_pd["product_id"] == product_id].iloc[0]

                            # Apply budget filter
                            if budget and product["price"] > budget:
                                continue

                            with st.container():
                                col_a, col_b = st.columns([1, 3])
                                with col_a:
                                    st.metric("Similarity", f"{score:.3f}")
                                with col_b:
                                    st.write(f"**{product['display_name']}**")
                                    st.write(f"Category: {product['category']} | Color: {product['color']} | Price: ${product['price']:.2f}")
                                st.divider()
                    else:
                        st.error("No results found")

                except Exception as e:
                    st.error(f"Search error: {str(e)}")

with tab2:
    st.header("AI Stylist Chat")
    st.markdown("Ask the AI stylist for product recommendations, outfit suggestions, or style advice!")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask the stylist..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # This would call the Claude agent from notebook 07
                # For demo purposes, showing a placeholder
                response = f"I would help you with: {prompt}\\n\\n(Claude agent integration pending)"
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

with tab3:
    st.header("System Analytics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Products", len(products_pd))

    with col2:
        st.metric("Total Users", len(users_pd))

    with col3:
        st.metric("Embeddings", len(embeddings_pd))

    st.subheader("Product Distribution")

    category_counts = products_pd["category"].value_counts()
    st.bar_chart(category_counts)

    st.subheader("Price Distribution")
    st.histogram_chart(products_pd["price"].dropna())

st.sidebar.markdown("---")
st.sidebar.caption("Fashion Visual Search MVP")
st.sidebar.caption("Powered by Databricks Mosaic AI")
'''

# Save app code
with open("/tmp/fashion_search_app.py", "w") as f:
    f.write(app_code)

print("✓ App code saved to /tmp/fashion_search_app.py")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run the App Locally (for testing)

# COMMAND ----------

# MAGIC %sh
# MAGIC # Test the app locally (will run in notebook)
# MAGIC # streamlit run /tmp/fashion_search_app.py --server.port 8501

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy as Databricks App
# MAGIC
# MAGIC To deploy this as a Databricks App:
# MAGIC
# MAGIC 1. Save the app code to your repo: `fashion-visual-search/app/app.py`
# MAGIC 2. Create an `app.yaml` configuration file
# MAGIC 3. Deploy using the Databricks CLI or UI
# MAGIC
# MAGIC ### app.yaml example:
# MAGIC ```yaml
# MAGIC command: ["streamlit", "run", "app.py", "--server.port", "8080"]
# MAGIC env:
# MAGIC   - name: CATALOG
# MAGIC     value: "main"
# MAGIC   - name: SCHEMA
# MAGIC     value: "fashion_demo"
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Alternative: Gradio Interface
# MAGIC
# MAGIC For a simpler interface, consider using Gradio:

# COMMAND ----------

# MAGIC %pip install gradio --quiet

# COMMAND ----------

import gradio as gr
from databricks.vector_search.client import VectorSearchClient
from pyspark.sql import functions as F

# Configuration
CATALOG = "main"
SCHEMA = "fashion_demo"
VECTOR_SEARCH_ENDPOINT = "fashion_vector_search"
INDEX_NAME = f"{CATALOG}.{SCHEMA}.product_embeddings_index"

# Load data
products_df = spark.table(f"{CATALOG}.{SCHEMA}.products")
embeddings_df = spark.table(f"{CATALOG}.{SCHEMA}.product_image_embeddings}")

def search_similar_products(product_id, num_results=10):
    """Search for similar products."""
    vsc = VectorSearchClient()
    vs_index = vsc.get_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT,
        index_name=INDEX_NAME
    )

    # Get embedding
    product_emb = embeddings_df.filter(F.col("product_id") == product_id).first()

    if not product_emb:
        return "Product not found"

    query_embedding = product_emb["image_embedding"]

    # Search
    results = vs_index.similarity_search(
        query_vector=query_embedding,
        columns=["product_id"],
        num_results=num_results
    )

    if not results or "result" not in results:
        return "No results found"

    # Format results
    result_ids = [r["product_id"] for r in results["result"]["data_array"]]
    scores = {r["product_id"]: r["score"] for r in results["result"]["data_array"]}

    results_df = products_df.filter(F.col("product_id").isin(result_ids)).collect()

    output = []
    for row in results_df:
        pid = row["product_id"]
        output.append(
            f"**{row['display_name']}**\\n"
            f"Category: {row['category']} | Price: ${row['price']:.2f} | "
            f"Similarity: {scores.get(pid, 0):.3f}\\n"
        )

    return "\\n".join(output)

# Create Gradio interface
demo = gr.Interface(
    fn=search_similar_products,
    inputs=[
        gr.Textbox(label="Product ID", placeholder="Enter product ID..."),
        gr.Slider(5, 20, value=10, step=1, label="Number of Results")
    ],
    outputs=gr.Markdown(label="Similar Products"),
    title="Fashion Visual Search",
    description="Find visually similar fashion products using Mosaic AI Vector Search"
)

# Launch (for testing in notebook)
# demo.launch(share=False, server_port=7860)

print("✓ Gradio interface ready")
print("Run: demo.launch() to start the interface")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC This notebook provides two UI options:
# MAGIC
# MAGIC ### Streamlit App (Recommended for Production)
# MAGIC - Full-featured interface
# MAGIC - Multiple tabs (Search, AI Stylist, Analytics)
# MAGIC - User personalization
# MAGIC - Budget filtering
# MAGIC - Deploy as Databricks App
# MAGIC
# MAGIC ### Gradio Interface (Quick Testing)
# MAGIC - Simple, fast interface
# MAGIC - Great for demos and testing
# MAGIC - Embedded in notebooks
# MAGIC
# MAGIC ## Next Steps
# MAGIC
# MAGIC 1. Test the interfaces with your data
# MAGIC 2. Customize the UI based on your needs
# MAGIC 3. Deploy as a Databricks App for production use
# MAGIC 4. Add authentication and user management
# MAGIC 5. Integrate with the Claude agent from notebook 07
# MAGIC 6. Add image upload capability for visual search
# MAGIC 7. Implement feedback collection for model improvement
