import streamlit as st
import numpy as np
from databricks.vector_search.client import VectorSearchClient
from databricks.sdk import WorkspaceClient
from databricks.sql import connect
import pandas as pd
from PIL import Image
import json
from typing import List, Dict, Optional
from dataclasses import dataclass
import random
import os

# Configuration
CATALOG = "main"
SCHEMA = "fashion_demo"
VECTOR_SEARCH_ENDPOINT = "fashion_vector_search"
INDEX_NAME = f"{CATALOG}.{SCHEMA}.product_embeddings_index"
CLAUDE_ENDPOINT = "databricks-claude-sonnet-4-5"

# SQL Warehouse ID - configured in app settings
SQL_WAREHOUSE_ID = os.getenv("SQL_WAREHOUSE_ID")

# Secret scope configuration for authentication
SECRET_SCOPE = "redditscope"
SECRET_KEY = "redditkey"

# Page config
st.set_page_config(
    page_title="Fashion Boutique - AI-Powered Shopping",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for ecommerce look
st.markdown("""
<style>
    .product-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .product-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    .product-name {
        font-size: 16px;
        font-weight: 600;
        color: #333;
        margin-bottom: 8px;
    }
    .product-price {
        font-size: 20px;
        font-weight: 700;
        color: #2e7d32;
        margin: 8px 0;
    }
    .product-meta {
        font-size: 13px;
        color: #666;
        margin: 4px 0;
    }
    .similarity-badge {
        background: #1976d2;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
    }
    .header-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
    }
    .filter-section {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize clients - using default user authentication
@st.cache_resource
def get_workspace_client():
    """Initialize Workspace client with user auth."""
    return WorkspaceClient()

@st.cache_resource
def get_vector_search_client():
    """Initialize Vector Search client with service principal auth from secrets."""
    try:
        # Get workspace client first
        w = get_workspace_client()

        # Retrieve PAT from Databricks secrets
        pat_token = w.secrets.get_secret(scope=SECRET_SCOPE, key=SECRET_KEY).value

        # Initialize VectorSearchClient with workspace URL and PAT from secrets
        return VectorSearchClient(
            workspace_url=w.config.host,
            personal_access_token=pat_token,
            disable_notice=True
        )
    except Exception as e:
        st.error(f"Failed to initialize Vector Search client: {e}")
        st.info(f"Ensure the secret '{SECRET_KEY}' exists in scope '{SECRET_SCOPE}'")
        return None

@st.cache_resource
def get_sql_connection():
    """Create SQL Warehouse connection using service principal authentication."""
    if not SQL_WAREHOUSE_ID:
        st.error("⚠️ SQL_WAREHOUSE_ID not configured. Please set it in app configuration.")
        return None

    try:
        # Get workspace client and retrieve PAT from secrets
        w = get_workspace_client()
        pat_token = w.secrets.get_secret(scope=SECRET_SCOPE, key=SECRET_KEY).value

        # Connect to SQL Warehouse with PAT authentication
        conn = connect(
            server_hostname=w.config.host.replace("https://", ""),
            http_path=f"/sql/1.0/warehouses/{SQL_WAREHOUSE_ID}",
            access_token=pat_token
        )
        return conn
    except Exception as e:
        st.error(f"Failed to connect to SQL Warehouse: {e}")
        st.info(f"Ensure the secret '{SECRET_KEY}' exists in scope '{SECRET_SCOPE}' and you have access to the SQL Warehouse.")
        return None

# Initialize clients in order (workspace client first for auth context)
w = get_workspace_client()
vsc = get_vector_search_client()
sql_conn = get_sql_connection()

# Load data using SQL Warehouse with user permissions
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_products():
    """Load products using SQL Warehouse connection with user auth."""
    if sql_conn is None:
        return pd.DataFrame(columns=[
            'product_id', 'product_display_name', 'master_category', 
            'base_color', 'price', 'article_type', 'image_path'
        ])
    
    try:
        query = f"SELECT * FROM {CATALOG}.{SCHEMA}.products"
        with sql_conn.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
        
        return pd.DataFrame(rows, columns=columns)
    except Exception as e:
        st.error(f"Error loading products: {e}")
        st.info("Ensure you have SELECT permission on the products table.")
        return pd.DataFrame(columns=[
            'product_id', 'product_display_name', 'master_category', 
            'base_color', 'price', 'article_type', 'image_path'
        ])

@st.cache_data(ttl=300)
def load_embeddings():
    """Load embeddings using SQL Warehouse connection with user auth."""
    if sql_conn is None:
        return pd.DataFrame(columns=['product_id', 'image_embedding'])
    
    try:
        query = f"SELECT product_id, image_embedding FROM {CATALOG}.{SCHEMA}.product_image_embeddings"
        with sql_conn.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
        
        return pd.DataFrame(rows, columns=columns)
    except Exception as e:
        st.error(f"Error loading embeddings: {e}")
        st.info("Ensure you have SELECT permission on the embeddings table.")
        return pd.DataFrame(columns=['product_id', 'image_embedding'])

@st.cache_data(ttl=300)
def load_user_features():
    """Load user features using SQL Warehouse connection with user auth."""
    if sql_conn is None:
        return pd.DataFrame(columns=['user_id', 'segment', 'category_prefs'])
    
    try:
        query = f"SELECT * FROM {CATALOG}.{SCHEMA}.user_style_features LIMIT 100"
        with sql_conn.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
        
        return pd.DataFrame(rows, columns=columns)
    except Exception as e:
        # User features are optional
        return pd.DataFrame(columns=['user_id', 'segment', 'category_prefs'])

# Load data
try:
    products_pd = load_products()
    embeddings_pd = load_embeddings()
    user_features_pd = load_user_features()
    
    if products_pd.empty:
        st.warning("⚠️ No product data loaded. Please check your permissions and SQL Warehouse configuration.")
except Exception as e:
    st.error(f"Error initializing data: {e}")
    products_pd = pd.DataFrame()
    embeddings_pd = pd.DataFrame()
    user_features_pd = pd.DataFrame()

# Helper function to parse category preferences
def parse_category_prefs(prefs):
    """Parse category preferences from JSON string or dict."""
    if prefs is None:
        return {}
    if isinstance(prefs, str):
        try:
            return json.loads(prefs)
        except:
            return {}
    return prefs if isinstance(prefs, dict) else {}

# Helper function to convert embeddings
def get_embedding_vector(embedding):
    """Convert embedding to numpy array."""
    if isinstance(embedding, str):
        try:
            return np.array(json.loads(embedding))
        except:
            return None
    elif isinstance(embedding, (list, np.ndarray)):
        return np.array(embedding)
    elif hasattr(embedding, 'tolist'):
        return np.array(embedding.tolist() if callable(embedding.tolist) else embedding)
    return None

# Helper functions
def search_similar_products(product_id: int, num_results: int = 12, budget: float = None):
    """Search for visually similar products using Vector Search."""
    try:
        # Check if Vector Search client is initialized
        if vsc is None:
            st.error("Vector Search client not initialized. Check authentication configuration.")
            return []

        # Get embedding
        product_emb = embeddings_pd[embeddings_pd["product_id"] == product_id]
        if product_emb.empty:
            return []

        query_embedding = product_emb.iloc[0]["image_embedding"]
        query_vector = get_embedding_vector(query_embedding)

        if query_vector is None:
            return []

        # Search using service principal authentication
        vs_index = vsc.get_index(
            endpoint_name=VECTOR_SEARCH_ENDPOINT,
            index_name=INDEX_NAME
        )
        
        results = vs_index.similarity_search(
            query_vector=query_vector.tolist(),
            columns=["product_id"],
            num_results=num_results * 2
        )
        
        if not results or "result" not in results:
            return []
        
        # Parse and enrich results
        result_data = results["result"]["data_array"]
        similar_products = []
        
        for r in result_data:
            pid = int(r[0])
            score = float(r[1])
            
            product = products_pd[products_pd["product_id"] == pid]
            if not product.empty:
                product = product.iloc[0]
                
                # Apply budget filter
                if budget and product["price"] > budget:
                    continue
                
                similar_products.append({
                    "product_id": pid,
                    "name": product["product_display_name"],
                    "category": product["master_category"],
                    "color": product["base_color"],
                    "price": product["price"],
                    "similarity": score,
                    "image_path": product["image_path"]
                })
                
                if len(similar_products) >= num_results:
                    break
        
        return similar_products
    
    except Exception as e:
        st.error(f"Error searching for similar products: {str(e)}")
        st.info("Ensure you have access to the Vector Search endpoint.")
        return []

# Header
st.markdown("""
<div class='header-banner'>
    <h1 style='margin: 0; font-size: 42px;'>👗 Fashion Boutique</h1>
    <p style='margin: 10px 0 0 0; font-size: 18px; opacity: 0.9;'>AI-Powered Visual Search & Personalized Recommendations</p>
</div>
""", unsafe_allow_html=True)

# Check if data is loaded
if products_pd.empty:
    st.error("⚠️ Unable to load product data.")
    st.markdown("""
    **Please verify:**
    1. SQL_WAREHOUSE_ID is configured in app settings
    2. The SQL Warehouse is running
    3. You have SELECT permissions on:
       - `{CATALOG}.{SCHEMA}.products`
       - `{CATALOG}.{SCHEMA}.product_image_embeddings`
       - `{CATALOG}.{SCHEMA}.user_style_features` (optional)
    4. The tables exist and contain data
    """.format(CATALOG=CATALOG, SCHEMA=SCHEMA))
    st.stop()

# Sidebar - Filters
st.sidebar.header("🎯 Shopping Preferences")

# User selection for personalization
use_personalization = st.sidebar.checkbox("Enable Personalized Recommendations", value=False)
selected_user = None

if use_personalization and not user_features_pd.empty:
    user_list = user_features_pd["user_id"].tolist()[:100]
    selected_user = st.sidebar.selectbox("Your Profile", user_list)
    
    if selected_user:
        user_info = user_features_pd[user_features_pd["user_id"] == selected_user].iloc[0]
        st.sidebar.success(f"**Style:** {user_info['segment'].title()}")
        
        category_prefs = parse_category_prefs(user_info.get('category_prefs'))
        if category_prefs:
            try:
                top_cat = max(category_prefs.items(), key=lambda x: x[1])
                st.sidebar.info(f"**Favorite:** {top_cat[0]}")
            except:
                pass

# Filters
st.sidebar.markdown("### 🔍 Filters")

categories = ["All Categories"] + sorted(products_pd["master_category"].unique().tolist())
selected_category = st.sidebar.selectbox("Category", categories)

colors = ["All Colors"] + sorted(products_pd["base_color"].dropna().unique().tolist())
selected_color = st.sidebar.selectbox("Color", colors)

use_budget = st.sidebar.checkbox("Set Budget")
budget = None
if use_budget:
    budget = st.sidebar.slider("Maximum Price ($)", 0, 300, 100, 10)
    st.sidebar.caption(f"Showing items under ${budget}")

num_results = st.sidebar.slider("Results to Show", 6, 24, 12, 6)

st.sidebar.markdown("---")
st.sidebar.caption("Powered by Databricks Mosaic AI")
st.sidebar.caption("Vector Search + CLIP Embeddings")

# Main content
tab1, tab2, tab3 = st.tabs(["🛍️ Shop by Style", "💬 AI Stylist Chat", "📊 Insights"])

with tab1:
    st.header("Discover Similar Styles")
    
    # Product browser
    st.markdown("<div class='filter-section'>", unsafe_allow_html=True)
    st.subheader("Select a Product to Find Similar Items")
    
    # Filter products for selection
    filtered_products = products_pd.copy()
    if selected_category != "All Categories":
        filtered_products = filtered_products[filtered_products["master_category"] == selected_category]
    if selected_color != "All Colors":
        filtered_products = filtered_products[filtered_products["base_color"] == selected_color]
    if budget:
        filtered_products = filtered_products[filtered_products["price"] <= budget]
    
    st.caption(f"Showing {len(filtered_products):,} products")
    
    # Initialize session state for random selection
    if 'random_seed' not in st.session_state:
        st.session_state.random_seed = 0
    
    # Product selection
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        product_names = filtered_products["product_display_name"].tolist()[:500]
        
        if not product_names:
            st.warning("No products match your filters.")
            st.stop()
        
        if 'selected_product_index' not in st.session_state:
            st.session_state.selected_product_index = 0
        
        selected_product_name = st.selectbox(
            "Choose a product",
            product_names,
            index=min(st.session_state.selected_product_index, len(product_names) - 1),
            label_visibility="collapsed",
            key=f"product_select_{st.session_state.random_seed}"
        )
    
    with col2:
        search_button = st.button("🔍 Find Similar Styles", type="primary", use_container_width=True)
    
    with col3:
        if st.button("🎲 Random", use_container_width=True):
            st.session_state.selected_product_index = random.randint(0, len(product_names) - 1)
            st.session_state.random_seed += 1
            st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Get selected product
    selected_product = filtered_products[
        filtered_products["product_display_name"] == selected_product_name
    ].iloc[0]
    
    # Display selected product
    st.markdown("### 📌 Your Selection")
    col_a, col_b = st.columns([1, 3])
    
    with col_a:
        st.markdown(f"<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); "
                   f"height: 200px; border-radius: 8px; display: flex; align-items: center; "
                   f"justify-content: center; color: white; font-size: 48px;'>👔</div>", 
                   unsafe_allow_html=True)
    
    with col_b:
        st.markdown(f"### {selected_product['product_display_name']}")
        st.markdown(f"**Category:** {selected_product['master_category']} | "
                   f"**Color:** {selected_product['base_color']} | "
                   f"**Article:** {selected_product['article_type']}")
        st.markdown(f"### ${selected_product['price']:.2f}")
        st.caption(f"Product ID: {selected_product['product_id']}")
    
    st.markdown("---")
    
    # Search results
    if search_button or 'last_search' in st.session_state:
        if search_button:
            st.session_state.last_search = selected_product['product_id']
        
        with st.spinner("🔍 Finding similar styles..."):
            similar_products = search_similar_products(
                product_id=int(selected_product['product_id']),
                num_results=num_results,
                budget=budget
            )
        
        if similar_products:
            st.markdown(f"### ✨ {len(similar_products)} Similar Products Found")
            
            # Display in grid
            cols_per_row = 3
            for i in range(0, len(similar_products), cols_per_row):
                cols = st.columns(cols_per_row)
                
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx < len(similar_products):
                        product = similar_products[idx]
                        
                        with col:
                            st.markdown(f"<div class='product-card'>", unsafe_allow_html=True)
                            
                            category_emoji = {
                                "Apparel": "👕",
                                "Footwear": "👟",
                                "Accessories": "👜",
                                "Personal Care": "💄"
                            }
                            emoji = category_emoji.get(product['category'], "🛍️")
                            
                            st.markdown(f"<div style='background: #f5f5f5; height: 180px; "
                                       f"border-radius: 8px; display: flex; align-items: center; "
                                       f"justify-content: center; font-size: 64px; margin-bottom: 12px;'>"
                                       f"{emoji}</div>", unsafe_allow_html=True)
                            
                            st.markdown(f"<div class='product-name'>{product['name'][:50]}...</div>", 
                                       unsafe_allow_html=True)
                            st.markdown(f"<div class='product-meta'>{product['category']} • {product['color']}</div>", 
                                       unsafe_allow_html=True)
                            st.markdown(f"<div class='product-price'>${product['price']:.2f}</div>", 
                                       unsafe_allow_html=True)
                            
                            similarity_pct = int(product['similarity'] * 100)
                            st.markdown(f"<div style='text-align: center; margin: 10px 0;'>"
                                       f"<span class='similarity-badge'>{similarity_pct}% Match</span></div>", 
                                       unsafe_allow_html=True)
                            
                            col_x, col_y = st.columns(2)
                            with col_x:
                                st.button("🛒 Add", key=f"add_{product['product_id']}", use_container_width=True)
                            with col_y:
                                st.button("❤️", key=f"like_{product['product_id']}", use_container_width=True)
                            
                            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("No similar products found. Try adjusting your filters.")

with tab2:
    st.header("💬 AI Fashion Stylist")
    st.markdown("Chat with our AI stylist powered by Claude Sonnet 4.5")
    
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask me anything about fashion, products, or recommendations..."):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
                    
                    chat_messages = []
                    for msg in st.session_state.chat_messages[-10:]:
                        role = ChatMessageRole.USER if msg["role"] == "user" else ChatMessageRole.ASSISTANT
                        chat_messages.append(ChatMessage(role=role, content=msg["content"]))
                    
                    # Use user's permissions to call Claude
                    response = w.serving_endpoints.query(
                        name=CLAUDE_ENDPOINT,
                        messages=chat_messages,
                        max_tokens=1000,
                        temperature=0.7
                    )
                    
                    assistant_response = response.choices[0].message.content
                    st.markdown(assistant_response)
                    
                    st.session_state.chat_messages.append({
                        "role": "assistant", 
                        "content": assistant_response
                    })
                    
                except Exception as e:
                    error_msg = f"I apologize, but I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.info("Ensure you have access to the Claude endpoint.")
                    st.session_state.chat_messages.append({
                        "role": "assistant", 
                        "content": error_msg
                    })
    
    st.markdown("---")
    st.markdown("**💡 Try asking:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎨 What's trending?"):
            st.session_state.chat_messages.append({"role": "user", "content": "What are the trending fashion items?"})
            st.rerun()
    
    with col2:
        if st.button("👔 Complete my outfit"):
            st.session_state.chat_messages.append({"role": "user", "content": "Help me complete my outfit"})
            st.rerun()
    
    with col3:
        if st.button("💰 Best deals"):
            st.session_state.chat_messages.append({"role": "user", "content": "Show me the best deals under $50"})
            st.rerun()

with tab3:
    st.header("📊 Shopping Insights")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Products", f"{len(products_pd):,}")
    
    with col2:
        st.metric("Categories", products_pd["master_category"].nunique())
    
    with col3:
        st.metric("Avg Price", f"${products_pd['price'].mean():.2f}")
    
    with col4:
        st.metric("Users", f"{len(user_features_pd):,}" if not user_features_pd.empty else "0")
    
    st.markdown("---")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Products by Category")
        category_counts = products_pd["master_category"].value_counts()
        st.bar_chart(category_counts)
    
    with col_b:
        st.subheader("Price Distribution")
        st.bar_chart(products_pd["price"].value_counts(bins=20).sort_index())
    
    st.markdown("---")
    
    st.subheader("Top Products by Category")
    for category in products_pd["master_category"].unique()[:3]:
        with st.expander(f"📦 {category}"):
            cat_products = products_pd[products_pd["master_category"] == category].nlargest(5, "price")
            for _, product in cat_products.iterrows():
                st.markdown(f"**{product['product_display_name']}** - ${product['price']:.2f}")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🔒 Secure Checkout**")
    st.caption("SSL encrypted transactions")

with col2:
    st.markdown("**🚚 Free Shipping**")
    st.caption("On orders over $50")

with col3:
    st.markdown("**↩️ Easy Returns**")
    st.caption("30-day return policy")