# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Generate Synthetic Users and Transactions
# MAGIC
# MAGIC This notebook generates synthetic user and transaction data for testing the recommendation system.
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - `main.fashion_demo.products` table exists
# MAGIC
# MAGIC **Output:**
# MAGIC - `main.fashion_demo.users` Delta table
# MAGIC - `main.fashion_demo.transactions` Delta table

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Configuration
CATALOG = "main"
SCHEMA = "fashion_demo"
PRODUCTS_TABLE = "products"
USERS_TABLE = "users"
TRANSACTIONS_TABLE = "transactions"

# Synthetic data parameters
NUM_USERS = 10000
TRANSACTIONS_PER_USER_MIN = 5
TRANSACTIONS_PER_USER_MAX = 50

RANDOM_SEED = 42

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import *
import sys

# Add src to path
sys.path.append("/Workspace/Users/kevin.ippen@databricks.com/fashion-visual-search/src")

from fashion_visual_search.data_generation import SyntheticDataGenerator, compute_user_statistics
from fashion_visual_search.utils import add_table_comment

# COMMAND ----------

# Initialize data generator
generator = SyntheticDataGenerator(seed=RANDOM_SEED)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Users

# COMMAND ----------

# Generate synthetic users
users_list = generator.generate_users(num_users=NUM_USERS)

print(f"Generated {len(users_list)} synthetic users")

# COMMAND ----------

# Convert to DataFrame
users_df = spark.createDataFrame(users_list)

# Display sample
display(users_df.limit(10))

# COMMAND ----------

# Show segment distribution
print("User segment distribution:")
users_df.groupBy("segment").count().orderBy(F.desc("count")).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Transactions

# COMMAND ----------

# DBTITLE 1,Optimized Transaction Generator with Pre-indexing
from collections import defaultdict
import random
from datetime import datetime, timedelta
import uuid
import numpy as np

def generate_transactions_realistic(
    generator,
    users,
    products,
    sessions_per_user_range=(5, 25),  # Increased volume
    events_per_session_range=(3, 12)  # Increased volume
):
    """
    Generate realistic transaction data with:
    - Power law distribution (80/20 rule - heavy hitter products)
    - Session-based browsing (multiple events per session)
    - Funnel sequences (view -> add_to_cart -> purchase)
    - Realistic conversion rates (2-3%)
    - Time-of-day patterns (peak hours)
    - Repeat purchases (users return to products)
    
    Note: "Transaction" = any event (view/cart/purchase), "Purchase" = completed sale
    """
    print("Pre-indexing products by category and price segment...")
    
    # Pre-index products by category
    products_by_category = defaultdict(list)
    for product in products:
        category = product.get('master_category') or product.get('category') or product.get('sub_category')
        if category:
            products_by_category[category].append(product)
    
    all_products = products
    
    # Pre-index products by price segment
    segments = ["casual", "formal", "athletic", "trendy", "vintage", "minimalist", "luxury", "budget"]
    products_by_segment = {}
    
    for segment in segments:
        price_range = generator._get_price_range_for_segment(segment)
        products_by_segment[segment] = [
            p for p in products
            if price_range[0] <= p.get("price", 0) <= price_range[1]
        ]
    
    # Create power law distribution for product popularity
    # Top 20% of products should get ~80% of attention
    print("\nCreating power law distribution for product popularity...")
    num_products = len(all_products)
    
    # Assign popularity scores using Zipf distribution (power law)
    # Higher alpha = more concentration on top products
    popularity_scores = np.random.zipf(a=1.5, size=num_products)
    popularity_scores = popularity_scores / popularity_scores.sum()  # Normalize to probabilities
    
    # Create weighted product selection
    product_weights = {p['product_id']: score for p, score in zip(all_products, popularity_scores)}
    
    # Identify "trending" products (top 5% get extra boost)
    sorted_products = sorted(all_products, key=lambda p: product_weights[p['product_id']], reverse=True)
    trending_products = sorted_products[:int(num_products * 0.05)]
    
    print(f"Indexed {len(products_by_category)} categories")
    print(f"Indexed {len(products_by_segment)} price segments")
    print(f"Created popularity distribution with {len(trending_products)} trending products")
    print("\nGenerating realistic sessions and transactions...")
    
    transactions = []
    transaction_id = 0
    
    # Progress tracking
    total_users = len(users)
    checkpoint = max(1, total_users // 10)
    
    for idx, user in enumerate(users):
        if idx % checkpoint == 0:
            print(f"Progress: {idx}/{total_users} users ({100*idx//total_users}%)")
        
        user_segment = user["segment"]
        preferred_categories = user.get("preferred_categories", [])
        segment_products = products_by_segment.get(user_segment, all_products)
        
        # Track user's browsing history for repeat behavior
        viewed_products = []
        
        # Generate multiple sessions for this user
        num_sessions = random.randint(*sessions_per_user_range)
        
        for session_idx in range(num_sessions):
            session_id = str(uuid.uuid4())
            
            # Generate realistic session timestamp with time-of-day patterns
            session_start = _generate_realistic_timestamp()
            
            # Determine if this session will result in a purchase (2.5% conversion)
            will_purchase = random.random() < 0.025
            
            # Number of events in this session
            num_events = random.randint(*events_per_session_range)
            
            # If purchasing, ensure we have view -> cart -> purchase funnel
            if will_purchase:
                num_events = max(num_events, 3)  # At least 3 events for funnel
            
            for event_idx in range(num_events):
                # Timestamp increments within session (30 sec - 5 minutes between events)
                minutes_increment = random.uniform(0.5, 5) * event_idx
                event_timestamp = session_start + timedelta(minutes=minutes_increment)
                
                # Select product with power law distribution
                # 30% chance to revisit previously viewed
                # 20% chance to select trending product
                # 50% chance to select based on preferences + popularity
                
                if viewed_products and random.random() < 0.3:
                    # Revisit previously viewed product
                    product = random.choice(viewed_products)
                elif random.random() < 0.2:
                    # Select trending product
                    product = random.choice(trending_products)
                else:
                    # Select based on preferences + popularity weights
                    product = _select_product_with_popularity(
                        products_by_category,
                        segment_products,
                        all_products,
                        product_weights,
                        user_segment,
                        preferred_categories
                    )
                
                viewed_products.append(product)
                
                # Determine event type based on funnel logic
                if will_purchase and event_idx == num_events - 1:
                    # Last event in purchase session = purchase
                    event_type = "purchase"
                elif will_purchase and event_idx == num_events - 2:
                    # Second to last = add_to_cart
                    event_type = "add_to_cart"
                elif will_purchase and event_idx >= num_events - 3:
                    # Third to last could be view or cart
                    event_type = random.choice(["view", "add_to_cart"])
                else:
                    # Regular browsing: mostly views, some add_to_cart
                    event_type = "view" if random.random() < 0.75 else "add_to_cart"
                
                transaction = {
                    "transaction_id": f"txn_{transaction_id:08d}",
                    "user_id": user["user_id"],
                    "product_id": product["product_id"],
                    "event_type": event_type,
                    "timestamp": event_timestamp.isoformat(),
                    "session_id": session_id
                }
                
                # Add purchase-specific fields
                if event_type == "purchase":
                    transaction["purchase_amount"] = product.get("price", 0.0)
                    transaction["quantity"] = random.randint(1, 2)  # Most purchases are 1-2 items
                
                transactions.append(transaction)
                transaction_id += 1
    
    print(f"Progress: {total_users}/{total_users} users (100%)")
    return transactions


def _select_product_with_popularity(products_by_category, segment_products, all_products, 
                                     product_weights, segment, preferred_categories):
    """Select a product based on user preferences AND popularity weights."""
    candidates = []
    
    # Try to match preferred categories (60% of the time)
    if random.random() < 0.6 and preferred_categories:
        for cat in preferred_categories:
            candidates.extend(products_by_category.get(cat, []))
        
        if candidates:
            # Further filter by price segment (50% of the time)
            if random.random() < 0.5 and segment_products:
                segment_product_ids = {p['product_id'] for p in segment_products}
                candidates = [p for p in candidates if p['product_id'] in segment_product_ids]
    
    # Fallback to segment products
    if not candidates and segment_products:
        candidates = segment_products
    
    # Final fallback
    if not candidates:
        candidates = all_products
    
    # Select from candidates using popularity weights
    candidate_weights = [product_weights.get(p['product_id'], 1.0) for p in candidates]
    total_weight = sum(candidate_weights)
    probabilities = [w / total_weight for w in candidate_weights]
    
    return np.random.choice(candidates, p=probabilities)


def _generate_realistic_timestamp(days_back=180):
    """Generate timestamp with realistic time-of-day patterns."""
    # Random date in the past
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    delta = end_date - start_date
    random_days = random.randint(0, delta.days)
    base_date = start_date + timedelta(days=random_days)
    
    # Time-of-day patterns (peak hours: 12-2pm, 7-10pm)
    hour_weights = [
        1, 1, 1, 1, 1, 2,  # 0-5am: very low
        3, 4, 5, 6, 7, 8,  # 6-11am: morning ramp up
        10, 10, 8, 7, 6, 5,  # 12-5pm: lunch peak then decline
        6, 9, 10, 9, 7, 4  # 6-11pm: evening peak
    ]
    
    hour = random.choices(range(24), weights=hour_weights)[0]
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    
    return base_date.replace(hour=hour, minute=minute, second=second, microsecond=0)

# COMMAND ----------

# Load products for transaction generation
products_table = spark.table(f"{CATALOG}.{SCHEMA}.{PRODUCTS_TABLE}")

# Use a subset of products to create realistic concentration
# Real e-commerce: most traffic goes to a small % of products
products_sample = products_table.sample(fraction=0.15, seed=RANDOM_SEED)  # ~6600 products
products_list = [row.asDict() for row in products_sample.collect()]

print(f"Loaded {len(products_list):,} products for transaction generation")
print(f"(Using subset to create realistic product popularity distribution)")

# COMMAND ----------

# Generate realistic synthetic transactions with power law distribution
# Note: "Transaction" = any event (view/cart/purchase), "Purchase" = completed sale
print("Starting realistic transaction generation...")
print(f"Users: {len(users_list):,}")
print(f"Products: {len(products_list):,}")
print(f"Sessions per user: 5-25 (increased for more volume)")
print(f"Events per session: 3-12 (increased for more volume)")
print(f"Expected conversion rate: ~2.5%")
print(f"Expected total events: ~1-1.5M (transactions)")
print(f"Expected purchases: ~25-40K")
print()

transactions_list = generate_transactions_realistic(
    generator=generator,
    users=users_list,
    products=products_list,
    sessions_per_user_range=(5, 25),
    events_per_session_range=(3, 12)
)

print(f"\n✓ Generated {len(transactions_list):,} total events (transactions)")
print(f"  (This includes views, add-to-carts, and purchases)")

# COMMAND ----------

# Compute statistics
stats = compute_user_statistics(transactions_list)

print("\nTransaction Statistics:")
print(f"  Total transactions: {stats['total_transactions']:,}")
print(f"  Total purchases: {stats['total_purchases']:,}")
print(f"  Total revenue: ${stats['total_revenue']:,.2f}")
print(f"  Conversion rate: {stats['conversion_rate']:.2%}")
print(f"  Avg purchase value: ${stats['avg_purchase_value']:.2f}")

# COMMAND ----------

# Convert to DataFrame
transactions_df = spark.createDataFrame(transactions_list)

# COMMAND ----------

# Display sample transactions
display(transactions_df.limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Quality Validation

# COMMAND ----------

# Verify all users have transactions
users_with_txns = transactions_df.select("user_id").distinct().count()
print(f"Users with transactions: {users_with_txns} / {NUM_USERS}")

# Verify transaction distribution
print("\nEvent type distribution:")
transactions_df.groupBy("event_type").count().orderBy(F.desc("count")).show()

# COMMAND ----------

# DBTITLE 1,Validate Realistic Data Patterns
print("=" * 60)
print("REALISM VALIDATION")
print("=" * 60)

# 1. Session-based browsing
print("\n1. SESSION PATTERNS:")
events_per_session = transactions_df.groupBy("session_id").count()
print(f"   Total sessions: {events_per_session.count():,}")
print(f"   Avg events per session: {events_per_session.agg(F.avg('count')).first()[0]:.1f}")
print("   Events per session distribution:")
events_per_session.groupBy("count").count().orderBy("count").show(10)

# 2. Funnel behavior - check sessions with purchases
print("\n2. PURCHASE FUNNEL ANALYSIS:")
purchase_sessions = transactions_df.filter(F.col("event_type") == "purchase").select("session_id").distinct()
print(f"   Sessions with purchases: {purchase_sessions.count():,}")

# Check if purchase sessions have preceding views/carts
funnel_check = (
    transactions_df
    .join(purchase_sessions, "session_id")
    .groupBy("session_id")
    .agg(
        F.sum(F.when(F.col("event_type") == "view", 1).otherwise(0)).alias("views"),
        F.sum(F.when(F.col("event_type") == "add_to_cart", 1).otherwise(0)).alias("carts"),
        F.sum(F.when(F.col("event_type") == "purchase", 1).otherwise(0)).alias("purchases")
    )
)
print("   Avg events in purchase sessions:")
funnel_check.agg(
    F.avg("views").alias("avg_views"),
    F.avg("carts").alias("avg_carts"),
    F.avg("purchases").alias("avg_purchases")
).show()

# 3. Conversion rate
print("\n3. CONVERSION RATE:")
total_sessions = transactions_df.select("session_id").distinct().count()
purchase_sessions_count = purchase_sessions.count()
conversion_rate = (purchase_sessions_count / total_sessions) * 100
print(f"   Session conversion rate: {conversion_rate:.2f}%")
print(f"   (Target: 2-3% - realistic e-commerce rate)")

# 4. Time patterns
print("\n4. TIME-OF-DAY PATTERNS:")
from pyspark.sql.functions import hour
transactions_with_hour = transactions_df.withColumn("hour", hour(F.col("timestamp")))
print("   Events by hour of day:")
transactions_with_hour.groupBy("hour").count().orderBy("hour").show(24)

print("\n" + "=" * 60)

# COMMAND ----------

# Verify products referenced in transactions exist
product_ids_in_txns = set([row["product_id"] for row in transactions_df.select("product_id").distinct().collect()])
product_ids_in_catalog = set([row["product_id"] for row in products_table.select("product_id").collect()])

orphaned_products = product_ids_in_txns - product_ids_in_catalog

if orphaned_products:
    print(f"WARNING: {len(orphaned_products)} product IDs in transactions not found in catalog")
else:
    print("✓ All transaction product IDs exist in catalog")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write to Unity Catalog

# COMMAND ----------

# Write users table
users_table_name = f"{CATALOG}.{SCHEMA}.{USERS_TABLE}"
users_df.write.format("delta").mode("overwrite").saveAsTable(users_table_name)

print(f"✓ Written {users_df.count()} users to {users_table_name}")

add_table_comment(
    CATALOG,
    SCHEMA,
    USERS_TABLE,
    "Synthetic user data for fashion recommendation testing"
)

# COMMAND ----------

# Write transactions table
transactions_table_name = f"{CATALOG}.{SCHEMA}.{TRANSACTIONS_TABLE}"
transactions_df.write.format("delta").mode("overwrite").saveAsTable(transactions_table_name)

print(f"✓ Written {transactions_df.count()} transactions to {transactions_table_name}")

add_table_comment(
    CATALOG,
    SCHEMA,
    TRANSACTIONS_TABLE,
    "Synthetic user-product interaction history (views, add-to-cart, purchases)"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary and Verification

# COMMAND ----------

print("=" * 60)
print("SYNTHETIC DATA GENERATION SUMMARY")
print("=" * 60)
print(f"\nUsers: {users_table_name}")
print(f"  Count: {spark.table(users_table_name).count():,}")
print(f"  Segments: {users_df.select('segment').distinct().count()}")

print(f"\nTransactions: {transactions_table_name}")
print(f"  Count: {spark.table(transactions_table_name).count():,}")
print(f"  Event types: {transactions_df.select('event_type').distinct().count()}")
print(f"  Date range: {transactions_df.agg(F.min('timestamp'), F.max('timestamp')).first()}")

print("\n" + "=" * 60)

# COMMAND ----------

# Show top active users
print("Top 10 most active users:")
spark.table(transactions_table_name).groupBy("user_id").count().orderBy(F.desc("count")).show(10)

# COMMAND ----------

# Show top purchased products
print("Top 10 most purchased products:")
(
    spark.table(transactions_table_name)
    .filter(F.col("event_type") == "purchase")
    .groupBy("product_id")
    .count()
    .orderBy(F.desc("count"))
    .join(products_table, "product_id")
    .select(
        "product_id", 
        "product_display_name", 
        "master_category", 
        F.col("count").alias("purchase_count")
    )
    .show(10, truncate=False)
)

# COMMAND ----------

# DBTITLE 1,Analyze Product Popularity Distribution
print("CURRENT PRODUCT POPULARITY ANALYSIS")
print("=" * 60)

# Get product interaction counts
product_popularity = (
    transactions_df
    .groupBy("product_id")
    .count()
    .orderBy(F.desc("count"))
)

print(f"\nTotal unique products in transactions: {product_popularity.count():,}")
print(f"Total events: {transactions_df.count():,}")

# Show top products
print("\nTop 20 most popular products:")
product_popularity.show(20)

# Distribution analysis
stats = product_popularity.select("count").summary("min", "25%", "50%", "75%", "max", "mean")
print("\nDistribution of product popularity:")
stats.show()

# Check if we have heavy hitters (top 20% should have ~80% of traffic)
total_events = transactions_df.count()
top_20_pct_products = int(product_popularity.count() * 0.2)
top_20_pct_events = product_popularity.limit(top_20_pct_products).agg(F.sum("count")).first()[0]
concentration = (top_20_pct_events / total_events) * 100

print(f"\nPareto Analysis (80/20 rule):")
print(f"  Top 20% of products ({top_20_pct_products:,} products) account for {concentration:.1f}% of events")
print(f"  Expected for realistic data: ~80%")
print(f"  Current: {'✓ GOOD' if concentration >= 75 else '✗ TOO UNIFORM'}")

print("\n" + "=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC 1. Run notebook `03_image_embeddings_pipeline` to generate product embeddings
# MAGIC 2. Run notebook `05_user_style_features` to compute user preference features
