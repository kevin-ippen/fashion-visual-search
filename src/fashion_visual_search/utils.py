"""
General utility functions for the fashion visual search system.
"""

from typing import Any, Dict, List, Optional
import json
from datetime import datetime
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


def get_spark() -> SparkSession:
    """Get or create Spark session."""
    return SparkSession.builder.getOrCreate()


def create_catalog_schema(catalog: str, schema: str) -> None:
    """
    Create Unity Catalog schema if it doesn't exist.

    Args:
        catalog: Catalog name
        schema: Schema name
    """
    spark = get_spark()

    # Create schema
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
    print(f"Schema {catalog}.{schema} is ready")


def table_exists(catalog: str, schema: str, table: str) -> bool:
    """
    Check if a table exists in Unity Catalog.

    Args:
        catalog: Catalog name
        schema: Schema name
        table: Table name

    Returns:
        True if table exists
    """
    spark = get_spark()
    full_name = f"{catalog}.{schema}.{table}"

    try:
        spark.sql(f"DESCRIBE TABLE {full_name}")
        return True
    except Exception:
        return False


def get_table_count(catalog: str, schema: str, table: str) -> int:
    """
    Get row count for a table.

    Args:
        catalog: Catalog name
        schema: Schema name
        table: Table name

    Returns:
        Number of rows
    """
    spark = get_spark()
    full_name = f"{catalog}.{schema}.{table}"
    return spark.table(full_name).count()


def optimize_table(catalog: str, schema: str, table: str, zorder_cols: Optional[List[str]] = None) -> None:
    """
    Optimize a Delta table with optional Z-ordering.

    Args:
        catalog: Catalog name
        schema: Schema name
        table: Table name
        zorder_cols: Optional columns for Z-ordering
    """
    spark = get_spark()
    full_name = f"{catalog}.{schema}.{table}"

    optimize_sql = f"OPTIMIZE {full_name}"
    if zorder_cols:
        zorder_clause = ", ".join(zorder_cols)
        optimize_sql += f" ZORDER BY ({zorder_clause})"

    spark.sql(optimize_sql)
    print(f"Optimized {full_name}")


def add_table_comment(catalog: str, schema: str, table: str, comment: str) -> None:
    """
    Add comment to a table.

    Args:
        catalog: Catalog name
        schema: Schema name
        table: Table name
        comment: Comment text
    """
    spark = get_spark()
    full_name = f"{catalog}.{schema}.{table}"
    spark.sql(f"COMMENT ON TABLE {full_name} IS '{comment}'")


def log_job_metrics(
    job_name: str,
    metrics: Dict[str, Any],
    log_table: str = "main.fashion_demo.job_metrics"
) -> None:
    """
    Log job execution metrics to a tracking table.

    Args:
        job_name: Name of the job
        metrics: Dictionary of metrics to log
        log_table: Full name of the metrics table
    """
    spark = get_spark()

    log_data = [{
        "job_name": job_name,
        "timestamp": datetime.now().isoformat(),
        "metrics": json.dumps(metrics)
    }]

    log_df = spark.createDataFrame(log_data)

    # Append to log table
    log_df.write.format("delta").mode("append").saveAsTable(log_table)


def sample_products_for_testing(
    catalog: str,
    schema: str,
    table: str = "products",
    sample_size: int = 100,
    categories: Optional[List[str]] = None
) -> DataFrame:
    """
    Sample products for testing purposes.

    Args:
        catalog: Catalog name
        schema: Schema name
        table: Products table name
        sample_size: Number of products to sample
        categories: Optional list of categories to filter

    Returns:
        Sampled products DataFrame
    """
    spark = get_spark()
    full_name = f"{catalog}.{schema}.{table}"

    df = spark.table(full_name)

    if categories:
        df = df.filter(F.col("category").isin(categories))

    return df.sample(fraction=min(1.0, sample_size / df.count()), seed=42).limit(sample_size)


def get_config_from_table(
    config_table: str = "main.fashion_demo.config",
    config_key: str = "scoring_weights"
) -> Dict[str, Any]:
    """
    Retrieve configuration from a Unity Catalog table.

    Args:
        config_table: Full name of config table
        config_key: Key for the configuration

    Returns:
        Configuration dictionary
    """
    spark = get_spark()

    config_df = spark.table(config_table).filter(F.col("config_key") == config_key)

    if config_df.count() == 0:
        return {}

    config_json = config_df.select("config_value").first()[0]
    return json.loads(config_json)


def save_config_to_table(
    config_key: str,
    config_value: Dict[str, Any],
    config_table: str = "main.fashion_demo.config"
) -> None:
    """
    Save configuration to a Unity Catalog table.

    Args:
        config_key: Key for the configuration
        config_value: Configuration dictionary
        config_table: Full name of config table
    """
    spark = get_spark()

    config_data = [{
        "config_key": config_key,
        "config_value": json.dumps(config_value),
        "updated_at": datetime.now().isoformat()
    }]

    config_df = spark.createDataFrame(config_data)

    # Upsert to config table
    config_df.write.format("delta").mode("append").saveAsTable(config_table)


class DataQualityChecker:
    """Data quality validation utilities."""

    @staticmethod
    def check_nulls(df: DataFrame, required_columns: List[str]) -> Dict[str, int]:
        """
        Check for null values in required columns.

        Args:
            df: DataFrame to check
            required_columns: Columns that should not be null

        Returns:
            Dictionary of column -> null count
        """
        null_counts = {}

        for col in required_columns:
            null_count = df.filter(F.col(col).isNull()).count()
            if null_count > 0:
                null_counts[col] = null_count

        return null_counts

    @staticmethod
    def check_duplicates(df: DataFrame, key_columns: List[str]) -> int:
        """
        Check for duplicate records based on key columns.

        Args:
            df: DataFrame to check
            key_columns: Columns that should be unique

        Returns:
            Number of duplicate records
        """
        total_count = df.count()
        distinct_count = df.select(key_columns).distinct().count()
        return total_count - distinct_count

    @staticmethod
    def check_data_freshness(
        df: DataFrame,
        timestamp_column: str,
        max_age_hours: int = 24
    ) -> bool:
        """
        Check if data is fresh (within max age).

        Args:
            df: DataFrame to check
            timestamp_column: Column containing timestamp
            max_age_hours: Maximum age in hours

        Returns:
            True if data is fresh
        """
        max_timestamp = df.agg(F.max(timestamp_column)).first()[0]

        if max_timestamp is None:
            return False

        age_hours = (datetime.now() - max_timestamp).total_seconds() / 3600
        return age_hours <= max_age_hours
