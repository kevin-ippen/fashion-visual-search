"""
Utilities for interacting with Databricks Mosaic AI Vector Search.
"""

from typing import List, Dict, Any, Optional
import requests
import numpy as np


class VectorSearchClient:
    """Client for Databricks Mosaic AI Vector Search."""

    def __init__(self, workspace_url: str, token: str):
        """
        Initialize the vector search client.

        Args:
            workspace_url: Databricks workspace URL
            token: Databricks API token
        """
        self.workspace_url = workspace_url.rstrip("/")
        self.token = token
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

    def create_endpoint(self, endpoint_name: str, endpoint_type: str = "STANDARD") -> Dict:
        """
        Create a vector search endpoint.

        Args:
            endpoint_name: Name for the endpoint
            endpoint_type: Type of endpoint (STANDARD)

        Returns:
            Response from endpoint creation
        """
        url = f"{self.workspace_url}/api/2.0/vector-search/endpoints"
        payload = {
            "name": endpoint_name,
            "endpoint_type": endpoint_type
        }

        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()

    def create_index(
        self,
        endpoint_name: str,
        index_name: str,
        primary_key: str,
        embedding_dimension: int,
        embedding_vector_column: str,
        source_table: str,
        distance_metric: str = "COSINE"
    ) -> Dict:
        """
        Create a vector search index.

        Args:
            endpoint_name: Name of the vector search endpoint
            index_name: Full name of the index (catalog.schema.index_name)
            primary_key: Primary key column name
            embedding_dimension: Dimension of embedding vectors
            embedding_vector_column: Column containing embeddings
            source_table: Source Delta table (catalog.schema.table)
            distance_metric: Distance metric (COSINE, L2, INNER_PRODUCT)

        Returns:
            Response from index creation
        """
        url = f"{self.workspace_url}/api/2.0/vector-search/indexes"
        payload = {
            "name": index_name,
            "endpoint_name": endpoint_name,
            "primary_key": primary_key,
            "index_type": "DELTA_SYNC",
            "delta_sync_index_spec": {
                "source_table": source_table,
                "embedding_dimension": embedding_dimension,
                "embedding_vector_column": embedding_vector_column
            },
            "distance_metric": distance_metric
        }

        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()

    def query_index(
        self,
        index_name: str,
        query_vector: np.ndarray,
        num_results: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Query a vector search index.

        Args:
            index_name: Full name of the index
            query_vector: Query embedding vector
            num_results: Number of results to return
            filters: Optional metadata filters

        Returns:
            List of search results with product IDs and scores
        """
        url = f"{self.workspace_url}/api/2.0/vector-search/indexes/{index_name}/query"

        # Convert numpy array to list
        query_list = query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector

        payload = {
            "query_vector": query_list,
            "num_results": num_results
        }

        if filters:
            payload["filters"] = filters

        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json().get("result", {}).get("data_array", [])

    def sync_index(self, index_name: str) -> Dict:
        """
        Trigger sync for a vector search index.

        Args:
            index_name: Full name of the index

        Returns:
            Response from sync trigger
        """
        url = f"{self.workspace_url}/api/2.0/vector-search/indexes/{index_name}/sync"
        response = requests.post(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def get_index_status(self, index_name: str) -> Dict:
        """
        Get status of a vector search index.

        Args:
            index_name: Full name of the index

        Returns:
            Index status information
        """
        url = f"{self.workspace_url}/api/2.0/vector-search/indexes/{index_name}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()


class VectorSearchHelper:
    """Helper functions for working with vector search results."""

    @staticmethod
    def parse_search_results(
        results: List[Dict[str, Any]],
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Parse and filter vector search results.

        Args:
            results: Raw results from vector search
            score_threshold: Minimum score threshold for results

        Returns:
            Filtered list of results
        """
        parsed = []
        for result in results:
            score = result.get("score", 0.0)

            if score_threshold and score < score_threshold:
                continue

            parsed.append({
                "product_id": result.get("product_id"),
                "score": score,
                "metadata": result
            })

        return parsed

    @staticmethod
    def create_filters(
        category: Optional[str] = None,
        brand: Optional[str] = None,
        price_range: Optional[tuple[float, float]] = None
    ) -> Dict[str, Any]:
        """
        Create filter dict for vector search queries.

        Args:
            category: Filter by category
            brand: Filter by brand
            price_range: Filter by price range (min, max)

        Returns:
            Filter dictionary
        """
        filters = {}

        if category:
            filters["category"] = category

        if brand:
            filters["brand"] = brand

        if price_range:
            min_price, max_price = price_range
            filters["price"] = {
                "$gte": min_price,
                "$lte": max_price
            }

        return filters
