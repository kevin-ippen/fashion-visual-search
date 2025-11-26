"""
Image embedding generation utilities using CLIP model via Databricks Model Serving.
"""

from typing import List, Optional
import requests
import base64
from io import BytesIO
from PIL import Image
import numpy as np


class ImageEmbedder:
    """Client for generating image embeddings via Databricks Model Serving endpoint."""

    def __init__(self, endpoint_url: str, token: str):
        """
        Initialize the image embedder.

        Args:
            endpoint_url: Full URL to the Databricks Model Serving endpoint
            token: Databricks API token for authentication
        """
        self.endpoint_url = endpoint_url
        self.token = token
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

    def encode_image_to_base64(self, image_path: str) -> str:
        """
        Encode an image file to base64 string.

        Args:
            image_path: Path to the image file

        Returns:
            Base64 encoded string of the image
        """
        with Image.open(image_path) as img:
            # Resize if too large (CLIP typically uses 224x224)
            if img.size[0] > 512 or img.size[1] > 512:
                img.thumbnail((512, 512), Image.Resampling.LANCZOS)

            buffer = BytesIO()
            img.save(buffer, format="PNG")
            img_bytes = buffer.getvalue()
            return base64.b64encode(img_bytes).decode("utf-8")

    def get_embedding(self, image_path: Optional[str] = None,
                     image_base64: Optional[str] = None) -> np.ndarray:
        """
        Get embedding for a single image.

        Args:
            image_path: Path to image file (alternative to image_base64)
            image_base64: Base64 encoded image string (alternative to image_path)

        Returns:
            Numpy array of embedding vector
        """
        if image_path:
            image_base64 = self.encode_image_to_base64(image_path)
        elif not image_base64:
            raise ValueError("Must provide either image_path or image_base64")

        payload = {
            "inputs": {
                "image": image_base64
            }
        }

        response = requests.post(
            self.endpoint_url,
            headers=self.headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()

        result = response.json()
        # Adjust based on actual endpoint response structure
        embedding = result.get("predictions", [result.get("embedding")])[0]
        return np.array(embedding)

    def get_embeddings_batch(self, image_paths: List[str]) -> List[np.ndarray]:
        """
        Get embeddings for a batch of images.

        Args:
            image_paths: List of paths to image files

        Returns:
            List of numpy arrays containing embedding vectors
        """
        embeddings = []
        for img_path in image_paths:
            try:
                emb = self.get_embedding(image_path=img_path)
                embeddings.append(emb)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                # Return zero vector on failure
                embeddings.append(np.zeros(512))  # Adjust dimension as needed

        return embeddings


def compute_cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Compute cosine similarity between two embeddings.

    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector

    Returns:
        Cosine similarity score (0-1)
    """
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))


def aggregate_embeddings(embeddings: List[np.ndarray], method: str = "mean") -> np.ndarray:
    """
    Aggregate multiple embeddings into a single embedding.

    Args:
        embeddings: List of embedding vectors
        method: Aggregation method - 'mean', 'max', or 'weighted_mean'

    Returns:
        Aggregated embedding vector
    """
    if not embeddings:
        raise ValueError("Cannot aggregate empty list of embeddings")

    embeddings_array = np.array(embeddings)

    if method == "mean":
        return np.mean(embeddings_array, axis=0)
    elif method == "max":
        return np.max(embeddings_array, axis=0)
    elif method == "weighted_mean":
        # Could implement decay weights based on recency
        weights = np.linspace(0.5, 1.0, len(embeddings))
        weights = weights / weights.sum()
        return np.average(embeddings_array, axis=0, weights=weights)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
