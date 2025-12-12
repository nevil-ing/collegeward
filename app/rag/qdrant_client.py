from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import uuid

from app.core.config import settings
from app.core.logging import get_logger
from app.utils.exceptions import DatabaseError

logger = get_logger(__name__)

class QdrantManager:
    def __init__(self):
        self.client = None
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        self._initialize_client()

    def _initialize_client(self):
        try:
            host = settings.QDRANT_HOST
            if host.startswith(("http://", "https://")):
                url_parts = host.split("://")
                protocol = url_parts[0]
                host_part = url_parts[1].split(":")[0]
                url = f"{protocol}://{host_part}:{settings.QDRANT_PORT}"
                self.client = QdrantClient(
                    url=url,
                    api_key=settings.QDRANT_API_KEY,
                    timeout=settings.QDRANT_TIMEOUT,
                )
                logger.info(f"Connected to Qdrant at {url}")
            else:
                self.client = QdrantClient(
                    host=host,
                    port=settings.QDRANT_PORT,
                    api_key=settings.QDRANT_API_KEY,
                    timeout=settings.QDRANT_TIMEOUT,
                )
                logger.info(f"Connected to Qdrant at {host}:{settings.QDRANT_PORT}")
            self._ensure_collection_exists()

        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
           #  if settings.ENVIRONMENT != "development":
           #  raise DatabaseError(f"Vector database connection failed: {e}")

    def _ensure_collection_exists(self):
        """Create collection if it doesn't exist"""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]

            if self.collection_name not in collection_names:
                # Create collection with vector configuration
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=384,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"Qdrant collection already exists: {self.collection_name}")

        except Exception as e:
            logger.error(f"Failed to create/verify Qdrant collection: {e}")
            raise DatabaseError(f"Vector database collection setup failed: {e}")

    async def add_document_chunks(
            self,
            user_id: str,
            note_id: str,
            chunks: List[Dict[str, Any]]
    ) -> bool:
        """Add document chunks to vector database"""
        try:
            if self.client is None:
                logger.warning("Qdrant client not initialized, skipping vector storage")
                return False

            points = []
            for i, chunk in enumerate(chunks):
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=chunk["embedding"],
                    payload={
                        "user_id": user_id,
                        "note_id": note_id,
                        "chunk_text": chunk["text"],
                        "chunk_index": i,
                        "subject_tags": chunk.get("subject_tags", []),
                        "file_type": chunk.get("file_type", ""),
                        "created_at": chunk.get("created_at", "")
                    }
                )
                points.append(point)

            # Upload points to collection
            operation_info = self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

            logger.info(f"Added {len(points)} document chunks for user {user_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to add document chunks: {e}")
            raise DatabaseError(f"Failed to store document embeddings: {e}")

    async def search_similar_chunks(
            self,
            user_id: str,
            query_vector: List[float],
            limit: int = 10,
            score_threshold: float = 0.7,
            subject_tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar document chunks"""
        try:
            if self.client is None:
                logger.warning("Qdrant client not initialized, returning empty results")
                return []

            # Build search filter
            search_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(value=user_id)
                    )
                ]
            )

            # Add subject tag filter if provided
            if subject_tags:
                search_filter.must.append(
                    models.FieldCondition(
                        key="subject_tags",
                        match=models.MatchAny(any=subject_tags)
                    )
                )

            # Perform similarity search
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=search_filter,
                limit=limit,
                score_threshold=score_threshold
            )

            # Format results
            results = []
            for scored_point in search_result:
                results.append({
                    "id": scored_point.id,
                    "score": scored_point.score,
                    "text": scored_point.payload["chunk_text"],
                    "note_id": scored_point.payload["note_id"],
                    "chunk_index": scored_point.payload["chunk_index"],
                    "subject_tags": scored_point.payload.get("subject_tags", []),
                    "file_type": scored_point.payload.get("file_type", "")
                })

            logger.info(f"Found {len(results)} similar chunks for user {user_id}")
            return results

        except Exception as e:
            logger.error(f"Failed to search similar chunks: {e}")
            raise DatabaseError(f"Vector search failed: {e}")

    async def delete_user_chunks(self, user_id: str) -> bool:
        """Delete all chunks for a specific user"""
        try:
            if self.client is None:
                logger.warning("Qdrant client not initialized, skipping deletion")
                return False

            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="user_id",
                                match=models.MatchValue(value=user_id)
                            )
                        ]
                    )
                )
            )
            logger.info(f"Deleted all chunks for user {user_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete user chunks: {e}")
            raise DatabaseError(f"Failed to delete user embeddings: {e}")

    async def delete_note_chunks(self, user_id: str, note_id: str) -> bool:
        """Delete chunks for a specific note"""
        try:
            if self.client is None:
                logger.warning("Qdrant client not initialized, skipping deletion")
                return False

            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="user_id",
                                match=models.MatchValue(value=user_id)
                            ),
                            models.FieldCondition(
                                key="note_id",
                                match=models.MatchValue(value=note_id)
                            )
                        ]
                    )
                )
            )
            logger.info(f"Deleted chunks for note {note_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete note chunks: {e}")
            raise DatabaseError(f"Failed to delete note embeddings: {e}")

qdrant_manager = QdrantManager()
