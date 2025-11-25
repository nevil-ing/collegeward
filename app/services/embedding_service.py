from typing import List, Dict, Any
import asyncio
from sentence_transformers import SentenceTransformer
import numpy as np

from app.core.logging import get_logger
from app.utils.exceptions import ProcessingError

logger = get_logger(__name__)

class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.embedding_dimension = 384
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the sentence transformer model"""
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Initialized embedding model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise ProcessingError(f"Embedding model initialization failed: {e}")

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        if not texts:
            return []

        try:
            # Run embedding generation in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                self._generate_embeddings_sync,
                texts
            )

            logger.info(f"Generated embeddings for {len(texts)} texts")
            return embeddings

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise ProcessingError(f"Embedding generation failed: {e}")

    def _generate_embeddings_sync(self, texts: List[str]) -> List[List[float]]:
        """Synchronous embedding generation"""
        # Filter out empty texts
        valid_texts = [text for text in texts if text and text.strip()]

        if not valid_texts:
            return []

        # Generate embeddings
        embeddings = self.model.encode(
            valid_texts,
            convert_to_tensor=False,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        # Convert to list of lists
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()

        return embeddings

    async def generate_single_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        if not text or not text.strip():
            return [0.0] * self.embedding_dimension

        embeddings = await self.generate_embeddings([text])
        return embeddings[0] if embeddings else [0.0] * self.embedding_dimension

    async def process_document_chunks(
            self,
            chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process document chunks and add embeddings"""
        if not chunks:
            return []

        try:
            # Extract text from chunks
            texts = [chunk.get("text", "") for chunk in chunks]

            # Generate embeddings
            embeddings = await self.generate_embeddings(texts)

            # Add embeddings to chunks
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                if i < len(embeddings):
                    chunk_with_embedding = chunk.copy()
                    chunk_with_embedding["embedding"] = embeddings[i]
                    processed_chunks.append(chunk_with_embedding)

            logger.info(f"Processed {len(processed_chunks)} chunks with embeddings")
            return processed_chunks

        except Exception as e:
            logger.error(f"Failed to process document chunks: {e}")
            raise ProcessingError(f"Document chunk processing failed: {e}")

    def get_embedding_dimension(self) -> int:

        return self.embedding_dimension


embedding_service = EmbeddingService()



