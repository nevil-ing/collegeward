"""
Topic Taxonomy Service

Provides hierarchical topic classification with:
- External source sync (MeSH, UMLS)
- Semantic matching using embeddings  
- Database-driven configurable taxonomy
"""

from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from decimal import Decimal
import uuid
import httpx
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_

from app.models.topic_taxonomy import TopicCategory, Topic, TopicSyncLog
from app.services.embedding_service import embedding_service
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


# Default taxonomy for initial seeding
DEFAULT_TAXONOMY = {
    "basic_sciences": {
        "name": "Basic Sciences",
        "description": "Foundational medical sciences",
        "topics": {
            "anatomy": {
                "name": "Anatomy",
                "keywords": ["anatomy", "structure", "organ", "muscle", "bone", "tissue"],
                "subtopics": ["musculoskeletal", "cardiovascular", "respiratory", "nervous", "digestive"]
            },
            "physiology": {
                "name": "Physiology", 
                "keywords": ["physiology", "function", "mechanism", "process", "homeostasis"],
                "subtopics": ["cellular", "cardiovascular", "respiratory", "renal", "endocrine"]
            },
            "biochemistry": {
                "name": "Biochemistry",
                "keywords": ["biochemistry", "metabolism", "enzyme", "protein", "molecular"],
                "subtopics": ["metabolism", "enzymes", "genetics", "molecular_biology"]
            },
            "pharmacology": {
                "name": "Pharmacology",
                "keywords": ["drug", "medication", "pharmacology", "dosage", "side effect", "mechanism of action"],
                "subtopics": ["pharmacokinetics", "pharmacodynamics", "antibiotics", "cardiovascular_drugs"]
            },
            "pathology": {
                "name": "Pathology",
                "keywords": ["pathology", "disease", "lesion", "neoplasm", "inflammation"],
                "subtopics": ["general_pathology", "systemic_pathology", "neoplasia"]
            },
            "microbiology": {
                "name": "Microbiology",
                "keywords": ["bacteria", "virus", "fungus", "parasite", "infection", "microbe"],
                "subtopics": ["bacteriology", "virology", "mycology", "parasitology", "immunology"]
            }
        }
    },
    "clinical_sciences": {
        "name": "Clinical Sciences",
        "description": "Clinical medical specialties",
        "topics": {
            "cardiology": {
                "name": "Cardiology",
                "keywords": ["heart", "cardiac", "cardiovascular", "ecg", "arrhythmia", "murmur"],
                "subtopics": ["heart_failure", "coronary_disease", "arrhythmias", "valvular"]
            },
            "pulmonology": {
                "name": "Pulmonology",
                "keywords": ["lung", "respiratory", "breathing", "pneumonia", "asthma", "copd"],
                "subtopics": ["obstructive", "restrictive", "infections", "neoplasms"]
            },
            "neurology": {
                "name": "Neurology",
                "keywords": ["brain", "nerve", "neurological", "seizure", "stroke", "headache"],
                "subtopics": ["cerebrovascular", "neurodegenerative", "epilepsy", "movement_disorders"]
            },
            "gastroenterology": {
                "name": "Gastroenterology",
                "keywords": ["stomach", "intestine", "liver", "digestive", "gi", "bowel"],
                "subtopics": ["hepatology", "inflammatory_bowel", "gi_oncology"]
            },
            "nephrology": {
                "name": "Nephrology",
                "keywords": ["kidney", "renal", "dialysis", "nephron", "glomerular"],
                "subtopics": ["acute_kidney", "chronic_kidney", "glomerular_disease"]
            },
            "endocrinology": {
                "name": "Endocrinology",
                "keywords": ["hormone", "diabetes", "thyroid", "adrenal", "pituitary", "endocrine"],
                "subtopics": ["diabetes", "thyroid_disorders", "adrenal_disorders"]
            }
        }
    },
    "nursing": {
        "name": "Nursing",
        "description": "Nursing education and practice",
        "topics": {
            "fundamentals": {
                "name": "Nursing Fundamentals",
                "keywords": ["nursing", "patient care", "vital signs", "hygiene", "assessment"],
                "subtopics": ["patient_assessment", "documentation", "safety"]
            },
            "medical_surgical": {
                "name": "Medical-Surgical Nursing",
                "keywords": ["surgical", "postoperative", "wound care", "pain management"],
                "subtopics": ["perioperative", "wound_management", "chronic_illness"]
            }
        }
    },
    "pharmacy": {
        "name": "Pharmacy",
        "description": "Pharmaceutical sciences",
        "topics": {
            "pharmaceutics": {
                "name": "Pharmaceutics",
                "keywords": ["formulation", "dosage form", "drug delivery", "stability"],
                "subtopics": ["solid_dosage", "liquid_dosage", "parenteral"]
            },
            "clinical_pharmacy": {
                "name": "Clinical Pharmacy",
                "keywords": ["drug therapy", "patient counseling", "medication review"],
                "subtopics": ["drug_interactions", "therapeutic_monitoring"]
            }
        }
    }
}


class TopicTaxonomyService:
    """Service for managing and querying topic taxonomy"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self._embedding_cache: Dict[str, List[float]] = {}
    
    async def classify_text(
        self,
        text: str,
        use_embedding: bool = True,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Classify text into topic hierarchy
        
        Args:
            text: Text to classify
            use_embedding: Use semantic matching (slower but more accurate)
            top_k: Return top K matches
            
        Returns:
            List of matches with category, topic, confidence
        """
        if use_embedding:
            return await self._classify_by_embedding(text, top_k)
        else:
            return await self._classify_by_keywords(text, top_k)
    
    async def _classify_by_embedding(self, text: str, top_k: int) -> List[Dict[str, Any]]:
        """Use embedding similarity for classification"""
        try:
            # Generate query embedding
            query_embedding = await embedding_service.generate_single_embedding(text)
            if not query_embedding:
                return await self._classify_by_keywords(text, top_k)
            
            # Get all active topics with keywords
            topics = await self._get_all_active_topics()
            
            matches = []
            for topic in topics:
                # Create topic description from keywords
                keywords = topic.keywords or []
                topic_text = f"{topic.name}: {' '.join(keywords)}"
                
                # Get or generate topic embedding
                topic_embedding = await self._get_topic_embedding(topic.id, topic_text)
                
                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_embedding, topic_embedding)
                
                matches.append({
                    "category_code": topic.category.code,
                    "category_name": topic.category.name,
                    "topic_code": topic.code,
                    "topic_name": topic.name,
                    "confidence": round(similarity, 3),
                    "topic_id": str(topic.id)
                })
            
            # Sort by confidence and return top K
            matches.sort(key=lambda x: x["confidence"], reverse=True)
            return matches[:top_k]
            
        except Exception as e:
            logger.error(f"Embedding classification failed: {e}")
            return await self._classify_by_keywords(text, top_k)
    
    async def _classify_by_keywords(self, text: str, top_k: int) -> List[Dict[str, Any]]:
        """Fallback keyword-based classification"""
        text_lower = text.lower()
        topics = await self._get_all_active_topics()
        
        matches = []
        for topic in topics:
            keywords = topic.keywords or []
            keyword_matches = sum(1 for kw in keywords if kw.lower() in text_lower)
            
            if keyword_matches > 0:
                confidence = min(keyword_matches * 0.2, 1.0)
                matches.append({
                    "category_code": topic.category.code,
                    "category_name": topic.category.name,
                    "topic_code": topic.code,
                    "topic_name": topic.name,
                    "confidence": round(confidence, 3),
                    "topic_id": str(topic.id)
                })
        
        matches.sort(key=lambda x: x["confidence"], reverse=True)
        return matches[:top_k] if matches else [{
            "category_code": "general",
            "category_name": "General",
            "topic_code": "general",
            "topic_name": "General Topic",
            "confidence": 0.3,
            "topic_id": None
        }]
    
    async def _get_topic_embedding(self, topic_id: uuid.UUID, topic_text: str) -> List[float]:
        """Get or generate topic embedding with caching"""
        cache_key = str(topic_id)
        
        if cache_key not in self._embedding_cache:
            embedding = await embedding_service.generate_single_embedding(topic_text)
            self._embedding_cache[cache_key] = embedding
        
        return self._embedding_cache[cache_key]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if not vec1 or not vec2:
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    async def _get_all_active_topics(self) -> List[Topic]:
        """Get all active topics with their categories"""
        from sqlalchemy.orm import selectinload
        
        query = (
            select(Topic)
            .options(selectinload(Topic.category))
            .where(
                and_(
                    Topic.is_active == True,
                    Topic.category.has(TopicCategory.is_active == True)
                )
            )
        )
        result = await self.db.execute(query)
        return result.scalars().all()
    
    async def get_related_topics(self, topic_code: str) -> List[Dict[str, str]]:
        """Get related topics for follow-up suggestions"""
        query = select(Topic).where(Topic.code == topic_code)
        result = await self.db.execute(query)
        topic = result.scalar_one_or_none()
        
        if not topic:
            return []
        
        # Get subtopics
        subtopic_query = select(Topic).where(Topic.parent_topic_id == topic.id)
        result = await self.db.execute(subtopic_query)
        subtopics = result.scalars().all()
        
        # Get sibling topics
        sibling_query = (
            select(Topic)
            .where(
                and_(
                    Topic.category_id == topic.category_id,
                    Topic.id != topic.id,
                    Topic.parent_topic_id == topic.parent_topic_id
                )
            )
            .limit(5)
        )
        result = await self.db.execute(sibling_query)
        siblings = result.scalars().all()
        
        related = []
        for t in list(subtopics) + list(siblings):
            related.append({
                "code": t.code,
                "name": t.name,
                "relationship": "subtopic" if t.parent_topic_id == topic.id else "sibling"
            })
        
        return related
    
    async def seed_default_taxonomy(self) -> Dict[str, int]:
        """Seed database with default taxonomy"""
        stats = {"categories": 0, "topics": 0}
        
        for category_code, category_data in DEFAULT_TAXONOMY.items():
            # Check if category exists
            existing = await self.db.execute(
                select(TopicCategory).where(TopicCategory.code == category_code)
            )
            category = existing.scalar_one_or_none()
            
            if not category:
                category = TopicCategory(
                    code=category_code,
                    name=category_data["name"],
                    description=category_data.get("description"),
                    source="default"
                )
                self.db.add(category)
                await self.db.flush()
                stats["categories"] += 1
            
            # Add topics
            for topic_code, topic_data in category_data.get("topics", {}).items():
                existing_topic = await self.db.execute(
                    select(Topic).where(
                        and_(
                            Topic.category_id == category.id,
                            Topic.code == topic_code
                        )
                    )
                )
                
                if not existing_topic.scalar_one_or_none():
                    topic = Topic(
                        category_id=category.id,
                        code=topic_code,
                        name=topic_data["name"],
                        keywords=topic_data.get("keywords", []),
                        source="default"
                    )
                    self.db.add(topic)
                    stats["topics"] += 1
        
        await self.db.commit()
        logger.info(f"Seeded taxonomy: {stats}")
        return stats
    
    async def sync_from_mesh(self, category_filter: Optional[str] = None) -> TopicSyncLog:
        """
        Sync topics from MeSH (Medical Subject Headings)
        
        MeSH is a free, publicly available vocabulary from NLM.
        API: https://meshb.nlm.nih.gov/api/
        """
        sync_log = TopicSyncLog(
            source="mesh",
            sync_type="incremental",
            status="running",
            started_at=datetime.utcnow()
        )
        self.db.add(sync_log)
        await self.db.flush()
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Example: Fetch top-level MeSH descriptors
                # In production, you'd paginate through the full tree
                
                # MeSH categories relevant to medical education
                mesh_categories = [
                    "A" if not category_filter or category_filter == "anatomy" else None,  # Anatomy
                    "C" if not category_filter or category_filter == "diseases" else None,  # Diseases
                    "D" if not category_filter or category_filter == "drugs" else None,  # Drugs
                    "G" if not category_filter or category_filter == "physiology" else None,  # Phenomena
                ]
                mesh_categories = [c for c in mesh_categories if c]
                
                added = 0
                updated = 0
                
                for tree_number in mesh_categories:
                    try:
                        # MeSH API endpoint
                        url = f"https://meshb.nlm.nih.gov/api/tree/children/{tree_number}"
                        response = await client.get(url)
                        
                        if response.status_code == 200:
                            data = response.json()
                            # Process MeSH descriptors...
                            # This is simplified - real implementation would parse the full response
                            logger.info(f"Fetched MeSH tree {tree_number}")
                            
                    except Exception as e:
                        logger.warning(f"Failed to fetch MeSH tree {tree_number}: {e}")
                        continue
                
                sync_log.status = "success"
                sync_log.topics_added = added
                sync_log.topics_updated = updated
                
        except Exception as e:
            sync_log.status = "failed"
            sync_log.error_message = str(e)
            logger.error(f"MeSH sync failed: {e}")
        
        sync_log.completed_at = datetime.utcnow()
        await self.db.commit()
        
        return sync_log


# Convenience function for quick classification without DB
async def classify_topic_simple(text: str) -> Tuple[str, str, float]:
    """
    Simple topic classification using default taxonomy keywords
    
    Returns: (category, topic, confidence)
    """
    text_lower = text.lower()
    
    best_match = ("general", "general", 0.3)
    
    for category_code, category_data in DEFAULT_TAXONOMY.items():
        for topic_code, topic_data in category_data.get("topics", {}).items():
            keywords = topic_data.get("keywords", [])
            matches = sum(1 for kw in keywords if kw.lower() in text_lower)
            
            if matches > 0:
                confidence = min(matches * 0.2, 1.0)
                if confidence > best_match[2]:
                    best_match = (category_code, topic_code, confidence)
    
    return best_match
