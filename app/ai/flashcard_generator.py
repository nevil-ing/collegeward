import re
import json
import logging
from typing import List, Optional, Dict, Any, Tuple
from uuid import UUID

from app.services.ai_service import AIServiceManager, ConversationMessage
from app.schemas.flashcard_schema import FlashcardCreate
from app.core.config import settings

logger = logging.getLogger(__name__)


class MedicalFlashcardGenerator:
    """
    Specialized flashcard generator for medical content using AI
    """

    # Medical subject categories for automatic tagging
    MEDICAL_SUBJECTS = {
        'anatomy': ['anatomy', 'structure', 'organ', 'system', 'body', 'tissue', 'cell'],
        'physiology': ['function', 'process', 'mechanism', 'pathway', 'regulation', 'homeostasis'],
        'pathology': ['disease', 'disorder', 'pathology', 'abnormal', 'lesion', 'tumor', 'cancer'],
        'pharmacology': ['drug', 'medication', 'treatment', 'therapy', 'dose', 'side effect'],
        'microbiology': ['bacteria', 'virus', 'infection', 'microbe', 'pathogen', 'antibiotic'],
        'immunology': ['immune', 'antibody', 'antigen', 'vaccine', 'allergy', 'autoimmune'],
        'cardiology': ['heart', 'cardiac', 'blood', 'circulation', 'vessel', 'pressure'],
        'neurology': ['brain', 'nerve', 'nervous', 'neuron', 'cognitive', 'mental'],
        'endocrinology': ['hormone', 'gland', 'diabetes', 'thyroid', 'insulin', 'metabolism'],
        'gastroenterology': ['digestive', 'stomach', 'intestine', 'liver', 'pancreas', 'gut'],
        'respiratory': ['lung', 'breathing', 'respiratory', 'oxygen', 'airway', 'pulmonary'],
        'nephrology': ['kidney', 'renal', 'urine', 'filtration', 'electrolyte', 'fluid'],
        'hematology': ['blood', 'anemia', 'clotting', 'platelet', 'hemoglobin', 'transfusion'],
        'oncology': ['cancer', 'tumor', 'malignant', 'chemotherapy', 'radiation', 'metastasis'],
        'pediatrics': ['child', 'infant', 'pediatric', 'development', 'growth', 'vaccination'],
        'obstetrics': ['pregnancy', 'birth', 'fetal', 'maternal', 'delivery', 'prenatal'],
        'psychiatry': ['mental', 'psychiatric', 'depression', 'anxiety', 'psychosis', 'therapy'],
        'surgery': ['surgical', 'operation', 'procedure', 'incision', 'suture', 'anesthesia'],
        'radiology': ['imaging', 'x-ray', 'ct', 'mri', 'ultrasound', 'scan'],
        'emergency': ['emergency', 'trauma', 'acute', 'critical', 'resuscitation', 'shock']
    }

    def __init__(self, ai_service: AIServiceManager):
        self.ai_service = ai_service

    async def generate_from_medical_text(
            self,
            text: str,
            source_type: str,
            source_id: Optional[UUID] = None,
            existing_tags: Optional[List[str]] = None,
            max_cards: int = 10,
            focus_areas: Optional[List[str]] = None
    ) -> List[FlashcardCreate]:
        """
        Generate byte-sized summary flashcards from medical text content.
        Each flashcard represents a digestible chunk of the note content, breaking
        down complex information into concise, reviewable summaries.

        Args:
            text: Source medical text content
            source_type: Type of source ('notes', 'chat', 'manual')
            source_id: ID of source document/conversation
            existing_tags: Existing subject tags from source
            max_cards: Maximum number of summary cards to generate
            focus_areas: Specific medical areas to focus on

        Returns:
            List of flashcard creation objects with byte-sized summaries
        """
        try:
            # Analyze text for medical content
            content_analysis = self._analyze_medical_content(text)

            # Generate subject-specific prompts
            prompt = self._create_medical_flashcard_prompt(
                text, max_cards, content_analysis, focus_areas
            )

            messages = [
                ConversationMessage(role="user", content=prompt)
            ]

            # Generate flashcards using AI with medical context
            response = await self.ai_service.generate_response(
                messages=messages,
                mode="ai_mode",
                temperature=0.2  # Lower temperature for medical accuracy
            )

            # Parse and enhance flashcards with medical tagging
            flashcards = self._parse_medical_flashcard_response(
                response.content,
                source_type,
                source_id,
                existing_tags,
                content_analysis
            )

            logger.info(f"Generated {len(flashcards)} medical flashcards from {source_type}")
            return flashcards

        except Exception as e:
            logger.error(f"Medical flashcard generation failed: {str(e)}")
            return []

    async def generate_clinical_scenarios(
            self,
            text: str,
            source_type: str,
            source_id: Optional[UUID] = None,
            max_scenarios: int = 5
    ) -> List[FlashcardCreate]:
        """
        Generate clinical scenario-based flashcards for case-based learning

        Args:
            text: Source medical text content
            source_type: Type of source
            source_id: ID of source
            max_scenarios: Maximum number of scenario cards

        Returns:
            List of clinical scenario flashcards
        """
        try:
            prompt = self._create_clinical_scenario_prompt(text, max_scenarios)

            messages = [
                ConversationMessage(role="user", content=prompt)
            ]

            response = await self.ai_service.generate_response(
                messages=messages,
                mode="ai_mode",
                temperature=0.3
            )

            scenarios = self._parse_clinical_scenario_response(
                response.content,
                source_type,
                source_id
            )

            logger.info(f"Generated {len(scenarios)} clinical scenario flashcards")
            return scenarios

        except Exception as e:
            logger.error(f"Clinical scenario generation failed: {str(e)}")
            return []

    def _analyze_medical_content(self, text: str) -> Dict[str, Any]:
        """
        Analyze text to identify medical subjects and complexity

        Args:
            text: Medical text content

        Returns:
            Dictionary with content analysis results
        """
        text_lower = text.lower()

        # Identify medical subjects
        detected_subjects = []
        subject_scores = {}

        for subject, keywords in self.MEDICAL_SUBJECTS.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                subject_scores[subject] = score
                detected_subjects.append(subject)

        # Sort subjects by relevance
        detected_subjects.sort(key=lambda x: subject_scores.get(x, 0), reverse=True)

        # Estimate complexity based on medical terminology density
        medical_terms = [
            'pathophysiology', 'etiology', 'diagnosis', 'prognosis', 'treatment',
            'syndrome', 'manifestation', 'complication', 'contraindication',
            'pharmacokinetics', 'pharmacodynamics', 'therapeutic', 'clinical'
        ]

        complexity_score = sum(1 for term in medical_terms if term in text_lower)

        # Determine difficulty level
        if complexity_score >= 8:
            difficulty = 5  # Advanced
        elif complexity_score >= 5:
            difficulty = 4  # Intermediate-Advanced
        elif complexity_score >= 3:
            difficulty = 3  # Intermediate
        elif complexity_score >= 1:
            difficulty = 2  # Basic-Intermediate
        else:
            difficulty = 1  # Basic

        return {
            'detected_subjects': detected_subjects[:5],  # Top 5 subjects
            'subject_scores': subject_scores,
            'complexity_score': complexity_score,
            'estimated_difficulty': difficulty,
            'text_length': len(text),
            'has_clinical_content': any(term in text_lower for term in [
                'patient', 'clinical', 'diagnosis', 'treatment', 'symptoms'
            ])
        }

    def _create_medical_flashcard_prompt(
            self,
            text: str,
            max_cards: int,
            content_analysis: Dict[str, Any],
            focus_areas: Optional[List[str]] = None
    ) -> str:
        """Create specialized prompt for medical flashcard generation"""

        subjects_context = ""
        if content_analysis['detected_subjects']:
            subjects_context = f"Focus on these medical areas: {', '.join(content_analysis['detected_subjects'][:3])}"

        focus_context = ""
        if focus_areas:
            focus_context = f"Pay special attention to: {', '.join(focus_areas)}"

        difficulty_guidance = self._get_difficulty_guidance(content_analysis['estimated_difficulty'])

        return f"""As a medical education expert, generate {max_cards} byte-sized summary flashcards from the following medical content. Each flashcard should be a concise, digestible summary of a key concept or section from the notes.

{subjects_context}
{focus_context}

Difficulty Level Guidance: {difficulty_guidance}

Format your response as a JSON array with this exact structure:
[
  {{
    "question": "Key concept or topic title (concise, 5-10 words)",
    "answer": "Byte-sized summary of the concept (2-4 sentences, max 150 words). Include essential information: definitions, mechanisms, clinical significance, or key facts. Make it digestible and easy to review.",
    "subject_tags": ["primary_subject", "secondary_subject"],
    "difficulty_level": 1-5
  }}
]

Flashcard Creation Guidelines:
1. QUESTIONS should be:
   - Concise topic titles or concept names (not full questions)
   - Clear and specific (e.g., "Acute Myocarditis", "Cardiac Output Regulation", "ECG Interpretation")
   - Represent the main concept being summarized

2. ANSWERS should be:
   - Byte-sized summaries (2-4 sentences, max 150 words)
   - Medically accurate and evidence-based
   - Include essential information: definitions, key mechanisms, clinical significance, or important facts
   - Break down complex concepts into digestible chunks
   - Use proper medical terminology with brief explanations
   - Each flashcard should represent a distinct, reviewable chunk of the note content

3. SUBJECT TAGS should include:
   - Primary medical specialty/system
   - Secondary relevant areas
   - Use standard medical categories

4. DIFFICULTY LEVELS:
   - 1: Basic facts, definitions, simple anatomy
   - 2: Understanding mechanisms, basic pathophysiology
   - 3: Clinical application, diagnosis, treatment basics
   - 4: Complex pathophysiology, differential diagnosis
   - 5: Advanced clinical reasoning, rare conditions

5. SUMMARY APPROACH:
   - Break the note content into logical, digestible chunks
   - Each flashcard should cover one key concept or section
   - Summaries should be comprehensive enough to be useful but concise enough to review quickly
   - Focus on the most important information from each section

Medical Content:
{text[:4000]}

Generate exactly {max_cards} byte-sized summary flashcards in the JSON format above. Each flashcard should be a digestible summary chunk of the note content."""

    def _create_clinical_scenario_prompt(self, text: str, max_scenarios: int) -> str:
        """Create prompt for clinical scenario flashcards"""
        return f"""Create {max_scenarios} clinical scenario flashcards based on the following medical content. Each flashcard should present a realistic patient case that tests clinical reasoning and application of medical knowledge.

Format as JSON array:
[
  {{
    "question": "Patient presentation and clinical scenario",
    "answer": "Diagnosis, reasoning, and management approach",
    "subject_tags": ["clinical_reasoning", "relevant_specialty"],
    "difficulty_level": 3-5
  }}
]

Guidelines for Clinical Scenarios:
1. Present realistic patient cases with relevant history, symptoms, and findings
2. Include age, gender, and pertinent background when relevant
3. Test diagnostic reasoning, differential diagnosis, or management decisions
4. Answers should explain the reasoning process and key clinical points
5. Focus on practical clinical application

Medical Content:
{text[:3000]}

Create {max_scenarios} clinical scenario flashcards in JSON format."""

    def _get_difficulty_guidance(self, estimated_difficulty: int) -> str:
        """Get difficulty-specific guidance for flashcard creation"""
        guidance = {
            1: "Focus on basic definitions, simple facts, and fundamental concepts",
            2: "Include basic mechanisms and straightforward clinical applications",
            3: "Emphasize clinical reasoning, diagnosis, and treatment principles",
            4: "Create complex scenarios requiring analysis and differential diagnosis",
            5: "Design advanced cases with rare conditions and complex reasoning"
        }
        return guidance.get(estimated_difficulty, guidance[3])

    def _parse_medical_flashcard_response(
            self,
            ai_response: str,
            source_type: str,
            source_id: Optional[UUID],
            existing_tags: Optional[List[str]],
            content_analysis: Dict[str, Any]
    ) -> List[FlashcardCreate]:
        """Parse AI response and enhance with medical subject tagging"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', ai_response, re.DOTALL)
            if not json_match:
                logger.warning("No JSON found in AI medical flashcard response")
                return []

            flashcard_data = json.loads(json_match.group())
            flashcards = []

            for item in flashcard_data:
                if not isinstance(item, dict) or not all(key in item for key in ['question', 'answer']):
                    continue

                # Enhanced subject tagging
                tags = self._enhance_subject_tags(
                    item.get('subject_tags', []),
                    item['question'] + ' ' + item['answer'],
                    existing_tags,
                    content_analysis['detected_subjects']
                )

                # Validate and adjust difficulty
                difficulty = self._validate_difficulty(
                    item.get('difficulty_level', content_analysis['estimated_difficulty']),
                    item['question'],
                    item['answer']
                )

                flashcard = FlashcardCreate(
                    question=item['question'].strip(),
                    answer=item['answer'].strip(),
                    subject_tags=tags if tags else None,
                    difficulty_level=difficulty,
                    created_from=source_type,
                    source_reference=source_id
                )
                flashcards.append(flashcard)

            return flashcards

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse medical flashcard response: {str(e)}")
            return []

    def _parse_clinical_scenario_response(
            self,
            ai_response: str,
            source_type: str,
            source_id: Optional[UUID]
    ) -> List[FlashcardCreate]:
        """Parse clinical scenario response"""
        try:
            json_match = re.search(r'\[.*\]', ai_response, re.DOTALL)
            if not json_match:
                return []

            scenario_data = json.loads(json_match.group())
            scenarios = []

            for item in scenario_data:
                if not isinstance(item, dict) or not all(key in item for key in ['question', 'answer']):
                    continue

                # Ensure clinical reasoning tag
                tags = item.get('subject_tags', [])
                if 'clinical_reasoning' not in tags:
                    tags.append('clinical_reasoning')

                scenario = FlashcardCreate(
                    question=item['question'].strip(),
                    answer=item['answer'].strip(),
                    subject_tags=tags,
                    difficulty_level=max(3, min(5, item.get('difficulty_level', 4))),
                    # Clinical scenarios are at least intermediate
                    created_from=source_type,
                    source_reference=source_id
                )
                scenarios.append(scenario)

            return scenarios

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse clinical scenario response: {str(e)}")
            return []

    def _enhance_subject_tags(
            self,
            ai_tags: List[str],
            content: str,
            existing_tags: Optional[List[str]],
            detected_subjects: List[str]
    ) -> List[str]:
        """Enhance subject tags with intelligent medical categorization"""
        tags = set()

        # Add AI-generated tags
        tags.update(ai_tags)

        # Add existing tags from source
        if existing_tags:
            tags.update(existing_tags)

        # Add detected subjects from content analysis
        tags.update(detected_subjects[:3])  # Top 3 detected subjects

        # Add tags based on content keywords
        content_lower = content.lower()
        for subject, keywords in self.MEDICAL_SUBJECTS.items():
            if any(keyword in content_lower for keyword in keywords):
                tags.add(subject)

        # Limit to reasonable number of tags
        return list(tags)[:8]

    def _validate_difficulty(self, ai_difficulty: int, question: str, answer: str) -> int:
        """Validate and adjust difficulty level based on content complexity"""
        # Ensure difficulty is in valid range
        difficulty = max(1, min(5, ai_difficulty))

        # Adjust based on content indicators
        content = (question + ' ' + answer).lower()

        # Indicators of higher difficulty
        complex_indicators = [
            'differential diagnosis', 'pathophysiology', 'mechanism',
            'contraindication', 'adverse effect', 'complication',
            'rare', 'syndrome', 'multisystem'
        ]

        # Indicators of lower difficulty
        basic_indicators = [
            'definition', 'location', 'function', 'structure',
            'normal', 'basic', 'simple'
        ]

        complex_count = sum(1 for indicator in complex_indicators if indicator in content)
        basic_count = sum(1 for indicator in basic_indicators if indicator in content)

        # Adjust difficulty based on content analysis
        if complex_count >= 2 and difficulty < 4:
            difficulty = min(difficulty + 1, 5)
        elif basic_count >= 2 and difficulty > 2:
            difficulty = max(difficulty - 1, 1)

        return difficulty