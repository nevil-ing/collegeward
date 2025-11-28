import re
import json
import logging
from typing import List, Optional, Dict, Any, Tuple
from uuid import UUID
from decimal import Decimal

from app.services.ai_service import AIServiceManager, ConversationMessage
from app.schemas.quiz_schema import QuizCreate, QuizQuestionCreate
from app.core.config import settings

logger = logging.getLogger(__name__)


class MedicalQuizGenerator:
    """
    Specialized quiz generator for medical content using AI
    """

    # Medical subject categories for quiz generation
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

    async def generate_quiz_from_materials(
            self,
            text_content: str,
            num_questions: int = 10,
            difficulty_level: Optional[int] = None,
            focus_subjects: Optional[List[str]] = None,
            quiz_title: Optional[str] = None,
            include_clinical_scenarios: bool = True
    ) -> QuizCreate:
        """
        Generate a comprehensive quiz from medical study materials

        Args:
            text_content: Source medical text content
            num_questions: Number of questions to generate
            difficulty_level: Target difficulty (1-5), auto-detected if None
            focus_subjects: Specific medical subjects to focus on
            quiz_title: Custom quiz title
            include_clinical_scenarios: Include clinical reasoning questions

        Returns:
            QuizCreate object with generated questions
        """
        try:
            # Analyze content for medical subjects and complexity
            content_analysis = self._analyze_medical_content(text_content)

            # Determine quiz parameters
            target_difficulty = difficulty_level or content_analysis['estimated_difficulty']
            # Use provided focus_subjects if available, otherwise use detected subjects
            if focus_subjects:
                # Normalize to lowercase and filter to valid medical subjects
                subjects = [s.lower() for s in focus_subjects if s.lower() in self.MEDICAL_SUBJECTS]
                if not subjects:
                    # If none match, use detected subjects
                    subjects = content_analysis['detected_subjects'][:3]
            else:
                subjects = content_analysis['detected_subjects'][:3]
            title = quiz_title or self._generate_quiz_title(subjects, content_analysis)

            # Generate questions with balanced distribution
            questions = await self._generate_balanced_questions(
                text_content,
                num_questions,
                target_difficulty,
                subjects,
                include_clinical_scenarios,
                content_analysis
            )

            quiz = QuizCreate(
                title=title,
                subject_tags=subjects,
                questions=questions
            )

            logger.info(f"Generated quiz '{title}' with {len(questions)} questions")
            return quiz

        except Exception as e:
            logger.error(f"Quiz generation failed: {str(e)}")
            raise

    async def generate_targeted_quiz(
            self,
            weak_areas: List[str],
            user_notes_content: str,
            num_questions: int = 15,
            focus_on_weaknesses: bool = True
    ) -> QuizCreate:
        """
        Generate a targeted quiz focusing on user's weak areas

        Args:
            weak_areas: List of subject areas where user needs improvement
            user_notes_content: User's study materials content
            num_questions: Number of questions to generate
            focus_on_weaknesses: Whether to heavily focus on weak areas

        Returns:
            QuizCreate object targeting weak areas
        """
        try:
            # Analyze content for weak area coverage
            content_analysis = self._analyze_medical_content(user_notes_content)

            # Prioritize weak areas in question generation
            if focus_on_weaknesses:
                # 70% questions from weak areas, 30% from other areas
                weak_questions = int(num_questions * 0.7)
                other_questions = num_questions - weak_questions
            else:
                # 50/50 split
                weak_questions = num_questions // 2
                other_questions = num_questions - weak_questions

            questions = []

            # Generate questions for weak areas
            if weak_questions > 0:
                weak_area_questions = await self._generate_subject_specific_questions(
                    user_notes_content,
                    weak_questions,
                    weak_areas,
                    difficulty_boost=1  # Slightly harder for weak areas
                )
                questions.extend(weak_area_questions)

            # Generate questions for other areas
            if other_questions > 0:
                other_subjects = [s for s in content_analysis['detected_subjects']
                                  if s not in weak_areas][:3]
                if other_subjects:
                    other_area_questions = await self._generate_subject_specific_questions(
                        user_notes_content,
                        other_questions,
                        other_subjects
                    )
                    questions.extend(other_area_questions)

            title = f"Targeted Review: {', '.join(weak_areas[:2])}"
            if len(weak_areas) > 2:
                title += f" and {len(weak_areas) - 2} more"

            quiz = QuizCreate(
                title=title,
                subject_tags=weak_areas + content_analysis['detected_subjects'][:2],
                questions=questions
            )

            logger.info(f"Generated targeted quiz for weak areas: {weak_areas}")
            return quiz

        except Exception as e:
            logger.error(f"Targeted quiz generation failed: {str(e)}")
            raise

    async def generate_adaptive_followup_quiz(
            self,
            previous_quiz_results: Dict[str, Any],
            user_notes_content: str,
            num_questions: int = 10
    ) -> QuizCreate:
        """
        Generate an adaptive follow-up quiz based on previous performance

        Args:
            previous_quiz_results: Results from previous quiz including wrong answers
            user_notes_content: User's study materials
            num_questions: Number of questions to generate

        Returns:
            QuizCreate object with adaptive questions
        """
        try:
            # Analyze previous performance
            incorrect_subjects = self._extract_weak_subjects_from_results(previous_quiz_results)
            missed_concepts = self._extract_missed_concepts(previous_quiz_results)

            # Generate follow-up questions targeting missed concepts
            questions = await self._generate_remedial_questions(
                user_notes_content,
                num_questions,
                incorrect_subjects,
                missed_concepts
            )

            title = "Follow-up Review Quiz"

            quiz = QuizCreate(
                title=title,
                subject_tags=incorrect_subjects,
                questions=questions
            )

            logger.info(f"Generated adaptive follow-up quiz targeting: {incorrect_subjects}")
            return quiz

        except Exception as e:
            logger.error(f"Adaptive quiz generation failed: {str(e)}")
            raise

    async def _generate_balanced_questions(
            self,
            text_content: str,
            num_questions: int,
            target_difficulty: int,
            subjects: List[str],
            include_clinical: bool,
            content_analysis: Dict[str, Any]
    ) -> List[QuizQuestionCreate]:
        """Generate a balanced set of questions with varied types and difficulties"""
        questions = []

        # Question type distribution
        if include_clinical and content_analysis['has_clinical_content']:
            # 40% factual, 35% conceptual, 25% clinical reasoning
            factual_count = int(num_questions * 0.4)
            conceptual_count = int(num_questions * 0.35)
            clinical_count = num_questions - factual_count - conceptual_count
        else:
            # 50% factual, 50% conceptual
            factual_count = num_questions // 2
            conceptual_count = num_questions - factual_count
            clinical_count = 0

        # Generate factual questions
        if factual_count > 0:
            factual_questions = await self._generate_question_type(
                text_content, factual_count, "factual", target_difficulty, subjects
            )
            questions.extend(factual_questions)

        # Generate conceptual questions
        if conceptual_count > 0:
            conceptual_questions = await self._generate_question_type(
                text_content, conceptual_count, "conceptual", target_difficulty, subjects
            )
            questions.extend(conceptual_questions)

        # Generate clinical reasoning questions
        if clinical_count > 0:
            clinical_questions = await self._generate_question_type(
                text_content, clinical_count, "clinical", target_difficulty + 1, subjects
            )
            questions.extend(clinical_questions)

        # Ensure we have the exact number of questions requested
        # If we have fewer, generate more to fill the gap
        if len(questions) < num_questions:
            remaining = num_questions - len(questions)
            logger.warning(
                f"Only generated {len(questions)} questions, need {num_questions}. Generating {remaining} more.")
            # Generate additional questions of mixed types
            additional_questions = await self._generate_question_type(
                text_content, remaining, "factual", target_difficulty, subjects
            )
            questions.extend(additional_questions)

        # If we have more, trim to the exact number
        if len(questions) > num_questions:
            questions = questions[:num_questions]

        # Shuffle and assign order
        import random
        random.shuffle(questions)
        for i, question in enumerate(questions, 1):
            question.question_order = i

        return questions

    async def _generate_question_type(
            self,
            text_content: str,
            count: int,
            question_type: str,
            difficulty: int,
            subjects: List[str]
    ) -> List[QuizQuestionCreate]:
        """Generate questions of a specific type"""
        prompt = self._create_question_generation_prompt(
            text_content, count, question_type, difficulty, subjects
        )

        messages = [ConversationMessage(role="user", content=prompt)]

        response = await self.ai_service.generate_response(
            messages=messages,
            mode="ai_mode",
            temperature=0.3  # Lower temperature for consistency
        )

        return self._parse_quiz_questions(response.content, question_type)

    async def _generate_subject_specific_questions(
            self,
            text_content: str,
            count: int,
            subjects: List[str],
            difficulty_boost: int = 0
    ) -> List[QuizQuestionCreate]:
        """Generate questions focused on specific subjects"""
        prompt = f"""Generate {count} multiple choice questions focused specifically on these medical subjects: {', '.join(subjects)}.

Use the following medical content as your source material:

{text_content[:3000]}

Format as JSON array:
[
  {{
    "question_text": "Clear, specific question",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "correct_answer": 0,
    "explanation": "Detailed explanation with reasoning",
    "subject_focus": "primary_subject"
  }}
]

Requirements:
- Focus heavily on the specified subjects: {', '.join(subjects)}
- Create questions that test understanding of key concepts in these areas
- Include 4 plausible answer options
- Provide comprehensive explanations
- Ensure medical accuracy

Generate exactly {count} questions in JSON format."""

        messages = [ConversationMessage(role="user", content=prompt)]

        response = await self.ai_service.generate_response(
            messages=messages,
            mode="ai_mode",
            temperature=0.3
        )

        return self._parse_quiz_questions(response.content, "subject_specific")

    async def _generate_remedial_questions(
            self,
            text_content: str,
            count: int,
            weak_subjects: List[str],
            missed_concepts: List[str]
    ) -> List[QuizQuestionCreate]:
        """Generate remedial questions for concepts the user missed"""
        concepts_text = ', '.join(missed_concepts) if missed_concepts else "key concepts"

        prompt = f"""Generate {count} remedial multiple choice questions to help reinforce these missed concepts: {concepts_text}

Focus on these weak subject areas: {', '.join(weak_subjects)}

Use this medical content as reference:

{text_content[:3000]}

Create questions that:
1. Re-teach the missed concepts from a different angle
2. Provide clear, educational explanations
3. Help build understanding step by step
4. Include common misconceptions as distractors

Format as JSON array:
[
  {{
    "question_text": "Question that reinforces missed concept",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "correct_answer": 0,
    "explanation": "Teaching-focused explanation that clarifies the concept",
    "remedial_focus": "concept being reinforced"
  }}
]

Generate exactly {count} remedial questions in JSON format."""

        messages = [ConversationMessage(role="user", content=prompt)]

        response = await self.ai_service.generate_response(
            messages=messages,
            mode="ai_mode",
            temperature=0.4  # Slightly higher for varied approaches
        )

        return self._parse_quiz_questions(response.content, "remedial")

    def _create_question_generation_prompt(
            self,
            text_content: str,
            count: int,
            question_type: str,
            difficulty: int,
            subjects: List[str]
    ) -> str:
        """Create specialized prompt for different question types"""

        type_instructions = {
            "factual": """
            - Test recall of specific facts, definitions, and basic information
            - Focus on "what", "where", "when" type questions
            - Include anatomical structures, normal values, basic classifications
            """,
            "conceptual": """
            - Test understanding of mechanisms, processes, and relationships
            - Focus on "how" and "why" questions
            - Include pathophysiology, cause-and-effect relationships
            """,
            "clinical": """
            - Present realistic patient scenarios requiring clinical reasoning
            - Test application of knowledge to patient care situations
            - Include diagnosis, treatment decisions, and clinical judgment
            """
        }

        difficulty_guidance = {
            1: "Basic level - simple recall and recognition",
            2: "Basic-intermediate - understanding of simple concepts",
            3: "Intermediate - application of knowledge",
            4: "Advanced - analysis and synthesis",
            5: "Expert - complex clinical reasoning"
        }

        return f"""Generate {count} high-quality multiple choice questions for medical students.

Question Type: {question_type.title()}
{type_instructions.get(question_type, "")}

Difficulty Level: {difficulty} - {difficulty_guidance.get(difficulty, "")}
Focus Subjects: {', '.join(subjects)}

Medical Content:
{text_content[:4000]}

Format as JSON array:
[
  {{
    "question_text": "Clear, clinically relevant question",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "correct_answer": 0,
    "explanation": "Comprehensive explanation with medical reasoning"
  }}
]

Requirements:
1. Create exactly 4 plausible answer options
2. Ensure one clearly correct answer
3. Make distractors believable but clearly incorrect
4. Provide detailed explanations that teach the concept
5. Use proper medical terminology
6. Ensure factual accuracy
7. Test the specified difficulty level

Generate exactly {count} questions in JSON format."""

    def _parse_quiz_questions(
            self,
            ai_response: str,
            question_type: str
    ) -> List[QuizQuestionCreate]:
        """Parse AI response into quiz questions"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', ai_response, re.DOTALL)
            if not json_match:
                logger.warning(f"No JSON found in AI quiz response for {question_type}")
                return []

            question_data = json.loads(json_match.group())
            questions = []

            for item in question_data:
                if not isinstance(item, dict):
                    continue

                # Validate required fields
                required_fields = ['question_text', 'options', 'correct_answer']
                if not all(field in item for field in required_fields):
                    logger.warning(f"Missing required fields in question: {item}")
                    continue

                # Validate options format
                options = item['options']
                if not isinstance(options, list) or len(options) < 2:
                    logger.warning(f"Invalid options format: {options}")
                    continue

                # Validate correct answer index
                correct_answer = item['correct_answer']
                if not isinstance(correct_answer, int) or correct_answer >= len(options):
                    logger.warning(f"Invalid correct answer index: {correct_answer}")
                    continue

                question = QuizQuestionCreate(
                    question_text=item['question_text'].strip(),
                    options=options,
                    correct_answer=correct_answer,
                    explanation=item.get('explanation', '').strip() or None,
                    question_order=1  # Will be set later
                )
                questions.append(question)

            return questions

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse quiz questions: {str(e)}")
            return []

    def _analyze_medical_content(self, text: str) -> Dict[str, Any]:
        """Analyze text to identify medical subjects and complexity"""
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

        # Estimate complexity
        medical_terms = [
            'pathophysiology', 'etiology', 'diagnosis', 'prognosis', 'treatment',
            'syndrome', 'manifestation', 'complication', 'contraindication',
            'pharmacokinetics', 'pharmacodynamics', 'therapeutic', 'clinical'
        ]

        complexity_score = sum(1 for term in medical_terms if term in text_lower)

        # Determine difficulty level
        if complexity_score >= 8:
            difficulty = 5
        elif complexity_score >= 5:
            difficulty = 4
        elif complexity_score >= 3:
            difficulty = 3
        elif complexity_score >= 1:
            difficulty = 2
        else:
            difficulty = 1

        return {
            'detected_subjects': detected_subjects[:5],
            'subject_scores': subject_scores,
            'complexity_score': complexity_score,
            'estimated_difficulty': difficulty,
            'text_length': len(text),
            'has_clinical_content': any(term in text_lower for term in [
                'patient', 'clinical', 'diagnosis', 'treatment', 'symptoms', 'case'
            ])
        }

    def _generate_quiz_title(
            self,
            subjects: List[str],
            content_analysis: Dict[str, Any]
    ) -> str:
        """Generate an appropriate quiz title"""
        if not subjects:
            return "Medical Knowledge Quiz"

        if len(subjects) == 1:
            return f"{subjects[0].title()} Quiz"
        elif len(subjects) == 2:
            return f"{subjects[0].title()} and {subjects[1].title()} Quiz"
        else:
            return f"{subjects[0].title()}, {subjects[1].title()} and More"

    def _extract_weak_subjects_from_results(
            self,
            quiz_results: Dict[str, Any]
    ) -> List[str]:
        """Extract subject areas where user performed poorly"""
        # This would analyze quiz results to identify weak subjects
        # For now, return a placeholder implementation
        return quiz_results.get('weak_subjects', [])

    def _extract_missed_concepts(
            self,
            quiz_results: Dict[str, Any]
    ) -> List[str]:
        """Extract specific concepts that were missed"""
        # This would analyze incorrect answers to identify missed concepts
        # For now, return a placeholder implementation
        return quiz_results.get('missed_concepts', [])