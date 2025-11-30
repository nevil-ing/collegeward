import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum

from pydantic import BaseModel

from app.services.ai_service import ai_service_manager, ConversationMessage, AIResponse
from app.core.config import settings
from app.utils.exceptions import AIServiceError

logger = logging.getLogger(__name__)


class TutorMode(str, Enum):
    """AI Tutor operating modes"""
    AI_MODE = "ai_mode"  # Fast mode using only Groq API
    VERIFIED_MODE = "verified_mode"  # Includes external medical sources


class ReasoningStep(BaseModel):
    """Individual step in reasoning process"""
    step_number: int
    title: str
    content: str
    questions: List[str] = []
    key_concepts: List[str] = []


class TutorResponse(BaseModel):
    """Structured tutor response with reasoning steps"""
    main_answer: str
    reasoning_steps: List[ReasoningStep] = []
    guided_questions: List[str] = []
    key_takeaways: List[str] = []
    follow_up_topics: List[str] = []
    sources_used: List[str] = []
    confidence_level: str = "medium"  # low, medium, high


class ConversationAnalysis(BaseModel):
    """Analysis of conversation context and learning needs"""
    topic_focus: str
    difficulty_level: str  # beginner, intermediate, advanced
    learning_gaps: List[str] = []
    suggested_approach: str
    prior_knowledge_assumed: List[str] = []


class AITutorService:
    """
    Specialized AI tutor service implementing medical education pedagogy

    Features:
    - Step-by-step reasoning breakdown
    - Guided questioning for active learning
    - Clinical scenario integration
    - Adaptive difficulty based on student responses
    - Evidence-based medical information
    """

    def __init__(self):
        self.system_prompts = {
            "base_tutor": self._get_base_tutor_prompt(),
            "step_by_step": self._get_step_by_step_prompt(),
            "guided_questions": self._get_guided_questions_prompt(),
            "clinical_reasoning": self._get_clinical_reasoning_prompt()
        }

    async def generate_tutor_response(
            self,
            user_message: str,
            conversation_history: List[ConversationMessage],
            context: Optional[str] = None,
            mode: TutorMode = TutorMode.AI_MODE,
            enable_step_by_step: bool = True,
            enable_guided_questions: bool = True
    ) -> TutorResponse:
        """
        Generate comprehensive tutor response with reasoning and guidance

        Args:
            user_message: Student's question or response
            conversation_history: Previous conversation messages
            context: Relevant study material context
            mode: Tutor operating mode (AI or Verified)
            enable_step_by_step: Whether to include step-by-step reasoning
            enable_guided_questions: Whether to include guided questions

        Returns:
            Structured tutor response with reasoning steps and guidance
        """
        try:
            # Analyze conversation context
            analysis = await self._analyze_conversation_context(
                user_message, conversation_history, context
            )

            # Generate main response
            main_response = await self._generate_main_response(
                user_message=user_message,
                conversation_history=conversation_history,
                context=context,
                mode=mode,
                analysis=analysis
            )

            # Extract reasoning steps if enabled
            reasoning_steps = []
            if enable_step_by_step:
                reasoning_steps = await self._extract_reasoning_steps(
                    main_response.content, analysis
                )

            # Generate guided questions if enabled
            guided_questions = []
            if enable_guided_questions:
                guided_questions = await self._generate_guided_questions(
                    user_message, main_response.content, analysis
                )

            # Extract key takeaways and follow-up topics
            key_takeaways = self._extract_key_takeaways(main_response.content)
            follow_up_topics = self._suggest_follow_up_topics(user_message, analysis)

            # Determine confidence level
            confidence_level = self._assess_confidence_level(
                main_response, context, mode
            )

            return TutorResponse(
                main_answer=main_response.content,
                reasoning_steps=reasoning_steps,
                guided_questions=guided_questions,
                key_takeaways=key_takeaways,
                follow_up_topics=follow_up_topics,
                sources_used=main_response.citations,
                confidence_level=confidence_level
            )

        except Exception as e:
            logger.error(f"Tutor response generation failed: {e}")
            raise AIServiceError(f"Failed to generate tutor response: {str(e)}")

    async def generate_clinical_scenario(
            self,
            topic: str,
            difficulty_level: str = "intermediate",
            context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate clinical scenario for active learning

        Args:
            topic: Medical topic for scenario
            difficulty_level: Scenario complexity
            context: Additional context from study materials

        Returns:
            Clinical scenario with questions and learning objectives
        """
        try:
            scenario_prompt = self._build_clinical_scenario_prompt(
                topic, difficulty_level, context
            )

            messages = [
                ConversationMessage(role="system", content=scenario_prompt),
                ConversationMessage(
                    role="user",
                    content=f"Create a clinical scenario for: {topic}"
                )
            ]

            response = await ai_service_manager.generate_response(
                messages=messages,
                mode="ai_mode",
                temperature=0.8,  # Higher creativity for scenarios
                max_tokens=1000
            )

            # Parse scenario response
            scenario_data = self._parse_clinical_scenario(response.content)

            return scenario_data

        except Exception as e:
            logger.error(f"Clinical scenario generation failed: {e}")
            raise AIServiceError(f"Failed to generate clinical scenario: {str(e)}")

    async def assess_student_understanding(
            self,
            student_response: str,
            expected_concepts: List[str],
            conversation_history: List[ConversationMessage]
    ) -> Dict[str, Any]:
        """
        Assess student's understanding based on their response

        Args:
            student_response: Student's answer or explanation
            expected_concepts: Key concepts that should be covered
            conversation_history: Previous conversation for context

        Returns:
            Assessment with feedback and recommendations
        """
        try:
            assessment_prompt = self._build_assessment_prompt(
                student_response, expected_concepts
            )

            messages = conversation_history + [
                ConversationMessage(role="system", content=assessment_prompt),
                ConversationMessage(
                    role="user",
                    content=f"Assess this student response: {student_response}"
                )
            ]

            response = await ai_service_manager.generate_response(
                messages=messages,
                mode="ai_mode",
                temperature=0.3,  # Lower temperature for consistent assessment
                max_tokens=800
            )

            # Parse assessment response
            assessment_data = self._parse_assessment_response(response.content)

            return assessment_data

        except Exception as e:
            logger.error(f"Student assessment failed: {e}")
            raise AIServiceError(f"Failed to assess student understanding: {str(e)}")

    async def _analyze_conversation_context(
            self,
            user_message: str,
            conversation_history: List[ConversationMessage],
            context: Optional[str]
    ) -> ConversationAnalysis:
        """Analyze conversation to understand learning context and needs"""
        try:
            # Simple analysis for now - could be enhanced with ML models
            topic_focus = self._extract_topic_focus(user_message, conversation_history)
            difficulty_level = self._assess_difficulty_level(user_message, conversation_history)
            learning_gaps = self._identify_learning_gaps(conversation_history)
            suggested_approach = self._suggest_teaching_approach(topic_focus, difficulty_level)

            return ConversationAnalysis(
                topic_focus=topic_focus,
                difficulty_level=difficulty_level,
                learning_gaps=learning_gaps,
                suggested_approach=suggested_approach,
                prior_knowledge_assumed=[]
            )

        except Exception as e:
            logger.warning(f"Conversation analysis failed: {e}")
            # Return default analysis
            return ConversationAnalysis(
                topic_focus="general medical topic",
                difficulty_level="intermediate",
                suggested_approach="step_by_step_explanation"
            )

    async def _generate_main_response(
            self,
            user_message: str,
            conversation_history: List[ConversationMessage],
            context: Optional[str],
            mode: TutorMode,
            analysis: ConversationAnalysis
    ) -> AIResponse:
        """Generate main AI response with appropriate system prompt"""
        # Select appropriate system prompt based on analysis
        system_prompt = self._select_system_prompt(analysis)

        # Build messages with context
        messages = [ConversationMessage(role="system", content=system_prompt)]

        if context:
            messages.append(ConversationMessage(
                role="system",
                content=f"Relevant study material context:\n{context}"
            ))

        # Add conversation history
        messages.extend(conversation_history)

        # Add current user message
        messages.append(ConversationMessage(role="user", content=user_message))

        # Determine medical query for verified mode
        medical_query = user_message if mode == TutorMode.VERIFIED_MODE else None

        return await ai_service_manager.generate_response(
            messages=messages,
            mode=mode.value,
            medical_query=medical_query,
            temperature=0.7,
            max_tokens=1500
        )

    async def _extract_reasoning_steps(
            self,
            response_content: str,
            analysis: ConversationAnalysis
    ) -> List[ReasoningStep]:
        """Extract step-by-step reasoning from AI response"""
        try:
            # Use AI to structure the reasoning steps
            step_prompt = self._get_step_extraction_prompt()

            messages = [
                ConversationMessage(role="system", content=step_prompt),
                ConversationMessage(
                    role="user",
                    content=f"Extract reasoning steps from: {response_content}"
                )
            ]

            response = await ai_service_manager.generate_response(
                messages=messages,
                mode="ai_mode",
                temperature=0.3,
                max_tokens=800
            )

            # Parse structured steps
            steps = self._parse_reasoning_steps(response.content)
            return steps

        except Exception as e:
            logger.warning(f"Reasoning step extraction failed: {e}")
            return []

    async def _generate_guided_questions(
            self,
            user_message: str,
            response_content: str,
            analysis: ConversationAnalysis
    ) -> List[str]:
        """Generate guided questions to promote active learning"""
        try:
            questions_prompt = self._get_guided_questions_prompt()

            messages = [
                ConversationMessage(role="system", content=questions_prompt),
                ConversationMessage(
                    role="user",
                    content=f"Generate guided questions for topic: {analysis.topic_focus}\n"
                            f"Student question: {user_message}\n"
                            f"Response given: {response_content[:500]}..."
                )
            ]

            response = await ai_service_manager.generate_response(
                messages=messages,
                mode="ai_mode",
                temperature=0.8,
                max_tokens=400
            )

            # Parse questions from response
            questions = self._parse_guided_questions(response.content)
            return questions

        except Exception as e:
            logger.warning(f"Guided questions generation failed: {e}")
            return []

    def _extract_topic_focus(
            self,
            user_message: str,
            conversation_history: List[ConversationMessage]
    ) -> str:
        """Extract main topic from conversation"""
        # Simple keyword-based extraction
        medical_topics = {
            "cardiology": ["heart", "cardiac", "cardiovascular", "ecg", "ekg"],
            "respiratory": ["lung", "respiratory", "breathing", "pneumonia", "asthma"],
            "neurology": ["brain", "neurological", "seizure", "stroke", "nervous"],
            "pharmacology": ["drug", "medication", "pharmacology", "dosage", "side effects"],
            "anatomy": ["anatomy", "structure", "organ", "system", "body"],
            "pathology": ["disease", "pathology", "diagnosis", "symptoms", "condition"]
        }

        text = user_message.lower()
        for topic, keywords in medical_topics.items():
            if any(keyword in text for keyword in keywords):
                return topic

        return "general medical topic"

    def _assess_difficulty_level(
            self,
            user_message: str,
            conversation_history: List[ConversationMessage]
    ) -> str:
        """Assess appropriate difficulty level"""
        # Check for advanced keywords first
        advanced_keywords = ["mechanism", "pathophysiology", "differential", "diagnosis", "ketoacidosis", "etiology"]
        if any(word in user_message.lower() for word in advanced_keywords):
            return "advanced"

        # Simple heuristic based on question complexity
        if len(user_message.split()) < 6:
            return "beginner"
        else:
            return "intermediate"

    def _identify_learning_gaps(self, conversation_history: List[ConversationMessage]) -> List[str]:
        """Identify potential learning gaps from conversation"""
        # Placeholder implementation
        return []

    def _suggest_teaching_approach(self, topic_focus: str, difficulty_level: str) -> str:
        """Suggest appropriate teaching approach"""
        if difficulty_level == "beginner":
            return "basic_explanation_with_examples"
        elif difficulty_level == "advanced":
            return "detailed_analysis_with_clinical_correlation"
        else:
            return "step_by_step_explanation"

    def _select_system_prompt(self, analysis: ConversationAnalysis) -> str:
        """Select appropriate system prompt based on analysis"""
        if analysis.suggested_approach == "step_by_step_explanation":
            return self.system_prompts["step_by_step"]
        elif analysis.suggested_approach == "detailed_analysis_with_clinical_correlation":
            return self.system_prompts["clinical_reasoning"]
        else:
            return self.system_prompts["base_tutor"]

    def _extract_key_takeaways(self, response_content: str) -> List[str]:
        """Extract key takeaways from response"""
        # Simple extraction - could be enhanced
        sentences = response_content.split('. ')
        takeaways = []

        for sentence in sentences[:3]:  # Take first 3 key sentences
            if len(sentence.strip()) > 20:
                takeaways.append(sentence.strip())

        return takeaways

    def _suggest_follow_up_topics(self, user_message: str, analysis: ConversationAnalysis) -> List[str]:
        """Suggest related topics for further study"""
        topic_map = {
            "cardiology": ["ECG interpretation", "Heart failure management", "Arrhythmias"],
            "respiratory": ["Pulmonary function tests", "Chest X-ray interpretation", "Oxygen therapy"],
            "neurology": ["Neurological examination", "Brain imaging", "Neuropharmacology"],
            "pharmacology": ["Drug interactions", "Pharmacokinetics", "Adverse effects"],
            "anatomy": ["Histology", "Embryology", "Clinical correlations"],
            "pathology": ["Laboratory diagnostics", "Imaging studies", "Treatment protocols"]
        }

        return topic_map.get(analysis.topic_focus, ["Related clinical cases", "Further reading", "Practice questions"])

    def _assess_confidence_level(self, response: AIResponse, context: Optional[str], mode: TutorMode) -> str:
        """Assess confidence level of the response"""
        if mode == TutorMode.VERIFIED_MODE and response.sources:
            return "high"
        elif context and len(context) > 500:
            return "medium"
        else:
            return "medium"

    def _parse_reasoning_steps(self, content: str) -> List[ReasoningStep]:
        """Parse reasoning steps from AI response"""
        # Simple parsing - could be enhanced with structured output
        steps = []
        lines = content.split('\n')

        current_step = None
        step_number = 1

        for line in lines:
            line = line.strip()
            if line.startswith(('Step', 'step', f'{step_number}.', f'{step_number}:')):
                if current_step:
                    steps.append(current_step)

                current_step = ReasoningStep(
                    step_number=step_number,
                    title=line,
                    content=""
                )
                step_number += 1
            elif current_step and line:
                current_step.content += line + " "

        if current_step:
            steps.append(current_step)

        return steps

    def _parse_guided_questions(self, content: str) -> List[str]:
        """Parse guided questions from AI response"""
        questions = []
        lines = content.split('\n')

        for line in lines:
            line = line.strip()
            if line.endswith('?') and len(line) > 10:
                # Remove numbering or bullet points
                clean_question = line.lstrip('0123456789.-• ')
                questions.append(clean_question)

        return questions[:5]  # Limit to 5 questions

    def _build_clinical_scenario_prompt(self, topic: str, difficulty: str, context: Optional[str]) -> str:
        """Build prompt for clinical scenario generation"""
        return f"""You are creating clinical scenarios for medical education.

Topic: {topic}
Difficulty: {difficulty}
Context: {context or "General medical education"}

Create a realistic clinical scenario that:
1. Presents a patient case relevant to the topic
2. Includes appropriate clinical details for the difficulty level
3. Promotes critical thinking and clinical reasoning
4. Includes 3-5 thought-provoking questions
5. Has clear learning objectives

Format your response with clear sections for:
- Patient Presentation
- Clinical Questions
- Learning Objectives
"""

    def _build_assessment_prompt(self, student_response: str, expected_concepts: List[str]) -> str:
        """Build prompt for student assessment"""
        return f"""You are assessing a medical student's understanding.

Expected key concepts: {', '.join(expected_concepts)}

Evaluate the student's response for:
1. Accuracy of medical information
2. Understanding of key concepts
3. Clinical reasoning ability
4. Areas needing improvement
5. Specific recommendations for further study

Provide constructive feedback that encourages learning.
"""

    def _parse_clinical_scenario(self, content: str) -> Dict[str, Any]:
        """Parse clinical scenario from AI response"""
        # Simple parsing - could be enhanced
        return {
            "scenario": content,
            "questions": [],
            "learning_objectives": [],
            "difficulty": "intermediate"
        }

    def _parse_assessment_response(self, content: str) -> Dict[str, Any]:
        """Parse assessment response"""
        return {
            "assessment": content,
            "score": "satisfactory",
            "recommendations": [],
            "strengths": [],
            "areas_for_improvement": []
        }

    def _get_base_tutor_prompt(self) -> str:
        """Base system prompt for AI tutor"""
        return """You are StudyBlitzAI, an expert medical tutor designed to help medical students learn through step-by-step reasoning and clinical examples.

Your teaching approach:
1. Break down complex medical concepts into digestible steps
2. Use clinical scenarios and real-world examples
3. Encourage critical thinking with guided questions
4. Provide clear explanations of pathophysiology, diagnosis, and treatment
5. Connect theoretical knowledge to practical applications

Guidelines:
- Always prioritize accuracy and evidence-based information
- Use medical terminology appropriately but explain complex terms
- Encourage active learning through questions and case-based reasoning
- When uncertain, acknowledge limitations and suggest consulting authoritative sources
- Focus on helping students understand underlying principles, not just memorization

Remember: You are an educational tool. Always encourage students to verify information with their instructors and authoritative medical sources for clinical decision-making."""

    def _get_step_by_step_prompt(self) -> str:
        """System prompt for step-by-step reasoning"""
        return self._get_base_tutor_prompt() + """

For this interaction, structure your response with clear step-by-step reasoning:
1. Start with the fundamental concept or principle
2. Build complexity gradually with each step
3. Explain the "why" behind each step
4. Use clinical examples to illustrate each point
5. Connect each step to the overall understanding

Format your steps clearly and make logical connections between them."""

    def _get_guided_questions_prompt(self) -> str:
        """System prompt for generating guided questions"""
        return """You are generating guided questions to promote active learning in medical education.

Create questions that:
1. Encourage critical thinking and analysis
2. Build on the current topic progressively
3. Connect theory to clinical practice
4. Help students identify knowledge gaps
5. Promote deeper understanding

Generate 3-5 questions that guide the student to discover answers through reasoning rather than memorization."""

    def _get_clinical_reasoning_prompt(self) -> str:
        """System prompt for clinical reasoning"""
        return self._get_base_tutor_prompt() + """

Focus on clinical reasoning and decision-making:
1. Present information as it would appear in clinical practice
2. Discuss differential diagnoses and decision trees
3. Explain the reasoning behind clinical decisions
4. Include relevant clinical guidelines and evidence
5. Connect pathophysiology to clinical presentation and management

Emphasize the thought process that experienced clinicians use."""

    def _get_step_extraction_prompt(self) -> str:
        """System prompt for extracting reasoning steps"""
        return """Extract and structure the reasoning steps from the given medical explanation.

Format each step as:
Step X: [Title]
[Content explanation]

Focus on:
1. Logical progression of concepts
2. Clear cause-and-effect relationships
3. Clinical relevance of each step
4. Key decision points or concepts

Limit to 3-5 main steps for clarity."""
