import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from decimal import Decimal
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

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
    # New: Assessment question for mastery tracking
    assessment_question: Optional[str] = None
    expected_concepts: List[str] = Field(default_factory=list)
    topic_classified: Optional[str] = None


class MasteryContext(BaseModel):
    """Student's mastery profile for adaptive teaching"""
    all_masteries: Dict[str, float] = Field(default_factory=dict)
    strong_topics: List[str] = Field(default_factory=list)
    weak_topics: List[str] = Field(default_factory=list)
    overall_level: str = "intermediate"  # beginner, intermediate, advanced
    total_interactions: int = 0


class ConversationAnalysis(BaseModel):
    """Analysis of conversation context and learning needs"""
    topic_focus: str
    topic_code: Optional[str] = None
    difficulty_level: str  # beginner, intermediate, advanced
    learning_gaps: List[str] = Field(default_factory=list)
    suggested_approach: str
    prior_knowledge_assumed: List[str] = Field(default_factory=list)
    mastery_context: Optional[MasteryContext] = None


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
            user_id: Optional[UUID] = None,
            db: Optional[AsyncSession] = None,
            enable_step_by_step: bool = True,
            enable_guided_questions: bool = True,
            generate_assessment: bool = True
    ) -> TutorResponse:
        """
        Generate comprehensive tutor response with reasoning and guidance

        Args:
            user_message: Student's question or response
            conversation_history: Previous conversation messages
            context: Relevant study material context
            mode: Tutor operating mode (AI or Verified)
            user_id: User UUID for mastery tracking
            db: Database session for mastery queries
            enable_step_by_step: Whether to include step-by-step reasoning
            enable_guided_questions: Whether to include guided questions
            generate_assessment: Whether to generate assessment question for mastery

        Returns:
            Structured tutor response with reasoning steps and guidance
        """
        try:
            # Get student's mastery context if user_id provided
            mastery_context = None
            if user_id and db:
                mastery_context = await self._get_mastery_context(user_id, db)
            
            # Analyze conversation context with mastery data
            analysis = await self._analyze_conversation_context(
                user_message, conversation_history, context, mastery_context
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

            # Generate assessment question for mastery tracking
            assessment_question = None
            expected_concepts = []
            if generate_assessment:
                assessment_question, expected_concepts = await self._generate_assessment_question(
                    user_message, main_response.content, analysis
                )

            return TutorResponse(
                main_answer=main_response.content,
                reasoning_steps=reasoning_steps,
                guided_questions=guided_questions,
                key_takeaways=key_takeaways,
                follow_up_topics=follow_up_topics,
                sources_used=main_response.citations,
                confidence_level=confidence_level,
                assessment_question=assessment_question,
                expected_concepts=expected_concepts,
                topic_classified=analysis.topic_code or analysis.topic_focus
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
            context: Optional[str],
            mastery_context: Optional[MasteryContext] = None
    ) -> ConversationAnalysis:
        """Analyze conversation to understand learning context and needs"""
        try:
            # Extract topic using taxonomy service if available
            topic_focus = self._extract_topic_focus(user_message, conversation_history)
            topic_code = None
            
            # Try to classify topic more precisely
            try:
                from app.services.topic_taxonomy_service import classify_topic_simple
                category, topic_code, confidence = await classify_topic_simple(user_message)
                if confidence > 0.5:
                    topic_focus = topic_code
            except Exception as e:
                logger.debug(f"Topic classification unavailable: {e}")
            
            # Assess difficulty using mastery data
            difficulty_level = self._assess_difficulty_level(
                user_message, conversation_history, mastery_context
            )
            
            # Identify learning gaps from mastery data
            learning_gaps = self._identify_learning_gaps(
                conversation_history, mastery_context
            )
            
            suggested_approach = self._suggest_teaching_approach(
                topic_focus, difficulty_level, mastery_context
            )

            return ConversationAnalysis(
                topic_focus=topic_focus,
                topic_code=topic_code,
                difficulty_level=difficulty_level,
                learning_gaps=learning_gaps,
                suggested_approach=suggested_approach,
                prior_knowledge_assumed=[],
                mastery_context=mastery_context
            )

        except Exception as e:
            logger.warning(f"Conversation analysis failed: {e}")
            # Return default analysis
            return ConversationAnalysis(
                topic_focus="general medical topic",
                difficulty_level="intermediate",
                suggested_approach="step_by_step_explanation"
            )
    
    async def _get_mastery_context(
            self,
            user_id: UUID,
            db: AsyncSession
    ) -> Optional[MasteryContext]:
        """Get student's mastery profile for adaptive teaching"""
        try:
            from app.services.mastery_service import MasteryCalculationService
            from app.models.subject_mastery import SubjectMastery
            from sqlalchemy import select, func
            
            # Get current masteries
            query = select(
                SubjectMastery.subject_tag,
                SubjectMastery.mastery_percentage,
                SubjectMastery.total_questions_answered
            ).where(SubjectMastery.user_id == user_id)
            
            result = await db.execute(query)
            rows = result.fetchall()
            
            if not rows:
                return MasteryContext(overall_level="beginner")
            
            masteries = {}
            strong_topics = []
            weak_topics = []
            total_interactions = 0
            
            for subject_tag, mastery_pct, questions in rows:
                pct = float(mastery_pct) if mastery_pct else 0.0
                masteries[subject_tag] = pct
                total_interactions += questions or 0
                
                if pct >= 80:
                    strong_topics.append(subject_tag)
                elif pct < 60:
                    weak_topics.append(subject_tag)
            
            # Calculate overall level
            avg_mastery = sum(masteries.values()) / len(masteries) if masteries else 0
            if avg_mastery >= 75:
                overall_level = "advanced"
            elif avg_mastery >= 45:
                overall_level = "intermediate"
            else:
                overall_level = "beginner"
            
            return MasteryContext(
                all_masteries=masteries,
                strong_topics=strong_topics,
                weak_topics=weak_topics,
                overall_level=overall_level,
                total_interactions=total_interactions
            )
            
        except Exception as e:
            logger.warning(f"Failed to get mastery context: {e}")
            return None
    
    async def _generate_assessment_question(
            self,
            user_message: str,
            response_content: str,
            analysis: ConversationAnalysis
    ) -> Tuple[Optional[str], List[str]]:
        """
        Generate a follow-up assessment question to test understanding.
        
        Returns:
            Tuple of (assessment_question, expected_concepts)
        """
        try:
            assessment_prompt = f"""Based on this educational exchange, generate ONE short follow-up question to assess the student's understanding.

Student asked: {user_message[:200]}
Topic: {analysis.topic_focus}
Difficulty: {analysis.difficulty_level}

Generate a question that:
1. Tests understanding of the key concept discussed
2. Requires application of knowledge, not just recall
3. Is clear and answerable in 1-2 sentences

Output format:
QUESTION: [your question]
CONCEPTS: [concept1], [concept2], [concept3]
"""
            
            messages = [
                ConversationMessage(role="system", content="You are an educational assessment expert."),
                ConversationMessage(role="user", content=assessment_prompt)
            ]
            
            response = await ai_service_manager.generate_response(
                messages=messages,
                mode="ai_mode",
                temperature=0.5,
                max_tokens=200
            )
            
            # Parse response
            content = response.content
            question = None
            concepts = []
            
            for line in content.split('\n'):
                if line.startswith('QUESTION:'):
                    question = line.replace('QUESTION:', '').strip()
                elif line.startswith('CONCEPTS:'):
                    concepts_str = line.replace('CONCEPTS:', '').strip()
                    concepts = [c.strip() for c in concepts_str.split(',')]
            
            return question, concepts
            
        except Exception as e:
            logger.warning(f"Assessment question generation failed: {e}")
            return None, []

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
            conversation_history: List[ConversationMessage],
            mastery_context: Optional[MasteryContext] = None
    ) -> str:
        """Assess appropriate difficulty level using mastery data + message analysis"""
        
        # Start with message complexity analysis
        base_level = self._assess_message_complexity(user_message)
        
        # If we have mastery data, use it to adjust
        if mastery_context:
            overall_level = mastery_context.overall_level
            
            # Check if this topic is in their strong or weak areas
            # Try to match topic from message
            topic = self._extract_topic_focus(user_message, conversation_history)
            
            # Check mastery for this topic
            topic_mastery = mastery_context.all_masteries.get(topic, 50.0)
            
            if topic_mastery >= 80:
                # They know this topic well - can handle advanced content
                return "advanced"
            elif topic_mastery < 40:
                # They're struggling with this topic - keep it simpler
                return "beginner"
            elif mastery_context.total_interactions < 10:
                # New student - start with intermediate
                return "intermediate"
            else:
                # Use their overall level as baseline
                return overall_level
        
        return base_level
    
    def _assess_message_complexity(self, message: str) -> str:
        """Analyze message complexity indicators"""
        advanced_indicators = [
            "mechanism", "pathophysiology", "differential", 
            "etiology", "prognosis", "contraindication",
            "ketoacidosis", "hemodynamics", "pharmacokinetics"
        ]
        
        msg_lower = message.lower()
        advanced_count = sum(1 for ind in advanced_indicators if ind in msg_lower)
        
        if advanced_count >= 2:
            return "advanced"
        elif advanced_count == 1 or len(message.split()) > 15:
            return "intermediate"
        else:
            return "beginner"

    def _identify_learning_gaps(
            self,
            conversation_history: List[ConversationMessage],
            mastery_context: Optional[MasteryContext] = None
    ) -> List[str]:
        """Identify learning gaps from mastery data + conversation patterns"""
        gaps = []
        
        # 1. Gaps from mastery data (topics with low scores)
        if mastery_context:
            weak_topics = mastery_context.weak_topics
            for topic in weak_topics[:3]:
                mastery_pct = mastery_context.all_masteries.get(topic, 0)
                gaps.append(f"Low mastery in {topic}: {mastery_pct:.0f}%")
        
        # 2. Gaps from conversation patterns - look for confusion indicators
        confusion_phrases = [
            "i don't understand", "what do you mean", "can you explain",
            "i'm confused", "wait", "sorry", "not sure", "help"
        ]
        
        for msg in conversation_history[-5:]:  # Check last 5 messages
            if msg.role == "user":
                msg_lower = msg.content.lower()
                if any(phrase in msg_lower for phrase in confusion_phrases):
                    gaps.append("Confusion detected in recent messages")
                    break
        
        # 3. Gaps from repeated questions on same topic
        user_messages = [m.content.lower() for m in conversation_history if m.role == "user"]
        if len(user_messages) >= 3:
            # Check for repeated keywords suggesting struggle
            from collections import Counter
            all_words = []
            for msg in user_messages[-3:]:
                words = [w for w in msg.split() if len(w) > 4]
                all_words.extend(words)
            
            word_counts = Counter(all_words)
            repeated = [w for w, c in word_counts.items() if c >= 2]
            if repeated:
                gaps.append(f"Repeated questions about: {', '.join(repeated[:3])}")
        
        return gaps

    def _suggest_teaching_approach(
            self,
            topic_focus: str,
            difficulty_level: str,
            mastery_context: Optional[MasteryContext] = None
    ) -> str:
        """Suggest appropriate teaching approach based on mastery and difficulty"""
        
        # Check if topic is in weak areas
        is_weak_topic = False
        if mastery_context and topic_focus in mastery_context.weak_topics:
            is_weak_topic = True
        
        if difficulty_level == "beginner" or is_weak_topic:
            return "basic_explanation_with_examples"
        elif difficulty_level == "advanced":
            if mastery_context and mastery_context.overall_level == "advanced":
                return "detailed_analysis_with_clinical_correlation"
            else:
                return "step_by_step_explanation"
        else:
            return "step_by_step_explanation"

    def _select_system_prompt(self, analysis: ConversationAnalysis) -> str:
        """Select appropriate system prompt based on analysis and mastery context"""
        # Get base prompt
        if analysis.suggested_approach == "step_by_step_explanation":
            base_prompt = self.system_prompts["step_by_step"]
        elif analysis.suggested_approach == "detailed_analysis_with_clinical_correlation":
            base_prompt = self.system_prompts["clinical_reasoning"]
        else:
            base_prompt = self.system_prompts["base_tutor"]
        
        # Add adaptive student context if mastery data available
        if analysis.mastery_context:
            adaptive_context = self._build_adaptive_context(analysis.mastery_context, analysis)
            return base_prompt + adaptive_context
        
        return base_prompt
    
    def _build_adaptive_context(
            self, 
            mastery_context: MasteryContext, 
            analysis: ConversationAnalysis
    ) -> str:
        """Build adaptive prompt context based on student's mastery profile"""
        
        strong_topics = ", ".join(mastery_context.strong_topics[:5]) or "None yet"
        weak_topics = ", ".join(mastery_context.weak_topics[:5]) or "None identified"
        
        level_instructions = {
            "beginner": """
- Use simple, clear language and avoid unexplained medical jargon
- Provide analogies and real-world examples for complex concepts
- Break explanations into smaller, digestible steps
- Check understanding frequently with simple follow-up questions
""",
            "intermediate": """
- Connect new concepts to prior knowledge they've demonstrated
- Use clinical correlations to reinforce learning
- Encourage deeper reasoning with "what if" scenarios
- Build on fundamentals towards more complex applications
""",
            "advanced": """
- Use precise medical terminology appropriate for their level
- Discuss edge cases, exceptions, and nuances
- Present complex differential diagnoses and decision trees
- Challenge with research-level questions and evidence-based reasoning
"""
        }
        
        level = mastery_context.overall_level
        instructions = level_instructions.get(level, level_instructions["intermediate"])
        
        # Check if current topic is in weak areas
        current_topic = analysis.topic_focus
        is_weak_topic = current_topic in mastery_context.weak_topics
        
        weak_topic_note = ""
        if is_weak_topic:
            mastery_pct = mastery_context.all_masteries.get(current_topic, 0)
            weak_topic_note = f"""
NOTE: The student is struggling with this topic ({current_topic}: {mastery_pct:.0f}% mastery).
- Provide extra scaffolding and examples
- Check understanding more frequently  
- Be encouraging and patient
"""
        
        adaptive_prompt = f"""

=== STUDENT PROFILE ===
Overall Level: {level.upper()}
Total Interactions: {mastery_context.total_interactions}
Strong Topics: {strong_topics}
Weak Topics: {weak_topics}

=== TEACHING INSTRUCTIONS ===
{instructions}
{weak_topic_note}
"""
        return adaptive_prompt

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
        return """You are CollegeWard, an expert medical tutor designed to help medical students learn through step-by-step reasoning and clinical examples.

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
