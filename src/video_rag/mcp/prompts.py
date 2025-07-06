"""
Prompt Management System for MCP Video RAG.

This module provides prompt templates and management for different AI model tasks,
including vision analysis, query enhancement, and answer generation.
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class PromptType(Enum):
    """Types of prompts available."""
    VISION_ANALYSIS = "vision_analysis"
    VIDEO_FRAME_ANALYSIS = "video_frame_analysis"
    QUERY_ENHANCEMENT = "query_enhancement"
    ANSWER_GENERATION = "answer_generation"
    CONTENT_SUMMARIZATION = "content_summarization"
    KEYWORD_EXTRACTION = "keyword_extraction"
    INTENT_CLASSIFICATION = "intent_classification"


@dataclass
class PromptTemplate:
    """Template for generating prompts with variables."""
    name: str
    template: str
    variables: List[str]
    description: str
    model_type: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    
    def format(self, **kwargs) -> str:
        """Format template with provided variables."""
        missing_vars = [var for var in self.variables if var not in kwargs]
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")
        
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Invalid variable in template: {e}")
    
    def validate_variables(self, **kwargs) -> bool:
        """Validate that all required variables are provided."""
        return all(var in kwargs for var in self.variables)


class PromptManager:
    """Manager for prompt templates and generation."""
    
    def __init__(self, templates_dir: Optional[Path] = None):
        self.templates_dir = templates_dir
        self.templates: Dict[PromptType, Dict[str, PromptTemplate]] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize default templates
        self._load_default_templates()
        
        # Load custom templates if directory provided
        if templates_dir and templates_dir.exists():
            self._load_custom_templates()
    
    def _load_default_templates(self) -> None:
        """Load default prompt templates."""
        self.templates[PromptType.VISION_ANALYSIS] = {
            "detailed_analysis": PromptTemplate(
                name="detailed_analysis",
                template="""
                Analyze this image in detail and provide a comprehensive description.
                
                Focus on:
                1. Main subjects and objects
                2. Actions and activities
                3. Setting and environment
                4. Visual details that would be useful for search
                5. Any text or readable content
                
                {additional_instructions}
                
                Provide a detailed but concise description.
                """,
                variables=["additional_instructions"],
                description="Detailed image analysis for general purposes",
                model_type="gpt-4-vision",
                max_tokens=1000,
                temperature=0.3
            ),
            
            "scene_description": PromptTemplate(
                name="scene_description",
                template="""
                Describe the scene in this image, focusing on:
                
                - What is happening
                - Who is present
                - Where it's taking place
                - When it might be (if temporal clues exist)
                - Emotional tone or atmosphere
                
                Context: {context}
                
                Keep the description focused and searchable.
                """,
                variables=["context"],
                description="Scene-focused analysis for video content",
                model_type="gpt-4-vision",
                max_tokens=500,
                temperature=0.4
            )
        }
        
        self.templates[PromptType.VIDEO_FRAME_ANALYSIS] = {
            "temporal_analysis": PromptTemplate(
                name="temporal_analysis",
                template="""
                Analyze this video frame captured at timestamp {timestamp:.2f} seconds.
                
                Video Context: {video_context}
                Previous Context: {previous_context}
                
                Please describe:
                1. What is happening at this moment
                2. How it relates to the overall video context
                3. Key visual elements for search and retrieval
                4. Any dialogue or text visible
                5. Emotional or dramatic significance
                
                Frame Analysis:
                """,
                variables=["timestamp", "video_context", "previous_context"],
                description="Temporal analysis of video frames",
                model_type="gpt-4-vision",
                max_tokens=800,
                temperature=0.3
            ),
            
            "action_detection": PromptTemplate(
                name="action_detection",
                template="""
                Focus on detecting and describing actions in this video frame.
                
                Timestamp: {timestamp:.2f}s
                Context: {context}
                
                Identify:
                - Primary actions being performed
                - Secondary or background activities
                - Movement patterns
                - Interactions between people/objects
                - Beginning or end of actions
                
                Action Description:
                """,
                variables=["timestamp", "context"],
                description="Action-focused frame analysis",
                model_type="gpt-4-vision",
                max_tokens=600,
                temperature=0.4
            )
        }
        
        self.templates[PromptType.QUERY_ENHANCEMENT] = {
            "semantic_expansion": PromptTemplate(
                name="semantic_expansion",
                template="""
                Enhance this search query to improve video search results:
                
                Original Query: {original_query}
                Context: {context}
                
                Expand the query by:
                1. Adding relevant synonyms and related terms
                2. Including temporal keywords if relevant
                3. Adding action verbs and descriptive adjectives
                4. Including potential visual or audio cues
                
                Enhanced Query:
                """,
                variables=["original_query", "context"],
                description="Semantic expansion of search queries",
                model_type="gpt-4-turbo",
                max_tokens=200,
                temperature=0.5
            ),
            
            "intent_focused": PromptTemplate(
                name="intent_focused",
                template="""
                Analyze and enhance this query based on search intent:
                
                Query: {query}
                Detected Intent: {intent}
                Confidence: {confidence}
                
                Enhance the query to better match the detected intent:
                - For factual searches: Add specific question words
                - For moment searches: Add temporal and situational context
                - For person searches: Add identifying characteristics
                - For action searches: Add movement and activity terms
                
                Enhanced Query:
                """,
                variables=["query", "intent", "confidence"],
                description="Intent-based query enhancement",
                model_type="gpt-4-turbo",
                max_tokens=150,
                temperature=0.4
            )
        }
        
        self.templates[PromptType.ANSWER_GENERATION] = {
            "comprehensive_answer": PromptTemplate(
                name="comprehensive_answer",
                template="""
                Based on the provided context from video analysis, answer the following question:
                
                Question: {question}
                
                Context from videos:
                {context_snippets}
                
                Video Metadata:
                {video_metadata}
                
                Please provide a comprehensive answer that:
                1. Directly addresses the question
                2. References specific video content and timestamps
                3. Provides relevant details from the context
                4. Indicates confidence level and any limitations
                
                Answer:
                """,
                variables=["question", "context_snippets", "video_metadata"],
                description="Comprehensive answer generation from video context",
                model_type="claude-3-5-sonnet",
                max_tokens=2000,
                temperature=0.6
            ),
            
            "quick_answer": PromptTemplate(
                name="quick_answer",
                template="""
                Provide a concise answer based on the video content:
                
                Question: {question}
                Context: {context}
                
                Give a direct, brief answer focusing on the most relevant information.
                
                Answer:
                """,
                variables=["question", "context"],
                description="Quick, concise answer generation",
                model_type="gpt-4-turbo",
                max_tokens=300,
                temperature=0.4
            )
        }
        
        self.templates[PromptType.CONTENT_SUMMARIZATION] = {
            "video_summary": PromptTemplate(
                name="video_summary",
                template="""
                Summarize this video content in approximately {max_words} words:
                
                Content: {content}
                Video Duration: {duration}
                Key Topics: {topics}
                
                Create a summary that captures:
                1. Main themes and topics
                2. Key moments and highlights
                3. Important information or conclusions
                4. Overall structure and flow
                
                Summary:
                """,
                variables=["content", "duration", "topics", "max_words"],
                description="Video content summarization",
                model_type="claude-3-5-sonnet",
                max_tokens=500,
                temperature=0.4
            )
        }
        
        self.templates[PromptType.KEYWORD_EXTRACTION] = {
            "search_keywords": PromptTemplate(
                name="search_keywords",
                template="""
                Extract the most important keywords from this query for video search:
                
                Query: {query}
                
                Focus on:
                - Main subjects and objects
                - Actions and verbs
                - Descriptive adjectives
                - Temporal indicators
                - Emotional or tonal keywords
                
                Return only the keywords, separated by commas, without explanations.
                
                Keywords:
                """,
                variables=["query"],
                description="Keyword extraction for search optimization",
                model_type="gpt-4-turbo",
                max_tokens=100,
                temperature=0.2
            )
        }
        
        self.templates[PromptType.INTENT_CLASSIFICATION] = {
            "classify_intent": PromptTemplate(
                name="classify_intent",
                template="""
                Classify the intent of this video search query:
                
                Query: {query}
                
                Categories and their meanings:
                - factual_search: Seeking specific information or facts
                - moment_search: Looking for a specific scene or moment
                - person_search: Searching for a particular person
                - action_search: Looking for specific actions or activities
                - object_search: Searching for specific objects or items
                - emotional_search: Looking for emotional content or reactions
                - temporal_search: Searching based on time or sequence
                
                Provide confidence scores (0-1) for each category.
                
                Response format: {{"category": confidence, ...}}
                
                Classification:
                """,
                variables=["query"],
                description="Query intent classification",
                model_type="gpt-4-turbo",
                max_tokens=200,
                temperature=0.1
            )
        }
    
    def _load_custom_templates(self) -> None:
        """Load custom templates from directory."""
        if not self.templates_dir:
            return
        
        try:
            for template_file in self.templates_dir.glob("*.json"):
                with open(template_file, 'r', encoding='utf-8') as f:
                    template_data = json.load(f)
                
                prompt_type = PromptType(template_data["type"])
                template_name = template_data["name"]
                
                template = PromptTemplate(
                    name=template_name,
                    template=template_data["template"],
                    variables=template_data["variables"],
                    description=template_data["description"],
                    model_type=template_data.get("model_type"),
                    max_tokens=template_data.get("max_tokens"),
                    temperature=template_data.get("temperature")
                )
                
                if prompt_type not in self.templates:
                    self.templates[prompt_type] = {}
                
                self.templates[prompt_type][template_name] = template
                self.logger.info(f"Loaded custom template: {template_name}")
        
        except Exception as e:
            self.logger.warning(f"Failed to load custom templates: {e}")
    
    def get_template(self, prompt_type: PromptType, template_name: str) -> Optional[PromptTemplate]:
        """Get a specific template."""
        return self.templates.get(prompt_type, {}).get(template_name)
    
    def get_templates(self, prompt_type: PromptType) -> Dict[str, PromptTemplate]:
        """Get all templates for a prompt type."""
        return self.templates.get(prompt_type, {})
    
    def list_templates(self) -> Dict[str, List[str]]:
        """List all available templates."""
        return {
            prompt_type.value: list(templates.keys())
            for prompt_type, templates in self.templates.items()
        }
    
    def generate_prompt(
        self,
        prompt_type: PromptType,
        template_name: str,
        **kwargs
    ) -> str:
        """Generate a prompt from template."""
        template = self.get_template(prompt_type, template_name)
        if not template:
            raise ValueError(f"Template not found: {prompt_type.value}/{template_name}")
        
        return template.format(**kwargs)
    
    def add_template(self, prompt_type: PromptType, template: PromptTemplate) -> None:
        """Add a new template."""
        if prompt_type not in self.templates:
            self.templates[prompt_type] = {}
        
        self.templates[prompt_type][template.name] = template
        self.logger.info(f"Added template: {template.name}")
    
    def save_template(self, prompt_type: PromptType, template_name: str, filepath: Path) -> None:
        """Save a template to file."""
        template = self.get_template(prompt_type, template_name)
        if not template:
            raise ValueError(f"Template not found: {prompt_type.value}/{template_name}")
        
        template_data = {
            "type": prompt_type.value,
            "name": template.name,
            "template": template.template,
            "variables": template.variables,
            "description": template.description,
            "model_type": template.model_type,
            "max_tokens": template.max_tokens,
            "temperature": template.temperature
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(template_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved template to: {filepath}")


class VisionPrompts:
    """Convenience class for vision-related prompts."""
    
    def __init__(self, prompt_manager: PromptManager):
        self.prompt_manager = prompt_manager
    
    def analyze_image(self, additional_instructions: str = "") -> str:
        """Get detailed image analysis prompt."""
        return self.prompt_manager.generate_prompt(
            PromptType.VISION_ANALYSIS,
            "detailed_analysis",
            additional_instructions=additional_instructions
        )
    
    def describe_scene(self, context: str) -> str:
        """Get scene description prompt."""
        return self.prompt_manager.generate_prompt(
            PromptType.VISION_ANALYSIS,
            "scene_description",
            context=context
        )
    
    def analyze_video_frame(
        self,
        timestamp: float,
        video_context: str,
        previous_context: str = ""
    ) -> str:
        """Get video frame analysis prompt."""
        return self.prompt_manager.generate_prompt(
            PromptType.VIDEO_FRAME_ANALYSIS,
            "temporal_analysis",
            timestamp=timestamp,
            video_context=video_context,
            previous_context=previous_context
        )
    
    def detect_actions(self, timestamp: float, context: str) -> str:
        """Get action detection prompt."""
        return self.prompt_manager.generate_prompt(
            PromptType.VIDEO_FRAME_ANALYSIS,
            "action_detection",
            timestamp=timestamp,
            context=context
        )


class QueryPrompts:
    """Convenience class for query-related prompts."""
    
    def __init__(self, prompt_manager: PromptManager):
        self.prompt_manager = prompt_manager
    
    def enhance_query(self, original_query: str, context: str = "") -> str:
        """Get query enhancement prompt."""
        return self.prompt_manager.generate_prompt(
            PromptType.QUERY_ENHANCEMENT,
            "semantic_expansion",
            original_query=original_query,
            context=context
        )
    
    def enhance_by_intent(self, query: str, intent: str, confidence: float) -> str:
        """Get intent-focused query enhancement prompt."""
        return self.prompt_manager.generate_prompt(
            PromptType.QUERY_ENHANCEMENT,
            "intent_focused",
            query=query,
            intent=intent,
            confidence=confidence
        )
    
    def extract_keywords(self, query: str) -> str:
        """Get keyword extraction prompt."""
        return self.prompt_manager.generate_prompt(
            PromptType.KEYWORD_EXTRACTION,
            "search_keywords",
            query=query
        )
    
    def classify_intent(self, query: str) -> str:
        """Get intent classification prompt."""
        return self.prompt_manager.generate_prompt(
            PromptType.INTENT_CLASSIFICATION,
            "classify_intent",
            query=query
        )


class AnswerPrompts:
    """Convenience class for answer generation prompts."""
    
    def __init__(self, prompt_manager: PromptManager):
        self.prompt_manager = prompt_manager
    
    def generate_comprehensive_answer(
        self,
        question: str,
        context_snippets: str,
        video_metadata: str = ""
    ) -> str:
        """Get comprehensive answer generation prompt."""
        return self.prompt_manager.generate_prompt(
            PromptType.ANSWER_GENERATION,
            "comprehensive_answer",
            question=question,
            context_snippets=context_snippets,
            video_metadata=video_metadata
        )
    
    def generate_quick_answer(self, question: str, context: str) -> str:
        """Get quick answer generation prompt."""
        return self.prompt_manager.generate_prompt(
            PromptType.ANSWER_GENERATION,
            "quick_answer",
            question=question,
            context=context
        )
    
    def summarize_video(
        self,
        content: str,
        duration: str,
        topics: str,
        max_words: int = 200
    ) -> str:
        """Get video summarization prompt."""
        return self.prompt_manager.generate_prompt(
            PromptType.CONTENT_SUMMARIZATION,
            "video_summary",
            content=content,
            duration=duration,
            topics=topics,
            max_words=max_words
        ) 