"""
Phase 5: Prompt Templates for ChatGroq LLM Integration

Implements the prompt templates specified in the development plan:
- System prompt instructing to cite timestamps and video IDs
- Document prompt for formatting retrieved segments
- QA chain creation with LangChain integration

As per development plan:
- Temperature=0 for deterministic answers
- Proper timestamp and video ID citation format
- Integration with Phase 4 Document format
"""

from typing import List, Optional
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document as LangChainDocument
from langchain_groq import ChatGroq

# System prompt template as specified in development plan
SYSTEM_PROMPT_TEMPLATE = """You are a helpful video analysis assistant that answers questions about video content based on retrieved segments.

IMPORTANT INSTRUCTIONS:
1. Always cite timestamps and video IDs in your answers using the format: [video_id: start_time-end_time]
2. Base your answers ONLY on the provided video segments - do not add information not present in the segments
3. If the retrieved segments don't contain enough information to answer the question, say so clearly
4. For audio segments, use the transcript text directly
5. For visual frame segments marked as "<IMAGE_FRAME>", describe what would be visible based on the timestamp
6. Combine information from multiple relevant segments when appropriate
7. Be concise but comprehensive in your answers
8. Always maintain factual accuracy - never hallucinate or invent information

When citing sources, use this exact format:
- For audio segments: [video_id: start_time-end_time] (audio)
- For visual segments: [video_id: start_time-end_time] (visual)

Question: {question}

Retrieved Video Segments:
{context}

Answer:"""


# Document prompt template for formatting individual segments
DOCUMENT_PROMPT_TEMPLATE = """
[{metadata[video_id]}: {metadata[start]:.1f}s-{metadata[end]:.1f}s | {metadata[modality]}]
{page_content}
---"""


class SystemPrompt:
    """System prompt configuration for ChatGroq LLM."""
    
    @staticmethod
    def get_template() -> str:
        """Get the system prompt template."""
        return SYSTEM_PROMPT_TEMPLATE
    
    @staticmethod
    def create_chat_prompt() -> ChatPromptTemplate:
        """Create LangChain ChatPromptTemplate for ChatGroq."""
        return ChatPromptTemplate.from_template(SYSTEM_PROMPT_TEMPLATE)


class DocumentPrompt:
    """Document formatting for retrieved segments."""
    
    @staticmethod
    def get_template() -> str:
        """Get the document prompt template."""
        return DOCUMENT_PROMPT_TEMPLATE
    
    @staticmethod
    def create_prompt() -> PromptTemplate:
        """Create LangChain PromptTemplate for document formatting."""
        return PromptTemplate(
            template=DOCUMENT_PROMPT_TEMPLATE,
            input_variables=["page_content", "metadata"]
        )
    
    @staticmethod
    def format_document(doc) -> str:
        """Format a single document for inclusion in prompt."""
        return DOCUMENT_PROMPT_TEMPLATE.format(
            page_content=doc.page_content,
            metadata=doc.metadata
        )


def format_documents_for_prompt(documents: List) -> str:
    """
    Format list of Phase 4 Document objects for LLM prompt.
    
    Args:
        documents: List of Document objects from Phase 4 retrieval
        
    Returns:
        Formatted string with all documents for LLM context
    """
    if not documents:
        return "No relevant video segments found."
    
    formatted_docs = []
    for i, doc in enumerate(documents, 1):
        # Get document details
        video_id = doc.metadata.get('video_id', 'unknown')
        start = doc.metadata.get('start', 0)
        end = doc.metadata.get('end', 0) 
        modality = doc.metadata.get('modality', 'unknown')
        
        # Format content based on modality
        if doc.is_audio_segment():
            content = doc.page_content
        elif doc.is_frame_segment():
            content = f"<IMAGE_FRAME> - Visual content from {start:.1f}s to {end:.1f}s"
        else:
            content = doc.page_content
        
        # Create formatted entry
        formatted_doc = f"""
{i}. [{video_id}: {start:.1f}s-{end:.1f}s | {modality}]
{content}
---"""
        formatted_docs.append(formatted_doc)
    
    return "\n".join(formatted_docs)


def create_qa_chain(llm: ChatGroq, retriever=None, verbose: bool = False):
    """
    Create QA chain for ChatGroq LLM as specified in development plan.
    
    Args:
        llm: ChatGroq LLM instance with temperature=0
        retriever: Optional retriever for LangChain integration
        verbose: Enable verbose logging
        
    Returns:
        Configured QA chain for video RAG
    """
    
    # Create the main prompt template
    prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT_TEMPLATE)
    
    if retriever is not None:
        # Create RetrievalQA chain as specified in development plan
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            verbose=verbose,
            chain_type_kwargs={
                "prompt": prompt,
                "document_prompt": DocumentPrompt.create_prompt()
            },
            return_source_documents=True  # Include source metadata
        )
    else:
        # Create simple LLM chain for direct document processing
        from langchain.chains.llm import LLMChain
        qa_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            verbose=verbose
        )
    
    return qa_chain


def create_chatgroq_llm(api_key: str, model: str = "llama-3.1-8b-instant") -> ChatGroq:
    """
    Create ChatGroq LLM instance as specified in development plan.
    
    Args:
        api_key: GROQ_API_KEY from environment
        model: Model name (default: llama-3.1-8b-instant)
        
    Returns:
        Configured ChatGroq LLM with temperature=0
    """
    return ChatGroq(
        groq_api_key=api_key,
        model=model,
        temperature=0,  # Deterministic, fact-focused answers
        max_tokens=None,  # No hard limit
        max_retries=3,
        timeout=30.0
    ) 