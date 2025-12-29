"""
LLM utilities for PubMed Sentence Matcher.
Simplified version supporting common LLM providers.
"""
import os
from typing import Any, Optional


async def create_chat_completion(
    messages: list[dict[str, str]],
    model: str,
    llm_provider: str = "openai",
    temperature: float = 0.4,
    max_tokens: int = 4000,
    llm_kwargs: Optional[dict[str, Any]] = None,
    **kwargs
) -> str:
    """
    Create a chat completion using LLM provider.
    
    Args:
        messages: List of message dictionaries
        model: Model name
        llm_provider: Provider name (openai, azure_openai, etc.)
        temperature: Temperature for generation
        max_tokens: Maximum tokens
        llm_kwargs: Additional LLM keyword arguments
    
    Returns:
        Response string from LLM
    """
    if llm_kwargs is None:
        llm_kwargs = {}
    
    # Get provider class
    if llm_provider == "openai":
        from langchain_openai import ChatOpenAI
        
        # Get API key
        api_key = llm_kwargs.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set")
        
        base_url = llm_kwargs.get("openai_api_base") or os.getenv("OPENAI_BASE_URL")
        
        llm_kwargs_clean = {k: v for k, v in llm_kwargs.items() if k not in ["openai_api_key", "openai_api_base"]}
        llm_kwargs_clean.update({
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        })
        if base_url:
            llm_kwargs_clean["base_url"] = base_url
        
        llm = ChatOpenAI(**llm_kwargs_clean)
    
    elif llm_provider == "azure_openai":
        from langchain_openai import AzureChatOpenAI
        
        llm = AzureChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            **llm_kwargs
        )
    
    elif llm_provider == "ollama":
        from langchain_ollama import ChatOllama
        
        llm = ChatOllama(
            model=model,
            temperature=temperature,
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            **llm_kwargs
        )
    
    else:
        # Default to OpenAI
        from langchain_openai import ChatOpenAI
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(f"Unsupported provider {llm_provider} and no OPENAI_API_KEY set")
        
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            **llm_kwargs
        )
    
    # Invoke LLM
    response = await llm.ainvoke(messages)
    
    # Extract content from response
    if hasattr(response, 'content'):
        return response.content
    elif isinstance(response, str):
        return response
    else:
        return str(response)

