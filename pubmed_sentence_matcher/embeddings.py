"""
Embeddings module for PubMed Sentence Matcher.
Simplified version supporting common embedding providers.
"""
import os
from typing import Any


class Memory:
    """Memory class for managing embeddings."""
    
    def __init__(self, embedding_provider: str, model: str, **embedding_kwargs: Any):
        """Initialize embeddings based on provider."""
        _embeddings = None
        
        match embedding_provider:
            case "openai":
                from langchain_openai import OpenAIEmbeddings
                _embeddings = OpenAIEmbeddings(model=model, **embedding_kwargs)
            
            case "custom":
                from langchain_openai import OpenAIEmbeddings
                _embeddings = OpenAIEmbeddings(
                    model=model,
                    openai_api_key=os.getenv("OPENAI_API_KEY", "custom"),
                    openai_api_base=os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1"),
                    check_embedding_ctx_length=False,
                    **embedding_kwargs,
                )
            
            case "azure_openai":
                from langchain_openai import AzureOpenAIEmbeddings
                _embeddings = AzureOpenAIEmbeddings(
                    model=model,
                    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
                    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
                    **embedding_kwargs,
                )
            
            case "cohere":
                from langchain_cohere import CohereEmbeddings
                _embeddings = CohereEmbeddings(model=model, **embedding_kwargs)
            
            case "ollama":
                from langchain_ollama import OllamaEmbeddings
                _embeddings = OllamaEmbeddings(
                    model=model,
                    base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
                    **embedding_kwargs,
                )
            
            case _:
                # Default to OpenAI
                from langchain_openai import OpenAIEmbeddings
                _embeddings = OpenAIEmbeddings(model=model, **embedding_kwargs)
        
        self._embeddings = _embeddings
    
    def get_embeddings(self):
        """Get the embeddings instance."""
        return self._embeddings

