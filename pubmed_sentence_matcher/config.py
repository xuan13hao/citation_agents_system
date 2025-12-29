"""
Configuration module for PubMed Sentence Matcher.
"""
import os
from typing import Dict, Any, Optional


class Config:
    """Simple configuration class for PubMed Sentence Matcher."""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize configuration from dict or environment variables."""
        if config_dict is None:
            config_dict = {}
        
        # Embedding configuration
        self.embedding = config_dict.get(
            "EMBEDDING", 
            os.getenv("EMBEDDING", "openai:text-embedding-3-small")
        )
        self.embedding_provider, self.embedding_model = self.parse_embedding(self.embedding)
        self.embedding_kwargs = config_dict.get("EMBEDDING_KWARGS", {})
        
        # LLM configuration
        self.smart_llm = config_dict.get(
            "SMART_LLM",
            os.getenv("SMART_LLM", "openai:gpt-4-turbo")
        )
        self.smart_llm_provider, self.smart_llm_model = self.parse_llm(self.smart_llm)
        self.llm_kwargs = config_dict.get("LLM_KWARGS", {})
        
        # Parse API keys from environment
        if self.embedding_provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.embedding_kwargs["openai_api_key"] = api_key
        
        if self.smart_llm_provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.llm_kwargs["openai_api_key"] = api_key
    
    @staticmethod
    def parse_embedding(embedding_str: str) -> tuple[str, str]:
        """Parse embedding string format: 'provider:model'."""
        if ':' in embedding_str:
            provider, model = embedding_str.split(':', 1)
            return provider.strip(), model.strip()
        return "openai", embedding_str
    
    @staticmethod
    def parse_llm(llm_str: str) -> tuple[str, str]:
        """Parse LLM string format: 'provider:model'."""
        if ':' in llm_str:
            provider, model = llm_str.split(':', 1)
            return provider.strip(), model.strip()
        return "openai", llm_str

