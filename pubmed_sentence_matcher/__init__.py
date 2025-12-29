"""
PubMed Sentence Matcher - Standalone Package

A high-accuracy sentence matching system for finding relevant sentences
in PubMed articles using multi-stage processing with LLM enhancement.
"""

from .pubmed_sentence_matcher import PubMedSentenceMatcher
from .pubmed_citation import PubMedCitationEnhancer

__all__ = ["PubMedSentenceMatcher", "PubMedCitationEnhancer"]
__version__ = "1.0.0"

