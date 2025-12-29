"""
Sentence-level Citation Matcher

Given a piece of text, find the top 5 most relevant PubMed citations for each sentence.
"""
import asyncio
import json
import os
import re
import sys
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv

# Add project path
sys.path.insert(0, os.path.dirname(__file__))

from pubmed_sentence_matcher.pubmed_sentence_matcher_mcp import PubMedSentenceMatcherMCP
from pubmed_sentence_matcher.config import Config

try:
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
        NLTK_AVAILABLE = True
    except LookupError:
        NLTK_AVAILABLE = False
except ImportError:
    NLTK_AVAILABLE = False

load_dotenv()


class SentenceCitationMatcher:
    """Tool for matching relevant citations for each sentence"""
    
    def __init__(
        self,
        similarity_threshold: float = 0.3,
        top_k_per_sentence: int = 5,
        max_articles_per_sentence: int = 20,
        citation_format: str = "mla"
    ):
        """
        Initialize sentence citation matcher
        
        Args:
            similarity_threshold: Similarity threshold
            top_k_per_sentence: Number of top citations to return per sentence
            max_articles_per_sentence: Maximum number of articles to search per sentence
            citation_format: Citation format (mla, apa, nature)
        """
        self.config = Config()
        self.similarity_threshold = similarity_threshold
        self.top_k_per_sentence = top_k_per_sentence
        self.max_articles_per_sentence = max_articles_per_sentence
        self.citation_format = citation_format
        
        # Initialize sentence tokenizer
        self._init_sentence_tokenizer()
        
        # MCP matcher will be initialized when needed
        self.matcher = None
    
    def _init_sentence_tokenizer(self):
        """Initialize sentence tokenizer"""
        if NLTK_AVAILABLE:
            self.sent_tokenize = nltk.sent_tokenize
        else:
            # Simple sentence splitting fallback
            def simple_tokenize(text: str) -> List[str]:
                # Split by period, question mark, exclamation mark
                sentences = re.split(r'[.!?]+\s+', text)
                return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
            self.sent_tokenize = simple_tokenize
    
    async def _ensure_matcher_initialized(self):
        """Ensure MCP matcher is initialized"""
        if self.matcher is None:
            self.matcher = PubMedSentenceMatcherMCP(
                email=os.getenv("PUBMED_EMAIL", "citation-matcher@example.com"),
                config=self.config,
                similarity_threshold=self.similarity_threshold,
            )
    
    async def close(self):
        """Close connections"""
        if self.matcher:
            await self.matcher.close()
            self.matcher = None
    
    def split_into_sentences(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text into sentences
        
        Args:
            text: Input text
            
        Returns:
            List of sentences, each containing text and position information
        """
        sentences = self.sent_tokenize(text)
        
        sentences_with_info = []
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            # Filter out very short sentences
            if len(sentence) < 15:
                continue
            
            sentences_with_info.append({
                "sentence_index": i + 1,
                "sentence": sentence,
                "length": len(sentence)
            })
        
        return sentences_with_info
    
    async def find_citations_for_sentence(
        self,
        sentence: str,
        sentence_index: int,
        total_sentences: int
    ) -> Dict[str, Any]:
        """
        Find citations for a single sentence
        
        Args:
            sentence: Sentence text
            sentence_index: Sentence index
            total_sentences: Total number of sentences
            
        Returns:
            Dictionary containing sentence information and citations
        """
        await self._ensure_matcher_initialized()
        
        print(f"\n[{sentence_index}/{total_sentences}] Processing sentence: {sentence[:80]}...")
        
        try:
            results = await self.matcher.find_matching_sentences(
                input_sentence=sentence,
                top_k=self.top_k_per_sentence,
                max_articles=self.max_articles_per_sentence,
                citation_format=self.citation_format
            )
            
            return {
                "sentence": sentence,
                "sentence_index": sentence_index,
                "citations_found": len(results),
                "citations": results
            }
        except Exception as e:
            print(f"  Error: {e}")
            return {
                "sentence": sentence,
                "sentence_index": sentence_index,
                "citations_found": 0,
                "citations": [],
                "error": str(e)
            }
    
    async def find_citations_for_text(
        self,
        text: str,
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        Find citations for each sentence in the text
        
        Args:
            text: Input text
            progress_callback: Optional progress callback function
            
        Returns:
            Complete results containing all sentences and citations
        """
        print("=" * 80)
        print("📝 Sentence-level Citation Matching")
        print("=" * 80)
        print(f"\nInput text length: {len(text)} characters")
        
        # Split sentences
        print("\n🔪 Splitting sentences...")
        sentences = self.split_into_sentences(text)
        print(f"   Found {len(sentences)} sentences")
        
        if not sentences:
            return {
                "input_text": text,
                "total_sentences": 0,
                "sentences": [],
                "timestamp": datetime.now().isoformat()
            }
        
        # Find citations for each sentence
        print(f"\n🔍 Starting to find top {self.top_k_per_sentence} citations for each sentence...")
        print(f"   Similarity threshold: {self.similarity_threshold}")
        print(f"   Max articles per sentence: {self.max_articles_per_sentence}")
        
        sentence_results = []
        for i, sentence_info in enumerate(sentences):
            if progress_callback:
                progress_callback(i + 1, len(sentences))
            
            result = await self.find_citations_for_sentence(
                sentence_info["sentence"],
                sentence_info["sentence_index"],
                len(sentences)
            )
            sentence_results.append(result)
        
        # Statistics
        total_citations = sum(r["citations_found"] for r in sentence_results)
        sentences_with_citations = sum(1 for r in sentence_results if r["citations_found"] > 0)
        
        print("\n" + "=" * 80)
        print("📊 Statistics")
        print("=" * 80)
        print(f"Total sentences: {len(sentences)}")
        print(f"Sentences with citations: {sentences_with_citations}")
        print(f"Total citations: {total_citations}")
        print(f"Average citations per sentence: {total_citations / len(sentences):.1f}")
        
        return {
            "input_text": text,
            "total_sentences": len(sentences),
            "sentences_with_citations": sentences_with_citations,
            "total_citations": total_citations,
            "query_parameters": {
                "similarity_threshold": self.similarity_threshold,
                "top_k_per_sentence": self.top_k_per_sentence,
                "max_articles_per_sentence": self.max_articles_per_sentence,
                "citation_format": self.citation_format
            },
            "sentences": sentence_results,
            "timestamp": datetime.now().isoformat()
        }


async def main():
    """Main function - command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Find top 5 relevant PubMed citations for each sentence"
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Input text file path (if not provided, will read from stdin)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output JSON file path (default: auto-generated)"
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.3,
        help="Similarity threshold (default: 0.3)"
    )
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=5,
        help="Number of top citations to return per sentence (default: 5)"
    )
    parser.add_argument(
        "-m", "--max-articles",
        type=int,
        default=20,
        help="Maximum number of articles to search per sentence (default: 20)"
    )
    parser.add_argument(
        "-f", "--format",
        choices=["mla", "apa", "nature"],
        default="mla",
        help="Citation format (default: mla)"
    )
    
    args = parser.parse_args()
    
    # Read input text
    if args.input:
        with open(args.input, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        print("Please enter text (press Ctrl+D or Ctrl+Z when finished):")
        text = sys.stdin.read()
    
    if not text.strip():
        print("Error: Input text is empty")
        return 1
    
    # Check environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set: export OPENAI_API_KEY='your-api-key'")
        return 1
    
    # Create matcher
    matcher = SentenceCitationMatcher(
        similarity_threshold=args.threshold,
        top_k_per_sentence=args.top_k,
        max_articles_per_sentence=args.max_articles,
        citation_format=args.format
    )
    
    try:
        # Find citations
        results = await matcher.find_citations_for_text(text)
        
        # Save results
        if args.output:
            output_file = args.output
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"sentence_citations_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Results saved to: {output_file}")
        
        # Display summary
        print("\n" + "=" * 80)
        print("📋 Results Summary")
        print("=" * 80)
        for sentence_result in results["sentences"]:
            idx = sentence_result["sentence_index"]
            citations_count = sentence_result["citations_found"]
            sentence_preview = sentence_result["sentence"][:60] + "..."
            print(f"\nSentence {idx}: {citations_count} citations")
            print(f"  {sentence_preview}")
            if citations_count > 0:
                top_citation = sentence_result["citations"][0]
                print(f"  Top 1: {top_citation.get('citation', 'N/A')[:80]}...")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  User interrupted")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        await matcher.close()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

