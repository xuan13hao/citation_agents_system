"""
PubMed Sentence Matcher - High Accuracy Multi-Stage Matching System

This module implements a sophisticated sentence matching system that finds the most
relevant sentences in PubMed articles for a given input sentence, using:
1. LLM-powered query optimization
2. Multi-strategy PubMed search
3. Full-text retrieval and intelligent sentence splitting
4. Multi-signal similarity calculation (semantic + keyword + structure + context)
5. LLM verification
6. Final ranking and deduplication
"""

import asyncio
import json
import re
import time
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Tuple, Any
import xml.etree.ElementTree as ET

try:
    from Bio import Entrez
    ENTREZ_AVAILABLE = True
except ImportError:
    ENTREZ_AVAILABLE = False

try:
    import nltk
    NLTK_AVAILABLE = True
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        NLTK_AVAILABLE = False
except ImportError:
    NLTK_AVAILABLE = False

from .llm_utils import create_chat_completion
from .embeddings import Memory
from .config import Config
from .pubmed_citation import PubMedCitationEnhancer


class PubMedSentenceMatcher:
    """
    High-accuracy sentence matcher for PubMed articles.
    
    Uses multi-stage approach with LLM enhancement and multi-signal similarity.
    """
    
    def __init__(
        self,
        email: str = "hrteam@h2-alpha.com",
        tool: str = "GPT-Researcher",
        config: Optional[Config] = None,
        similarity_threshold: float = 0.5,
        semantic_weight: float = 0.5,
        keyword_weight: float = 0.25,
        structure_weight: float = 0.15,
        context_weight: float = 0.1,
    ):
        """
        Initialize the PubMed sentence matcher.
        
        Args:
            email: Email for PubMed API identification
            tool: Tool name for PubMed API
            config: Config object (if None, creates new one)
            similarity_threshold: Minimum similarity score to consider
            semantic_weight: Weight for semantic similarity (0-1)
            keyword_weight: Weight for keyword overlap (0-1)
            structure_weight: Weight for structure similarity (0-1)
            context_weight: Weight for context relevance (0-1)
        """
        self.email = email
        self.tool = tool
        self.config = config or Config()
        self.similarity_threshold = similarity_threshold
        
        # Validate weights sum to 1.0
        total_weight = semantic_weight + keyword_weight + structure_weight + context_weight
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
        
        self.weights = {
            'semantic': semantic_weight,
            'keyword': keyword_weight,
            'structure': structure_weight,
            'context': context_weight
        }
        
        # Initialize PubMed enhancer
        self.pubmed_enhancer = PubMedCitationEnhancer(email=email, tool=tool)
        
        # Initialize embedding model
        self.memory = Memory(
            self.config.embedding_provider,
            self.config.embedding_model,
            **self.config.embedding_kwargs
        )
        self.embeddings = self.memory.get_embeddings()
        
        # Initialize sentence tokenizer
        self._init_sentence_tokenizer()
        
        # Cache for embeddings and API calls
        self._embedding_cache = {}
        self._article_cache = {}
    
    def _init_sentence_tokenizer(self):
        """Initialize sentence tokenizer (nltk or fallback)."""
        if NLTK_AVAILABLE:
            try:
                from nltk.tokenize import sent_tokenize
                self.sent_tokenize = sent_tokenize
            except:
                self.sent_tokenize = self._simple_sentence_split
        else:
            self.sent_tokenize = self._simple_sentence_split
    
    def _simple_sentence_split(self, text: str) -> List[str]:
        """Simple sentence splitting fallback."""
        # Split on sentence-ending punctuation
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    # ==================== Stage 1: LLM Query Optimization ====================
    
    async def _generate_optimized_queries(self, input_sentence: str) -> List[str]:
        """
        Stage 1: Use LLM to understand sentence intent and generate optimized PubMed queries.
        
        Args:
            input_sentence: The input sentence to match
            
        Returns:
            List of optimized PubMed search queries
        """
        messages = [
            {
                "role": "system",
                "content": "You are a biomedical research expert specializing in PubMed search query optimization."
            },
            {
                "role": "user",
                "content": f"""Given this sentence: "{input_sentence}"

Generate 3-5 optimized PubMed search queries that would best find articles containing similar statements. Consider:
1. Key biomedical concepts and terminology
2. Synonyms and related terms
3. MeSH terms if applicable
4. Different phrasings of the same concept
5. Both broad and specific queries

Respond with ONLY a JSON array of query strings, no other text:
["query1", "query2", "query3"]"""
            }
        ]
        
        try:
            response = await create_chat_completion(
                messages=messages,
                model=self.config.smart_llm_model,
                llm_provider=self.config.smart_llm_provider,
                temperature=0.3,
                max_tokens=500,
                llm_kwargs=self.config.llm_kwargs,
            )
            
            # Parse JSON response
            response = response.strip()
            # Remove markdown code blocks if present
            if response.startswith("```"):
                response = re.sub(r'^```(?:json)?\s*', '', response)
                response = re.sub(r'\s*```$', '', response)
            
            queries = json.loads(response)
            if isinstance(queries, list) and len(queries) > 0:
                return queries[:5]  # Limit to 5 queries
            else:
                return [input_sentence]  # Fallback to original sentence
                
        except Exception as e:
            print(f"Error generating optimized queries: {e}")
            # Fallback: extract keywords from sentence
            return self._extract_keyword_queries(input_sentence)
    
    def _extract_keyword_queries(self, sentence: str) -> List[str]:
        """Fallback: extract keywords for search."""
        # Remove common stop words
        stop_words = {
            'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'must', 'this', 'that', 'these', 'those'
        }
        
        words = re.findall(r'\b\w+\b', sentence.lower())
        keywords = [w for w in words if len(w) > 3 and w not in stop_words]
        
        if keywords:
            # Create queries with different combinations
            queries = [
                ' '.join(keywords[:3]),  # First 3 keywords
                ' '.join(keywords),      # All keywords
                ' AND '.join(keywords[:2])  # First 2 with AND
            ]
            return queries
        return [sentence]
    
    # ==================== Stage 2: Multi-Strategy PubMed Search ====================
    
    async def _search_pubmed_multi_strategy(
        self, 
        input_sentence: str, 
        optimized_queries: List[str]
    ) -> List[str]:
        """
        Stage 2: Search PubMed using multiple strategies and merge results.
        
        Args:
            input_sentence: Original input sentence
            optimized_queries: LLM-generated queries
            
        Returns:
            List of unique PMIDs
        """
        all_pmids = set()
        
        # Strategy 1: Search with optimized queries
        for query in optimized_queries:
            pmids = await self._search_pubmed_with_query(query, max_results=20)
            all_pmids.update(pmids)
            time.sleep(0.35)  # Respect rate limits
        
        # Strategy 2: Direct keyword search (as supplement)
        keyword_query = ' '.join(self._extract_keywords(input_sentence))
        if keyword_query:
            pmids = await self._search_pubmed_with_query(keyword_query, max_results=10)
            all_pmids.update(pmids)
            time.sleep(0.35)
        
        return list(all_pmids)[:100]  # Limit to top 100 PMIDs
    
    def _extract_keywords(self, sentence: str) -> List[str]:
        """Extract meaningful keywords from sentence."""
        stop_words = {
            'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'must', 'this', 'that', 'these', 'those'
        }
        
        words = re.findall(r'\b\w+\b', sentence.lower())
        keywords = [w for w in words if len(w) > 3 and w not in stop_words]
        return keywords[:5]  # Top 5 keywords
    
    async def _search_pubmed_with_query(self, query: str, max_results: int = 20) -> List[str]:
        """Search PubMed with a query and return PMIDs."""
        try:
            if ENTREZ_AVAILABLE:
                Entrez.email = self.email
                handle = Entrez.esearch(
                    db="pubmed",
                    term=query,
                    retmax=max_results,
                    retmode="json"
                )
                # When retmode="json", read as text and parse JSON
                json_data = handle.read().decode('utf-8')
                handle.close()
                record = json.loads(json_data)
                return record.get('esearchresult', {}).get('idlist', [])
            else:
                # Fallback to urllib
                params = {
                    'db': 'pubmed',
                    'term': query,
                    'retmax': max_results,
                    'retmode': 'json',
                    'tool': self.tool,
                    'email': self.email
                }
                url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?" + urllib.parse.urlencode(params)
                
                with urllib.request.urlopen(url) as response:
                    data = response.read().decode('utf-8')
                    result = json.loads(data)
                    return result.get('esearchresult', {}).get('idlist', [])
        except Exception as e:
            print(f"Error searching PubMed with query '{query}': {e}")
            return []
    
    # ==================== Stage 3: Full-Text Retrieval and Sentence Splitting ====================
    
    async def _fetch_and_parse_articles(self, pmids: List[str]) -> List[Dict]:
        """
        Stage 3: Fetch article content and intelligently split into sentences.
        
        Args:
            pmids: List of PubMed IDs
            
        Returns:
            List of article dictionaries with sentences
        """
        articles = []
        
        # Process in batches to respect rate limits
        batch_size = 10
        for i in range(0, len(pmids), batch_size):
            batch_pmids = pmids[i:i+batch_size]
            
            tasks = [self._fetch_single_article(pmid) for pmid in batch_pmids]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, dict) and result:
                    articles.append(result)
            
            # Rate limiting
            if i + batch_size < len(pmids):
                time.sleep(0.5)
        
        return articles
    
    async def _fetch_single_article(self, pmid: str) -> Optional[Dict]:
        """Fetch and parse a single article."""
        # Check cache
        if pmid in self._article_cache:
            return self._article_cache[pmid]
        
        try:
            # Fetch article metadata and abstract
            article_data = self.pubmed_enhancer._fetch_article_details(pmid)
            if not article_data:
                return None
            
            # Get abstract text - PubMedCitationEnhancer doesn't extract abstract by default
            # So we fetch it separately
            abstract = await self._fetch_abstract(pmid)
            
            # Try to fetch full text if available (from PMC)
            full_text = await self._fetch_fulltext_if_available(pmid)
            
            # Combine abstract and full text
            combined_text = abstract
            if full_text:
                combined_text += "\n\n" + full_text
            
            if not combined_text.strip():
                return None
            
            # Split into sentences with context
            sentences_with_context = self._intelligent_sentence_split(
                combined_text,
                article_data
            )
            
            article_dict = {
                'pmid': pmid,
                'metadata': article_data,
                'sentences': sentences_with_context,
                'abstract': abstract,
                'full_text': full_text,
                'title': article_data.get('title', ''),
                'authors': article_data.get('authors', []),
                'journal': article_data.get('journal', ''),
                'year': article_data.get('pub_year', ''),
            }
            
            # Cache result
            self._article_cache[pmid] = article_dict
            return article_dict
            
        except Exception as e:
            print(f"Error fetching article {pmid}: {e}")
            return None
    
    async def _fetch_abstract(self, pmid: str) -> str:
        """Fetch abstract from PubMed."""
        try:
            if ENTREZ_AVAILABLE:
                Entrez.email = self.email
                handle = Entrez.efetch(db="pubmed", id=pmid, retmode="xml")
                xml_data = handle.read().decode('utf-8')
                handle.close()
                
                root = ET.fromstring(xml_data)
                abstract_elem = root.find('.//AbstractText')
                if abstract_elem is not None:
                    return ' '.join(abstract_elem.itertext())
            return ""
        except Exception as e:
            print(f"Error fetching abstract for {pmid}: {e}")
            return ""
    
    async def _fetch_fulltext_if_available(self, pmid: str) -> str:
        """Try to fetch full text from PMC if available."""
        try:
            # First check if article is in PMC
            if ENTREZ_AVAILABLE:
                Entrez.email = self.email
                handle = Entrez.elink(
                    dbfrom="pubmed",
                    db="pmc",
                    id=pmid
                )
                record = Entrez.read(handle)
                handle.close()
                
                if record and record[0].get('LinkSetDb'):
                    pmc_id = record[0]['LinkSetDb'][0]['Link'][0]['Id']
                    # Fetch PMC full text (simplified - would need full PMC parsing)
                    # For now, return empty - can be enhanced later
                    return ""
        except Exception as e:
            pass
        return ""
    
    def _intelligent_sentence_split(self, text: str, article_data: Dict) -> List[Dict]:
        """
        Split text into sentences while preserving context information.
        
        Args:
            text: Text to split
            article_data: Article metadata for context
            
        Returns:
            List of sentence dictionaries with context
        """
        sentences = self.sent_tokenize(text)
        
        sentences_with_context = []
        for i, sentence in enumerate(sentences):
            # Clean sentence
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                continue
            
            # Get context (previous and next sentences)
            prev_sentence = sentences[i-1].strip() if i > 0 else ""
            next_sentence = sentences[i+1].strip() if i < len(sentences)-1 else ""
            context = f"{prev_sentence} {sentence} {next_sentence}".strip()
            
            sentences_with_context.append({
                'text': sentence,
                'context': context,
                'position': i,
                'article': article_data
            })
        
        return sentences_with_context
    
    # ==================== Stage 4: Multi-Signal Similarity Calculation ====================
    
    async def _calculate_multi_signal_similarity(
        self,
        input_sentence: str,
        candidate_sentences: List[Dict]
    ) -> List[Dict]:
        """
        Stage 4: Calculate similarity using multiple signals and fuse them.
        
        Args:
            input_sentence: Input sentence to match
            candidate_sentences: List of candidate sentences with context
            
        Returns:
            List of matches with similarity scores
        """
        if not candidate_sentences:
            return []
        
        results = []
        
        # Extract all sentence texts for batch processing
        sentence_texts = [c['text'] for c in candidate_sentences]
        
        # Calculate all similarities in parallel where possible
        semantic_scores = await self._batch_semantic_similarity(input_sentence, sentence_texts)
        keyword_scores = [self._keyword_overlap_score(input_sentence, text) for text in sentence_texts]
        structure_scores = [self._sentence_structure_similarity(input_sentence, text) for text in sentence_texts]
        context_scores = await self._batch_context_relevance(input_sentence, candidate_sentences)
        
        # Combine scores
        for i, candidate in enumerate(candidate_sentences):
            final_score = (
                self.weights['semantic'] * semantic_scores[i] +
                self.weights['keyword'] * keyword_scores[i] +
                self.weights['structure'] * structure_scores[i] +
                self.weights['context'] * context_scores[i]
            )
            
            # Only include if above threshold
            if final_score >= self.similarity_threshold:
                results.append({
                    'sentence': candidate['text'],
                    'article': candidate['article'],
                    'context': candidate.get('context', ''),
                    'scores': {
                        'semantic': semantic_scores[i],
                        'keyword': keyword_scores[i],
                        'structure': structure_scores[i],
                        'context': context_scores[i],
                        'final': final_score
                    },
                    'position': candidate.get('position', 0)
                })
        
        return results
    
    async def _batch_semantic_similarity(
        self,
        input_sentence: str,
        candidate_sentences: List[str]
    ) -> List[float]:
        """Calculate semantic similarity using embeddings (batch)."""
        try:
            # Get embeddings
            input_embedding = await self._get_embedding(input_sentence)
            candidate_embeddings = await asyncio.gather(*[
                self._get_embedding(sent) for sent in candidate_sentences
            ])
            
            # Calculate cosine similarity
            scores = []
            for cand_emb in candidate_embeddings:
                if input_embedding and cand_emb:
                    similarity = self._cosine_similarity(input_embedding, cand_emb)
                    scores.append(float(similarity))
                else:
                    scores.append(0.0)
            
            return scores
        except Exception as e:
            print(f"Error calculating semantic similarity: {e}")
            return [0.0] * len(candidate_sentences)
    
    async def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for text (with caching)."""
        if text in self._embedding_cache:
            return self._embedding_cache[text]
        
        try:
            # Use langchain embeddings
            embedding = await self.embeddings.aembed_query(text)
            self._embedding_cache[text] = embedding
            return embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            import numpy as np
            
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot_product / (norm1 * norm2))
        except ImportError:
            # Fallback to manual calculation if numpy not available
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = sum(a * a for a in vec1) ** 0.5
            norm2 = sum(b * b for b in vec2) ** 0.5
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot_product / (norm1 * norm2))
    
    def _keyword_overlap_score(self, sentence1: str, sentence2: str) -> float:
        """Calculate keyword overlap score using TF-IDF-like approach."""
        # Extract keywords from both sentences
        words1 = set(re.findall(r'\b\w+\b', sentence1.lower()))
        words2 = set(re.findall(r'\b\w+\b', sentence2.lower()))
        
        # Remove stop words
        stop_words = {
            'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'must', 'this', 'that', 'these', 'those'
        }
        
        words1 = {w for w in words1 if len(w) > 3 and w not in stop_words}
        words2 = {w for w in words2 if len(w) > 3 and w not in stop_words}
        
        if not words1 or not words2:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _sentence_structure_similarity(self, sentence1: str, sentence2: str) -> float:
        """Calculate sentence structure similarity."""
        # Simple approach: compare sentence length and punctuation patterns
        len1, len2 = len(sentence1), len(sentence2)
        len_similarity = 1.0 - abs(len1 - len2) / max(len1, len2, 1)
        
        # Punctuation pattern similarity
        punct1 = re.findall(r'[.!?,;:]', sentence1)
        punct2 = re.findall(r'[.!?,;:]', sentence2)
        punct_similarity = 1.0 if punct1 == punct2 else 0.5
        
        return (len_similarity + punct_similarity) / 2.0
    
    async def _batch_context_relevance(
        self,
        input_sentence: str,
        candidates: List[Dict]
    ) -> List[float]:
        """Calculate context relevance scores (batch)."""
        # For now, use simple keyword matching in context
        # Can be enhanced with embedding-based context matching
        input_keywords = set(re.findall(r'\b\w+\b', input_sentence.lower()))
        input_keywords = {w for w in input_keywords if len(w) > 3}
        
        scores = []
        for candidate in candidates:
            context = candidate.get('context', candidate.get('text', ''))
            context_keywords = set(re.findall(r'\b\w+\b', context.lower()))
            context_keywords = {w for w in context_keywords if len(w) > 3}
            
            if input_keywords and context_keywords:
                overlap = len(input_keywords & context_keywords) / len(input_keywords)
                scores.append(overlap)
            else:
                scores.append(0.0)
        
        return scores
    
    # ==================== Stage 5: LLM Verification ====================
    
    async def _llm_verify_matches(
        self,
        input_sentence: str,
        top_matches: List[Dict]
    ) -> List[Dict]:
        """
        Stage 5: Use LLM to verify match accuracy.
        
        Args:
            input_sentence: Original input sentence
            top_matches: Top matches from similarity calculation
            
        Returns:
            Verified matches with LLM scores
        """
        verified_matches = []
        
        # Process in batches to avoid too many LLM calls
        batch_size = 5
        for i in range(0, len(top_matches), batch_size):
            batch = top_matches[i:i+batch_size]
            
            tasks = [
                self._verify_single_match(input_sentence, match)
                for match in batch
            ]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for match, verification in zip(batch, batch_results):
                if isinstance(verification, dict) and verification.get('is_valid'):
                    match['llm_verification'] = verification
                    # Adjust final score with LLM verification
                    match['scores']['final'] *= verification.get('relevance_score', 1.0)
                    verified_matches.append(match)
                elif not isinstance(verification, Exception):
                    # If verification failed but not an exception, still include
                    verified_matches.append(match)
        
        return verified_matches
    
    async def _verify_single_match(
        self,
        input_sentence: str,
        match: Dict
    ) -> Dict:
        """Verify a single match using LLM."""
        messages = [
            {
                "role": "system",
                "content": "You are an expert at evaluating semantic similarity between scientific statements."
            },
            {
                "role": "user",
                "content": f"""Input sentence: "{input_sentence}"

Matched sentence from article: "{match['sentence']}"
Article context: "{match.get('context', '')[:500]}"
Article title: "{match['article'].get('title', '')}"

Rate the semantic similarity and relevance on a scale of 0-1:
1. Do they express the same or very similar meaning?
2. Is the matched sentence from a relevant scientific context?
3. Would this citation be appropriate for the input sentence?

Respond with ONLY valid JSON, no other text:
{{
    "relevance_score": 0.0-1.0,
    "reasoning": "brief explanation",
    "is_valid": true/false
}}"""
            }
        ]
        
        try:
            response = await create_chat_completion(
                messages=messages,
                model=self.config.smart_llm_model,
                llm_provider=self.config.smart_llm_provider,
                temperature=0.2,
                max_tokens=200,
                llm_kwargs=self.config.llm_kwargs,
            )
            
            # Parse JSON
            response = response.strip()
            if response.startswith("```"):
                response = re.sub(r'^```(?:json)?\s*', '', response)
                response = re.sub(r'\s*```$', '', response)
            
            verification = json.loads(response)
            return verification
            
        except Exception as e:
            print(f"Error in LLM verification: {e}")
            # Return default verification
            return {
                "relevance_score": 0.8,
                "reasoning": "Verification failed, using default score",
                "is_valid": True
            }
    
    # ==================== Stage 6: Final Ranking ====================
    
    def _final_ranking(
        self,
        verified_matches: List[Dict],
        top_k: int = 10
    ) -> List[Dict]:
        """
        Stage 6: Final ranking considering multiple factors.
        
        Args:
            verified_matches: Verified matches
            top_k: Number of top results to return
            
        Returns:
            Final ranked and deduplicated results
        """
        # Calculate article quality scores
        for match in verified_matches:
            article_quality = self._calculate_article_quality(match['article'])
            
            # Final rank score = similarity * LLM verification * article quality
            llm_score = match.get('llm_verification', {}).get('relevance_score', 1.0)
            match['final_rank_score'] = (
                match['scores']['final'] *
                llm_score *
                article_quality
            )
        
        # Sort by final rank score
        sorted_matches = sorted(
            verified_matches,
            key=lambda x: x.get('final_rank_score', 0),
            reverse=True
        )
        
        # Deduplicate: keep only best match per article
        deduplicated = self._deduplicate_by_article(sorted_matches)
        
        return deduplicated[:top_k]
    
    def _calculate_article_quality(self, article_data: Dict) -> float:
        """Calculate article quality score based on metadata."""
        score = 1.0
        
        # Prefer articles with more authors (usually indicates collaboration)
        authors = article_data.get('authors', [])
        if len(authors) > 0:
            score *= 1.0 + min(len(authors) / 10.0, 0.2)  # Up to 20% boost
        
        # Prefer recent articles (within last 10 years)
        year = article_data.get('pub_year', '')
        if year:
            try:
                year_int = int(year)
                current_year = 2024
                if current_year - year_int <= 10:
                    score *= 1.1  # 10% boost for recent articles
            except:
                pass
        
        # Prefer articles with journal information
        if article_data.get('journal'):
            score *= 1.05  # 5% boost
        
        return min(score, 1.5)  # Cap at 1.5x
    
    def _deduplicate_by_article(self, matches: List[Dict]) -> List[Dict]:
        """Keep only the best match per article."""
        seen_pmids = {}
        deduplicated = []
        
        for match in matches:
            pmid = match['article'].get('pmid', '')
            if not pmid:
                continue
            
            if pmid not in seen_pmids:
                seen_pmids[pmid] = match
                deduplicated.append(match)
            else:
                # Keep the one with higher score
                if match['final_rank_score'] > seen_pmids[pmid]['final_rank_score']:
                    deduplicated.remove(seen_pmids[pmid])
                    seen_pmids[pmid] = match
                    deduplicated.append(match)
        
        # Re-sort after deduplication
        return sorted(
            deduplicated,
            key=lambda x: x.get('final_rank_score', 0),
            reverse=True
        )
    
    # ==================== Main Function ====================
    
    async def find_matching_sentences(
        self,
        input_sentence: str,
        top_k: int = 10,
        max_articles: int = 50,
        citation_format: str = "mla"
    ) -> List[Dict]:
        """
        Main function: Find top matching sentences in PubMed articles.
        
        Args:
            input_sentence: Input sentence to match
            top_k: Number of top results to return
            max_articles: Maximum number of articles to process
            citation_format: Citation format (mla, apa, nature)
            
        Returns:
            List of matched sentences with citations
        """
        print(f"🔍 Starting sentence matching for: '{input_sentence[:100]}...'")
        
        # Stage 1: LLM Query Optimization
        print("📝 Stage 1: Generating optimized queries...")
        optimized_queries = await self._generate_optimized_queries(input_sentence)
        print(f"   Generated {len(optimized_queries)} queries")
        
        # Stage 2: Multi-Strategy Search
        print("🔎 Stage 2: Searching PubMed...")
        pmids = await self._search_pubmed_multi_strategy(input_sentence, optimized_queries)
        pmids = pmids[:max_articles]  # Limit articles
        print(f"   Found {len(pmids)} articles")
        
        if not pmids:
            print("   No articles found")
            return []
        
        # Stage 3: Fetch and Parse Articles
        print("📚 Stage 3: Fetching article content...")
        articles = await self._fetch_and_parse_articles(pmids)
        print(f"   Fetched {len(articles)} articles")
        
        if not articles:
            print("   No article content available")
            return []
        
        # Extract all candidate sentences
        all_candidates = []
        for article in articles:
            all_candidates.extend(article['sentences'])
        
        print(f"   Extracted {len(all_candidates)} candidate sentences")
        
        # Stage 4: Multi-Signal Similarity
        print("🧮 Stage 4: Calculating similarity scores...")
        matches = await self._calculate_multi_signal_similarity(
            input_sentence,
            all_candidates
        )
        print(f"   Found {len(matches)} matches above threshold")
        
        if not matches:
            print("   No matches found")
            return []
        
        # Sort by similarity score
        matches = sorted(
            matches,
            key=lambda x: x['scores']['final'],
            reverse=True
        )
        
        # Take top candidates for LLM verification
        top_candidates = matches[:top_k * 2]  # Verify more than we need
        
        # Stage 5: LLM Verification
        print("✅ Stage 5: Verifying matches with LLM...")
        verified_matches = await self._llm_verify_matches(input_sentence, top_candidates)
        print(f"   Verified {len(verified_matches)} matches")
        
        # Stage 6: Final Ranking
        print("🏆 Stage 6: Final ranking...")
        final_results = self._final_ranking(verified_matches, top_k)
        
        # Format citations
        for result in final_results:
            article_data = result['article']
            citation = self.pubmed_enhancer.format_mla_citation(article_data, "")
            if citation_format == "apa":
                citation = self.pubmed_enhancer.format_apa_citation(article_data, "")
            elif citation_format == "nature":
                citation = self.pubmed_enhancer.format_nature_citation(article_data, "")
            
            result['citation'] = citation
            result['citation_format'] = citation_format
        
        print(f"✅ Found {len(final_results)} final matches")
        return final_results

