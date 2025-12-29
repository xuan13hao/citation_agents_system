"""
PubMed Sentence Matcher - MCP Version

This module implements the sentence matching system using MCP (Model Context Protocol)
for tool integration. It provides the same interface as the original but uses MCP
servers for PubMed and OpenAI API calls.
"""
import asyncio
import json
import re
from typing import Dict, List, Optional, Tuple, Any

try:
    import nltk
    NLTK_AVAILABLE = True
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        NLTK_AVAILABLE = False
except ImportError:
    NLTK_AVAILABLE = False

from .mcp_client import MCPToolManager
from .config import Config
from .pubmed_citation import PubMedCitationEnhancer


class PubMedSentenceMatcherMCP:
    """
    High-accuracy sentence matcher for PubMed articles using MCP.
    
    Uses multi-stage approach with LLM enhancement and multi-signal similarity.
    All external API calls go through MCP servers.
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
        mcp_manager: Optional[MCPToolManager] = None,
    ):
        """
        Initialize the PubMed sentence matcher with MCP.
        
        Args:
            email: Email for PubMed API identification
            tool: Tool name for PubMed API
            config: Config object (if None, creates new one)
            similarity_threshold: Minimum similarity score to consider
            semantic_weight: Weight for semantic similarity (0-1)
            keyword_weight: Weight for keyword overlap (0-1)
            structure_weight: Weight for structure similarity (0-1)
            context_weight: Weight for context relevance (0-1)
            mcp_manager: Optional pre-initialized MCP manager
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
        
        # Initialize PubMed enhancer (still uses direct API for citation formatting)
        self.pubmed_enhancer = PubMedCitationEnhancer(email=email, tool=tool)
        
        # Initialize MCP manager
        self.mcp_manager = mcp_manager or MCPToolManager()
        self._mcp_initialized = False
        
        # Initialize sentence tokenizer
        self._init_sentence_tokenizer()
        
        # Cache for embeddings
        self._embedding_cache = {}
    
    async def _ensure_mcp_initialized(self):
        """Ensure MCP clients are initialized."""
        if not self._mcp_initialized:
            await self.mcp_manager.initialize()
            self._mcp_initialized = True
    
    async def close(self):
        """Close MCP connections."""
        if self._mcp_initialized:
            await self.mcp_manager.shutdown()
            self._mcp_initialized = False
    
    def _init_sentence_tokenizer(self):
        """Initialize sentence tokenizer."""
        if NLTK_AVAILABLE:
            self.sent_tokenize = nltk.sent_tokenize
        else:
            # Fallback to simple regex-based tokenization
            def simple_tokenize(text: str) -> List[str]:
                # Simple sentence splitting
                sentences = re.split(r'[.!?]+\s+', text)
                return [s.strip() for s in sentences if s.strip()]
            self.sent_tokenize = simple_tokenize
    
    # ==================== Stage 1: LLM Query Optimization ====================
    
    def _is_paragraph_or_multiple_sentences(self, text: str) -> bool:
        """
        Detect if input is a paragraph or multiple sentences.
        
        Args:
            text: Input text to check
            
        Returns:
            True if text appears to be a paragraph or multiple sentences
        """
        # Split into sentences
        sentences = self.sent_tokenize(text)
        
        # Consider it a paragraph if:
        # 1. More than 1 sentence, OR
        # 2. Single sentence but very long (more than 200 characters)
        if len(sentences) > 1:
            return True
        if len(sentences) == 1 and len(text.strip()) > 200:
            return True
        
        return False
    
    async def _understand_context_background(self, text: str) -> str:
        """
        Use LLM to understand the overall context and background of a paragraph or multiple sentences.
        
        Args:
            text: Input paragraph or multiple sentences
            
        Returns:
            Context summary/background understanding
        """
        await self._ensure_mcp_initialized()
        
        messages = [
            {
                "role": "system",
                "content": "You are an expert at understanding biomedical and scientific text. Analyze the given text and provide a concise summary of the main concepts, background, and key themes."
            },
            {
                "role": "user",
                "content": f"""Analyze the following text and provide a concise summary of:
1. Main research topic or theme
2. Key biomedical/scientific concepts mentioned
3. Overall context and background
4. Important terminology and relationships

Text: "{text}"

Respond with a clear, structured summary that captures the essential context and background information."""
            }
        ]
        
        try:
            response = await self.mcp_manager.openai_chat(
                messages=messages,
                model=self.config.smart_llm_model,
                temperature=0.3,
                max_tokens=500,
            )
            
            if not response or not response.strip():
                print("Warning: Empty response from context understanding, using original text")
                return text
            
            return response.strip()
        except Exception as e:
            print(f"Error understanding context: {e}")
            return text
    
    async def _generate_optimized_queries(self, input_sentence: str) -> List[str]:
        """
        Stage 1: Use LLM to generate optimized PubMed search queries.
        If input is a paragraph or multiple sentences, first understand the context.
        
        Args:
            input_sentence: Input sentence to match (can be a paragraph or multiple sentences)
            
        Returns:
            List of optimized search queries
        """
        await self._ensure_mcp_initialized()
        
        # Check if input is a paragraph or multiple sentences
        if self._is_paragraph_or_multiple_sentences(input_sentence):
            print("   Detected paragraph/multiple sentences, understanding context first...")
            # First understand the context and background
            context_summary = await self._understand_context_background(input_sentence)
            print(f"   Context understood: {context_summary[:100]}...")
            
            # Use both original text and context summary for query generation
            query_input = f"Context: {context_summary}\n\nOriginal text: {input_sentence}"
        else:
            query_input = input_sentence
        
        messages = [
            {
                "role": "system",
                "content": "You are an expert at generating PubMed search queries. Generate 3-5 optimized search queries that would help find relevant articles for the given text."
            },
            {
                "role": "user",
                "content": f"""Generate optimized PubMed search queries for this text: "{query_input}"

Consider:
1. Key biomedical concepts and terminology
2. Synonyms and related terms
3. MeSH terms if applicable
4. Different phrasings of the same concept
5. Both broad and specific queries
6. If context was provided, focus on the main themes and concepts from the context

Respond with ONLY a JSON array of query strings, no other text:
["query1", "query2", "query3"]"""
            }
        ]
        
        try:
            response = await self.mcp_manager.openai_chat(
                messages=messages,
                model=self.config.smart_llm_model,
                temperature=0.3,
                max_tokens=500,
            )
            
            if not response or not response.strip():
                print("Warning: Empty response from LLM, using fallback")
                return self._extract_keyword_queries(input_sentence)
            
            # Parse JSON response with improved error handling
            response = response.strip()
            
            # Remove markdown code blocks if present
            if response.startswith("```"):
                response = re.sub(r'^```(?:json)?\s*', '', response)
                response = re.sub(r'\s*```$', '', response)
                response = response.strip()
            
            # Try to extract JSON array from response
            # Look for JSON array pattern (with proper bracket matching)
            json_match = re.search(r'\[(?:[^\[\]]|\[[^\]]*\])*\]', response)
            if json_match:
                response = json_match.group(0)
            
            # Try to parse JSON
            try:
                queries = json.loads(response)
            except json.JSONDecodeError as json_err:
                # If direct parse fails, try multiple extraction strategies
                print(f"Warning: JSON parse error: {json_err}")
                
                # Strategy 1: Extract quoted strings (handles both single and double quotes)
                query_matches = re.findall(r'["\']([^"\']+)["\']', response)
                if query_matches and len(query_matches) > 0:
                    queries = query_matches[:5]
                    print(f"Extracted {len(queries)} queries using quote extraction")
                else:
                    # Strategy 2: Extract text between brackets (handles Python list format)
                    bracket_match = re.search(r'\[(.*?)\]', response, re.DOTALL)
                    if bracket_match:
                        content = bracket_match.group(1)
                        # Split by comma and clean
                        items = [item.strip().strip('"\'') for item in content.split(',')]
                        queries = [item for item in items if item and len(item) > 3][:5]
                        if queries:
                            print(f"Extracted {len(queries)} queries from bracket content")
                        else:
                            raise json_err
                    else:
                        raise json_err
            
            if isinstance(queries, list) and len(queries) > 0:
                # Filter out empty strings
                queries = [q for q in queries if q and isinstance(q, str) and q.strip()]
                if queries:
                    return queries[:5]  # Limit to 5 queries
                else:
                    return self._extract_keyword_queries(input_sentence)
            else:
                return self._extract_keyword_queries(input_sentence)
                
        except Exception as e:
            print(f"Error generating optimized queries: {e}")
            # Fallback: extract keywords from sentence
            return self._extract_keyword_queries(input_sentence)
    
    def _extract_keyword_queries(self, sentence: str) -> List[str]:
        """Fallback: extract keywords for search."""
        stop_words = {
            'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'must', 'this', 'that', 'these', 'those'
        }
        
        words = re.findall(r'\b\w+\b', sentence.lower())
        keywords = [w for w in words if len(w) > 3 and w not in stop_words]
        
        if keywords:
            queries = [
                ' '.join(keywords[:3]),
                ' '.join(keywords),
                ' AND '.join(keywords[:2])
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
        Stage 2: Search PubMed using multiple strategies via MCP.
        
        Args:
            input_sentence: Original input sentence
            optimized_queries: LLM-generated queries
            
        Returns:
            List of unique PMIDs
        """
        await self._ensure_mcp_initialized()
        
        all_pmids = set()
        
        # Strategy 1: Search with optimized queries
        for query in optimized_queries:
            pmids = await self.mcp_manager.pubmed_search(query, max_results=20)
            all_pmids.update(pmids)
            await asyncio.sleep(0.35)  # Respect rate limits
        
        # Strategy 2: Direct keyword search (as supplement)
        keyword_query = ' '.join(self._extract_keywords(input_sentence))
        if keyword_query:
            pmids = await self.mcp_manager.pubmed_search(keyword_query, max_results=10)
            all_pmids.update(pmids)
            await asyncio.sleep(0.35)
        
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
    
    # ==================== Stage 3: Full-Text Retrieval and Sentence Splitting ====================
    
    async def _fetch_and_parse_articles(self, pmids: List[str]) -> List[Dict]:
        """
        Stage 3: Fetch article content via MCP and intelligently split into sentences.
        
        Args:
            pmids: List of PubMed IDs
            
        Returns:
            List of article dictionaries with sentences
        """
        await self._ensure_mcp_initialized()
        
        if not pmids:
            return []
        
        # Fetch articles in batches
        articles = []
        batch_size = 10
        
        for i in range(0, len(pmids), batch_size):
            batch_pmids = pmids[i:i + batch_size]
            batch_articles = await self.mcp_manager.pubmed_fetch(batch_pmids)
            
            for article_data in batch_articles:
                # Extract text (abstract)
                text = article_data.get('abstract', '')
                if not text:
                    continue
                
                # Split into sentences
                sentences = self._intelligent_sentence_split(text, article_data)
                
                if sentences:
                    articles.append({
                        'pmid': article_data.get('pmid', ''),
                        'title': article_data.get('title', ''),
                        'authors': article_data.get('authors', []),
                        'journal': article_data.get('journal', ''),
                        'year': article_data.get('year', ''),
                        'sentences': sentences
                    })
            
            # Rate limiting
            if i + batch_size < len(pmids):
                await asyncio.sleep(0.5)
        
        return articles
    
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
        Stage 4: Calculate similarity using multiple signals via MCP.
        
        Args:
            input_sentence: Input sentence to match
            candidate_sentences: List of candidate sentences with context
            
        Returns:
            List of matches with similarity scores
        """
        if not candidate_sentences:
            return []
        
        await self._ensure_mcp_initialized()
        
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
                    }
                })
        
        return results
    
    async def _batch_semantic_similarity(
        self,
        input_sentence: str,
        candidate_sentences: List[str]
    ) -> List[float]:
        """Calculate semantic similarity using embeddings via MCP."""
        await self._ensure_mcp_initialized()
        
        try:
            # Get embeddings for input and all candidates
            all_texts = [input_sentence] + candidate_sentences
            
            # Check cache
            cache_key = tuple(all_texts)
            if cache_key in self._embedding_cache:
                embeddings = self._embedding_cache[cache_key]
            else:
                embeddings = await self.mcp_manager.openai_embed(
                    all_texts,
                    model=self.config.embedding_model
                )
                self._embedding_cache[cache_key] = embeddings
            
            if not embeddings or len(embeddings) < 2:
                return [0.0] * len(candidate_sentences)
            
            input_embedding = embeddings[0]
            candidate_embeddings = embeddings[1:]
            
            # Calculate cosine similarity
            similarities = []
            for candidate_emb in candidate_embeddings:
                similarity = self._cosine_similarity(input_embedding, candidate_emb)
                similarities.append(similarity)
            
            return similarities
        except Exception as e:
            print(f"Error calculating semantic similarity: {e}")
            return [0.0] * len(candidate_sentences)
    
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
            # Fallback without numpy
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = sum(a * a for a in vec1) ** 0.5
            norm2 = sum(b * b for b in vec2) ** 0.5
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return dot_product / (norm1 * norm2)
    
    def _keyword_overlap_score(self, sentence1: str, sentence2: str) -> float:
        """Calculate keyword overlap score."""
        words1 = set(re.findall(r'\b\w+\b', sentence1.lower()))
        words2 = set(re.findall(r'\b\w+\b', sentence2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0
    
    def _sentence_structure_similarity(self, sentence1: str, sentence2: str) -> float:
        """Calculate sentence structure similarity."""
        # Simple length-based similarity
        len1 = len(sentence1)
        len2 = len(sentence2)
        
        if len1 == 0 or len2 == 0:
            return 0.0
        
        ratio = min(len1, len2) / max(len1, len2)
        return ratio * 0.5  # Scale down
    
    async def _batch_context_relevance(
        self,
        input_sentence: str,
        candidates: List[Dict]
    ) -> List[float]:
        """Calculate context relevance scores."""
        # Simple implementation: use semantic similarity of context
        contexts = [c.get('context', c['text']) for c in candidates]
        return await self._batch_semantic_similarity(input_sentence, contexts)
    
    # ==================== Stage 5: LLM Verification ====================
    
    async def _llm_verify_match(
        self,
        input_sentence: str,
        match: Dict
    ) -> Dict[str, Any]:
        """
        Stage 5: Use LLM to verify match quality via MCP.
        
        Args:
            input_sentence: Original input sentence
            match: Match dictionary with sentence and article info
            
        Returns:
            Verification result with relevance score
        """
        await self._ensure_mcp_initialized()
        
        messages = [
            {
                "role": "system",
                "content": "You are an expert at evaluating the relevance of scientific citations. Rate how well a matched sentence relates to an input sentence."
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
            response = await self.mcp_manager.openai_chat(
                messages=messages,
                model=self.config.smart_llm_model,
                temperature=0.2,
                max_tokens=200,
            )
            
            if not response or not response.strip():
                return {
                    "relevance_score": 0.8,
                    "reasoning": "Empty response, using default score",
                    "is_valid": True
                }
            
            # Parse JSON with improved error handling
            response = response.strip()
            if response.startswith("```"):
                response = re.sub(r'^```(?:json)?\s*', '', response)
                response = re.sub(r'\s*```$', '', response)
                response = response.strip()
            
            # Try to extract JSON object from response
            json_match = re.search(r'\{.*?\}', response, re.DOTALL)
            if json_match:
                response = json_match.group(0)
            
            try:
                verification = json.loads(response)
                # Validate required fields
                if not isinstance(verification, dict):
                    raise ValueError("Response is not a JSON object")
                
                # Ensure required fields exist
                if "relevance_score" not in verification:
                    verification["relevance_score"] = 0.8
                if "reasoning" not in verification:
                    verification["reasoning"] = "Default reasoning"
                if "is_valid" not in verification:
                    verification["is_valid"] = True
                
                return verification
            except json.JSONDecodeError as json_err:
                print(f"Warning: JSON parse error in verification: {json_err}")
                # Try to extract score from response
                score_match = re.search(r'"relevance_score"\s*:\s*([0-9.]+)', response)
                if score_match:
                    score = float(score_match.group(1))
                    return {
                        "relevance_score": min(max(score, 0.0), 1.0),
                        "reasoning": "Extracted from response",
                        "is_valid": True
                    }
                raise json_err
            
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
        
        # Prefer articles with more authors
        authors = article_data.get('authors', [])
        if len(authors) > 0:
            score *= 1.0 + min(len(authors) / 10.0, 0.2)
        
        # Prefer recent articles
        year = article_data.get('year', '')
        if year:
            try:
                year_int = int(year)
                current_year = 2024
                if current_year - year_int <= 10:
                    score *= 1.1  # 10% boost for recent articles
            except ValueError:
                pass
        
        return score
    
    def _deduplicate_by_article(self, matches: List[Dict]) -> List[Dict]:
        """Remove duplicate matches from the same article."""
        seen_articles = {}
        
        for match in matches:
            article_id = match['article'].get('pmid', '')
            if article_id not in seen_articles:
                seen_articles[article_id] = match
            else:
                # Keep the one with higher score
                if match.get('final_rank_score', 0) > seen_articles[article_id].get('final_rank_score', 0):
                    seen_articles[article_id] = match
        
        return list(seen_articles.values())
    
    # ==================== Main Function ====================
    
    async def find_matching_sentences(
        self,
        input_sentence: str,
        top_k: int = 10,
        max_articles: int = 50,
        citation_format: str = "mla"
    ) -> List[Dict]:
        """
        Main function: Find top matching sentences in PubMed articles using MCP.
        
        Args:
            input_sentence: Input sentence to match
            top_k: Number of top results to return
            max_articles: Maximum number of articles to process
            citation_format: Citation format (mla, apa, nature)
            
        Returns:
            List of matched sentences with citations
        """
        print(f"🔍 Starting sentence matching (MCP) for: '{input_sentence[:100]}...'")
        
        # Stage 1: LLM Query Optimization
        print("📝 Stage 1: Generating optimized queries...")
        optimized_queries = await self._generate_optimized_queries(input_sentence)
        print(f"   Generated {len(optimized_queries)} queries")
        
        # Stage 2: Multi-Strategy Search
        print("🔎 Stage 2: Searching PubMed via MCP...")
        pmids = await self._search_pubmed_multi_strategy(input_sentence, optimized_queries)
        pmids = pmids[:max_articles]
        print(f"   Found {len(pmids)} articles")
        
        if not pmids:
            print("   No articles found")
            return []
        
        # Stage 3: Fetch and Parse Articles
        print("📚 Stage 3: Fetching article content via MCP...")
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
        print("🔢 Stage 4: Calculating similarity scores via MCP...")
        matches = await self._calculate_multi_signal_similarity(input_sentence, all_candidates)
        print(f"   Found {len(matches)} matches above threshold")
        
        if not matches:
            print("   No matches found")
            return []
        
        # Stage 5: LLM Verification
        print("✅ Stage 5: Verifying matches via MCP...")
        verified_matches = []
        for match in matches[:50]:  # Limit verification to top 50
            verification = await self._llm_verify_match(input_sentence, match)
            match['llm_verification'] = verification
            verified_matches.append(match)
        
        # Stage 6: Final Ranking
        print("🏆 Stage 6: Final ranking...")
        final_results = self._final_ranking(verified_matches, top_k)
        
        # Add citations
        for result in final_results:
            article = result['article']
            # Format article data for citation
            article_data = {
                'title': article.get('title', ''),
                'authors': article.get('authors', []),
                'journal': article.get('journal', ''),
                'year': article.get('year', ''),
                'pmid': article.get('pmid', ''),
            }
            
            # Format citation based on format
            if citation_format.lower() == "mla":
                citation = self.pubmed_enhancer.format_mla_citation(article_data, "")
            elif citation_format.lower() == "apa":
                citation = self.pubmed_enhancer.format_apa_citation(article_data, "")
            elif citation_format.lower() == "nature":
                citation = self.pubmed_enhancer.format_nature_citation(article_data, "")
            else:
                # Default to MLA
                citation = self.pubmed_enhancer.format_mla_citation(article_data, "")
            
            result['citation'] = citation
            result['citation_format'] = citation_format
        
        print(f"✅ Completed! Returning {len(final_results)} results")
        return final_results

