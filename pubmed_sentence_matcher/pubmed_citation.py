"""
PubMed API integration for citation enhancement.
This module provides functions to search PubMed by title and format citations in MLA style.
"""
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Dict, Optional, List
import time

try:
    from Bio import Entrez
    ENTREZ_AVAILABLE = True
except ImportError:
    ENTREZ_AVAILABLE = False
    print("Warning: Bio.Entrez not available. Install biopython: pip install biopython")


class PubMedCitationEnhancer:
    """Enhances citations by searching PubMed API and formatting in MLA style."""
    
    def __init__(self, email: str = "hrteam@h2-alpha.com", tool: str = "GPT-Researcher"):
        """
        Initialize the PubMed citation enhancer.
        
        Args:
            email: Email address for API identification
            tool: Tool name for API identification
        """
        self.email = email
        self.tool = tool
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        
    def search_pubmed_by_title(self, title: str) -> Optional[Dict]:
        """
        Search PubMed for an article by title and return detailed information.
        
        Args:
            title: The article title to search for
            
        Returns:
            Dictionary containing article details or None if not found
        """
        try:
            # Clean the title for search
            clean_title = self._clean_title_for_search(title)
            
            # Step 1: Search for PMIDs using ESearch
            pmid = self._search_for_pmid(clean_title)
            if not pmid:
                return None
                
            # Step 2: Get detailed information using EFetch
            article_data = self._fetch_article_details(pmid)
            if not article_data:
                return None
                
            return article_data
            
        except Exception as e:
            print(f"Error searching PubMed for title '{title}': {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _clean_title_for_search(self, title: str) -> str:
        """Clean title for PubMed search."""
        # Remove common prefixes and suffixes
        title = re.sub(r'^[^\w]*', '', title)  # Remove leading non-word chars
        title = re.sub(r'[^\w]*$', '', title)  # Remove trailing non-word chars
        
        # Remove extra whitespace
        title = re.sub(r'\s+', ' ', title).strip()
        
        # Return cleaned title (no URL encoding here)
        return title
    
    def _search_for_pmid(self, title: str) -> Optional[str]:
        """Search PubMed for PMID using title."""
        try:
            if ENTREZ_AVAILABLE:
                return self._search_with_entrez(title)
            else:
                return self._search_with_urllib(title)
        except Exception as e:
            print(f"Error searching for PMID: {e}")
            return None
    
    def _search_with_entrez(self, title: str) -> Optional[str]:
        """Search using Bio.Entrez (preferred method)."""
        try:
            # Set email for Entrez
            Entrez.email = self.email
            
            # Strategy 1: Try exact title match with quotes (as suggested by user)
            search_term = f'"{title}"[ti]'
            handle = Entrez.esearch(db="pubmed", term=search_term, retmax=50)
            record = Entrez.read(handle)
            handle.close()
            
            if record['IdList']:
                return record['IdList'][0]
            
            # Strategy 2: Try lowercase title match with quotes
            search_term = f'"{title.lower()}"[ti]'
            handle = Entrez.esearch(db="pubmed", term=search_term, retmax=50)
            record = Entrez.read(handle)
            handle.close()
            
            if record['IdList']:
                return record['IdList'][0]
            
            # Strategy 3: Try without quotes (most flexible)
            search_term = f"{title}[ti]"
            handle = Entrez.esearch(db="pubmed", term=search_term, retmax=50)
            record = Entrez.read(handle)
            handle.close()
            
            if record['IdList']:
                return record['IdList'][0]
            
            # Strategy 4: Try partial title match (first 50 characters)
            if len(title) > 50:
                partial_title = title[:50]
                search_term = f'"{partial_title}"[ti]'
                handle = Entrez.esearch(db="pubmed", term=search_term, retmax=50)
                record = Entrez.read(handle)
                handle.close()
                
                if record['IdList']:
                    return record['IdList'][0]
            
            # Strategy 5: Try key terms search
            key_terms = self._extract_key_terms(title)
            if key_terms:
                time.sleep(0.5)  # Add delay to respect rate limits
                
                search_term = " AND ".join(key_terms[:2]) + "[ti]"
                handle = Entrez.esearch(db="pubmed", term=search_term, retmax=50)
                record = Entrez.read(handle)
                handle.close()
                
                if record['IdList']:
                    return record['IdList'][0]
            return None
            
        except Exception as e:
            print(f"Error with Entrez search: {e}")
            return None
    
    def _search_with_urllib(self, title: str) -> Optional[str]:
        """Fallback search using urllib."""
        try:
            # Strategy 1: Try exact title match (case-insensitive)
            params = {
                'db': 'pubmed',
                'term': f'"{title.lower()}"[Title]',
                'retmax': 1,
                'retmode': 'json',
                'tool': self.tool,
                'email': self.email
            }
            
            url = f"{self.base_url}/esearch.fcgi?" + urllib.parse.urlencode(params)
            
            with urllib.request.urlopen(url) as response:
                data = response.read().decode('utf-8')
                import json
                result = json.loads(data)
                
                if result.get('esearchresult', {}).get('idlist'):
                    return result['esearchresult']['idlist'][0]
            
            # Strategy 2: Try original case exact match
            params['term'] = f'"{title}"[Title]'
            url = f"{self.base_url}/esearch.fcgi?" + urllib.parse.urlencode(params)
            
            with urllib.request.urlopen(url) as response:
                data = response.read().decode('utf-8')
                result = json.loads(data)
                
                if result.get('esearchresult', {}).get('idlist'):
                    return result['esearchresult']['idlist'][0]
            
            # Strategy 3: Try key terms search
            key_terms = self._extract_key_terms(title)
            if key_terms:
                import time
                time.sleep(0.5)  # Add delay to respect rate limits
                
                search_term = " AND ".join(key_terms[:2])
                params['term'] = f'{search_term}[Title]'
                url = f"{self.base_url}/esearch.fcgi?" + urllib.parse.urlencode(params)
                
                with urllib.request.urlopen(url) as response:
                    data = response.read().decode('utf-8')
                    result = json.loads(data)
                    
                    if result.get('esearchresult', {}).get('idlist'):
                        return result['esearchresult']['idlist'][0]
                    
            return None
            
        except Exception as e:
            print(f"Error with urllib search: {e}")
            return None
    
    def _extract_key_terms(self, title: str) -> list:
        """Extract key terms from title for flexible searching."""
        # Remove common words and extract meaningful terms
        common_words = {
            'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall'
        }
        
        # Split title into words and filter
        words = title.lower().split()
        key_terms = [word.strip('.,;:!?()[]{}"\'') for word in words 
                    if len(word.strip('.,;:!?()[]{}"\'')) > 2 and word.lower() not in common_words]
        
        # Return up to 4 most relevant terms for better matching
        return key_terms[:4]
    
    def _fetch_article_details(self, pmid: str) -> Optional[Dict]:
        """Fetch detailed article information using PMID."""
        try:
            if ENTREZ_AVAILABLE:
                return self._fetch_with_entrez(pmid)
            else:
                return self._fetch_with_urllib(pmid)
        except Exception as e:
            print(f"Error fetching article details for PMID {pmid}: {e}")
            return None
    
    def _fetch_with_entrez(self, pmid: str) -> Optional[Dict]:
        """Fetch using Bio.Entrez (preferred method)."""
        try:
            # Set email for Entrez
            Entrez.email = self.email
            
            # Fetch article data
            handle = Entrez.efetch(db="pubmed", id=pmid, retmode="xml")
            xml_data = handle.read().decode('utf-8')
            handle.close()
            
            # Parse XML response
            root = ET.fromstring(xml_data)
            
            # Extract article information
            article_info = self._parse_pubmed_xml(root)
            return article_info
            
        except Exception as e:
            print(f"Error with Entrez fetch: {e}")
            return None
    
    def _fetch_with_urllib(self, pmid: str) -> Optional[Dict]:
        """Fallback fetch using urllib."""
        try:
            params = {
                'db': 'pubmed',
                'id': pmid,
                'retmode': 'xml',
                'tool': self.tool,
                'email': self.email
            }
            
            url = f"{self.base_url}/efetch.fcgi?" + urllib.parse.urlencode(params)
            
            with urllib.request.urlopen(url) as response:
                xml_data = response.read().decode('utf-8')
                
            # Parse XML response
            root = ET.fromstring(xml_data)
            
            # Extract article information
            article_info = self._parse_pubmed_xml(root)
            return article_info
            
        except Exception as e:
            print(f"Error with urllib fetch: {e}")
            return None
    
    def _parse_pubmed_xml(self, root: ET.Element) -> Dict:
        """Parse PubMed XML response to extract article details."""
        article_info = {}
        
        try:
            # Find the article element
            article = root.find('.//PubmedArticle')
            if article is None:
                return {}
            
            # Extract title
            title_elem = article.find('.//ArticleTitle')
            if title_elem is not None:
                article_info['title'] = self._clean_xml_text(title_elem.text)
            
            # Extract authors
            authors = []
            author_list = article.find('.//AuthorList')
            if author_list is not None:
                for author in author_list.findall('.//Author'):
                    last_name = author.find('LastName')
                    first_name = author.find('ForeName')
                    initials = author.find('Initials')
                    
                    if last_name is not None:
                        name = self._clean_xml_text(last_name.text)
                        if first_name is not None:
                            name += f", {self._clean_xml_text(first_name.text)}"
                        elif initials is not None:
                            name += f", {self._clean_xml_text(initials.text)}"
                        authors.append(name)
            
            article_info['authors'] = authors
            
            # Extract journal information
            journal = article.find('.//Journal')
            if journal is not None:
                journal_title = journal.find('.//Title')
                if journal_title is not None:
                    article_info['journal'] = self._clean_xml_text(journal_title.text)
                
                # Extract volume, issue, pages
                journal_issue = journal.find('.//JournalIssue')
                if journal_issue is not None:
                    volume = journal_issue.find('Volume')
                    if volume is not None:
                        article_info['volume'] = self._clean_xml_text(volume.text)
                    
                    issue = journal_issue.find('Issue')
                    if issue is not None:
                        article_info['issue'] = self._clean_xml_text(issue.text)
                    
                    pub_date = journal_issue.find('PubDate')
                    if pub_date is not None:
                        year = pub_date.find('Year')
                        if year is not None:
                            article_info['pub_year'] = self._clean_xml_text(year.text)
                        
                        month = pub_date.find('Month')
                        if month is not None:
                            article_info['pub_month'] = self._clean_xml_text(month.text)
                        
                        day = pub_date.find('Day')
                        if day is not None:
                            article_info['pub_day'] = self._clean_xml_text(day.text)
            
            # Extract pages
            pages = article.find('.//MedlinePgn')
            if pages is not None:
                article_info['pages'] = self._clean_xml_text(pages.text)
            
            # Extract DOI
            doi_elem = article.find('.//ELocationID[@EIdType="doi"]')
            if doi_elem is not None:
                article_info['doi'] = self._clean_xml_text(doi_elem.text)
            
            # Extract PMID
            pmid_elem = article.find('.//PMID')
            if pmid_elem is not None:
                article_info['pmid'] = self._clean_xml_text(pmid_elem.text)
            
            return article_info
            
        except Exception as e:
            print(f"Error parsing PubMed XML: {e}")
            return {}
    
    def _clean_xml_text(self, text: str) -> str:
        """Clean XML text content."""
        if text is None:
            return ""
        return text.strip()
    
    def format_mla_citation(self, article_data: Dict, original_url: str = "") -> str:
        """
        Format article data into MLA citation style.
        
        Args:
            article_data: Dictionary containing article information
            original_url: Original URL from the reference
            
        Returns:
            MLA formatted citation string
        """
        try:
            citation_parts = []
            
            # Authors
            authors = article_data.get('authors', [])
            if authors:
                if len(authors) == 1:
                    citation_parts.append(authors[0])
                elif len(authors) == 2:
                    citation_parts.append(f"{authors[0]} and {authors[1]}")
                elif len(authors) > 2:
                    citation_parts.append(f"{authors[0]} et al.")
            
            # Title
            title = article_data.get('title', '')
            if title:
                citation_parts.append(f'"{title}"')
            
            # Journal
            journal = article_data.get('journal', '')
            if journal:
                citation_parts.append(f"<em>{journal}</em>")
            
            # Volume and Issue
            volume = article_data.get('volume', '')
            issue = article_data.get('issue', '')
            if volume:
                if issue:
                    citation_parts.append(f"{volume}, no. {issue}")
                else:
                    citation_parts.append(f"{volume}")
            
            # Publication date
            pub_year = article_data.get('pub_year', '')
            pub_month = article_data.get('pub_month', '')
            pub_day = article_data.get('pub_day', '')
            
            if pub_year:
                date_parts = [pub_year]
                if pub_month:
                    date_parts.insert(0, pub_month)
                if pub_day:
                    date_parts.insert(1, pub_day)
                citation_parts.append(', '.join(date_parts))
            
            # Pages
            pages = article_data.get('pages', '')
            if pages:
                citation_parts.append(f"pp. {pages}")
            
            # DOI or URL
            doi = article_data.get('doi', '')
            if doi:
                citation_parts.append(f"doi:{doi}")
            elif original_url:
                citation_parts.append(f"<{original_url}>")
            
            return ', '.join(citation_parts)
            
        except Exception as e:
            print(f"Error formatting MLA citation: {e}")
            return ""
    
    def format_apa_citation(self, article_data: Dict, original_url: str = "") -> str:
        """
        Format article data into APA citation style.
        
        Args:
            article_data: Dictionary containing article information
            original_url: Original URL from the reference
            
        Returns:
            APA formatted citation string
        """
        try:
            citation_parts = []
            
            # Authors
            authors = article_data.get('authors', [])
            if authors:
                if len(authors) == 1:
                    citation_parts.append(authors[0])
                elif len(authors) <= 7:
                    author_list = []
                    for i, author in enumerate(authors):
                        if i == len(authors) - 1:
                            author_list.append(f"& {author}")
                        else:
                            author_list.append(author)
                    citation_parts.append(', '.join(author_list))
                else:
                    citation_parts.append(f"{authors[0]} et al.")
            
            # Publication date
            pub_year = article_data.get('pub_year', '')
            if pub_year:
                citation_parts.append(f"({pub_year})")
            
            # Title
            title = article_data.get('title', '')
            if title:
                citation_parts.append(title + ".")
            
            # Journal
            journal = article_data.get('journal', '')
            if journal:
                citation_parts.append(f"<em>{journal}</em>")
            
            # Volume and Issue
            volume = article_data.get('volume', '')
            issue = article_data.get('issue', '')
            if volume:
                if issue:
                    citation_parts.append(f"{volume}({issue})")
                else:
                    citation_parts.append(f"{volume}")
            
            # Pages
            pages = article_data.get('pages', '')
            if pages:
                citation_parts.append(f"pp. {pages}")
            
            # DOI or URL
            doi = article_data.get('doi', '')
            if doi:
                citation_parts.append(f"https://doi.org/{doi}")
            elif original_url:
                citation_parts.append(f"Retrieved from {original_url}")
            
            return ' '.join(citation_parts)
            
        except Exception as e:
            print(f"Error formatting APA citation: {e}")
            return ""
    
    def format_nature_citation(self, article_data: Dict, original_url: str = "") -> str:
        """
        Format article data into Nature citation style.
        
        Args:
            article_data: Dictionary containing article information
            original_url: Original URL from the reference
            
        Returns:
            Nature formatted citation string
        """
        try:
            citation_parts = []
            
            # Authors
            authors = article_data.get('authors', [])
            if authors:
                if len(authors) == 1:
                    citation_parts.append(authors[0])
                elif len(authors) <= 6:
                    citation_parts.append(', '.join(authors))
                else:
                    citation_parts.append(f"{authors[0]} et al.")
            
            # Title
            title = article_data.get('title', '')
            if title:
                citation_parts.append(title)
            
            # Journal
            journal = article_data.get('journal', '')
            if journal:
                citation_parts.append(f"<em>{journal}</em>")
            
            # Volume, Issue, Pages, Year
            volume = article_data.get('volume', '')
            issue = article_data.get('issue', '')
            pages = article_data.get('pages', '')
            pub_year = article_data.get('pub_year', '')
            
            journal_info = []
            if volume:
                journal_info.append(f"{volume}")
            if issue:
                journal_info.append(f"({issue})")
            if pages:
                journal_info.append(f"{pages}")
            if pub_year:
                journal_info.append(f"({pub_year})")
            
            if journal_info:
                citation_parts.append(' '.join(journal_info))
            
            # DOI or URL
            doi = article_data.get('doi', '')
            if doi:
                citation_parts.append(f"doi:{doi}")
            elif original_url:
                citation_parts.append(f"({original_url})")
            
            return '. '.join(citation_parts)
            
        except Exception as e:
            print(f"Error formatting Nature citation: {e}")
            return ""

