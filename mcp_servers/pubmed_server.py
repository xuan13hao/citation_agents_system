"""
MCP Server for PubMed API tools.

This server exposes PubMed search and article retrieval tools via MCP protocol.
"""
import asyncio
import json
import sys
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional
import xml.etree.ElementTree as ET

try:
    from Bio import Entrez
    ENTREZ_AVAILABLE = True
except ImportError:
    ENTREZ_AVAILABLE = False

# MCP Protocol implementation (simplified version)
# In production, you would use the official mcp Python SDK


class MCPServer:
    """Simple MCP server implementation for PubMed tools."""
    
    def __init__(self, email: str = "pubmed-matcher@example.com", tool: str = "PubMed-MCP"):
        self.email = email
        self.tool = tool
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP protocol request."""
        method = request.get("method")
        params = request.get("params", {})
        
        if method == "tools/list":
            return await self.list_tools()
        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            return await self.call_tool(tool_name, arguments)
        elif method == "initialize":
            return {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "pubmed-mcp-server",
                    "version": "1.0.0"
                }
            }
        else:
            return {"error": {"code": -32601, "message": f"Method not found: {method}"}}
    
    async def list_tools(self) -> Dict[str, Any]:
        """List available tools."""
        return {
            "tools": [
                {
                    "name": "pubmed_search",
                    "description": "Search PubMed articles by query string",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query string"
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of results",
                                "default": 20
                            }
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "pubmed_fetch",
                    "description": "Fetch article details by PMID",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "pmids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of PubMed IDs"
                            }
                        },
                        "required": ["pmids"]
                    }
                },
                {
                    "name": "pubmed_get_fulltext",
                    "description": "Get full text from PMC if available",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "pmid": {
                                "type": "string",
                                "description": "PubMed ID"
                            }
                        },
                        "required": ["pmid"]
                    }
                }
            ]
        }
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool by name with arguments."""
        try:
            if tool_name == "pubmed_search":
                query = arguments.get("query")
                max_results = arguments.get("max_results", 20)
                pmids = await self._search_pubmed(query, max_results)
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps({"pmids": pmids})
                        }
                    ]
                }
            
            elif tool_name == "pubmed_fetch":
                pmids = arguments.get("pmids", [])
                articles = await self._fetch_articles(pmids)
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps({"articles": articles})
                        }
                    ]
                }
            
            elif tool_name == "pubmed_get_fulltext":
                pmid = arguments.get("pmid")
                fulltext = await self._get_fulltext(pmid)
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps({"fulltext": fulltext})
                        }
                    ]
                }
            
            else:
                return {
                    "error": {
                        "code": -32601,
                        "message": f"Unknown tool: {tool_name}"
                    }
                }
        except Exception as e:
            return {
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                },
                "isError": True
            }
    
    async def _search_pubmed(self, query: str, max_results: int = 20) -> List[str]:
        """Search PubMed and return PMIDs."""
        try:
            if ENTREZ_AVAILABLE:
                Entrez.email = self.email
                handle = Entrez.esearch(
                    db="pubmed",
                    term=query,
                    retmax=max_results,
                    retmode="json"
                )
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
                url = f"{self.base_url}/esearch.fcgi?" + urllib.parse.urlencode(params)
                
                with urllib.request.urlopen(url) as response:
                    data = response.read().decode('utf-8')
                    result = json.loads(data)
                    return result.get('esearchresult', {}).get('idlist', [])
        except Exception as e:
            print(f"Error searching PubMed: {e}", file=sys.stderr)
            return []
    
    async def _fetch_articles(self, pmids: List[str]) -> List[Dict[str, Any]]:
        """Fetch article details by PMIDs."""
        if not pmids:
            return []
        
        try:
            if ENTREZ_AVAILABLE:
                Entrez.email = self.email
                handle = Entrez.efetch(
                    db="pubmed",
                    id=','.join(pmids),
                    retmode="xml"
                )
                xml_data = handle.read()
                handle.close()
                
                # Parse XML
                root = ET.fromstring(xml_data)
                articles = []
                
                for article in root.findall('.//PubmedArticle'):
                    article_data = self._parse_article_xml(article)
                    if article_data:
                        articles.append(article_data)
                
                return articles
            else:
                # Fallback to urllib
                params = {
                    'db': 'pubmed',
                    'id': ','.join(pmids),
                    'retmode': 'xml',
                    'tool': self.tool,
                    'email': self.email
                }
                url = f"{self.base_url}/efetch.fcgi?" + urllib.parse.urlencode(params)
                
                with urllib.request.urlopen(url) as response:
                    xml_data = response.read()
                    root = ET.fromstring(xml_data)
                    articles = []
                    
                    for article in root.findall('.//PubmedArticle'):
                        article_data = self._parse_article_xml(article)
                        if article_data:
                            articles.append(article_data)
                    
                    return articles
        except Exception as e:
            print(f"Error fetching articles: {e}", file=sys.stderr)
            return []
    
    def _parse_article_xml(self, article_elem: ET.Element) -> Optional[Dict[str, Any]]:
        """Parse a PubmedArticle XML element into a dictionary."""
        try:
            # Extract PMID
            pmid_elem = article_elem.find('.//PMID')
            pmid = pmid_elem.text if pmid_elem is not None else ""
            
            # Extract title
            title_elem = article_elem.find('.//ArticleTitle')
            title = title_elem.text if title_elem is not None else ""
            
            # Extract abstract
            abstract_elems = article_elem.findall('.//AbstractText')
            abstract = " ".join([elem.text for elem in abstract_elems if elem.text])
            
            # Extract authors
            authors = []
            for author in article_elem.findall('.//Author'):
                lastname = author.find('LastName')
                firstname = author.find('ForeName')
                if lastname is not None and firstname is not None:
                    authors.append(f"{firstname.text} {lastname.text}")
            
            # Extract journal
            journal_elem = article_elem.find('.//Journal/Title')
            journal = journal_elem.text if journal_elem is not None else ""
            
            # Extract year
            year_elem = article_elem.find('.//PubDate/Year')
            year = year_elem.text if year_elem is not None else ""
            
            return {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "journal": journal,
                "year": year
            }
        except Exception as e:
            print(f"Error parsing article XML: {e}", file=sys.stderr)
            return None
    
    async def _get_fulltext(self, pmid: str) -> str:
        """Get full text from PMC if available."""
        try:
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
                    # For now, return empty - full PMC parsing would be more complex
                    return ""
        except Exception as e:
            print(f"Error getting fulltext: {e}", file=sys.stderr)
        return ""


async def main():
    """Main entry point for MCP server (stdio mode)."""
    server = MCPServer()
    
    # Read from stdin, write to stdout (MCP stdio protocol)
    while True:
        try:
            line = await asyncio.get_event_loop().run_in_executor(
                None, sys.stdin.readline
            )
            if not line:
                break
            
            request = json.loads(line.strip())
            request_id = request.get("id")
            
            # Handle request
            result = await server.handle_request(request)
            
            # Format response according to JSON-RPC 2.0
            if "error" in result:
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": result["error"]
                }
            else:
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": result
                }
            
            # Write response
            response_json = json.dumps(response)
            print(response_json, flush=True)
            
        except json.JSONDecodeError:
            error_response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32700,
                    "message": "Parse error"
                }
            }
            print(json.dumps(error_response), flush=True)
        except Exception as e:
            error_response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            }
            print(json.dumps(error_response), flush=True)


if __name__ == "__main__":
    asyncio.run(main())

