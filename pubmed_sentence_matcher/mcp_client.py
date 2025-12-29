"""
MCP Client for connecting to MCP servers.

This module provides a client interface to communicate with MCP servers
for PubMed and OpenAI tools.
"""
import asyncio
import json
import subprocess
from typing import Any, Dict, List, Optional
import sys


class MCPClient:
    """Client for communicating with MCP servers via stdio."""
    
    def __init__(self, server_command: List[str], server_name: str = "mcp-server"):
        """
        Initialize MCP client.
        
        Args:
            server_command: Command to start the MCP server (e.g., ["python", "server.py"])
            server_name: Name of the server for logging
        """
        self.server_command = server_command
        self.server_name = server_name
        self.process: Optional[subprocess.Popen] = None
        self.request_id = 0
        self.initialized = False
    
    async def start(self):
        """Start the MCP server process."""
        if self.process is not None:
            return
        
        self.process = subprocess.Popen(
            self.server_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0
        )
        
        # Initialize the connection
        init_request = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "pubmed-sentence-matcher",
                    "version": "1.0.0"
                }
            }
        }
        
        response = await self._send_request(init_request)
        if response and "error" not in response:
            self.initialized = True
    
    async def stop(self):
        """Stop the MCP server process."""
        if self.process:
            self.process.terminate()
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(self.process.wait),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                self.process.kill()
            self.process = None
            self.initialized = False
    
    def _next_id(self) -> int:
        """Get next request ID."""
        self.request_id += 1
        return self.request_id
    
    async def _send_request(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send a request to the MCP server and wait for response."""
        if not self.process:
            raise RuntimeError("Server not started")
        
        request_json = json.dumps(request) + "\n"
        
        try:
            # Write request
            self.process.stdin.write(request_json)
            self.process.stdin.flush()
            
            # Read response (simple line-based protocol)
            response_line = await asyncio.get_event_loop().run_in_executor(
                None, self.process.stdout.readline
            )
            
            if not response_line:
                return None
            
            response = json.loads(response_line.strip())
            return response
        except Exception as e:
            print(f"Error communicating with {self.server_name}: {e}", file=sys.stderr)
            return {"error": {"code": -32603, "message": str(e)}}
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from the server."""
        request = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/list",
            "params": {}
        }
        
        response = await self._send_request(request)
        if response and "result" in response:
            return response["result"].get("tools", [])
        return []
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call a tool on the MCP server.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Arguments for the tool
            
        Returns:
            Tool result (parsed from JSON if applicable)
        """
        request = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        
        response = await self._send_request(request)
        
        if response and "error" in response:
            error = response["error"]
            raise RuntimeError(f"Tool call failed: {error.get('message', 'Unknown error')}")
        
        if response and "result" in response:
            result = response["result"]
            # Extract content from MCP response format
            if "content" in result and len(result["content"]) > 0:
                content = result["content"][0]
                if content.get("type") == "text":
                    text = content.get("text", "")
                    # Try to parse as JSON, fallback to string
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        return text
            return result
        
        return None


class MCPToolManager:
    """Manager for multiple MCP clients."""
    
    def __init__(self):
        self.pubmed_client: Optional[MCPClient] = None
        self.openai_client: Optional[MCPClient] = None
    
    async def initialize(
        self,
        pubmed_server_path: Optional[str] = None,
        openai_server_path: Optional[str] = None
    ):
        """
        Initialize MCP clients.
        
        Args:
            pubmed_server_path: Path to PubMed MCP server script
            openai_server_path: Path to OpenAI MCP server script
        """
        import os
        
        # Default paths
        if pubmed_server_path is None:
            # Get project root (parent of pubmed_sentence_matcher directory)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            pubmed_server_path = os.path.join(
                project_root,
                "mcp_servers",
                "pubmed_server.py"
            )
        
        if openai_server_path is None:
            # Get project root (parent of pubmed_sentence_matcher directory)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            openai_server_path = os.path.join(
                project_root,
                "mcp_servers",
                "openai_server.py"
            )
        
        # Start PubMed server
        self.pubmed_client = MCPClient(
            ["python", pubmed_server_path],
            "pubmed-server"
        )
        await self.pubmed_client.start()
        
        # Start OpenAI server
        self.openai_client = MCPClient(
            ["python", openai_server_path],
            "openai-server"
        )
        await self.openai_client.start()
    
    async def shutdown(self):
        """Shutdown all MCP clients."""
        if self.pubmed_client:
            await self.pubmed_client.stop()
        if self.openai_client:
            await self.openai_client.stop()
    
    async def pubmed_search(self, query: str, max_results: int = 20) -> List[str]:
        """Search PubMed."""
        if not self.pubmed_client:
            raise RuntimeError("PubMed client not initialized")
        
        result = await self.pubmed_client.call_tool(
            "pubmed_search",
            {"query": query, "max_results": max_results}
        )
        
        if isinstance(result, dict) and "pmids" in result:
            return result["pmids"]
        return []
    
    async def pubmed_fetch(self, pmids: List[str]) -> List[Dict[str, Any]]:
        """Fetch PubMed articles."""
        if not self.pubmed_client:
            raise RuntimeError("PubMed client not initialized")
        
        result = await self.pubmed_client.call_tool(
            "pubmed_fetch",
            {"pmids": pmids}
        )
        
        if isinstance(result, dict) and "articles" in result:
            return result["articles"]
        return []
    
    async def openai_chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4-turbo",
        temperature: float = 0.4,
        max_tokens: int = 4000
    ) -> str:
        """Call OpenAI chat completion."""
        if not self.openai_client:
            raise RuntimeError("OpenAI client not initialized")
        
        result = await self.openai_client.call_tool(
            "openai_chat",
            {
                "messages": messages,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
        )
        
        if isinstance(result, str):
            return result
        return str(result)
    
    async def openai_embed(
        self,
        texts: List[str],
        model: str = "text-embedding-3-small"
    ) -> List[List[float]]:
        """Generate embeddings."""
        if not self.openai_client:
            raise RuntimeError("OpenAI client not initialized")
        
        result = await self.openai_client.call_tool(
            "openai_embed",
            {"texts": texts, "model": model}
        )
        
        if isinstance(result, dict) and "embeddings" in result:
            return result["embeddings"]
        return []

