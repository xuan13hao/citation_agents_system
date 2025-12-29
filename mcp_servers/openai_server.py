"""
MCP Server for OpenAI API tools.

This server exposes OpenAI LLM and embedding tools via MCP protocol.
"""
import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional

try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


class OpenAIMCPServer:
    """MCP server for OpenAI tools."""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL")
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable must be set")
    
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
                    "name": "openai-mcp-server",
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
                    "name": "openai_chat",
                    "description": "Chat completion using OpenAI models",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "messages": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "role": {"type": "string"},
                                        "content": {"type": "string"}
                                    }
                                },
                                "description": "List of messages"
                            },
                            "model": {
                                "type": "string",
                                "description": "Model name",
                                "default": "gpt-4-turbo"
                            },
                            "temperature": {
                                "type": "number",
                                "description": "Temperature",
                                "default": 0.4
                            },
                            "max_tokens": {
                                "type": "integer",
                                "description": "Maximum tokens",
                                "default": 4000
                            }
                        },
                        "required": ["messages"]
                    }
                },
                {
                    "name": "openai_embed",
                    "description": "Generate embeddings using OpenAI models",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "texts": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of texts to embed"
                            },
                            "model": {
                                "type": "string",
                                "description": "Embedding model",
                                "default": "text-embedding-3-small"
                            }
                        },
                        "required": ["texts"]
                    }
                }
            ]
        }
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool by name with arguments."""
        try:
            if tool_name == "openai_chat":
                messages = arguments.get("messages", [])
                model = arguments.get("model", "gpt-4-turbo")
                temperature = arguments.get("temperature", 0.4)
                max_tokens = arguments.get("max_tokens", 4000)
                
                response = await self._chat_completion(
                    messages, model, temperature, max_tokens
                )
                
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": response
                        }
                    ]
                }
            
            elif tool_name == "openai_embed":
                texts = arguments.get("texts", [])
                model = arguments.get("model", "text-embedding-3-small")
                
                embeddings = await self._generate_embeddings(texts, model)
                
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps({"embeddings": embeddings})
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
    
    async def _chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """Call OpenAI chat completion."""
        if not LANGCHAIN_AVAILABLE:
            raise RuntimeError("langchain-openai not available")
        
        llm_kwargs = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "openai_api_key": self.api_key,
        }
        
        if self.base_url:
            llm_kwargs["base_url"] = self.base_url
        
        llm = ChatOpenAI(**llm_kwargs)
        response = await llm.ainvoke(messages)
        
        if hasattr(response, 'content'):
            return response.content
        elif isinstance(response, str):
            return response
        else:
            return str(response)
    
    async def _generate_embeddings(self, texts: List[str], model: str) -> List[List[float]]:
        """Generate embeddings for texts."""
        if not LANGCHAIN_AVAILABLE:
            raise RuntimeError("langchain-openai not available")
        
        embeddings = OpenAIEmbeddings(
            model=model,
            openai_api_key=self.api_key,
            openai_api_base=self.base_url
        )
        
        # Generate embeddings
        result = await embeddings.aembed_documents(texts)
        return result


async def main():
    """Main entry point for MCP server (stdio mode)."""
    server = OpenAIMCPServer()
    
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

