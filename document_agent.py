import re
import json
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# Import the Agent directly from openai_agents
from agents import Agent, function_tool
from ollama_client import OllamaClient
from document_memory import DocumentMemory

# Import the agent adapter to add the run method to the Agent class
import agent_adapter

# Initialize document memory
document_memory = DocumentMemory()


# Define the tool schemas
class FetchDocumentInput(BaseModel):
    url: str = Field(..., description="URL of the document to fetch")


class FetchDocumentOutput(BaseModel):
    content: str = Field(..., description="Content of the document")


class ExtractInfoInput(BaseModel):
    text: str = Field(..., description="Text to extract information from")
    info_type: str = Field(
        ..., description="Type of information to extract (e.g., 'dates', 'names', 'key points')"
    )


class ExtractInfoOutput(BaseModel):
    information: List[str] = Field(..., description="List of extracted information")


class SearchDocumentInput(BaseModel):
    text: str = Field(..., description="Document text to search within")
    query: str = Field(..., description="Query to search for")


class SearchDocumentOutput(BaseModel):
    results: List[str] = Field(..., description="List of matching paragraphs or sentences")


# Implement tool functions
@function_tool
def fetch_document(url: str) -> Dict[str, Any]:
    """Fetches a document from a URL and returns its content.
    Checks document memory first before making a network request."""
    # Check if document already exists in memory
    cached_doc = document_memory.get_document_by_url(url)
    if cached_doc:
        print(f"Retrieved document from memory: {url}")
        return {"content": cached_doc["content"]}
    
    # If not in memory, fetch from URL
    try:
        print(f"Fetching document from URL: {url}")
        response = requests.get(url)
        response.raise_for_status()
        content = re.sub(r"<[^>]+>", "", response.text)  # Remove HTML tags
        
        # Store in document memory
        document_memory.store_document(url, content, {"fetched_at": str(datetime.now())})
        
        return {"content": content}
    except Exception as e:
        return {"content": f"Error fetching document: {str(e)}"}


@function_tool
def extract_info(text: str, info_type: str) -> Dict[str, Any]:
    """Extracts specified type of information from text using Ollama."""
    client = OllamaClient(model_name="mistral")

    prompt = f"""
    Extract all {info_type} from the following text.
    Return ONLY a JSON array with the items.

    TEXT:
    {text[:2000]}  # Limit text length to prevent context overflow

    JSON ARRAY OF {info_type.upper()}:
    """

    try:
        response = client.chat.completions.create(
            model="mistral",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Lower temperature for more deterministic output
        )

        result_text = response.choices[0].message.content
        print(f"Extract info response: {result_text[:100]}...")

        # Try to find JSON array in the response
        try:
            match = re.search(r"\[.*\]", result_text, re.DOTALL)
            if match:
                information = json.loads(match.group(0))
            else:
                # If no JSON array is found, try to parse the entire response as JSON
                try:
                    information = json.loads(result_text)
                    if not isinstance(information, list):
                        information = [result_text.strip()]
                except:
                    information = [result_text.strip()]
        except json.JSONDecodeError:
            # Split by commas or newlines if JSON parsing fails
            information = []
            for line in result_text.split('\n'):
                line = line.strip()
                if line and not line.startswith('```') and not line.endswith('```'):
                    information.append(line)
            if not information:
                information = [item.strip() for item in result_text.split(",")]
    except Exception as e:
        print(f"Error in extract_info: {str(e)}")
        information = [f"Error extracting information: {str(e)}"]

    return {"information": information}


@function_tool
def search_document(text: str, query: str) -> Dict[str, Any]:
    """Searches for relevant content in the document."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    client = OllamaClient(model_name="mistral")

    prompt = f"""
    You need to find paragraphs in a document that answer or relate to the query: "{query}"
    Rate each paragraph's relevance to the query on a scale of 0-10.
    Return the 3 most relevant paragraphs with their ratings as JSON.

    Document sections:
    {json.dumps(paragraphs[:15])}  # Limit to first 15 paragraphs for context limits

    Output format: [{"rating": 8, "text": "paragraph text"}, ...]
    """

    try:
        response = client.chat.completions.create(
            model="mistral",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Lower temperature for more deterministic output
        )

        result_text = response.choices[0].message.content
        print(f"Search document response: {result_text[:100]}...")

        # Try to find JSON array in the response
        try:
            match = re.search(r"\[.*\]", result_text, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
                results = [item["text"] for item in parsed if "text" in item]
            else:
                # Try to parse the entire response as JSON
                try:
                    parsed = json.loads(result_text)
                    if isinstance(parsed, list):
                        results = [item.get("text", str(item)) for item in parsed]
                    else:
                        results = [str(parsed)]
                except:
                    # If JSON parsing fails, extract quoted text
                    results = re.findall(r'"([^"]+)"', result_text)
                    if not results:
                        results = [result_text]
        except json.JSONDecodeError:
            # If JSON parsing fails completely
            results = [result_text]
    except Exception as e:
        print(f"Error in search_document: {str(e)}")
        results = [f"Error searching document: {str(e)}"]

    return {"results": results}


# Define additional tools for document memory management
class ListDocumentsOutput(BaseModel):
    documents: List[Dict] = Field(..., description="List of stored documents")

class GetDocumentInput(BaseModel):
    url: str = Field(..., description="URL of the document to retrieve")

class GetDocumentOutput(BaseModel):
    content: str = Field(..., description="Content of the retrieved document")
    metadata: Dict = Field(..., description="Metadata of the document")

@function_tool
def list_documents() -> Dict[str, Any]:
    """Lists all stored documents in memory."""
    documents = document_memory.list_documents()
    return {"documents": documents}

@function_tool
def get_document(url: str) -> Dict[str, Any]:
    """Retrieves a document from memory by URL."""
    doc = document_memory.get_document_by_url(url)
    if not doc:
        return {"content": "Document not found", "metadata": {}}
    return {"content": doc["content"], "metadata": doc["metadata"]}

# Create a Document Analysis Agent
def create_document_agent():
    """Creates and returns an AI agent for document analysis."""
    client = OllamaClient(model_name="mistral")
    
    # Collect all the tools decorated with function_tool
    tools = [
        fetch_document,
        extract_info,
        search_document,
        list_documents,
        get_document
    ]

    agent = Agent(
        name="DocumentAnalysisAgent",
        instructions=(
            "You are a Document Analysis Assistant that helps users extract valuable information from documents.\n\n"
            "When given a task:\n"
            "1. If you need to analyze a document, first use fetch_document to get its content.\n"
            "2. Use extract_info to identify specific information in the document.\n"
            "3. Use search_document to find answers to specific questions.\n"
            "4. Summarize your findings in a clear, organized manner.\n\n"
            "You can manage documents with:\n"
            "- list_documents to see all stored documents\n"
            "- get_document to retrieve a previously fetched document\n\n"
            "Always be thorough and accurate in your analysis. If the document content is too large, "
            "focus on the most relevant sections for the user's query."
        ),
        tools=tools,
        model=client,
    )

    return agent
