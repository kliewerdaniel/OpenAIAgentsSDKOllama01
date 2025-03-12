import os
import json
import hashlib
from typing import Dict, List, Optional

class DocumentMemory:
    """Simple document storage system for the agent."""
    
    def __init__(self, storage_dir: str = "./document_memory"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        
        self.index_file = os.path.join(storage_dir, "index.json")
        self.document_index = self._load_index()
    
    def _load_index(self) -> Dict:
        """Load document index from disk."""
        if os.path.exists(self.index_file):
            with open(self.index_file, 'r') as f:
                return json.load(f)
        return {"documents": {}}
    
    def _save_index(self):
        """Save document index to disk."""
        with open(self.index_file, 'w') as f:
            json.dump(self.document_index, f, indent=2)
    
    def _generate_doc_id(self, url: str) -> str:
        """Generate a unique ID for a document based on its URL."""
        return hashlib.md5(url.encode()).hexdigest()
    
    def store_document(self, url: str, content: str, metadata: Optional[Dict] = None) -> str:
        """Store a document and return its ID."""
        doc_id = self._generate_doc_id(url)
        doc_path = os.path.join(self.storage_dir, f"{doc_id}.txt")
        
        # Store document content
        with open(doc_path, 'w') as f:
            f.write(content)
        
        # Update index
        self.document_index["documents"][doc_id] = {
            "url": url,
            "path": doc_path,
            "metadata": metadata or {}
        }
        
        self._save_index()
        return doc_id
    
    def get_document(self, doc_id: str) -> Optional[Dict]:
        """Retrieve a document by ID."""
        if doc_id not in self.document_index["documents"]:
            return None
        
        doc_info = self.document_index["documents"][doc_id]
        
        try:
            with open(doc_info["path"], 'r') as f:
                content = f.read()
            return {
                "id": doc_id,
                "url": doc_info["url"],
                "content": content,
                "metadata": doc_info["metadata"]
            }
        except Exception as e:
            print(f"Error retrieving document {doc_id}: {e}")
            return None
    
    def get_document_by_url(self, url: str) -> Optional[Dict]:
        """Find and retrieve a document by URL."""
        doc_id = self._generate_doc_id(url)
        return self.get_document(doc_id)
    
    def list_documents(self) -> List[Dict]:
        """List all stored documents."""
        return [
            {"id": doc_id, "url": info["url"], "metadata": info["metadata"]}
            for doc_id, info in self.document_index["documents"].items()
        ]