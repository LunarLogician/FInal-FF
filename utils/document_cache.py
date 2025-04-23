from typing import Dict, Optional
from datetime import datetime

class DocumentCache:
    def __init__(self):
        self._cache: Dict[str, Dict] = {}

    def store(self, doc_id: str, text: str, metadata: Optional[Dict] = None) -> None:
        """Store document in cache"""
        self._cache[doc_id] = {
            "text": text,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }

    def get(self, doc_id: str) -> Optional[Dict]:
        """Get document from cache"""
        return self._cache.get(doc_id)

    def exists(self, doc_id: str) -> bool:
        """Check if document exists in cache"""
        return doc_id in self._cache

    def keys(self) -> list:
        """Get all document IDs in cache"""
        return list(self._cache.keys())

# Global instance
document_cache = DocumentCache() 