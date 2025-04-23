// Simple in-memory document cache
interface DocumentCacheEntry {
  text: string;
  fileName: string;
  timestamp: number;
}

class DocumentCache {
  private cache: Map<string, DocumentCacheEntry>;

  constructor() {
    this.cache = new Map();
  }

  store(docId: string, text: string, fileName: string): void {
    this.cache.set(docId, {
      text,
      fileName,
      timestamp: Date.now()
    });
  }

  get(docId: string): DocumentCacheEntry | undefined {
    return this.cache.get(docId);
  }

  exists(docId: string): boolean {
    return this.cache.has(docId);
  }

  clear(): void {
    this.cache.clear();
  }
}

// Export singleton instance
export const documentCache = new DocumentCache(); 