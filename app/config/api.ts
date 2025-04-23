export const API_CONFIG = {
  MAIN_API_URL: process.env.NEXT_PUBLIC_MAIN_API_URL || 'http://localhost:5001',
  ENDPOINTS: {
    UPLOAD: '/api/upload',
    ANALYZE: process.env.NEXT_PUBLIC_ANALYZE_ENDPOINT || '/analyze',
    CONSISTENCY: process.env.NEXT_PUBLIC_CONSISTENCY_ENDPOINT || '/consistency/consistency',
    ESG: process.env.NEXT_PUBLIC_ESG_ENDPOINT || '/esg',
    PINECONE: {
      UPLOAD: '/api/pinecone/upload',
      SEARCH: '/api/pinecone/search',
    },
    SUMMARY: '/api/analyze/summary',
    GRI: '/api/analyze/gri',
    CSRD: '/api/analyze/csrd',
    SASB: '/api/analyze/sasb',
  }
} as const;
