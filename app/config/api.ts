export const API_CONFIG = {
  MAIN_API_URL: process.env.NEXT_PUBLIC_MAIN_API_URL || 'https://final-ff.onrender.com',
  ENDPOINTS: {
    UPLOAD: '/api/upload',
    ANALYZE: '/api/analyze',
    CONSISTENCY: '/api/analyze/consistency',
    ESG: '/api/analyze/esg',
    PINECONE: {
      UPLOAD: '/api/pinecone/upload',
      SEARCH: '/api/pinecone/search'
    },
    SUMMARY: '/api/analyze/summary',
    GRI: '/api/analyze/gri',
    CSRD: '/api/analyze/csrd',
    SASB: '/api/analyze/sasb',
  }
} as const;
