# Document Analysis Platform

AI-powered document analysis platform with ESG, GRI, and CSRD compliance checking.

## Tech Stack

- Frontend: Next.js + TypeScript
- Backend: FastAPI
- Database: Pinecone
- AI: OpenAI

## Environment Setup

Required environment variables:
```env
# OpenAI
OPENAI_API_KEY=your-key

# Pinecone
PINECONE_API_KEY=your-key
PINECONE_ENVIRONMENT=us-east-1
PINECONE_INDEX=embed-upload

# API URLs
NEXT_PUBLIC_API_URL=http://localhost:5001
NEXT_PUBLIC_BACKEND_URL=http://localhost:5001
```

## Deployment

The application is configured for deployment on Render:
- Frontend: Deploy as a Static Site
- Backend: Deploy as a Web Service

Remember to set the environment variables in your Render dashboard.
