import pytest
import httpx
import asyncio
from typing import AsyncGenerator

BASE_URL = "http://localhost:3000"

@pytest.fixture
def chat_request_data():
    return {
        "messages": [{"role": "user", "content": "Hello"}],
        "docText": "Sample document text",
        "docName": "test.pdf"
    }

@pytest.fixture
def sasb_request_data():
    return {
        "docText": "Sample sustainability report text discussing environmental impact and social responsibility."
    }

@pytest.fixture
def embed_request_data():
    return {
        "text": "Sample text for testing",
        "namespace": "test-namespace"
    }

@pytest.fixture
async def client() -> AsyncGenerator[httpx.AsyncClient, None]:
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30.0) as client:
        yield client

@pytest.mark.asyncio
async def test_chat_endpoint_with_valid_data(client: httpx.AsyncClient, chat_request_data):
    """Test chat endpoint with valid data"""
    try:
        response = await client.post("/api/chat", json=chat_request_data)
        assert response.status_code in [200, 201, 202]
    except httpx.RequestError as e:
        pytest.skip(f"Server not available: {str(e)}")

@pytest.mark.asyncio
async def test_sasb_endpoint_with_valid_data(client: httpx.AsyncClient, sasb_request_data):
    """Test SASB endpoint with valid data"""
    try:
        response = await client.post("/api/analyze/sasb", json=sasb_request_data)
        assert response.status_code in [200, 201, 202]
    except httpx.RequestError as e:
        pytest.skip(f"Server not available: {str(e)}")

@pytest.mark.asyncio
async def test_embed_endpoint_with_valid_data(client: httpx.AsyncClient, embed_request_data):
    """Test embed endpoint with valid data"""
    try:
        response = await client.post("/api/embed", json=embed_request_data)
        assert response.status_code in [200, 201, 202]
    except httpx.RequestError as e:
        pytest.skip(f"Server not available: {str(e)}") 