import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone

# Load environment variables
load_dotenv()

# Initialize Pinecone
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment="us-east-1"  # Updated to correct environment
)

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_KEY")
)

# Example function to create and query a vector store
def create_vector_store(texts, index_name="my-rag-index"):
    """
    Create a vector store from a list of texts
    """
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            name=index_name,
            dimension=1536,  # OpenAI embeddings dimension
            metric="cosine"
        )
    
    return Pinecone.from_texts(
        texts=texts,
        embedding=embeddings,
        index_name=index_name
    )

def query_vector_store(vector_store, query, k=3):
    """
    Query the vector store for similar documents
    """
    results = vector_store.similarity_search(query, k=k)
    return results

if __name__ == "__main__":
    # Example usage
    example_texts = [
        "This is a sample document about artificial intelligence.",
        "Here's another document about machine learning.",
        "This document discusses natural language processing."
    ]
    
    # Create vector store
    vector_store = create_vector_store(example_texts)
    
    # Example query
    query = "What is AI?"
    results = query_vector_store(vector_store, query)
    
    print("\nQuery Results:")
    for i, doc in enumerate(results):
        print(f"\nDocument {i+1}:")
        print(doc.page_content) 