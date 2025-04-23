import { Pinecone } from '@pinecone-database/pinecone';

if (!process.env.PINECONE_API_KEY) {
  throw new Error("PINECONE_API_KEY environment variable is not set. Please set it to your Pinecone API key.");
}

// Read environment variables with fallbacks
const apiKey = process.env.PINECONE_API_KEY as string;
console.log("Initializing Pinecone client");

// Initialize the Pinecone client
const pinecone = new Pinecone({
  apiKey,
});

export default pinecone; 