import pinecone from '@/utils/pinecone';

export const dynamic = 'force-dynamic';

export async function GET() {
  try {
    console.log('Starting document fetch from Pinecone...');
    
    // List all available indexes to debug
    console.log('Listing all Pinecone indexes...');
    try {
      const indexes = await pinecone.listIndexes();
      console.log('Available Pinecone indexes:', indexes);
    } catch (indexListError) {
      console.error('Error listing Pinecone indexes:', indexListError);
    }
    
    console.log('Connecting to "resonate-report-rag" index...');
    const index = pinecone.index("resonate-report-rag");
    
    console.log('Querying Pinecone with dummy vector...');
    // Fetch all documents (up to 1000)
    const queryResponse = await index.query({
      vector: new Array(1536).fill(0), // Dummy vector to fetch all
      topK: 1000,
      includeMetadata: true
    });
    
    console.log(`Pinecone query returned ${queryResponse.matches?.length || 0} matches`);
    
    // Debug the first 3 matches to see their structure
    if (queryResponse.matches && queryResponse.matches.length > 0) {
      console.log('Sample matches (first 3):');
      queryResponse.matches.slice(0, 3).forEach((match, i) => {
        console.log(`Match ${i+1}:`, {
          id: match.id,
          score: match.score,
          metadata: match.metadata
        });
      });
    } else {
      console.log('No matches returned from Pinecone');
    }

    // Use a Map to track unique documents by document identifier
    const uniqueDocuments = new Map();
    
    if (queryResponse.matches) {
      queryResponse.matches.forEach(match => {
        // Look for document identifier in priority order: filename, report_title, pdf_name
        const docId = match.metadata?.filename || match.metadata?.report_title || match.metadata?.pdf_name;
        
        if (docId) {
          if (!uniqueDocuments.has(docId)) {
            // Create a display name that includes both report_title and pdf_name when available
            let displayName = docId;
            
            // If we're using report_title as the ID but also have pdf_name, append it
            if (match.metadata?.report_title === docId && match.metadata?.pdf_name) {
              displayName = `${docId} [${match.metadata.pdf_name}]`;
            }
            // If we're using pdf_name as the ID but also have report_title, prepend it
            else if (match.metadata?.pdf_name === docId && match.metadata?.report_title) {
              displayName = `${match.metadata.report_title} [${docId}]`;
            }
            
            uniqueDocuments.set(docId, {
              id: docId,
              name: displayName
            });
          }
        } else {
          console.log(`Match ${match.id} has no document identifier in metadata:`, match.metadata);
        }
      });
    }

    // Convert Map values to array
    const documents = Array.from(uniqueDocuments.values());
    console.log(`Found ${documents.length} unique documents:`, documents.map(d => d.name));

    return Response.json({ documents });
  } catch (error) {
    console.error('Error fetching documents:', error);
    console.error('Error details:', {
      name: error instanceof Error ? error.name : 'Unknown',
      message: error instanceof Error ? error.message : String(error),
      stack: error instanceof Error ? error.stack : 'No stack trace'
    });
    return Response.json({ 
      error: 'Failed to fetch documents', 
      details: error instanceof Error ? error.message : String(error) 
    }, { status: 500 });
  }
} 