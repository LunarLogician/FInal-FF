import { NextResponse } from 'next/server';
import pinecone from '@/utils/pinecone';

// Simple regex patterns for commitment and specificity analysis
const COMMITMENT_PATTERNS = [
  /will|shall|must|commit|pledge|promise|ensure|guarantee|dedicated to|aim to|target|goal|by \d{4}/gi,
  /we are committed|we will|we shall|we must|we promise|we guarantee|we ensure/gi
];

const SPECIFICITY_PATTERNS = [
  /\d+%|\d+ percent|\d+ tonnes|\d+MW|\d+ megawatts|\d+ GW|\d+ gigawatts/gi,
  /specific|measurable|timebound|quantifiable|detailed|precise|exact|defined/gi,
  /by \d{4}|by 20\d{2}|in \d{4}|in 20\d{2}/gi
];

function analyzeText(text: string) {
  // Count commitment indicators
  const commitmentCount = COMMITMENT_PATTERNS.reduce((count, pattern) => {
    const matches = text.match(pattern) || [];
    return count + matches.length;
  }, 0);

  // Count specificity indicators
  const specificityCount = SPECIFICITY_PATTERNS.reduce((count, pattern) => {
    const matches = text.match(pattern) || [];
    return count + matches.length;
  }, 0);

  // Normalize scores to probabilities between 0 and 1
  // Using word count as a normalizing factor
  const wordCount = text.split(/\s+/).length;
  const commitmentProb = Math.min(commitmentCount / (wordCount * 0.05), 1);
  const specificityProb = Math.min(specificityCount / (wordCount * 0.05), 1);

  return {
    commitment: commitmentProb,
    specificity: specificityProb
  };
}

export async function POST(req: Request) {
  try {
    const { documentIds } = await req.json();
    
    if (!documentIds?.length) {
      return NextResponse.json({ error: 'No documents provided' }, { status: 400 });
    }

    console.log('Analyzing documents:', documentIds);

    // Create a filter to find documents matching any of the provided IDs
    // in any of the possible identifier fields
    const filter = {
      $or: [
        // Look for matches in filename
        documentIds.length > 0 ? { filename: { $in: documentIds } } : null,
        // Look for matches in report_title
        documentIds.length > 0 ? { report_title: { $in: documentIds } } : null,
        // Look for matches in pdf_name
        documentIds.length > 0 ? { pdf_name: { $in: documentIds } } : null
      ].filter(Boolean) // Remove null values
    };

    // Get document contents from Pinecone
    const index = pinecone.index("resonate-report-rag");
    const queryResponse = await index.query({
      vector: new Array(1536).fill(0), // Dummy vector to fetch all
      topK: 1000,
      includeMetadata: true,
      filter: filter
    });

    console.log(`Found ${queryResponse.matches?.length || 0} matches for analysis`);

    // Combine all document text
    const documentText = queryResponse.matches
      .filter(match => match.metadata?.text)
      .map(match => match.metadata?.text)
      .join('\n\n');

    if (!documentText) {
      return NextResponse.json({ error: 'No text found in selected documents' }, { status: 400 });
    }

    // Analyze the text
    const { commitment: commitmentProb, specificity: specificityProb } = analyzeText(documentText);

    // Calculate cheap talk score (high commitment, low specificity)
    const cheapTalkProb = commitmentProb * (1 - specificityProb);

    return NextResponse.json({
      commitment_probability: commitmentProb,
      specificity_probability: specificityProb,
      cheap_talk_probability: cheapTalkProb
    });
  } catch (error) {
    console.error('Error in analyze endpoint:', error);
    return NextResponse.json({ error: 'Failed to analyze documents' }, { status: 500 });
  }
}