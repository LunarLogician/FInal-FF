import { NextRequest, NextResponse } from 'next/server';
import { OpenAI } from 'openai';

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

interface ScoreData {
  [key: string]: number;
}

export async function POST(req: NextRequest) {
  try {
    const { docText, docName, consistencyResult, analysisResult, esgResult } = await req.json();

    const response = await openai.chat.completions.create({
      model: "gpt-4-turbo-preview",
      response_format: { type: "json_object" },
      messages: [
        {
          role: "system",
          content: `You are an expert assistant generating brief explanations for scores from ESG, Reporting Quality, and Cheap Talk analysis of a company report. Keep each explanation short (2–3 sentences). Return a separate explanation for:
          - Consistency Score
          - Variability Score
          - Readability Score
          - Clarity Score
          - Commitment
          - Specificity
          - Cheap Talk Score
          - Safe Talk Score
          - ESG categories: Business Ethics & Values, Climate Change, Community Relations, Corporate Governance, Human Capital, Natural Capital, Non-ESG, Pollution & Waste, Product Liability

          Return the result as a JSON object like:
          {
            "reporting_quality": {
              "consistency": "...",
              "variability": "...", 
              "readability": "...",
              "clarity": "..."
            },
            "cheap_talk": {
              "commitment": "...",
              "specificity": "...",
              "cheap_talk_score": "...",
              "safe_talk_score": "..."
            },
            "esg": {
              "Business Ethics & Values": "...",
              "Climate Change": "...",
              "Community Relations": "...",
              "Corporate Governance": "...",
              "Human Capital": "...",
              "Natural Capital": "...",
              "Non-ESG": "...",
              "Pollution & Waste": "...",
              "Product Liability": "..."
            }
          }
          `

        },
        {
          role: "user",
          content: `
Analyze the following document and scores.

Document Name: ${docName}
Document Snippet: ${docText.slice(0, 1500)}


Scores:
- Consistency: ${consistencyResult?.consistency_score !== undefined ? (consistencyResult.consistency_score * 100).toFixed(1) + "%" : "N/A"}
- Variability: ${consistencyResult?.consistency_variability !== undefined ? (consistencyResult.consistency_variability * 100).toFixed(1) + "%" : "N/A"}
- Readability: ${consistencyResult?.readability_score !== undefined ? (consistencyResult.readability_score * 100).toFixed(1) + "%" : "N/A"}
- Clarity: ${consistencyResult?.clarity_score !== undefined ? (consistencyResult.clarity_score * 100).toFixed(1) + "%" : "N/A"}
- Commitment: ${analysisResult?.commitment_probability !== undefined ? (analysisResult.commitment_probability * 100).toFixed(1) + "%" : "N/A"}
- Specificity: ${analysisResult?.specificity_probability !== undefined ? (analysisResult.specificity_probability * 100).toFixed(1) + "%" : "N/A"}
- Cheap Talk: ${analysisResult?.cheap_talk_probability !== undefined ? (analysisResult.cheap_talk_probability * 100).toFixed(1) + "%" : "N/A"}
- Safe Talk: ${analysisResult?.safe_talk_probability !== undefined ? (analysisResult.safe_talk_probability * 100).toFixed(1) + "%" : "N/A"}
- ESG: ${esgResult ? JSON.stringify(
    Object.fromEntries(
      Object.entries(esgResult as ScoreData).map(([k, v]) => [k, `${(v * 100).toFixed(1)}%`])
    )
  ) : "N/A"}


Please return a JSON object like:
{
  "reporting_quality": "Explanation...",
  "cheap_talk": {
    "commitment": "Explanation...",
    "specificity": "Explanation...",
    "cheap_talk_score": "Explanation...",
    "safe_talk_score": "Explanation..."
  }
  "esg": "Explanation..."
}
        `
        }
      ]
    });

    const raw = response.choices[0]?.message?.content || '{}';

    try {
      const parsed = JSON.parse(raw);
      return NextResponse.json(parsed);
    } catch (parseErr) {
      console.error("Failed to parse OpenAI JSON:", raw);
      return NextResponse.json({ error: "Failed to parse OpenAI response" }, { status: 500 });
    }
  } catch (err) {
    console.error("Score insight generation failed:", err);
    return NextResponse.json({ error: "Score insight generation failed" }, { status: 500 });
  }
}