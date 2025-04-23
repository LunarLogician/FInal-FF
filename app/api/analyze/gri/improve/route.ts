import { OpenAI } from 'openai';

export const runtime = 'nodejs';

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});


export async function POST(req: Request) {
  try {
    const { question } = await req.json();
    if (!question) {
      return new Response(JSON.stringify({ error: 'Missing question' }), { status: 400 });
    }

    const systemPrompt = `
You are a sustainability disclosure expert helping companies improve GRI alignment.

Given a GRI-aligned disclosure requirement (e.g. "Report Scope 1 greenhouse gas emissions"), provide a concise, actionable tip that helps the company meet or improve on this requirement in their ESG or annual report.

Your advice should:
- Use plain language.
- Be no more than 2–3 sentences.
- Be specific and practical.
- Focus on what information to include and how.

Respond ONLY with a JSON object like:
{ "advice": "..." }
    `.trim();

    const completion = await openai.chat.completions.create({
      model: 'gpt-4-turbo-preview',
      response_format: { type: 'json_object' },
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: question }
      ]
    });

    const raw = completion.choices[0]?.message?.content || '{}';

    try {
      const parsed = JSON.parse(raw);
      return new Response(JSON.stringify(parsed), { status: 200 });
    } catch (err) {
      console.error("❌ Failed to parse GPT JSON:", raw);
      return new Response(JSON.stringify({ error: "OpenAI response format error" }), { status: 500 });
    }

  } catch (err) {
    console.error("🔥 GRI improvement error:", err);
    return new Response(JSON.stringify({ error: "Internal server error" }), { status: 500 });
  }
}