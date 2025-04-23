import { NextRequest, NextResponse } from 'next/server';
import { spawn } from 'child_process';
import path from 'path';

export const runtime = 'nodejs';

export async function POST(req: NextRequest): Promise<NextResponse> {  // Specify return type here
  const { documentId } = await req.json();
  if (!documentId) {
    return NextResponse.json({ error: 'Missing documentId' }, { status: 400 });
  }

  return new Promise((resolve) => {
    // 1) Spawn Python
    const pythonProcess = spawn('python3', [
      path.join(process.cwd(), 'app/api/analyze/engine.py'),
      documentId,
      'gpt-4'   // or your model
    ]);

    let output = '';
    let errorOutput = '';

    // 2) Log Python’s stdout in real-time
    pythonProcess.stdout.on('data', (data) => {
      console.log('Python stdout:', data.toString());  // <-- LOG IT
      output += data.toString();
    });

    // 3) Log Python’s stderr in real-time
    pythonProcess.stderr.on('data', (data) => {
      console.error('Python stderr:', data.toString()); // <-- LOG IT
      errorOutput += data.toString();
    });

    // 4) On exit, handle success/failure
    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        // If script crashed or returned non-zero, log errorOutput & send 500
        console.error('Python error (code):', code);
        console.error('Python errorOutput:', errorOutput);
        resolve(
          NextResponse.json({ error: 'Python script failed', details: errorOutput }, { status: 500 })
        );
      } else {
        // Attempt to parse JSON from stdout
        try {
          const parsed = JSON.parse(output);
          resolve(NextResponse.json(parsed));
        } catch (err) {
          resolve(
            NextResponse.json({ error: 'Bad JSON from engine', raw: output }, { status: 500 })
          );
        }
      }
    });
  });
}
