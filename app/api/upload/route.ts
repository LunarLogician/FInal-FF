import { NextRequest, NextResponse } from 'next/server';
import { writeFile, unlink } from 'fs/promises';
import { spawn } from 'child_process';
import path from 'path';
import { storeDocument } from '@/utils/documentStore';

export async function POST(req: NextRequest) {
  const data = await req.formData();
  const file = data.get('file') as File;

  if (!file) {
    return NextResponse.json(
      { error: 'No file uploaded' },
      { status: 400 }
    );
  }

  // Convert the uploaded file to a buffer
  const bytes = await file.arrayBuffer();

  // Validate allowed file extensions
  const ext = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
  const allowed = [".pdf", ".docx", ".txt"];
  if (!allowed.includes(ext)) {
    return NextResponse.json(
      { error: "Unsupported file type. Please upload a PDF, Word document, or text file (.pdf, .docx, .txt)" },
      { status: 400 }
    );
  }
  
  const buffer = Buffer.from(bytes);

  // Write to a temporary location
  const tempPath = `/tmp/${file.name}`;
  await writeFile(tempPath, buffer);

  try {
    // Determine the path to the Python script in the scripts folder
    const pyScript = path.join(process.cwd(), 'scripts', 'extract_text.py');

    // Spawn a child process to run the Python script with the temp file path as argument
    const proc = spawn('python3', [pyScript, tempPath]);

    let output = '';
    let errorOutput = '';

    proc.stdout.on('data', (data) => {
      output += data.toString();
    });

    proc.stderr.on('data', (err) => {
      errorOutput += err.toString();
      console.error('Python error:', err.toString());
    });

    // Wait until the Python script completes
    await new Promise((resolve, reject) => {
      proc.on('close', (code) => {
        if (code === 0) {
          resolve(code);
        } else {
          reject(new Error(`Python script failed with code ${code}: ${errorOutput}`));
        }
      });
    });

    if (!output.trim()) {
      throw new Error('No text content extracted from file');
    }

    // Store document in Pinecone
    try {
      await storeDocument(output, file.name);
      console.log('Successfully stored document:', file.name);
    } catch (error) {
      console.error('Error storing document:', error);
      return NextResponse.json(
        { error: 'Failed to process document for search' },
        { status: 500 }
      );
    }

    // Return success response
    return NextResponse.json({
      id: file.name, // Using filename as document ID
      name: file.name,
      text: output,
    });

  } catch (error) {
    console.error('Error processing file:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to process file' },
      { status: 500 }
    );
  } finally {
    // Clean up the temporary file
    try {
      await unlink(tempPath);
    } catch (error) {
      console.error('Error deleting temporary file:', error);
    }
  }
}
