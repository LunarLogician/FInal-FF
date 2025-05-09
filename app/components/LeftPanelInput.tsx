import React, { useState, useRef } from "react";
import { Upload, ClipboardPaste } from "lucide-react";
import { API_CONFIG } from '../config/api';

export default function LeftPanelInput({
  onUploadComplete,
  embedAndUploadToPinecone,
  onSummaryGenerated, // ✅ new prop
}: {
  onUploadComplete: (fileName: string, text: string) => void;
  embedAndUploadToPinecone: (text: string, namespace: string) => void;
  onSummaryGenerated?: (summary: string[]) => void; // ✅ optional
}) {
  const [textInput, setTextInput] = useState("");
  const [error, setError] = useState("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [message, setMessage] = useState('');

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleFileSelect = (file: File | null) => {
    if (!file) return;
    const ext = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
    const allowed = [".pdf", ".docx"];
    if (!allowed.includes(ext)) {
      setError("Only PDF and DOCX files are supported.");
      return;
    }
    if (file.size > 25 * 1024 * 1024) {
      setError("File is too large. Max 25MB.");
      return;
    }    
    setSelectedFile(file);
    setError("");
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] || null;
    handleFileSelect(file);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    handleFileSelect(file);
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleChooseClick = () => {
    fileInputRef.current?.click();
  };

  const generateSummary = async (text: string) => {
    try {
      const res = await fetch(`${API_CONFIG.MAIN_API_URL}${API_CONFIG.ENDPOINTS.SUMMARY}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ docText: text }),
      });
      const json = await res.json();
      if (Array.isArray(json.summary) && onSummaryGenerated) {
        onSummaryGenerated(json.summary);
      }
    } catch (err) {
      console.error("Summary generation failed", err);
    }
  };
  
  

  const handleFileUpload = async (file: File) => {
    try {
      setIsUploading(true);
      setError('');

      const formData = new FormData();
      formData.append('file', file);
      
      // First upload the file to get the text content
      const uploadResponse = await fetch(`${API_CONFIG.MAIN_API_URL}${API_CONFIG.ENDPOINTS.UPLOAD}`, {
        method: 'POST',
        body: formData,
      });
      
      if (!uploadResponse.ok) {
        throw new Error('File upload failed');
      }

      const uploadResult = await uploadResponse.json();
      console.log('File uploaded successfully:', uploadResult);
      
      if (!uploadResult.text) {
        throw new Error('No text content extracted from file');
      }

      // Then upload to Pinecone using the rewritten URL
      const pineconeResponse = await fetch(`${API_CONFIG.MAIN_API_URL}${API_CONFIG.ENDPOINTS.PINECONE.UPLOAD}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          text: uploadResult.text,
          namespace: 'default'
        }),
      });

      if (!pineconeResponse.ok) {
        throw new Error('Pinecone upload failed');
      }

      const pineconeResult = await pineconeResponse.json();
      console.log('Pinecone upload successful:', pineconeResult);

      // Call onUploadComplete with the extracted text
      onUploadComplete(uploadResult.filename, uploadResult.text);

      // Analyze the text using the correct endpoint
      const analyzeResponse = await fetch(`${API_CONFIG.MAIN_API_URL}${API_CONFIG.ENDPOINTS.ANALYZE}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: uploadResult.text }),
      });

      if (!analyzeResponse.ok) {
        throw new Error('Analysis failed');
      }

      const analyzeResult = await analyzeResponse.json();
      console.log('Analysis successful:', analyzeResult);

      setMessage('File uploaded and processed successfully');
      setError('');

      return uploadResult;
    } catch (err) {
      console.error('Operation failed:', err);
      setError(err instanceof Error ? err.message : 'Upload failed');
      setMessage('');
      return null;
    } finally {
      setIsUploading(false);
    }
  };

  const handlePaste = async () => {
    if (!textInput.trim()) {
      setError("Please paste some text before submitting.");
      return;
    }
    onUploadComplete("Pasted Text", textInput);
    embedAndUploadToPinecone(textInput, "pasted-text");
    await generateSummary(textInput); // ✅ summarize
    setError("");
  };

  return (
    <div className="space-y-4 w-full max-w-xl mx-auto px-3 sm:px-4 md:px-6 lg:px-8">
      {/* Upload Section */}
      <div className="p-3 sm:p-4 md:p-6 border rounded-xl bg-white shadow-sm space-y-3 sm:space-y-4">
        <div className="flex items-center gap-2 text-gray-700 text-sm sm:text-base">
          <Upload size={16} className="sm:w-5 sm:h-5" />
          <span>Upload PDF or DOCX</span>
        </div>
  
        <p className="text-xs sm:text-sm text-gray-500 mt-1">
          ⚠️ Only English-language documents are supported.
        </p>
  
        <input
          type="file"
          accept=".pdf,.docx"
          ref={fileInputRef}
          onChange={handleFileInputChange}
          className="hidden"
        />
  
        <div
          onClick={handleChooseClick}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          className={`border-2 border-dashed rounded-lg px-3 sm:px-4 py-6 sm:py-8 text-center text-sm cursor-pointer transition-all
            ${
              isDragging
                ? "border-blue-500 bg-blue-50"
                : "border-gray-300 hover:border-blue-500 hover:bg-blue-50"
            }`}
        >
          <div className="flex flex-col items-center gap-2">
            <Upload size={24} className="text-gray-400" />
            <div>
              <span className="text-blue-600 font-medium">Click to upload</span> or drag and drop
            </div>
            <p className="text-xs text-gray-500">
              PDF or DOCX up to 25MB
            </p>
          </div>
        </div>

        {selectedFile && (
          <div className="flex items-center justify-between p-2 bg-gray-50 rounded-lg">
            <span className="text-sm text-gray-600 truncate">{selectedFile.name}</span>
            <button
              onClick={() => handleFileUpload(selectedFile)}
              disabled={isUploading}
              className="px-3 py-1 text-sm text-white bg-blue-600 rounded hover:bg-blue-700 disabled:opacity-50"
            >
              {isUploading ? "Uploading..." : "Upload"}
            </button>
          </div>
        )}

        {error && (
          <p className="text-xs sm:text-sm text-red-600 px-3 sm:px-4">{error}</p>
        )}

        {message && <p className="text-green-600 mt-2">{message}</p>}
      </div>
    </div>
  );
}
