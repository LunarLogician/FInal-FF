import React, { useState } from 'react';
import { API_CONFIG } from '../config/api';

export default function FileUpload() {
    const [message, setMessage] = useState('');
    const [error, setError] = useState('');

    const handleFileUpload = async (file: File) => {
        try {
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

            // Then upload to Pinecone using the rewritten URL
            const pineconeResponse = await fetch(`${API_CONFIG.MAIN_API_URL}${API_CONFIG.ENDPOINTS.PINECONE.UPLOAD}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: uploadResult.text }),
            });

            if (!pineconeResponse.ok) {
                throw new Error('Pinecone upload failed');
            }

            const pineconeResult = await pineconeResponse.json();
            console.log('Pinecone upload successful:', pineconeResult);

            setMessage('File uploaded and processed successfully');
            setError('');
        } catch (err) {
            console.error('Operation failed:', err);
            setError(err instanceof Error ? err.message : 'Upload failed');
            setMessage('');
        }
    };

    return (
        <div className="w-full max-w-xl mx-auto p-4">
            <div className="mb-4">
                <input
                    type="file"
                    onChange={(e) => e.target.files?.[0] && handleFileUpload(e.target.files[0])}
                    accept=".pdf,.docx,.txt"
                    className="block w-full text-sm text-gray-500
                        file:mr-4 file:py-2 file:px-4
                        file:rounded-full file:border-0
                        file:text-sm file:font-semibold
                        file:bg-blue-50 file:text-blue-700
                        hover:file:bg-blue-100"
                />
            </div>
            {message && <p className="text-green-600 mt-2">{message}</p>}
            {error && <p className="text-red-600 mt-2">{error}</p>}
        </div>
    );
} 