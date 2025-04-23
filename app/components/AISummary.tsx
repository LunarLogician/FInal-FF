import React, { useState } from 'react';
import { API_CONFIG } from '../config/api';

const BACKEND_URL = 'https://final-ff.onrender.com';

export default function AISummary() {
    const [summary, setSummary] = useState('');
    const [error, setError] = useState('');

    const generateSummary = async (text: string) => {
        try {
            const response = await fetch(`${BACKEND_URL}${API_CONFIG.ENDPOINTS.SUMMARY}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text }),
            });

            if (!response.ok) {
                throw new Error('Summary generation failed');
            }

            const result = await response.json();
            setSummary(result.summary);
            setError('');
        } catch (err) {
            console.error('Operation failed:', err);
            setError(err instanceof Error ? err.message : 'Summary generation failed');
            setSummary('');
        }
    };

    return (
        <div className="w-full max-w-xl mx-auto p-4">
            {summary && <p className="text-green-600 mt-2">{summary}</p>}
            {error && <p className="text-red-600 mt-2">{error}</p>}
        </div>
    );
} 