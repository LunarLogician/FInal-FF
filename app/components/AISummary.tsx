import { useState, useEffect } from 'react';
import { API_CONFIG } from '../config/api';

interface AISummaryProps {
  text: string;
}

export default function AISummary({ text }: AISummaryProps) {
  const [summary, setSummary] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    const generateSummary = async () => {
      setIsLoading(true);
      try {
        const response = await fetch(`${API_CONFIG.SUMMARY_API_URL}${API_CONFIG.ENDPOINTS.SUMMARIZE}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text }),
        });
        const data = await response.json();
        setSummary(data.summary);
      } catch (error) {
        console.error('Error generating summary:', error);
      } finally {
        setIsLoading(false);
      }
    };

    if (text) {
      generateSummary();
    }
  }, [text]);

  if (!text) return null;

  return (
    <div className="bg-white p-4 rounded-xl border border-zinc-200 mb-4">
      <h3 className="text-sm font-medium mb-3">📝 AI Summary</h3>
      {isLoading ? (
        <div className="text-sm text-zinc-500">Generating summary...</div>
      ) : (
        <ul className="list-disc list-inside space-y-2 text-sm text-zinc-700">
          {summary.map((point, index) => (
            <li key={index}>{point}</li>
          ))}
        </ul>
      )}
    </div>
  );
} 