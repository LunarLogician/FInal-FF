import { useState, useEffect } from 'react';
import { API_CONFIG } from '../config/api';

interface AISummaryProps {
  text: string;
}

export default function AISummary({ text }: AISummaryProps) {
  const [summary, setSummary] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const generateSummary = async () => {
      setIsLoading(true);
      try {
        const response = await fetch(`${API_CONFIG.MAIN_API_URL}${API_CONFIG.ENDPOINTS.SUMMARY}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text }),
        });

        if (!response.ok) {
          throw new Error('Failed to generate summary');
        }

        const data = await response.json();
        setSummary(data.summary || []);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to generate summary');
      } finally {
        setIsLoading(false);
      }
    };

    if (text) {
      generateSummary();
    }
  }, [text]);

  if (isLoading) {
    return <div>Generating summary...</div>;
  }

  if (error) {
    return <div>Error: {error}</div>;
  }

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-medium">AI Summary</h3>
      {summary.map((item, index) => (
        <p key={index} className="text-sm text-gray-600">
          {item}
        </p>
      ))}
    </div>
  );
} 