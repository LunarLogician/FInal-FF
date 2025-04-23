'use client';

import LeftPanelInput from "./components/LeftPanelInput";
import { useState, useRef, useEffect } from 'react';
import {
  Settings,
  Send,
  MoreVertical,
  HelpCircle,
  ArrowRight,
  Bookmark,
  BookmarkCheck,
  Trash2,
  X,
  Info,
  Check,
  Copy,
  Download,
  Loader2,
  Sparkles,
  ChevronDown,
  GripVertical
} from 'lucide-react';
import Image from 'next/image';
import ReactMarkdown from 'react-markdown';
import ComplianceDashboard from "./components/ComplianceDashboard";
import { API_CONFIG } from './config/api';


// ---------- Type Definitions ----------
type Message = {
  id: string;
  role: string;
  content: string;
  sources?: string[];
};

type SentimentAnalysis = {
  positive_score: number;
  negative_score: number;
  neutral_score: number;
  overall_sentiment: string;
};

type Note = {
  id: string;
  content: string;
  timestamp: number;
};

type CheapTalkAnalysis = {
  commitment_probability: number;
  specificity_probability: number;
  cheap_talk_probability: number;
  safe_talk_probability: number; // ✅ Add this
};

type ConsistencyResult = {
  consistency_score: number | null;
  consistency_variability: number | null;  // 👈 Add this
  readability_score: number | null;
  clarity_score: number | null;
};


// ---------- Main Component ----------
export default function Page() {
  // Initialize all state hooks at the top level
  const [messages, setMessages] = useState<Message[]>([]);
  const [notes, setNotes] = useState<Note[]>([]);
  const [selectedNote, setSelectedNote] = useState<Note | null>(null);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  // Analysis States
  // --- Analysis Results ---
  const [analysisResult, setAnalysisResult] = useState<CheapTalkAnalysis | null>(null);
  const [sentimentResult, setSentimentResult] = useState<SentimentAnalysis | null>(null);
  const [consistencyResult, setConsistencyResult] = useState<ConsistencyResult | null>(null);
  const [esgResult, setESGResult] = useState<{ [key: string]: number } | null>(null);
  const [tcfdComplianceResults, setTCFDComplianceResults] = useState<any[] | null>(null);
  const [isAnalyzingTCFD, setIsAnalyzingTCFD] = useState(false);
  const [isAnalyzingGRI, setIsAnalyzingGRI] = useState(false);
  const [griComplianceResults, setGRIComplianceResults] = useState<any[] | null>(null);
  const [csrdComplianceResults, setCSRDComplianceResults] = useState<any[] | null>(null);
  const [isAnalyzingCSRD, setIsAnalyzingCSRD] = useState(false);
  const [sasbComplianceResults, setSASBComplianceResults] = useState<any[] | null>(null);
  const [isAnalyzingSASB, setIsAnalyzingSASB] = useState(false);



  // --- Loading Flags ---
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isAnalyzingSentiment, setIsAnalyzingSentiment] = useState(false);
  const [isTransitionAnalyzing, setIsTransitionAnalyzing] = useState(false);
  const [isAnalyzingESG, setIsAnalyzingESG] = useState(false);
  const [isAnalyzingConsistency, setIsAnalyzingConsistency] = useState(false);

  // --- Cheap Talk "Why" Toggles ---
  const [showWhyCommitment, setShowWhyCommitment] = useState(false);
  const [showWhySpecificity, setShowWhySpecificity] = useState(false);
  const [showWhyCheapTalkScore, setShowWhyCheapTalkScore] = useState(false);
  const [showWhySafeTalk, setShowWhySafeTalk] = useState(false);

  // --- Reporting Quality "Why" Toggles ---
  const [showWhyConsistency, setShowWhyConsistency] = useState(false);
  const [showWhyVariability, setShowWhyVariability] = useState(false);
  const [showWhyReadability, setShowWhyReadability] = useState(false);
  const [showWhyClarity, setShowWhyClarity] = useState(false);

  // --- ESG "Why" Toggles ---
  const [showWhyESGCategory, setShowWhyESGCategory] = useState<{ [key: string]: boolean }>({});
  // Single Uploaded Doc (or pasted text)
  const [uploadedDoc, setUploadedDoc] = useState<{ name: string; text: string } | null>(null);

  // --- Compliance How to improve Toggles ---
  const [showImprovementAdvice, setShowImprovementAdvice] = useState<{ [id: string]: boolean }>({});
  const [tcfdAdvice, setTCFDAdvice] = useState<{ [id: string]: string }>({});
  const [csrdAdvice, setCSRDAdvice] = useState<{ [id: string]: string }>({});
  const [griAdvice, setGRIAdvice] = useState<{ [id: string]: string }>({});
  const [sasbAdvice, setSASBAdvice] = useState<{ [id: string]: string }>({});
  
  // --- Advice Compliance toggles ---
  const [tcfdDrafts, setTCFDDrafts] = useState<{ [id: string]: string }>({});
  const [draftLoading, setDraftLoading] = useState<{ [id: string]: boolean }>({});
  const [copiedDraftId, setCopiedDraftId] = useState<string | null>(null);

  const [griDrafts, setGRIDrafts] = useState<{ [id: string]: string }>({});
  const [griDraftLoading, setGRIDraftLoading] = useState<{ [id: string]: boolean }>({});

  const [csrdDrafts, setCSRDDrafts] = useState<{ [id: string]: string }>({});
  const [csrdDraftLoading, setCSRDDraftLoading] = useState<{ [id: string]: boolean }>({});

  const [sasbDrafts, setSASBDrafts] = useState<{ [id: string]: string }>({});
  const [sasbDraftLoading, setSASBDraftLoading] = useState<{ [id: string]: boolean }>({});

  /// --- Loading for Show how to improve ----
  const [adviceLoading, setAdviceLoading] = useState<{ [id: string]: boolean }>({});

  // 🧠 Summary section state
  const [summary, setSummary] = useState<string[]>([]);
  const [copiedSummary, setCopiedSummary] = useState(false);


  // --- Smart recommendations ---
  const [extraSuggestions, setExtraSuggestions] = useState<string[]>([]);


  // ✅ Add this:
  const [scoreInsights, setScoreInsights] = useState<{
    reporting_quality?: {
      consistency?: string;
      variability?: string; 
      readability?: string;
      clarity?: string;
    };
    cheap_talk?: {
      commitment?: string;
      specificity?: string;
      cheap_talk_score?: string;
      safe_talk_score?: string;
    };
    esg?: { [category: string]: string }; 
  } | null>(null);
  
  // ✅ Add immediately after ⬆️ this
  const [showOnboardingTip, setShowOnboardingTip] = useState(false);

  useEffect(() => {
    const hasShownTip = sessionStorage.getItem("onboarding-tip-shown");

    if (!hasShownTip && uploadedDoc && scoreInsights && Object.keys(scoreInsights).length > 0) {
      const timeout = setTimeout(() => {
        setShowOnboardingTip(true);
        sessionStorage.setItem("onboarding-tip-shown", "true");
      }, 1500);

      return () => clearTimeout(timeout);
    }
  }, [uploadedDoc, scoreInsights]);


  // Modal / Misc. States
  const [showHelpModal, setShowHelpModal] = useState(false);

  const [showSummaryModal, setShowSummaryModal] = useState(false);
  const [downloadingPDF, setDownloadingPDF] = useState(false);
  const [pdfError, setPdfError] = useState<string | null>(null);

  const copySummaryToClipboard = () => {
    const combinedText = summary.join("\n");
    navigator.clipboard.writeText(combinedText)
      .then(() => {
        setCopiedSummary(true);
        setTimeout(() => setCopiedSummary(false), 2000);
      })
      .catch(err => {
        console.error('Failed to copy text: ', err);
      });
  };

  const downloadSummaryAsPDF = async () => {
    setDownloadingPDF(true);
    try {
      const response = await fetch('/api/convert-to-pdf', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: summary.join("\n") }),
      });
      
      if (!response.ok) throw new Error('Failed to generate PDF');
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'AI-Summary.pdf';
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error('Error downloading PDF:', error);
      setPdfError(error instanceof Error ? error.message : 'Failed to download PDF');
    } finally {
      setDownloadingPDF(false);
    }
  };


  // ✅ Add this below state hooks
  const [isNavOpen, setIsNavOpen] = useState(false);

  const BACKEND_URL = 'https://final-ff.onrender.com';

  const embedAndUploadToPinecone = async (text: string, name?: string) => {
    try {
      const response = await fetch(`${BACKEND_URL}${API_CONFIG.ENDPOINTS.PINECONE.UPLOAD}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text, name }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(`Failed to upload to Pinecone: ${errorData.message || 'Unknown error'}`);
      }

      const data = await response.json();
      console.log('Successfully uploaded to Pinecone:', data);
      return data.vector_id;
    } catch (error) {
      console.error('Error uploading to Pinecone:', error);
      throw error;
    }
  };


  // Scroll reference for chat messages
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Chat Submit Handler
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;
    const newMessage = { id: Math.random().toString(36).substring(7), role: 'user', content: input };
    setMessages((prev) => [...prev, newMessage]);
    setInput('');
    setIsLoading(true);
    setError(null);
    const assistantMessageId = Math.random().toString(36).substring(7);
    setMessages((prev) => [...prev, { id: assistantMessageId, role: 'assistant', content: '' }]);
    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: [...messages, newMessage].map(({ role, content }) => ({ role, content })),
          docText: uploadedDoc?.text || '',
          docName: uploadedDoc?.name || '',
          consistencyResult,
          analysisResult,
          esgResult,
        })
      });
      const extraSuggestionsHeader = response.headers.get('X-Extra-Suggestions');
      if (extraSuggestionsHeader) {
        try {
          setExtraSuggestions(JSON.parse(extraSuggestionsHeader));
        } catch (e) {
          console.warn("Failed to parse extra suggestions:", e);
          setExtraSuggestions([]);
        }
      } else {
        setExtraSuggestions([]);
      }      
      if (!response.ok) throw new Error('Failed to get response');
      const sourceDocs = response.headers.get('X-Source-Documents');
      const sources = sourceDocs ? JSON.parse(sourceDocs) : [];
      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      if (!reader) throw new Error('No reader available');
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value);
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantMessageId ? { ...msg, content: msg.content + chunk, sources } : msg
          )
        );
      }
    } catch (err) {
      setError(err as Error);
      setMessages((prev) => prev.filter((msg) => msg.id !== assistantMessageId));
    } finally {
      setIsLoading(false);
    }
  };

  // Predefined Questions
  const predefinedQuestions = [
    "What are the company's main sustainability initiatives, and what progress is reported?",
    "Does the report disclose Scope 1, 2, and 3 greenhouse gas emissions? If so, what are the reported values?",
    "Is there an external assurance or audit statement included in the report?",
    "Does the report follow any ESG reporting standards (e.g., GRI, SASB, TCFD)?",
  ];
  
  const handlePredefinedQuestion = (question: string) => {
    if (isLoading) return;
    setInput(question);
    const fakeEvent = { preventDefault: () => {} } as React.FormEvent;
    handleSubmit(fakeEvent);
  };

  // Cheap Talk Analysis
  const handleAnalyze = async () => {
    try {
      setIsAnalyzing(true);
      const response = await fetch(`${BACKEND_URL}${API_CONFIG.ENDPOINTS.ANALYZE}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text: input }),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Failed to analyze text");
      }
      
      const data = await response.json();
      setAnalysisResult(data.analysis);
    } catch (err) {
      console.error("Analysis failed:", err);
      setError(err as Error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // ESG Analysis
  const handleESGAnalysis = async () => {
    if (!uploadedDoc) {
      setError(new Error('Please upload (or paste) a document to analyze'));
      return;
    }
    setIsAnalyzingESG(true); // START loading indicator for ESG
    try {
      const response = await fetch(`${BACKEND_URL}${API_CONFIG.ENDPOINTS.ESG}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: uploadedDoc.text }),
      });
      if (!response.ok) throw new Error('Failed to analyze document for ESG');
      const result = await response.json();
      setESGResult(result);
    } catch (err) {
      console.error('Error during ESG analysis:', err);
      setError(err as Error);
    } finally {
      setIsAnalyzingESG(false); // END loading indicator for ESG
    }
  };
  





  // Sentiment Analysis
  const handleSentimentAnalysis = async () => {
    if (!uploadedDoc) {
      setError(new Error('Please upload (or paste) a document to analyze'));
      return;
    }
    setIsAnalyzingSentiment(true);
    try {
      await new Promise((resolve) => setTimeout(resolve, 1500));
      const result: SentimentAnalysis = {
        positive_score: Math.random() * 0.6 + 0.2,
        negative_score: Math.random() * 0.4,
        neutral_score: Math.random() * 0.3 + 0.1,
        overall_sentiment: 'positive'
      };
      if (
        result.positive_score > result.negative_score &&
        result.positive_score > result.neutral_score
      ) {
        result.overall_sentiment = 'positive';
      } else if (
        result.negative_score > result.positive_score &&
        result.negative_score > result.neutral_score
      ) {
        result.overall_sentiment = 'negative';
      } else {
        result.overall_sentiment = 'neutral';
      }
      setSentimentResult(result);
    } catch (err) {
      console.error('Error analyzing sentiment:', err);
      setError(err as Error);
    } finally {
      setIsAnalyzingSentiment(false);
    }
  };

  // Transition Audit
  const handleTransitionAudit = async () => {
    if (!uploadedDoc) {
      setError(new Error('Please upload (or paste) a document to run the Transition Audit'));
      return;
    }
    setIsTransitionAnalyzing(true);
    try {
      const response = await fetch('/api/analyze/transition', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ docText: uploadedDoc.text })
      });
      if (!response.ok) throw new Error('Failed to run Transition Audit');
      const result = await response.json();
      console.log('Transition Audit Result:', result);
      if (result.excel_path) {
        window.open(`/api/analyze/download?file=${encodeURIComponent(result.excel_path)}`, '_blank');
      }
    } catch (err) {
      console.error('Error running Transition Audit:', err);
      setError(err as Error);
    } finally {
      setIsTransitionAnalyzing(false);
    }
  };

  const handleTCFDComplianceCheck = async () => {
    if (!uploadedDoc) {
      setError(new Error("Please upload a document to check TCFD compliance"));
      return;
    }
  
    setIsAnalyzingTCFD(true);
    try {
      const response = await fetch(`${BACKEND_URL}/api/analyze/tcfd`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ docText: uploadedDoc.text }),
      });
  
      if (!response.ok) throw new Error("TCFD compliance check failed");
  
      const data = await response.json();
      setTCFDComplianceResults(data.results);
    } catch (error) {
      console.error("TCFD check error:", error);
      setError(error as Error);
    } finally {
      setIsAnalyzingTCFD(false);
    }
  };
  

  const handleGRIComplianceCheck = async () => {
    if (!uploadedDoc) return;
  
    setIsAnalyzingGRI(true);
    try {
      const res = await fetch(`${BACKEND_URL}${API_CONFIG.ENDPOINTS.GRI}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ docText: uploadedDoc.text }),
      });
  
      const data = await res.json();
      setGRIComplianceResults(data.results);
    } catch (err) {
      console.error("GRI check error:", err);
      setError(err as Error);
    } finally {
      setIsAnalyzingGRI(false);
    }
  };
  
  const handleCSRDComplianceCheck = async () => {
    if (!uploadedDoc) {
      setError(new Error("Please upload a document to check CSRD compliance"));
      return;
    }
  
    setIsAnalyzingCSRD(true);
    try {
      const response = await fetch(`${BACKEND_URL}${API_CONFIG.ENDPOINTS.CSRD}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ docText: uploadedDoc.text }),
      });
  
      if (!response.ok) throw new Error("CSRD compliance check failed");
  
      const data = await response.json();
      setCSRDComplianceResults(data.results);
    } catch (error) {
      console.error("CSRD check error:", error);
      setError(error as Error);
    } finally {
      setIsAnalyzingCSRD(false);
    }
  };
  
  const handleSASBComplianceCheck = async () => {
    if (!uploadedDoc) {
      setError(new Error("Please upload a document to check SASB compliance"));
      return;
    }
  
    setIsAnalyzingSASB(true);
    try {
      const response = await fetch(`${BACKEND_URL}${API_CONFIG.ENDPOINTS.SASB}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ docText: uploadedDoc.text }),
      });
  
      if (!response.ok) throw new Error("SASB compliance check failed");
  
      const data = await response.json();
      setSASBComplianceResults(data.results);
    } catch (error) {
      console.error("SASB check error:", error);
      setError(error as Error);
    } finally {
      setIsAnalyzingSASB(false);
    }
  };
  


  //consistency analysis
  const runConsistency = async () => {
    if (!uploadedDoc) return;
  
    setIsAnalyzingConsistency(true); // START
    try {
      const response = await fetch(`${BACKEND_URL}${API_CONFIG.ENDPOINTS.CONSISTENCY}`, {
        method: "POST",
        headers: { 
          "Content-Type": "application/json",
          "Accept": "application/json"
        },
        body: JSON.stringify({ chunks: [uploadedDoc.text] }),
      });
  
      if (!response.ok) throw new Error("Consistency analysis failed.");
  
      const data = await response.json();
      setConsistencyResult({
        consistency_score: data.consistency_score ?? null,
        consistency_variability: data.consistency_variability ?? null,
        readability_score: data.readability_score ?? null,
        clarity_score: data.clarity_score ?? null,
      });
    } catch (error) {
      console.error("Consistency error:", error);
      setConsistencyResult(null);
    } finally {
      setIsAnalyzingConsistency(false); // END
    }
  };
  
  
  

  // Bookmarking Messages
  const handleBookmark = (messageContent: string) => {
    const isAlreadySaved = notes.some((note) => note.content === messageContent);
    if (!isAlreadySaved) {
      setNotes((prev) => [
        ...prev,
        {
          id: Math.random().toString(36).substring(7),
          content: messageContent,
          timestamp: Date.now()
        }
      ]);
    }
  };

  const [rightColumnWidth, setRightColumnWidth] = useState(22); // Default width in rem
  const [isResizing, setIsResizing] = useState(false);
  const startX = useRef(0);
  const startWidth = useRef(0);

  const handleMouseDown = (e: React.MouseEvent) => {
    setIsResizing(true);
    startX.current = e.clientX;
    startWidth.current = rightColumnWidth;
    document.body.style.cursor = 'col-resize';
  };

  const handleMouseMove = (e: MouseEvent) => {
    if (!isResizing) return;
    
    const deltaX = e.clientX - startX.current;
    const deltaRem = deltaX / 16; // Convert pixels to rem (assuming 16px = 1rem)
    const newWidth = Math.max(16, Math.min(40, startWidth.current - deltaRem)); // Min 16rem, max 40rem
    
    setRightColumnWidth(newWidth);
  };

  const handleMouseUp = () => {
    setIsResizing(false);
    document.body.style.cursor = '';
  };

  useEffect(() => {
    if (isResizing) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
    }

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isResizing]);

  // ---------- Render ----------
  const [showAppOverviewModal, setShowAppOverviewModal] = useState(false);

  return (
    <div className="flex h-screen bg-[#f8f9fe] p-4">
      {/* LEFT SIDEBAR */}
      <div className="w-96 bg-white rounded-2xl shadow-sm mr-4 flex flex-col h-full">
        {/* Header (Logo, etc.) */}
        <div className="p-4 border-b border-zinc-100">
          <div className="flex items-center gap-2 mb-6">
            <Image src="/logo.png" alt="Resonate AI" width={100} height={100} />
            <span className="text-lg font-medium">Resonate AI</span>
          </div>
        </div>
        {/* Paste Text Input and Preview */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
        <LeftPanelInput
          onUploadComplete={async (name, text) => {
            setUploadedDoc({ name, text });
            setMessages([]);
            setAnalysisResult(null);
            setConsistencyResult(null);
            setESGResult(null);
            setSentimentResult(null);
            setSelectedNote(null);
            setError(null);
            setTCFDComplianceResults(null);
            setGRIComplianceResults(null);
            setCSRDComplianceResults(null);
            setSASBComplianceResults(null);
            setSummary([]); // Reset summary

            try {
              // Generate summary first
              const summaryRes = await fetch("/api/analyze/summary", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ docText: text }),
              });
              
              if (summaryRes.ok) {
                const summaryData = await summaryRes.json();
                if (Array.isArray(summaryData.summary)) {
                  setSummary(summaryData.summary);
                  // Show modal automatically
                  setShowSummaryModal(true);
                }
              }

              // Rest of the analysis
              // 1. Cheap Talk
              const cheapRes = await fetch(`${BACKEND_URL}${API_CONFIG.ENDPOINTS.ANALYZE}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text }),
              });
              const cheapData = await cheapRes.json();
              setAnalysisResult(cheapData.analysis);

              // 2. Consistency
              setIsAnalyzingConsistency(true);
              const consistencyRes = await fetch(`${BACKEND_URL}${API_CONFIG.ENDPOINTS.CONSISTENCY}`, {
                method: "POST",
                headers: { 
                  "Content-Type": "application/json",
                  "Accept": "application/json"
                },
                body: JSON.stringify({ chunks: [text] }),
              });
              const consistencyData = await consistencyRes.json();
              setConsistencyResult({
                consistency_score: consistencyData.consistency_score ?? null,
                consistency_variability: consistencyData.consistency_variability ?? null,
                readability_score: consistencyData.readability_score ?? null,
                clarity_score: consistencyData.clarity_score ?? null,
              });
              setIsAnalyzingConsistency(false);

              // 3. ESG
              setIsAnalyzingESG(true);
              const esgRes = await fetch(`${BACKEND_URL}${API_CONFIG.ENDPOINTS.ESG}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text }),
              });
              const esgData = await esgRes.json();
              setESGResult(esgData);
              setIsAnalyzingESG(false);

              // 4. Explanation
              const explanationRes = await fetch("/api/score-insights", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                  docText: text,
                  docName: name,
                  analysisResult: cheapData,
                  consistencyResult: consistencyData,
                  esgResult: esgData,
                }),
              });
              if (explanationRes.ok) {
                const explanationText = await explanationRes.text();
                if (explanationText) {
                  const explanation = JSON.parse(explanationText);
                  setScoreInsights(explanation);
                  setShowOnboardingTip(true);
                } else {
                  console.warn("No explanation returned from /api/score-insights");
                  setScoreInsights(null);
                }
              } else {
                throw new Error("Failed to fetch explanation");
              }

              // Call embedAndUploadToPinecone at the end
              await embedAndUploadToPinecone(text, name);

            } catch (err) {
              console.error("Upload error:", err);
              setError(err as Error);
            }
          }}  // <-- end of onUploadComplete callback

          onSummaryGenerated={(summary) => setSummary(summary)}
          embedAndUploadToPinecone={embedAndUploadToPinecone}
        />


          {uploadedDoc && (
            <div className="mt-4 flex flex-col" style={{ height: "50%" }}>
              <h3 className="text-sm font-semibold mb-2">{uploadedDoc.name}</h3>
              <div className="p-3 bg-zinc-50 border rounded-lg text-sm whitespace-pre-wrap overflow-y-auto h-full">
                {uploadedDoc.text ? uploadedDoc.text.slice(0, 5000) : 'No text content available'}
                {uploadedDoc.text && uploadedDoc.text.length > 5000 && '…'}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* MAIN CHAT AREA */}
      <div className="flex-1 bg-white rounded-2xl shadow-sm mx-4 flex flex-col min-w-0 max-w-[calc(100%-48rem)]">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-zinc-100">
          <h1 className="text-xl font-medium">Chat & Analysis</h1>
          <div className="flex items-center gap-2">
            {summary.length > 0 && (
              <button 
                onClick={() => setShowSummaryModal(true)}
                className="px-4 py-2 bg-yellow-100 hover:bg-yellow-200 rounded-lg relative group flex items-center gap-2"
              >
                <span className="text-sm font-medium text-yellow-800">AI Summary Generated</span>
                <Sparkles size={20} className="text-yellow-600" />
              </button>
            )}
            <button onClick={() => setShowAppOverviewModal(true)} className="p-2 hover:bg-zinc-50 rounded-full">
              <Info size={20} className="text-zinc-600" />
            </button>
          </div>
        </div>
        {/* Chat Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {showOnboardingTip && (
          <div className="p-3 mb-4 bg-blue-50 border border-blue-200 rounded-lg text-sm text-blue-800 shadow-sm">
            🧠 <strong>Heads up:</strong> I've analyzed this report for clarity, consistency, ESG focus, and commitment language. 
            Ask me anything — or try one of the <em>Quick Analysis</em> prompts!
            <button
              onClick={() => setShowOnboardingTip(false)}
              className="ml-4 text-xs text-blue-600 underline"
            >
              Dismiss
            </button>
          </div>
        )}
          {messages.map((message, idx) => (
            <div
              key={message.id}
              className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'} group`}
            >
              <div className={`relative max-w-[80%] ${message.role === 'user' ? 'ml-auto' : 'mr-auto'}`}>
                <div
                  className={`rounded-2xl p-3 ${
                    message.role === 'user'
                      ? 'bg-blue-200 text-blue-900'
                      : 'bg-[#f8f9fe] text-zinc-800'
                  }`}
                >
                  {message.content.length > 0 ? (
                    <div className="chat-content">
                      <ReactMarkdown>{message.content}</ReactMarkdown>
                    </div>
                  ) : (
                    isLoading &&
                    message.role === 'assistant' && (
                      <div className="flex space-x-2">
                        <div className="w-2 h-2 bg-zinc-400 rounded-full animate-bounce" />
                        <div className="w-2 h-2 bg-zinc-400 rounded-full animate-bounce [animation-delay:0.2s]" />
                        <div className="w-2 h-2 bg-zinc-400 rounded-full animate-bounce [animation-delay:0.4s]" />
                      </div>
                    )
                  )}

                  {/* Source reference */}
                  {message.role === 'assistant' && message.sources && message.sources.length > 0 && (
                    <div className="mt-2 text-xs text-zinc-500 border-t border-zinc-200 pt-2">
                      <span className="italic">
                        {message.sources.length === 1
                          ? `Source: ${message.sources[0]}`
                          : `Sources: ${Array.from(new Set(message.sources)).join(', ')}`}
                      </span>
                    </div>
                  )}
                </div>

                {/* ✅ Follow-up suggestions appear below last assistant message */}
                {message.role === 'assistant' && idx === messages.length - 1 && extraSuggestions.length > 0 && (
                  <div className="mt-2 space-y-1">
                    {extraSuggestions.map((suggestion, index) => (
                      <button
                        key={index}
                        onClick={() => handlePredefinedQuestion(suggestion)}
                        className="text-sm text-blue-600 underline hover:text-blue-800 transition"
                      >
                        👉 {suggestion}
                      </button>
                    ))}
                  </div>
                )}

                {/* Bookmark button */}
                {message.content && (
                  <button
                    onClick={() => handleBookmark(message.content)}
                    className={`absolute top-2 ${
                      message.role === 'user' ? 'left-2' : 'right-2'
                    } opacity-0 group-hover:opacity-100 transition-opacity p-1 rounded-full hover:bg-zinc-100`}
                  >
                    {notes.some((note) => note.content === message.content) ? (
                      <BookmarkCheck size={16} className="text-blue-600" />
                    ) : (
                      <Bookmark size={16} className="text-zinc-400" />
                    )}
                  </button>
                )}
              </div>
            </div>
          ))}

          <div ref={messagesEndRef} />
          {error && (
            <div className="flex justify-center">
              <div className="max-w-[80%] rounded-2xl p-3 bg-red-50 text-red-500 text-sm">
                Error: {error.message}
              </div>
            </div>
          )}
        </div>
        {/* Chat Input */}
        <div className="p-4 border-t border-zinc-100">
          <form onSubmit={handleSubmit} className="relative">
          <textarea
            className="w-full p-3 pr-12 rounded-xl border border-zinc-200 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 bg-[#f8f9fe] resize-none"
            value={input}
            placeholder="Type your message..."
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSubmit(e); // submit on Enter
              }
            }}
            rows={2}
            disabled={isLoading}
          />

            <button
              type="submit"
              className={`absolute right-3 top-1/2 -translate-y-1/2 p-2 rounded-full transition-colors ${
                isLoading ? 'text-zinc-400 hover:bg-transparent cursor-not-allowed' : 'text-blue-600 hover:bg-zinc-100'
              }`}
              disabled={isLoading}
            >
              <Send size={20} />
            </button>
          </form>
        </div>
      </div>

      <div className="hidden">
        bg-blue-500 bg-green-500 bg-yellow-500 bg-purple-500
        bg-red-500 bg-gray-400 bg-indigo-500 bg-emerald-500
        bg-orange-500 bg-teal-500
      </div>


      {/* RIGHT SIDEBAR - Tools & Insights */}
      <div 
        className="bg-white rounded-2xl shadow-sm ml-4 flex flex-col relative overflow-hidden"
        style={{ width: `${rightColumnWidth}rem` }}
      >
        {/* Resize handle */}
        <div
          className="absolute left-0 top-0 bottom-0 w-1 cursor-col-resize hover:bg-blue-100 active:bg-blue-200 transition-colors flex items-center justify-center"
          onMouseDown={handleMouseDown}
        >
          <GripVertical size={16} className="text-zinc-400" />
        </div>

        <div className="p-4 border-b border-zinc-100">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-medium">Tools & Insights</h2>
            <button onClick={() => setShowHelpModal(true)} className="p-2 hover:bg-zinc-50 rounded-full">
              <HelpCircle size={20} className="text-zinc-600" />
            </button>
          </div>
        </div>

        {/* Dropdown Navigation */}
        <div className="flex-1 overflow-y-auto">
          <div className="p-4 space-y-4">
            <div className="space-y-4">
              <h4 className="text-base font-semibold text-blue-700">Core Features</h4>
              <div className="grid grid-cols-2 gap-4">
                <div className="p-3 bg-blue-50 rounded-xl border border-blue-100">
                  <div className="flex items-center gap-2 mb-2">
                    <Send size={18} className="text-blue-600" />
                    <h5 className="font-medium text-sm">Intelligent Chat</h5>
                  </div>
                  <p className="text-xs text-zinc-600 pl-7">
                    Engage with our AI assistant to analyze documents, extract insights, and receive expert guidance.
                  </p>
                </div>
                <div className="p-3 bg-green-50 rounded-xl border border-green-100">
                  <div className="flex items-center gap-2 mb-2">
                    <ArrowRight size={18} className="text-green-600" />
                    <h5 className="font-medium text-sm">Document Analysis</h5>
                  </div>
                  <p className="text-xs text-zinc-600 pl-7">
                    Upload your sustainability reports for instant analysis across multiple dimensions.
                  </p>
                </div>
                <div className="p-3 bg-purple-50 rounded-xl border border-purple-100">
                  <div className="flex items-center gap-2 mb-2">
                    <ChevronDown size={18} className="text-purple-600" />
                    <h5 className="font-medium text-sm">Compliance Engine</h5>
                  </div>
                  <p className="text-xs text-zinc-600 pl-7">
                    Automatically evaluate your reports against TCFD, GRI, CSRD, and SASB frameworks.
                  </p>
                </div>
                <div className="p-3 bg-yellow-50 rounded-xl border border-yellow-100">
                  <div className="flex items-center gap-2 mb-2">
                    <Sparkles size={18} className="text-yellow-600" />
                    <h5 className="font-medium text-sm">AI Summary</h5>
                  </div>
                  <p className="text-xs text-zinc-600 pl-7">
                    Generate concise, exportable summaries with key insights for stakeholder presentations.
                  </p>
                </div>
              </div>
            </div>

            <div>
              <button 
                onClick={() => setIsNavOpen(!isNavOpen)}
                className="w-full flex items-center gap-2 p-2 hover:bg-zinc-50 rounded-lg transition-colors"
              >
                <ChevronDown size={16} className={`text-zinc-600 transition-transform ${isNavOpen ? 'rotate-180' : ''}`} />
                <span className="text-sm font-medium text-zinc-700">Jump to Section</span>
              </button>
              
              {isNavOpen && (
                <div className="mt-2 grid grid-cols-2 gap-1.5">
                  <a href="#quick-analysis" className="px-2 py-1.5 text-xs text-zinc-600 hover:bg-zinc-50 rounded-md transition-colors">Quick</a>
                  <a href="#reporting-quality" className="px-2 py-1.5 text-xs text-zinc-600 hover:bg-zinc-50 rounded-md transition-colors">Quality</a>
                  <a href="#cheap-talk" className="px-2 py-1.5 text-xs text-zinc-600 hover:bg-zinc-50 rounded-md transition-colors">Cheap Talk</a>
                  <a href="#esg-analysis" className="px-2 py-1.5 text-xs text-zinc-600 hover:bg-zinc-50 rounded-md transition-colors">ESG</a>
                  <a href="#compliance-analysis" className="px-2 py-1.5 text-xs text-zinc-600 hover:bg-zinc-50 rounded-md transition-colors">Compliance</a>
                  <a href="#transition-audit" className="px-2 py-1.5 text-xs text-zinc-600 hover:bg-zinc-50 rounded-md transition-colors">Audit</a>
                </div>
              )}
            </div>

            <details id="quick-analysis" open className="scroll-mt-16">
              <summary className="text-sm font-medium cursor-pointer mb-2">Quick Analysis</summary>
              <div className="p-4 rounded-xl border border-zinc-200 bg-[#f8f9fe]">
                <div className="space-y-2">
                  {predefinedQuestions.map((question, index) => (
                    <button
                      key={index}
                      onClick={() => handlePredefinedQuestion(question)}
                      disabled={isLoading}
                      className="w-full p-3 rounded-lg bg-white text-left text-sm text-zinc-600 hover:bg-zinc-50 flex items-center justify-between gap-2 border border-zinc-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      <span className="line-clamp-2">{question}</span>
                      <ArrowRight size={16} className="flex-shrink-0" />
                    </button>
                  ))}
                </div>
              </div>
            </details>


            <details id="reporting-quality" open className="scroll-mt-16">
              <summary className="text-sm font-medium cursor-pointer mb-2">Reporting Quality Analysis</summary>
              <div className="p-4 rounded-xl border border-zinc-200 bg-[#f8f9fe]">
                <div className="space-y-3">
                <button
                  onClick={runConsistency}
                  disabled={!uploadedDoc}
                  className="w-full p-3 rounded-lg bg-white text-left text-sm text-zinc-600 hover:bg-zinc-50 flex items-center justify-between gap-2 border border-zinc-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isAnalyzingConsistency ? (
                    <div className="flex items-center justify-between w-full">
                      <span>Analyzing...</span>
                      <div className="flex space-x-1">
                        <div className="w-2 h-2 bg-zinc-400 rounded-full animate-bounce" />
                        <div className="w-2 h-2 bg-zinc-400 rounded-full animate-bounce [animation-delay:0.2s]" />
                        <div className="w-2 h-2 bg-zinc-400 rounded-full animate-bounce [animation-delay:0.4s]" />
                      </div>
                    </div>
                  ) : (
                    <>
                      <span>{!uploadedDoc ? "Upload a document first" : "Analyze Uploaded Document"}</span>
                      <ArrowRight size={16} className="flex-shrink-0" />
                    </>
                  )}
                </button>
                  {consistencyResult && (
                    <div className="space-y-2">
                      {[
                        { label: "Consistency Score", score: consistencyResult.consistency_score, color: "bg-teal-500" },
                        { label: "Variability Score", score: consistencyResult.consistency_variability, color: "bg-purple-500" },
                        { label: "Comprehensibility Score", score: consistencyResult.readability_score, color: "bg-orange-500" },
                        { label: "Clarity Score", score: consistencyResult.clarity_score, color: "bg-blue-500" },
                      ].map(({ label, score, color }) => (
                        score !== null && score !== undefined && (
                          <div key={label} className="p-3 rounded-lg bg-white border border-zinc-200">
                            <div className="flex justify-between items-center mb-1">
                              <span className="text-xs text-zinc-500">{label}</span>
                              <span className="text-xs font-medium">{(score * 100).toFixed(2)}%</span>
                            </div>
                            <div className="h-2 bg-zinc-100 rounded-full overflow-hidden">
                              <div
                                className={`h-full ${color} rounded-full transition-all duration-500`}
                                style={{ width: `${(score * 100).toFixed(2)}%` }}
                              />
                            </div>

                            {label === "Consistency Score" && scoreInsights?.reporting_quality?.consistency && (
                              <>
                                <button
                                  className="text-xs text-blue-500 underline mt-1"
                                  onClick={() => setShowWhyConsistency((prev) => !prev)}
                                >
                                  Why?
                                </button>
                                {showWhyConsistency && (
                                  <p className="text-xs text-zinc-500 mt-1">{scoreInsights.reporting_quality.consistency}</p>
                                )}
                              </>
                            )}
                            {label === "Variability Score" && scoreInsights?.reporting_quality?.variability && (
                              <>
                                <button
                                  className="text-xs text-blue-500 underline mt-1"
                                  onClick={() => setShowWhyVariability((prev) => !prev)}
                                >
                                  Why?
                                </button>
                                {showWhyVariability && (
                                  <p className="text-xs text-zinc-500 mt-1">{scoreInsights.reporting_quality.variability}</p>
                                )}
                              </>
                            )}


                            {label === "Comprehensibility Score" && scoreInsights?.reporting_quality?.readability && (
                              <>
                                <button
                                  className="text-xs text-blue-500 underline mt-1"
                                  onClick={() => setShowWhyReadability((prev) => !prev)}
                                >
                                  Why?
                                </button>
                                {showWhyReadability && (
                                  <p className="text-xs text-zinc-500 mt-1">{scoreInsights.reporting_quality.readability}</p>
                                )}
                              </>
                            )}
                            {label === "Clarity Score" && scoreInsights?.reporting_quality?.clarity && (
                              <>
                                <button
                                  className="text-xs text-blue-500 underline mt-1"
                                  onClick={() => setShowWhyClarity((prev) => !prev)}
                                >
                                  Why?
                                </button>
                                {showWhyClarity && (
                                  <p className="text-xs text-zinc-500 mt-1">{scoreInsights.reporting_quality.clarity}</p>
                                )}
                              </>
                            )}
                          </div>
                        )
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </details>





          


            <details id="cheap-talk" open className="scroll-mt-16">
            <summary className="text-sm font-medium cursor-pointer mb-2">Cheap Talk Analysis</summary>

            <div className="p-4 rounded-xl border border-zinc-200 bg-[#f8f9fe]">
              <div className="space-y-3">
                <button
                  onClick={handleAnalyze}
                  disabled={isAnalyzing || !uploadedDoc}
                  className="w-full p-3 rounded-lg bg-white text-left text-sm text-zinc-600 hover:bg-zinc-50 flex items-center justify-between gap-2 border border-zinc-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <span>{!uploadedDoc ? 'Upload a document first' : 'Analyze Uploaded Document'}</span>
                  {isAnalyzing ? (
                    <div className="flex space-x-1">
                      <div className="w-2 h-2 bg-zinc-400 rounded-full animate-bounce" />
                      <div className="w-2 h-2 bg-zinc-400 rounded-full animate-bounce [animation-delay:0.2s]" />
                      <div className="w-2 h-2 bg-zinc-400 rounded-full animate-bounce [animation-delay:0.4s]" />
                    </div>
                  ) : (
                    <ArrowRight size={16} className="flex-shrink-0" />
                  )}
                </button>

                {analysisResult && (
                  <div className="space-y-2">
                    {[
                      { label: "Commitment", score: analysisResult.commitment_probability, color: "bg-blue-500", why: scoreInsights?.cheap_talk?.commitment, toggle: showWhyCommitment, setToggle: setShowWhyCommitment },
                      { label: "Specificity", score: analysisResult.specificity_probability, color: "bg-green-500", why: scoreInsights?.cheap_talk?.specificity, toggle: showWhySpecificity, setToggle: setShowWhySpecificity },
                      { label: "Cheap Talk Score", score: analysisResult.cheap_talk_probability, color: "bg-yellow-500", why: scoreInsights?.cheap_talk?.cheap_talk_score, toggle: showWhyCheapTalkScore, setToggle: setShowWhyCheapTalkScore },
                      { label: "Safe Talk Score", score: analysisResult.safe_talk_probability, color: "bg-purple-500", why: scoreInsights?.cheap_talk?.safe_talk_score, toggle: showWhySafeTalk, setToggle: setShowWhySafeTalk },
                    ].map(({ label, score, color, why, toggle, setToggle }) => (
                      <div key={label} className="p-3 rounded-lg bg-white border border-zinc-200">
                        <div className="flex justify-between items-center mb-1">
                          <span className="text-xs text-zinc-500">{label}</span>
                          <span className="text-xs font-medium">{(score * 100).toFixed(1)}%</span>
                        </div>
                        <div className="h-2 bg-zinc-100 rounded-full overflow-hidden">
                          <div
                            className={`h-full ${color} rounded-full transition-all duration-500`}
                            style={{ width: `${(score * 100).toFixed(1)}%` }}
                          />
                        </div>
                        {why && (
                          <>
                            <button
                              className="text-xs text-blue-500 underline mt-1"
                              onClick={() => setToggle((prev: boolean) => !prev)}
                            >
                              Why?
                            </button>
                            {toggle && (
                              <p className="text-xs text-zinc-500 mt-1">{why}</p>
                            )}
                          </>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </details>



          <details id="esg-analysis" open className="scroll-mt-16">      
            <summary className="text-sm font-medium cursor-pointer mb-2">ESG Analysis</summary>
            <div className="p-4 rounded-xl border border-zinc-200 bg-[#f8f9fe]">
              <div className="mb-3">
              </div>

              <div className="space-y-3">
                <button
                  onClick={handleESGAnalysis}
                  disabled={isAnalyzingESG || !uploadedDoc}
                  className="w-full p-3 rounded-lg bg-white text-left text-sm text-zinc-600 hover:bg-zinc-50 flex items-center justify-between gap-2 border border-zinc-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isAnalyzingESG ? (
                    <div className="flex items-center justify-between w-full">
                      <span>Analyzing...</span>
                      <div className="flex space-x-1">
                        <div className="w-2 h-2 bg-zinc-400 rounded-full animate-bounce" />
                        <div className="w-2 h-2 bg-zinc-400 rounded-full animate-bounce [animation-delay:0.2s]" />
                        <div className="w-2 h-2 bg-zinc-400 rounded-full animate-bounce [animation-delay:0.4s]" />
                      </div>
                    </div>
                  ) : (
                    <>
                      <span>{!uploadedDoc ? 'Upload a document first' : 'Analyze Uploaded Document'}</span>
                      <ArrowRight size={16} className="flex-shrink-0" />
                    </>
                  )}
                </button>



                {esgResult && (
                  <div className="space-y-2">
                    {Object.entries(esgResult).map(([category, score]) => (
                      <div key={category} className="p-3 rounded-lg bg-white border border-zinc-200">
                        <div className="flex justify-between items-center mb-1">
                          <span className="text-xs text-zinc-500">{category}</span>
                          <span className="text-xs font-medium">{(score * 100).toFixed(1)}%</span>
                        </div>
                        <div className="h-2 bg-zinc-100 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-indigo-500 rounded-full transition-all duration-500"
                            style={{ width: `${(score * 100).toFixed(1)}%` }}
                          />
                        </div>

                        {scoreInsights?.esg &&
                          typeof scoreInsights.esg === 'object' &&
                          scoreInsights.esg[category] && (
                            <>
                              <button
                                className="text-xs text-blue-500 underline mt-2"
                                onClick={() =>
                                  setShowWhyESGCategory((prev) => ({
                                    ...prev,
                                    [category]: !prev[category],
                                  }))
                                }
                              >
                                Why?
                              </button>
                              {showWhyESGCategory[category] && (
                                <p className="text-xs text-zinc-500 mt-1">
                                  {scoreInsights.esg[category]}
                                </p>
                              )}
                            </>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </details>

          {/* 📊 Compliance Dashboard */}
          {(tcfdComplianceResults || griComplianceResults) && (
            <section id="compliance-dashboard" className="scroll-mt-16">
            <div className="p-4 rounded-xl border border-zinc-200 bg-[#f8f9fe]">
              <ComplianceDashboard
                frameworks={[
                  ...(tcfdComplianceResults ? [{ name: "TCFD", results: tcfdComplianceResults }] : []),
                  ...(griComplianceResults ? [{ name: "GRI", results: griComplianceResults }] : []),
                  ...(csrdComplianceResults ? [{ name: "CSRD", results: csrdComplianceResults }] : []), 
                  ...(sasbComplianceResults ? [{ name: "SASB", results: sasbComplianceResults }] : []), 
                ]}
              />
            </div>
          </section>
          )}



          <details id="compliance-analysis" open className="scroll-mt-16">
          <summary className="text-sm font-medium cursor-pointer mb-2">Compliance Analysis</summary>
          <div className="p-4 rounded-xl border border-zinc-200 bg-[#f8f9fe]">
            <div className="space-y-3">
              {/* TCFD Button */}
              <button
                onClick={handleTCFDComplianceCheck}
                disabled={!uploadedDoc || isAnalyzingTCFD}
                className="w-full p-3 rounded-lg bg-white text-left text-sm text-zinc-600 hover:bg-zinc-50 flex items-center justify-between gap-2 border border-zinc-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <span>{!uploadedDoc ? "Upload a document first" : "Check TCFD Compliance"}</span>
                {isAnalyzingTCFD ? (
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-zinc-400 rounded-full animate-bounce" />
                    <div className="w-2 h-2 bg-zinc-400 rounded-full animate-bounce [animation-delay:0.2s]" />
                    <div className="w-2 h-2 bg-zinc-400 rounded-full animate-bounce [animation-delay:0.4s]" />
                  </div>
                ) : (
                  <ArrowRight size={16} className="flex-shrink-0" />
                )}
              </button>

              {/* TCFD Results */}
              {tcfdComplianceResults && (
                <div className="space-y-2">
                  {tcfdComplianceResults.map((item) => (
                    <div
                      key={item.id}
                      className={`p-3 rounded-lg border text-sm ${
                        item.status.includes("✅")
                          ? 'bg-green-50 border-green-200 text-green-800'
                          : item.status.includes("⚠️")
                          ? 'bg-yellow-50 border-yellow-200 text-yellow-800'
                          : 'bg-red-50 border-red-200 text-red-800'
                      }`}
                    >
                      <p><strong>{item.section}:</strong> {item.question}</p>
                      <p>
                        {item.status} (Match Score: <strong>{item.matchScore ?? Math.round(item.similarity * 100)}%</strong>) –{" "}
                        {
                          item.status === '✅ Likely Met'
                            ? 'This report covers this topic well.'
                            : item.status === '⚠️ Unclear'
                            ? 'This report partially addresses this topic.'
                            : 'This topic is likely missing from the report.'
                        }
                      </p>


                      {(item.status.includes("⚠️") || item.status.includes("❌")) && (
                        <>
                          <button
                            onClick={async () => {
                              if (!showImprovementAdvice[item.id]) {
                                setAdviceLoading((prev) => ({ ...prev, [item.id]: true })); // Start loading
                                try {
                                  const res = await fetch("/api/analyze/tcfd/improve", {
                                    method: "POST",
                                    headers: { "Content-Type": "application/json" },
                                    body: JSON.stringify({ question: item.question })
                                  });
                                  const data = await res.json();
                                  setTCFDAdvice((prev) => ({ ...prev, [item.id]: data.advice }));
                                } catch (err) {
                                  console.error("Advice fetch failed:", err);
                                } finally {
                                  setAdviceLoading((prev) => ({ ...prev, [item.id]: false })); // Stop loading
                                }
                              }

                              setShowImprovementAdvice((prev) => ({
                                ...prev,
                                [item.id]: !prev[item.id]
                              }));
                            }}
                            className="text-xs text-blue-500 underline mt-2"
                          >
                            {adviceLoading[item.id]
                              ? "Processing..."
                              : showImprovementAdvice[item.id]
                              ? "Hide Advice"
                              : "Show how to improve"}
                          </button>

                          {showImprovementAdvice[item.id] && tcfdAdvice[item.id] && (
                            <>
                              <p className="text-xs text-zinc-600 mt-2">{tcfdAdvice[item.id]}</p>
                              
                              <button
                                onClick={async () => {
                                  if (!tcfdDrafts[item.id]) {
                                    setDraftLoading((prev) => ({ ...prev, [item.id]: true }));
                                    try {
                                      const res = await fetch("/api/analyze/tcfd/draft", {
                                        method: "POST",
                                        headers: { "Content-Type": "application/json" },
                                        body: JSON.stringify({
                                          question: item.question,
                                          docText: uploadedDoc?.text || "", // 👈 Pass document text
                                        }),
                                      });
                                      const data = await res.json();
                                      setTCFDDrafts((prev) => ({ ...prev, [item.id]: data.draft }));
                                    } catch (err) {
                                      console.error("Draft fetch failed:", err);
                                    } finally {
                                      setDraftLoading((prev) => ({ ...prev, [item.id]: false }));
                                    }
                                  }
                                }}                                  
                                className="text-xs text-blue-600 underline mt-1"
                              >
                                {draftLoading[item.id] ? "Generating..." : "Generate Draft Text"}
                              </button>


                              {tcfdDrafts[item.id] && (
                                <div className="mt-3 p-3 bg-zinc-50 border border-zinc-200 rounded-lg">
                                  <div className="flex justify-between items-center mb-1">
                                    <span className="text-xs font-semibold text-zinc-600">📄 Report-style Draft</span>
                                    <button
                                      onClick={() => {
                                        const textToCopy = tcfdDrafts[item.id] || "";
                                        if (!textToCopy) return;

                                        // Fallback method using a temporary <textarea>
                                        const textarea = document.createElement("textarea");
                                        textarea.value = textToCopy;
                                        textarea.setAttribute("readonly", "");
                                        textarea.style.position = "absolute";
                                        textarea.style.left = "-9999px";
                                        document.body.appendChild(textarea);
                                        textarea.select();

                                        try {
                                          const successful = document.execCommand("copy");
                                          if (successful) {
                                            setCopiedDraftId(item.id);
                                            setTimeout(() => setCopiedDraftId(null), 2000);
                                          } else {
                                            console.warn("Fallback: Copy command was unsuccessful");
                                          }
                                        } catch (err) {
                                          console.error("Fallback: Unable to copy", err);
                                        }

                                        document.body.removeChild(textarea);
                                      }}
                                      className="text-xs text-blue-500 underline"
                                    >
                                      {copiedDraftId === item.id ? "Copied!" : "Copy to clipboard"}
                                    </button>


                                  </div>
                                  <div className="max-h-40 overflow-y-auto rounded-md border border-zinc-100 bg-white p-2">
                                    <p className="text-xs text-zinc-700 whitespace-pre-wrap">{tcfdDrafts[item.id]}</p>
                                  </div>
                                </div>
                              )}
                            </>
                          )}
                        </>
                      )}
                    </div>
                  ))}
                </div>
              )}


              {/* GRI Button */}
              <button
                onClick={handleGRIComplianceCheck}
                disabled={!uploadedDoc || isAnalyzingGRI}
                className="w-full p-3 rounded-lg bg-white text-left text-sm text-zinc-600 hover:bg-zinc-50 flex items-center justify-between gap-2 border border-zinc-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <span>{!uploadedDoc ? "Upload a document first" : "Check GRI Compliance"}</span>
                {isAnalyzingGRI ? (
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-zinc-400 rounded-full animate-bounce" />
                    <div className="w-2 h-2 bg-zinc-400 rounded-full animate-bounce [animation-delay:0.2s]" />
                    <div className="w-2 h-2 bg-zinc-400 rounded-full animate-bounce [animation-delay:0.4s]" />
                  </div>
                ) : (
                  <ArrowRight size={16} className="flex-shrink-0" />
                )}
              </button>

              {/* GRI Results */}
              {griComplianceResults && (
                <div className="space-y-2">
                  {griComplianceResults.map((item) => (
                    <div
                      key={item.id}
                      className={`p-3 rounded-lg border text-sm ${
                        item.status.includes("✅")
                          ? 'bg-green-50 border-green-200 text-green-800'
                          : item.status.includes("⚠️")
                          ? 'bg-yellow-50 border-yellow-200 text-yellow-800'
                          : 'bg-red-50 border-red-200 text-red-800'
                      }`}
                    >
                      <p><strong>{item.section}:</strong> {item.question}</p>
                      <p>
                        {item.status} (Match Score: <strong>{item.matchScore ?? Math.round(item.similarity * 100)}%</strong>) –{" "}
                        {
                          item.status === '✅ Likely Met'
                            ? 'This report covers this topic well.'
                            : item.status === '⚠️ Unclear'
                            ? 'This report partially addresses this topic.'
                            : 'This topic is likely missing from the report.'
                        }
                      </p>


                      {(item.status.includes("⚠️") || item.status.includes("❌")) && (
                        <>
                          <button
                            onClick={async () => {
                              if (!showImprovementAdvice[item.id]) {
                                setAdviceLoading((prev) => ({ ...prev, [item.id]: true })); // Start loading
                                try {
                                  const res = await fetch("/api/analyze/gri/improve", {
                                    method: "POST",
                                    headers: { "Content-Type": "application/json" },
                                    body: JSON.stringify({ question: item.question })
                                  });
                                  const data = await res.json();
                                  setGRIAdvice((prev) => ({ ...prev, [item.id]: data.advice }));
                                } catch (err) {
                                  console.error("GRI Advice fetch failed:", err);
                                } finally {
                                  setAdviceLoading((prev) => ({ ...prev, [item.id]: false })); // End loading
                                }
                              }

                              setShowImprovementAdvice((prev) => ({
                                ...prev,
                                [item.id]: !prev[item.id]
                              }));
                            }}
                            className="text-xs text-blue-500 underline mt-2"
                          >
                            {adviceLoading[item.id]
                              ? "Processing..."
                              : showImprovementAdvice[item.id]
                              ? "Hide Advice"
                              : "Show how to improve"}
                          </button>
                          {showImprovementAdvice[item.id] && griAdvice[item.id] && (
                            <>
                              <p className="text-xs text-zinc-600 mt-2">{griAdvice[item.id]}</p>

                              <button
                                onClick={async () => {
                                  if (!griDrafts[item.id]) {
                                    setDraftLoading((prev) => ({ ...prev, [item.id]: true }));
                                    try {
                                      const res = await fetch("/api/analyze/gri/draft", {
                                        method: "POST",
                                        headers: { "Content-Type": "application/json" },
                                        body: JSON.stringify({ question: item.question, docText: uploadedDoc?.text })
                                      });
                                      const data = await res.json();
                                      setGRIDrafts((prev) => ({ ...prev, [item.id]: data.draft }));
                                    } catch (err) {
                                      console.error("GRI Draft fetch failed:", err);
                                    } finally {
                                      setDraftLoading((prev) => ({ ...prev, [item.id]: false }));
                                    }
                                  }
                                }}
                                className="text-xs text-blue-600 underline mt-1"
                              >
                                {draftLoading[item.id] ? "Generating..." : "Generate Draft Text"}
                              </button>

                              {griDrafts[item.id] && (
                                <div className="mt-3 p-3 bg-zinc-50 border border-zinc-200 rounded-lg max-h-64 overflow-y-auto custom-scroll">
                                  <div className="flex justify-between items-center mb-1">
                                    <span className="text-xs font-semibold text-zinc-600">📄 Report-style Draft</span>
                                    <button
                                      onClick={() => {
                                        const textToCopy = griDrafts[item.id] || "";
                                        if (!textToCopy) return;

                                        const textarea = document.createElement("textarea");
                                        textarea.value = textToCopy;
                                        textarea.setAttribute("readonly", "");
                                        textarea.style.position = "absolute";
                                        textarea.style.left = "-9999px";
                                        document.body.appendChild(textarea);
                                        textarea.select();

                                        try {
                                          const successful = document.execCommand("copy");
                                          if (successful) {
                                            setCopiedDraftId(item.id);
                                            setTimeout(() => setCopiedDraftId(null), 2000);
                                          } else {
                                            console.warn("Fallback: Copy command was unsuccessful");
                                          }
                                        } catch (err) {
                                          console.error("Fallback: Unable to copy", err);
                                        }

                                        document.body.removeChild(textarea);
                                      }}
                                      className="text-xs text-blue-500 underline"
                                    >
                                      {copiedDraftId === item.id ? "Copied!" : "Copy to clipboard"}
                                    </button>

                                  </div>
                                  <p className="text-xs text-zinc-700 whitespace-pre-wrap">{griDrafts[item.id]}</p>
                                </div>
                              )}
                            </>
                          )}

                        </>
                      )}
                    </div>
                  ))}
                </div>
              )}


              {/* CSRD Button */}
              <button
                onClick={handleCSRDComplianceCheck}
                disabled={!uploadedDoc || isAnalyzingCSRD}
                className="w-full p-3 rounded-lg bg-white text-left text-sm text-zinc-600 hover:bg-zinc-50 flex items-center justify-between gap-2 border border-zinc-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <span>{!uploadedDoc ? "Upload a document first" : "Check CSRD Compliance"}</span>
                {isAnalyzingCSRD ? (
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-zinc-400 rounded-full animate-bounce" />
                    <div className="w-2 h-2 bg-zinc-400 rounded-full animate-bounce [animation-delay:0.2s]" />
                    <div className="w-2 h-2 bg-zinc-400 rounded-full animate-bounce [animation-delay:0.4s]" />
                  </div>
                ) : (
                  <ArrowRight size={16} className="flex-shrink-0" />
                )}
              </button>

              {/* CSRD Results */}
              {csrdComplianceResults && (
                <div className="space-y-2">
                  {csrdComplianceResults.map((item) => (
                    <div
                      key={item.id}
                      className={`p-3 rounded-lg border text-sm ${
                        item.status.includes("✅")
                          ? 'bg-green-50 border-green-200 text-green-800'
                          : item.status.includes("⚠️")
                          ? 'bg-yellow-50 border-yellow-200 text-yellow-800'
                          : 'bg-red-50 border-red-200 text-red-800'
                      }`}
                    >
                      <p><strong>{item.section}:</strong> {item.question}</p>
                      <p>
                        {item.status} (Match Score: <strong>{item.matchScore ?? Math.round(item.similarity * 100)}%</strong>) –{" "}
                        {
                          item.status === '✅ Likely Met'
                            ? 'This report covers this topic well.'
                            : item.status === '⚠️ Unclear'
                            ? 'This report partially addresses this topic.'
                            : 'This topic is likely missing from the report.'
                        }
                      </p>


                      {(item.status.includes("⚠️") || item.status.includes("❌")) && (
                        <>
                          <button
                            onClick={async () => {
                              if (!showImprovementAdvice[item.id]) {
                                setAdviceLoading((prev) => ({ ...prev, [item.id]: true })); // Start loading
                                try {
                                  const res = await fetch("/api/analyze/csrd/improve", {
                                    method: "POST",
                                    headers: { "Content-Type": "application/json" },
                                    body: JSON.stringify({ question: item.question })
                                  });
                                  const data = await res.json();
                                  setCSRDAdvice((prev) => ({ ...prev, [item.id]: data.advice }));
                                } catch (err) {
                                  console.error("CSRD Advice fetch failed:", err);
                                } finally {
                                  setAdviceLoading((prev) => ({ ...prev, [item.id]: false })); // End loading
                                }
                              }

                              setShowImprovementAdvice((prev) => ({
                                ...prev,
                                [item.id]: !prev[item.id]
                              }));
                            }}
                            className="text-xs text-blue-500 underline mt-2"
                          >
                            {adviceLoading[item.id]
                              ? "Processing..."
                              : showImprovementAdvice[item.id]
                              ? "Hide Advice"
                              : "Show how to improve"}
                          </button>

                          {showImprovementAdvice[item.id] && csrdAdvice[item.id] && (
                            <>
                              <p className="text-xs text-zinc-600 mt-2">{csrdAdvice[item.id]}</p>

                              <button
                                onClick={async () => {
                                  if (!csrdDrafts[item.id]) {
                                    setCSRDDraftLoading((prev) => ({ ...prev, [item.id]: true }));
                                    try {
                                      const res = await fetch("/api/analyze/csrd/draft", {
                                        method: "POST",
                                        headers: { "Content-Type": "application/json" },
                                        body: JSON.stringify({ question: item.question, docText: uploadedDoc?.text }),
                                      });
                                      const data = await res.json();
                                      setCSRDDrafts((prev) => ({ ...prev, [item.id]: data.draft }));
                                    } catch (err) {
                                      console.error("CSRD Draft fetch failed:", err);
                                    } finally {
                                      setCSRDDraftLoading((prev) => ({ ...prev, [item.id]: false }));
                                    }
                                  }
                                }}
                                className="text-xs text-blue-600 underline mt-1"
                              >
                                {csrdDraftLoading[item.id] ? "Generating..." : "Generate Draft Text"}
                              </button>

                              {csrdDrafts[item.id] && (
                                <div className="mt-3 p-3 bg-zinc-50 border border-zinc-200 rounded-lg max-h-64 overflow-y-auto custom-scroll">
                                  <div className="flex justify-between items-center mb-1">
                                    <span className="text-xs font-semibold text-zinc-600">📄 Report-style Draft</span>
                                    <button
                                      onClick={() => {
                                        const textToCopy = csrdDrafts[item.id] || "";
                                        if (!textToCopy) return;

                                        const textarea = document.createElement("textarea");
                                        textarea.value = textToCopy;
                                        textarea.setAttribute("readonly", "");
                                        textarea.style.position = "absolute";
                                        textarea.style.left = "-9999px";
                                        document.body.appendChild(textarea);
                                        textarea.select();

                                        try {
                                          const successful = document.execCommand("copy");
                                          if (successful) {
                                            setCopiedDraftId(item.id);
                                            setTimeout(() => setCopiedDraftId(null), 2000);
                                          } else {
                                            console.warn("Fallback: Copy command was unsuccessful");
                                          }
                                        } catch (err) {
                                          console.error("Fallback: Unable to copy", err);
                                        }

                                        document.body.removeChild(textarea);
                                      }}
                                      className="text-xs text-blue-500 underline"
                                    >
                                      {copiedDraftId === item.id ? "Copied!" : "Copy to clipboard"}
                                    </button>

                                  </div>
                                  <p className="text-xs text-zinc-700 whitespace-pre-wrap">{csrdDrafts[item.id]}</p>
                                </div>
                              )}
                            </>
                          )}

                        </>
                      )}
                    </div>
                  ))}
                </div>
              )}




              {/* SASB Button */}
              <button
                onClick={handleSASBComplianceCheck}
                disabled={!uploadedDoc || isAnalyzingSASB}
                className="w-full p-3 rounded-lg bg-white text-left text-sm text-zinc-600 hover:bg-zinc-50 flex items-center justify-between gap-2 border border-zinc-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <span>{!uploadedDoc ? "Upload a document first" : "Check SASB Compliance"}</span>
                {isAnalyzingSASB ? (
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-zinc-400 rounded-full animate-bounce" />
                    <div className="w-2 h-2 bg-zinc-400 rounded-full animate-bounce [animation-delay:0.2s]" />
                    <div className="w-2 h-2 bg-zinc-400 rounded-full animate-bounce [animation-delay:0.4s]" />
                  </div>
                ) : (
                  <ArrowRight size={16} className="flex-shrink-0" />
                )}
              </button>

              {/* SASB Results */}
              {sasbComplianceResults && (
                <div className="space-y-2">
                  {sasbComplianceResults.map((item) => (
                    <div
                      key={item.id}
                      className={`p-3 rounded-lg border text-sm ${
                        item.status.includes("✅")
                          ? 'bg-green-50 border-green-200 text-green-800'
                          : item.status.includes("⚠️")
                          ? 'bg-yellow-50 border-yellow-200 text-yellow-800'
                          : 'bg-red-50 border-red-200 text-red-800'
                      }`}
                    >
                      <div className="flex items-start justify-between gap-2">
                        <span className="font-medium">{item.section}</span>
                        <span>{item.status} ({item.matchScore ?? Math.round(item.similarity * 100)}%)</span>
                      </div>
                      <p className="text-xs mt-1">{item.question}</p>

                      {(item.status.includes("⚠️") || item.status.includes("❌")) && (
                        <>
                          <button
                            onClick={async () => {
                              if (!showImprovementAdvice[item.id]) {
                                setAdviceLoading((prev) => ({ ...prev, [item.id]: true })); // Start loading
                                try {
                                  const res = await fetch("/api/analyze/sasb/improve", {
                                    method: "POST",
                                    headers: { "Content-Type": "application/json" },
                                    body: JSON.stringify({ question: item.question })
                                  });
                                  const data = await res.json();
                                  setSASBAdvice((prev) => ({ ...prev, [item.id]: data.advice }));
                                } catch (err) {
                                  console.error("Advice fetch failed:", err);
                                } finally {
                                  setAdviceLoading((prev) => ({ ...prev, [item.id]: false })); // End loading
                                }
                              }

                              setShowImprovementAdvice((prev) => ({
                                ...prev,
                                [item.id]: !prev[item.id]
                              }));
                            }}
                            className="text-xs text-blue-500 underline mt-2"
                          >
                            {adviceLoading[item.id]
                              ? "Processing..."
                              : showImprovementAdvice[item.id]
                              ? "Hide Advice"
                              : "Show how to improve"}
                          </button>
                          {showImprovementAdvice[item.id] && sasbAdvice[item.id] && (
                            <>
                              <p className="text-xs text-zinc-600 mt-2">{sasbAdvice[item.id]}</p>

                              <button
                                onClick={async () => {
                                  if (!sasbDrafts[item.id]) {
                                    setSASBDraftLoading((prev) => ({ ...prev, [item.id]: true }));
                                    try {
                                      const res = await fetch("/api/analyze/sasb/draft", {
                                        method: "POST",
                                        headers: { "Content-Type": "application/json" },
                                        body: JSON.stringify({ question: item.question, docText: uploadedDoc?.text }),
                                      });
                                      const data = await res.json();
                                      setSASBDrafts((prev) => ({ ...prev, [item.id]: data.draft }));
                                    } catch (err) {
                                      console.error("SASB Draft fetch failed:", err);
                                    } finally {
                                      setSASBDraftLoading((prev) => ({ ...prev, [item.id]: false }));
                                    }
                                  }
                                }}
                                className="text-xs text-blue-600 underline mt-1"
                              >
                                {sasbDraftLoading[item.id] ? "Generating..." : "Generate Draft Text"}
                              </button>

                              {sasbDrafts[item.id] && (
                                <div className="mt-3 p-3 bg-zinc-50 border border-zinc-200 rounded-lg max-h-64 overflow-y-auto custom-scroll">
                                  <div className="flex justify-between items-center mb-1">
                                    <span className="text-xs font-semibold text-zinc-600">📄 Report-style Draft</span>
                                    <button
                                      onClick={() => {
                                        const textToCopy = sasbDrafts[item.id] || "";
                                        if (!textToCopy) return;

                                        const textarea = document.createElement("textarea");
                                        textarea.value = textToCopy;
                                        textarea.setAttribute("readonly", "");
                                        textarea.style.position = "absolute";
                                        textarea.style.left = "-9999px";
                                        document.body.appendChild(textarea);
                                        textarea.select();

                                        try {
                                          const successful = document.execCommand("copy");
                                          if (successful) {
                                            setCopiedDraftId(item.id);
                                            setTimeout(() => setCopiedDraftId(null), 2000);
                                          } else {
                                            console.warn("Fallback: Copy command was unsuccessful");
                                          }
                                        } catch (err) {
                                          console.error("Fallback: Unable to copy", err);
                                        }

                                        document.body.removeChild(textarea);
                                      }}
                                      className="text-xs text-blue-500 underline"
                                    >
                                      {copiedDraftId === item.id ? "Copied!" : "Copy to clipboard"}
                                    </button>
                                  </div>
                                  <p className="text-xs text-zinc-700 whitespace-pre-wrap">{sasbDrafts[item.id]}</p>
                                </div>
                              )}
                            </>
                          )}

                        </>
                      )}
                    </div>
                  ))}
                </div>
              )}


            </div>
          </div>
        </details>




        <details id="transition-audit" open className="scroll-mt-16">
        <summary className="text-sm font-medium text-zinc-400 cursor-not-allowed mb-2">🚧 Transition Audit (Coming Soon)</summary>
          <div className="p-4 rounded-xl border border-zinc-200 bg-[#f8f9fe]">
            <div className="space-y-3">
            <button
              disabled
              className="w-full p-3 rounded-lg bg-zinc-100 text-left text-sm text-zinc-400 flex items-center justify-between gap-2 border border-zinc-200 cursor-not-allowed"
            >
              <span>Coming soon</span>
              <ArrowRight size={16} className="flex-shrink-0" />
            </button>
            </div>
          </div>
        </details>



          <div className="p-4 rounded-xl border border-zinc-200 bg-[#f8f9fe]">
            <h3 className="text-sm font-medium mb-3">Notes</h3>
            {notes.length === 0 ? (
              <p className="text-sm text-zinc-600">
                Hover over any message and click the bookmark icon to save it as a note.
              </p>
            ) : (
              <div className="space-y-2">
                {notes.map((note) => (
                  <div
                    key={note.id}
                    className="p-3 rounded-lg bg-white border border-zinc-200 group cursor-pointer hover:border-blue-200 transition-colors"
                  >
                    <div className="flex justify-between items-start gap-2">
                      <p
                        onClick={() => setSelectedNote(note)}
                        className="text-sm text-zinc-600 line-clamp-3 flex-1"
                      >
                        {note.content.replace(/[#*_`]/g, '')}
                      </p>
                      <div className="flex items-center gap-1">
                        <button
                          onClick={async (e) => {
                            e.stopPropagation();
                            setDownloadingPDF(true);
                            setPdfError(null);
                            try {
                              const response = await fetch('/api/convert-to-pdf', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ text: note.content }),
                              });
                              
                              if (!response.ok) throw new Error('Failed to generate PDF');
                              
                              const blob = await response.blob();
                              const url = window.URL.createObjectURL(blob);
                              const a = document.createElement('a');
                              a.href = url;
                              a.download = `note-${new Date(note.timestamp).toISOString().split('T')[0]}.pdf`;
                              document.body.appendChild(a);
                              a.click();
                              window.URL.revokeObjectURL(url);
                              document.body.removeChild(a);
                            } catch (error) {
                              console.error('Error downloading PDF:', error);
                              setPdfError('Failed to generate PDF. Please try again later.');
                            } finally {
                              setDownloadingPDF(false);
                            }
                          }}
                          disabled={downloadingPDF}
                          className={`opacity-0 group-hover:opacity-100 transition-opacity p-1 rounded-full hover:bg-zinc-100 ${downloadingPDF ? 'cursor-not-allowed' : ''}`}
                        >
                          {downloadingPDF ? (
                            <Loader2 size={14} className="text-zinc-400 animate-spin" />
                          ) : (
                            <Download size={14} className="text-zinc-400" />
                          )}
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            setNotes((prev) => prev.filter((n) => n.id !== note.id));
                          }}
                          className="opacity-0 group-hover:opacity-100 transition-opacity p-1 rounded-full hover:bg-zinc-100"
                        >
                          <Trash2 size={14} className="text-zinc-400" />
                        </button>
                      </div>
                      {pdfError && (
                        <p className="text-xs text-red-500 mt-1">{pdfError}</p>
                      )}
                    </div>
                    <p className="text-xs text-zinc-400 mt-1">
                      {new Date(note.timestamp).toLocaleDateString()}{' '}
                      {new Date(note.timestamp).toLocaleTimeString()}
                    </p>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>

    {/* Note Modal */}
    {selectedNote && (
      <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
        <div className="bg-white rounded-2xl w-[600px] max-h-[80vh] flex flex-col">
          <div className="flex items-center justify-between p-4 border-b border-zinc-100">
            <h3 className="text-lg font-medium">Note</h3>
            <button onClick={() => setSelectedNote(null)} className="p-2 hover:bg-zinc-50 rounded-full">
              <X size={20} className="text-zinc-600" />
            </button>
          </div>
          <div className="p-6 overflow-y-auto">
            <div className="prose prose-sm max-w-none text-zinc-800 chat-content">
              <ReactMarkdown>{selectedNote.content}</ReactMarkdown>
            </div>
            <p className="text-sm text-zinc-400 mt-4">
              Saved on {new Date(selectedNote.timestamp).toLocaleDateString()} at{' '}
              {new Date(selectedNote.timestamp).toLocaleTimeString()}
            </p>
          </div>
        </div>
      </div>
    )}

    {/* Help Modal */}
    {showHelpModal && (
      <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
        <div className="bg-white rounded-2xl w-[600px] max-h-[80vh] flex flex-col">
          <div className="flex items-center justify-between p-4 border-b border-zinc-100">
            <h3 className="text-lg font-medium">Tools & Insights Help</h3>
            <button onClick={() => setShowHelpModal(false)} className="p-2 hover:bg-zinc-50 rounded-full">
              <X size={20} className="text-zinc-600" />
            </button>
          </div>
          <div className="p-6 overflow-y-auto">
          <div className="space-y-6">

            {/* Quick */}
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <div className="bg-blue-100 p-2 rounded-lg">
                  <ArrowRight size={20} className="text-blue-600" />
                </div>
                <h4 className="font-medium">Quick Analysis</h4>
              </div>
              <p className="text-zinc-600 text-sm pl-10">
                One-click prompts to help explore the uploaded ESG report quickly. Ideal for starting a conversation.
              </p>
            </div>

            {/* Cheap Talk */}
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <div className="bg-yellow-100 p-2 rounded-lg">
                  <Info size={20} className="text-yellow-600" />
                </div>
                <h4 className="font-medium">Cheap Talk Analysis</h4>
              </div>
              <p className="text-zinc-600 text-sm pl-10">
                Measures how specific and committed the report's language is. Combines both into a Cheap Talk Score and a Safe Talk Score.
              </p>
            </div>

            {/* Reporting Quality */}
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <div className="bg-teal-100 p-2 rounded-lg">
                  <Info size={20} className="text-teal-600" />
                </div>
                <h4 className="font-medium">Reporting Quality</h4>
              </div>
              <p className="text-zinc-600 text-sm pl-10">
                Analyzes four dimensions: Consistency, Variability, Comprehensibility, and Clarity — based on the structure and vocabulary of the report.
              </p>
            </div>

            {/* ESG */}
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <div className="bg-indigo-100 p-2 rounded-lg">
                  <Info size={20} className="text-indigo-600" />
                </div>
                <h4 className="font-medium">ESG Focus</h4>
              </div>
              <p className="text-zinc-600 text-sm pl-10">
                Evaluates how much of the uploaded report is devoted to each ESG topic: Business Ethics, Climate Change, Governance, Pollution, etc.
              </p>
            </div>

            {/* Compliance */}
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <div className="bg-pink-100 p-2 rounded-lg">
                  <Info size={20} className="text-pink-600" />
                </div>
                <h4 className="font-medium">Compliance Analysis</h4>
              </div>
              <p className="text-zinc-600 text-sm pl-10">
                Checks if your report aligns with TCFD, GRI, CSRD, or SASB requirements. For each unmet or unclear item, you get improvement advice and auto-generated draft text.
              </p>
            </div>

            {/* Dashboard */}
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <div className="bg-gray-100 p-2 rounded-lg">
                  <Info size={20} className="text-gray-600" />
                </div>
                <h4 className="font-medium">Compliance Dashboard</h4>
              </div>
              <p className="text-zinc-600 text-sm pl-10">
                Visual summary of your performance across all frameworks — helps identify strengths and blind spots.
              </p>
            </div>

            {/* Notes */}
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <div className="bg-purple-100 p-2 rounded-lg">
                  <Bookmark size={20} className="text-purple-600" />
                </div>
                <h4 className="font-medium">Notes</h4>
              </div>
              <p className="text-zinc-600 text-sm pl-10">
                Bookmark any assistant message for later reference. Notes are saved in the sidebar and include timestamps.
              </p>
            </div>
            </div>
          </div>
        </div>
      </div>
    )}

    {/* Summary Modal */}
    {showSummaryModal && (
      <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
        <div className="bg-white rounded-2xl w-[600px] max-h-[80vh] flex flex-col">
          <div className="flex items-center justify-between p-4 border-b border-zinc-100">
            <div className="flex items-center gap-2">
              <Sparkles size={20} className="text-yellow-600" />
              <h3 className="text-lg font-medium">AI Summary</h3>
            </div>
            <button onClick={() => setShowSummaryModal(false)} className="p-2 hover:bg-zinc-50 rounded-full">
              <X size={20} className="text-zinc-600" />
            </button>
          </div>
          <div className="p-6 overflow-y-auto">
            <div className="prose prose-sm max-w-none text-zinc-800">
              {summary.map((point, index) => (
                <div key={index} className="flex items-start gap-2 mb-4">
                  <div className="mt-1 text-yellow-600">•</div>
                  <p className="flex-1">{point}</p>
                </div>
              ))}
            </div>
            <div className="mt-6 flex justify-end gap-2">
              <button
                onClick={copySummaryToClipboard}
                className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors flex items-center gap-2"
              >
                {copiedSummary ? (
                  <>
                    <Check size={16} />
                    Copied!
                  </>
                ) : (
                  <>
                    <Copy size={16} />
                    Copy to Clipboard
                  </>
                )}
              </button>
              <button
                onClick={downloadSummaryAsPDF}
                disabled={downloadingPDF}
                className="px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {downloadingPDF ? (
                  <>
                    <Loader2 size={16} className="animate-spin" />
                    Generating...
                  </>
                ) : (
                  <>
                    <Download size={16} />
                    Download PDF
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      </div>
    )}

    {/* App Overview Modal */}
    {showAppOverviewModal && (
      <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
        <div className="bg-white rounded-2xl w-[700px] max-h-[85vh] flex flex-col">
          <div className="flex items-center justify-between p-4 border-b border-zinc-100">
            <div className="flex items-center gap-3">
              <Image src="/logo.png" alt="Resonate AI" width={32} height={32} className="rounded" />
              <h3 className="text-lg font-medium">Resonate AI Analyser Overview</h3>
            </div>
            <button onClick={() => setShowAppOverviewModal(false)} className="p-2 hover:bg-zinc-50 rounded-full">
              <X size={20} className="text-zinc-600" />
            </button>
          </div>
          <div className="p-6 overflow-y-auto">
            <div className="space-y-8">
              {/* Introduction */}
              <div>
                <h4 className="text-base font-semibold mb-2 text-blue-700">Welcome to Resonate AI</h4>
                <p className="text-zinc-600 text-sm leading-relaxed">
                  Resonate AI is a comprehensive platform designed to transform how organizations analyze, understand, and improve their sustainability reporting. Our advanced AI models provide real-time insights and recommendations to enhance reporting quality and compliance with international standards.
                </p>
              </div>

              {/* Core Features Section */}
              <div>
                <h4 className="text-base font-semibold mb-3 text-blue-700">Core Features</h4>
                <div className="grid grid-cols-2 gap-4">
                  <div className="p-3 bg-blue-50 rounded-xl border border-blue-100">
                    <div className="flex items-center gap-2 mb-2">
                      <Send size={18} className="text-blue-600" />
                      <h5 className="font-medium text-sm">Intelligent Chat</h5>
                    </div>
                    <p className="text-xs text-zinc-600 pl-7">
                      Engage with our AI assistant to analyze documents, extract insights, and receive expert guidance on improving your reporting.
                    </p>
                  </div>
                  <div className="p-3 bg-green-50 rounded-xl border border-green-100">
                    <div className="flex items-center gap-2 mb-2">
                      <ArrowRight size={18} className="text-green-600" />
                      <h5 className="font-medium text-sm">Document Analysis</h5>
                    </div>
                    <p className="text-xs text-zinc-600 pl-7">
                      Upload your sustainability reports for instant analysis across multiple dimensions including language quality, ESG focus, and regulatory compliance.
                    </p>
                  </div>
                  <div className="p-3 bg-purple-50 rounded-xl border border-purple-100">
                    <div className="flex items-center gap-2 mb-2">
                      <ChevronDown size={18} className="text-purple-600" />
                      <h5 className="font-medium text-sm">Compliance Engine</h5>
                    </div>
                    <p className="text-xs text-zinc-600 pl-7">
                      Automatically evaluate your reports against TCFD, GRI, CSRD, and SASB frameworks with actionable improvement suggestions.
                    </p>
                  </div>
                  <div className="p-3 bg-yellow-50 rounded-xl border border-yellow-100">
                    <div className="flex items-center gap-2 mb-2">
                      <Sparkles size={18} className="text-yellow-600" />
                      <h5 className="font-medium text-sm">AI Summary</h5>
                    </div>
                    <p className="text-xs text-zinc-600 pl-7">
                      Generate concise, exportable summaries of your reports with key insights highlighted for stakeholder presentations.
                    </p>
                  </div>
                </div>
              </div>

              {/* Analysis Tools Section */}
              <div>
                <h4 className="text-base font-semibold mb-3 text-blue-700">Advanced Analysis Tools</h4>
                <div className="space-y-3">
                  <div className="p-3 bg-zinc-50 rounded-xl border border-zinc-200">
                    <h5 className="font-medium text-sm mb-2">Reporting Quality Analysis</h5>
                    <p className="text-xs text-zinc-600 mb-2">
                      Evaluate the professional quality of your sustainability reporting with metrics for:
                    </p>
                    <ul className="list-disc pl-5 text-xs text-zinc-600 space-y-1">
                      <li><span className="font-medium">Consistency Score:</span> Measures alignment and coherence throughout your report</li>
                      <li><span className="font-medium">Variability Score:</span> Identifies balanced coverage of topics and concepts</li>
                      <li><span className="font-medium">Comprehensibility Score:</span> Analyzes readability and accessibility of your content</li>
                      <li><span className="font-medium">Clarity Score:</span> Evaluates precision and directness of language</li>
                    </ul>
                  </div>

                  <div className="p-3 bg-zinc-50 rounded-xl border border-zinc-200">
                    <h5 className="font-medium text-sm mb-2">Cheap Talk Analysis</h5>
                    <p className="text-xs text-zinc-600 mb-2">
                      Identify substantive commitments versus vague statements in your reports:
                    </p>
                    <ul className="list-disc pl-5 text-xs text-zinc-600 space-y-1">
                      <li><span className="font-medium">Commitment Score:</span> Detects language indicating genuine commitments</li>
                      <li><span className="font-medium">Specificity Score:</span> Measures concrete details vs. generic statements</li>
                      <li><span className="font-medium">Cheap Talk Score:</span> Identifies non-binding, low-risk language</li>
                      <li><span className="font-medium">Safe Talk Score:</span> Highlights cautious, liability-limiting language</li>
                    </ul>
                  </div>

                  <div className="p-3 bg-zinc-50 rounded-xl border border-zinc-200">
                    <h5 className="font-medium text-sm mb-2">ESG Focus Analysis</h5>
                    <p className="text-xs text-zinc-600 mb-2">
                      Quantify your report's focus across nine key ESG categories:
                    </p>
                    <div className="grid grid-cols-3 gap-2">
                      <div className="bg-blue-50 p-1.5 rounded text-xs text-center text-blue-700">Business Ethics</div>
                      <div className="bg-green-50 p-1.5 rounded text-xs text-center text-green-700">Climate Change</div>
                      <div className="bg-indigo-50 p-1.5 rounded text-xs text-center text-indigo-700">Community Relations</div>
                      <div className="bg-purple-50 p-1.5 rounded text-xs text-center text-purple-700">Corporate Governance</div>
                      <div className="bg-yellow-50 p-1.5 rounded text-xs text-center text-yellow-700">Human Capital</div>
                      <div className="bg-teal-50 p-1.5 rounded text-xs text-center text-teal-700">Natural Capital</div>
                      <div className="bg-orange-50 p-1.5 rounded text-xs text-center text-orange-700">Pollution & Waste</div>
                      <div className="bg-red-50 p-1.5 rounded text-xs text-center text-red-700">Product Liability</div>
                      <div className="bg-zinc-100 p-1.5 rounded text-xs text-center text-zinc-700">Non-ESG</div>
                    </div>
                  </div>

                  <div className="p-3 bg-zinc-50 rounded-xl border border-zinc-200">
                    <h5 className="font-medium text-sm mb-2">Compliance Dashboard</h5>
                    <p className="text-xs text-zinc-600">
                      Comprehensive assessment of your report's alignment with major frameworks (TCFD, GRI, CSRD, SASB), featuring automated gap identification, improvement recommendations, and AI-generated draft text to strengthen your disclosures.
                    </p>
                  </div>
                </div>
              </div>

              {/* Productivity Features */}
              <div>
                <h4 className="text-base font-semibold mb-3 text-blue-700">Productivity Features</h4>
                <div className="grid grid-cols-2 gap-3">
                  <div className="p-3 bg-zinc-50 rounded-xl border border-zinc-200">
                    <div className="flex items-center gap-2 mb-1">
                      <Bookmark size={16} className="text-blue-600" />
                      <h5 className="font-medium text-sm">Notes & Bookmarks</h5>
                    </div>
                    <p className="text-xs text-zinc-600 pl-7">
                      Save important insights and conversations for later reference or export as PDFs.
                    </p>
                  </div>
                  <div className="p-3 bg-zinc-50 rounded-xl border border-zinc-200">
                    <div className="flex items-center gap-2 mb-1">
                      <Download size={16} className="text-blue-600" />
                      <h5 className="font-medium text-sm">Export & Share</h5>
                    </div>
                    <p className="text-xs text-zinc-600 pl-7">
                      Export analyses, summaries, and insights in professional PDF format for stakeholder presentations.
                    </p>
                  </div>
                </div>
              </div>
              
              {/* Technical Details */}
              <div>
                <h4 className="text-base font-semibold mb-2 text-blue-700">Technical Details</h4>
                <p className="text-xs text-zinc-600 mb-2">
                  Built on cutting-edge technology to deliver accurate, actionable insights:
                </p>
                <ul className="list-disc pl-5 text-xs text-zinc-600 space-y-1">
                  <li>Advanced large language models fine-tuned for ESG reporting analysis</li>
                  <li>Real-time streaming responses for immediate feedback</li>
                  <li>Specialized models for each analysis dimension (TCFD, GRI, ESG categories, etc.)</li>
                  <li>Built with Next.js, TypeScript, and Tailwind CSS for a responsive, accessible interface</li>
                </ul>
              </div>

              <div className="text-center mt-2 pt-4 border-t border-zinc-100">
                <p className="text-xs text-zinc-500">© 2025 Resonate AI. All rights reserved.</p>
                <p className="text-xs text-blue-600 mt-1">www.resonateai.co.uk</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    )}
  </div>
);
}
