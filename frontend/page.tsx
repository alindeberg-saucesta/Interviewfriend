"use client";

import React, { useState, useRef, useEffect, KeyboardEvent } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/cjs/styles/prism';
import { Toaster, toast } from 'react-hot-toast';

// --- Types ---
interface Message {
  speaker: 'user' | 'assistant' | 'system';
  text: string;
}
type Theme = 'light' | 'dark';

// --- Component ---
export default function Home() {
  // --- State ---
  const [inputValue, setInputValue] = useState('');
  const [conversation, setConversation] = useState<Message[]>([]);
  const [isResponding, setIsResponding] = useState(false);
  const [role, setRole] = useState<'interviewer' | 'interviewee' | null>(null);
  const [theme, setTheme] = useState<Theme>('light');
  const [hasMounted, setHasMounted] = useState(false);

  // --- Refs ---
  const conversationEndRef = useRef<HTMLDivElement | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  // --- Backend URL (assumes same origin) ---
  const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || '';

  // --- Effects ---
  useEffect(() => {
    setHasMounted(true);
    // Load saved theme
    const stored = localStorage.getItem('theme') as Theme | null;
    if (stored) {
      setTheme(stored);
    }
  }, []);

  useEffect(() => {
    if (!hasMounted) return;
    const root = window.document.documentElement;
    if (theme === 'dark') {
      root.classList.add('dark');
      localStorage.setItem('theme', 'dark');
    } else {
      root.classList.remove('dark');
      localStorage.setItem('theme', 'light');
    }
  }, [theme, hasMounted]);

  useEffect(() => {
    if (conversationEndRef.current) {
      conversationEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [conversation]);

  // Clear chat when role changes
  useEffect(() => {
    if (role !== null) {
      setConversation([]);
      setInputValue('');
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
        abortControllerRef.current = null;
        setIsResponding(false);
      }
    }
  }, [role]);

  // --- Handlers ---
  async function handleSend() {
    if (!inputValue.trim() || !role) return;

    // Abort any in-flight request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    const userMsg: Message = { speaker: 'user', text: inputValue };
    setConversation(prev => [...prev, userMsg]);
    setInputValue('');
    setIsResponding(true);

    try {
      const response = await fetch(`${backendUrl}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          role,
          messages: [
            // include all previous messages in expected format
            ...conversation.map(m => ({
              role: m.speaker === 'user' ? 'user' : 'assistant',
              content: m.text
            })),
            { role: 'user', content: userMsg.text }
          ]
        }),
        signal: abortController.signal
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      // Stream SSE
      const reader = response.body!.getReader();
      const decoder = new TextDecoder();
      let assistantText = '';
      // seed assistant message
      setConversation(prev => [...prev, { speaker: 'assistant', text: '' }]);

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value);
        // Parse SSE lines
        for (const line of chunk.split('\n')) {
          if (line.startsWith('data: ')) {
            try {
              const payload = JSON.parse(line.slice(6));
              assistantText += payload.chunk;
              // update last assistant message
              setConversation(prev => {
                const updated = [...prev];
                const lastIdx = updated.findIndex((m, i) => i === updated.length - 1 && m.speaker === 'assistant');
                if (lastIdx !== -1) {
                  updated[lastIdx].text = assistantText;
                }
                return updated;
              });
            } catch (err) {
              console.error('Could not parse SSE chunk', err);
            }
          }
        }
      }
    } catch (err: any) {
      if (err.name !== 'AbortError') {
        toast.error(`Chat error: ${err.message}`);
      }
    } finally {
      setIsResponding(false);
    }
  }

  function handleClearChat() {
    setConversation([]);
    setInputValue('');
  }

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!isResponding) {
        handleSend();
      }
    }
  }

  return (
    <div className="flex flex-col h-screen p-4">
      <Toaster />
      {/* Theme Toggle */}
      <div className="flex justify-end mb-2">
        <button
          onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')}
          className="px-2 py-1 border rounded"
        >
          {theme === 'light' ? 'Dark' : 'Light'} Mode
        </button>
      </div>

      {/* Role Selection */}
      <div className="flex space-x-2 mb-4">
        <button
          onClick={() => setRole('interviewer')}
          className={`px-4 py-2 rounded ${role === 'interviewer' ? 'bg-blue-500 text-white' : 'border'}`}
        >
          Interviewer
        </button>
        <button
          onClick={() => setRole('interviewee')}
          className={`px-4 py-2 rounded ${role === 'interviewee' ? 'bg-green-500 text-white' : 'border'}`}
        >
          Interviewee
        </button>
        {role && (
          <button onClick={handleClearChat} className="px-4 py-2 border rounded">
            Clear Chat
          </button>
        )}
      </div>

      {/* Conversation Window */}
      <div className="flex-1 overflow-y-auto mb-4 space-y-2">
        {conversation.map((msg, idx) => (
          <div key={idx} className={`p-2 rounded ${msg.speaker === 'user' ? 'bg-gray-200 self-end' : 'bg-gray-800 text-white self-start'}`}>
            <ReactMarkdown remarkPlugins={[remarkGfm]} components={{
              code({ node, inline, className, children }) {
                return !inline ? (
                  <SyntaxHighlighter language="typescript" style={vscDarkPlus}>
                    {String(children).replace(/\n$/, '')}
                  </SyntaxHighlighter>
                ) : (
                  <code className={className}>{children}</code>
                );
              }
            }}>
              {msg.text}
            </ReactMarkdown>
          </div>
        ))}
        <div ref={conversationEndRef} />
      </div>

      {/* Input Box */}
      {role && (
        <div className="flex space-x-2">
          <textarea
            className="flex-1 p-2 border rounded resize-none"
            rows={2}
            value={inputValue}
            onChange={e => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isResponding}
            placeholder="Type your message..."
          />
          <button
            onClick={handleSend}
            disabled={isResponding || !inputValue.trim()}
            className="px-4 py-2 bg-blue-600 text-white rounded disabled:opacity-50"
          >
            {isResponding ? '…' : 'Send'}
          </button>
        </div>
      )}
    </div>
  );
}
