import { useEffect, useRef, useState } from 'react'
import type { FormEvent, KeyboardEvent } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import './App.css'

type Programme = 'BCA' | 'BBA' | 'BCom'

type ChatMessage = {
  role: 'user' | 'assistant'
  content: string
}

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'
const QUICK_PROMPTS = [
  'What is the minimum attendance required?',
  'Explain fee refund rules in simple points.',
  'What are promotion criteria for next semester?',
]

function App() {
  const [programme, setProgramme] = useState<Programme>('BCA')
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      role: 'assistant',
      content:
        'Hi! I am your college assistant. Ask me anything about academics or fee details.',
    },
  ])
  const chatWindowRef = useRef<HTMLElement | null>(null)
  const shouldAutoScrollRef = useRef(true)
  const endOfMessagesRef = useRef<HTMLDivElement | null>(null)
  const inputRef = useRef<HTMLTextAreaElement | null>(null)

  useEffect(() => {
    if (!shouldAutoScrollRef.current) {
      return
    }
    endOfMessagesRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages, loading])

  useEffect(() => {
    const container = chatWindowRef.current
    if (!container) {
      return
    }

    const updateAutoScrollPreference = () => {
      const distanceFromBottom = container.scrollHeight - container.scrollTop - container.clientHeight
      shouldAutoScrollRef.current = distanceFromBottom < 80
    }

    updateAutoScrollPreference()
    container.addEventListener('scroll', updateAutoScrollPreference)

    return () => {
      container.removeEventListener('scroll', updateAutoScrollPreference)
    }
  }, [])

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    const message = input.trim()
    if (!message || loading) {
      return
    }

    setError(null)
    setInput('')
    shouldAutoScrollRef.current = true
    setMessages((prev) => [...prev, { role: 'user', content: message }])
    setLoading(true)

    try {
      const response = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          programme,
          message,
        }),
      })

      if (!response.ok) {
        const details = await response.json().catch(() => ({}))
        const detailText = typeof details?.detail === 'string' ? details.detail : 'Unknown API error'
        throw new Error(detailText)
      }

      const data: { answer: string } = await response.json()
      setMessages((prev) => [...prev, { role: 'assistant', content: data.answer }])
    } catch (err) {
      const messageText = err instanceof Error ? err.message : 'Failed to reach backend'
      setError(messageText)
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: 'I could not reach the backend. Make sure the Python API server is running on port 8000.',
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  const handleInputKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      const form = event.currentTarget.form
      form?.requestSubmit()
    }
  }

  const useQuickPrompt = (prompt: string) => {
    setInput(prompt)
    inputRef.current?.focus()
  }

  return (
    <main className="chat-shell">
      <header className="chat-header">
        <div className="title-row">
          <p className="eyebrow">College Assistant</p>
          <p className="status-pill" role="status" aria-live="polite">
            <span className="status-dot" aria-hidden="true" /> API Connected
          </p>
        </div>
        <h1>Ask Academic And Fee Questions</h1>
        <p className="subtext">
          Production-ready React chat interface connected to your LangGraph backend.
        </p>

        <div className="header-controls">
          <label className="programme-control" htmlFor="programme">
            Programme
            <select
              id="programme"
              value={programme}
              onChange={(event) => setProgramme(event.target.value as Programme)}
            >
              <option value="BCA">BCA</option>
              <option value="BBA">BBA</option>
              <option value="BCom">BCom</option>
            </select>
          </label>

          <div className="quick-prompts" aria-label="Quick prompts">
            {QUICK_PROMPTS.map((prompt) => (
              <button
                key={prompt}
                type="button"
                className="prompt-chip"
                onClick={() => useQuickPrompt(prompt)}
                disabled={loading}
              >
                {prompt}
              </button>
            ))}
          </div>
        </div>
      </header>

      <section ref={chatWindowRef} className="chat-window" aria-live="polite">
        {messages.map((msg, index) => (
          <article key={`${msg.role}-${index}`} className={`bubble ${msg.role}`}>
            <p className="bubble-role">{msg.role === 'assistant' ? 'Assistant' : 'You'}</p>
            {msg.role === 'assistant' ? (
              <div className="markdown-content">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
              </div>
            ) : (
              <p className="plain-content">{msg.content}</p>
            )}
          </article>
        ))}
        {loading && (
          <article className="bubble assistant typing">
            <p className="bubble-role">Assistant</p>
            <p className="typing-text">
              Thinking
              <span className="dot-flow" aria-hidden="true" />
            </p>
          </article>
        )}
        <div ref={endOfMessagesRef} aria-hidden="true" />
      </section>

      <form className="chat-form" onSubmit={handleSubmit}>
        <textarea
          ref={inputRef}
          value={input}
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={handleInputKeyDown}
          placeholder="Ask about attendance, exams, tuition, refunds..."
          rows={2}
          aria-label="Your message"
        />
        <button type="submit" disabled={loading || !input.trim()}>
          Send
        </button>
        <p className="input-helper">Press Enter to send, Shift+Enter for a new line.</p>
      </form>

      {error && <p className="error-text">Backend error: {error}</p>}
    </main>
  )
}

export default App
