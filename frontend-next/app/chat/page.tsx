"use client";

import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import ReactMarkdown from "react-markdown";
import {
    Send,
    FileText,
    Bot,
    User,
    ChevronDown,
    ChevronRight,
    Search,
    Brain,
    PenLine,
    BarChart2,
    BookOpen,
    Sparkles,
} from "lucide-react";
import Badge from "@/components/ui/Badge";

/* ────────────────── Types ────────────────── */
interface Message {
    id: string;
    role: "user" | "assistant";
    content: string;
    citations?: string[];
    agentSteps?: AgentStep[];
}

interface AgentStep {
    agent: string;
    action: string;
    color: "cyan" | "violet" | "amber" | "green";
}

interface SourceItem {
    title: string;
    page: number;
    score: number;
    excerpt: string;
}

/* ────────────────── Mock Data ────────────────── */
const MOCK_DOCS = [
    "Attention Is All You Need.pdf",
    "REALM: Retrieval-Augmented Language Model Pre-Training.pdf",
    "Agentic RAG Survey 2024.pdf",
    "Knowledge Graphs + LLMs.pdf",
];

const MOCK_SOURCES: SourceItem[] = [
    { title: "Agentic RAG Survey 2024", page: 14, score: 0.97, excerpt: "Agentic RAG introduces autonomous multi-step reasoning where agents iteratively retrieve and refine..." },
    { title: "Attention Is All You Need", page: 3, score: 0.91, excerpt: "The Transformer architecture relies solely on attention mechanisms, dispensing with recurrence..." },
    { title: "REALM Pre-Training", page: 7, score: 0.88, excerpt: "Retrieval-augmented language model pre-training opens-books language models with knowledge retrieval..." },
];

const AGENT_STEPS: AgentStep[] = [
    { agent: "Planner", action: "Decomposing query into sub-tasks", color: "violet" },
    { agent: "Retriever", action: "Searching vector DB (k=8)", color: "cyan" },
    { agent: "Retriever", action: "Hybrid BM25 + dense reranking", color: "cyan" },
    { agent: "Generator", action: "Synthesizing with citations", color: "green" },
];

const DEMO_RESPONSE = `## Agentic RAG Overview

**Agentic RAG** extends traditional RAG pipelines by introducing **autonomous, multi-step reasoning agents** that can:

1. **Decompose** complex queries into sub-tasks
2. **Iteratively retrieve** from multiple sources
3. **Self-correct** based on retrieved evidence
4. **Generate** grounded, cited answers

### Key Differences from Naïve RAG

| Aspect | Naïve RAG | Agentic RAG |
|--------|-----------|-------------|
| Retrieval | Single-shot | Iterative |
| Planning | None | Multi-step |
| Memory | None | Persistent |
| Citations | Optional | Required |

> Studies show Agentic RAG improves answer accuracy by **~35%** on complex multi-hop questions. [1][2]
`;

/* ────────────────── Component ────────────────── */
export default function ChatPage() {
    const [messages, setMessages] = useState<Message[]>([
        {
            id: "0",
            role: "assistant",
            content: "Hello! I'm your Agentic RAG Research Assistant. Ask me anything about your uploaded documents, or any academic topic you're researching.",
        },
    ]);
    const [input, setInput] = useState("");
    const [isTyping, setIsTyping] = useState(false);
    const [expandedSteps, setExpandedSteps] = useState<string | null>(null);
    const [sources, setSources] = useState<SourceItem[]>([]);
    const bottomRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages, isTyping]);

    const sendMessage = () => {
        if (!input.trim()) return;
        const userMsg: Message = { id: crypto.randomUUID(), role: "user", content: input };
        setMessages((prev) => [...prev, userMsg]);
        setInput("");
        setIsTyping(true);
        setSources([]);

        setTimeout(() => {
            setIsTyping(false);
            const aiMsg: Message = {
                id: crypto.randomUUID(),
                role: "assistant",
                content: DEMO_RESPONSE,
                citations: ["Agentic RAG Survey 2024", "Attention Is All You Need"],
                agentSteps: AGENT_STEPS,
            };
            setMessages((prev) => [...prev, aiMsg]);
            setSources(MOCK_SOURCES);
        }, 2800);
    };

    return (
        <div
            className="h-[calc(100vh-4rem)] flex overflow-hidden"
            style={{ background: "var(--bg-primary)" }}
        >
            {/* ── Left Panel ── */}
            <aside
                className="w-64 xl:w-72 flex-shrink-0 flex flex-col border-r overflow-y-auto hidden md:flex"
                style={{ borderColor: "var(--border-card)", background: "var(--bg-secondary)" }}
            >
                {/* Documents */}
                <div className="p-4 border-b" style={{ borderColor: "var(--border-card)" }}>
                    <p className="text-xs font-semibold uppercase tracking-wider mb-3" style={{ color: "var(--text-muted)" }}>
                        Indexed Documents
                    </p>
                    <ul className="space-y-1.5">
                        {MOCK_DOCS.map((doc) => (
                            <li
                                key={doc}
                                className="flex items-center gap-2.5 px-3 py-2.5 rounded-lg cursor-pointer transition-colors"
                                style={{ color: "var(--text-secondary)" }}
                                onMouseEnter={(e) => (e.currentTarget.style.background = "rgba(255,255,255,0.04)")}
                                onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
                            >
                                <FileText className="w-3.5 h-3.5 flex-shrink-0" style={{ color: "var(--accent-indigo)" }} />
                                <span className="text-xs truncate">{doc}</span>
                            </li>
                        ))}
                    </ul>
                </div>

                {/* Agent Activity Log */}
                <div className="p-4 flex-1">
                    <p className="text-xs font-semibold uppercase tracking-wider mb-3" style={{ color: "var(--text-muted)" }}>
                        Agent Activity
                    </p>
                    <div className="space-y-2">
                        {[
                            { icon: Brain, label: "Planner", status: "Idle", color: "violet" as const },
                            { icon: Search, label: "Retriever", status: "Ready", color: "cyan" as const },
                            { icon: PenLine, label: "Generator", status: "Standby", color: "green" as const },
                        ].map(({ icon: Icon, label, status, color }) => (
                            <div
                                key={label}
                                className="flex items-center gap-2.5 px-3 py-2 rounded-lg"
                                style={{ background: "rgba(255,255,255,0.03)" }}
                            >
                                <Icon className="w-3.5 h-3.5" style={{ color: `var(--accent-${color === "violet" ? "violet" : color === "cyan" ? "cyan" : "cyan"})` }} />
                                <span className="text-xs flex-1" style={{ color: "var(--text-secondary)" }}>{label}</span>
                                <Badge color={color} className="text-[10px] px-1.5 py-0">{status}</Badge>
                            </div>
                        ))}
                    </div>
                </div>
            </aside>

            {/* ── Main Chat Panel ── */}
            <div className="flex-1 flex flex-col min-w-0">
                {/* Header */}
                <div
                    className="px-6 py-4 border-b flex items-center gap-3 flex-shrink-0"
                    style={{ borderColor: "var(--border-card)", background: "var(--bg-secondary)" }}
                >
                    <div
                        className="w-8 h-8 rounded-lg flex items-center justify-center"
                        style={{ background: "linear-gradient(135deg, #6366f1, #22d3ee)" }}
                    >
                        <Bot className="w-4 h-4 text-white" />
                    </div>
                    <div>
                        <p className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>Research Assistant</p>
                        <p className="text-xs" style={{ color: "var(--text-muted)" }}>Agentic RAG · {MOCK_DOCS.length} documents indexed</p>
                    </div>
                    <div className="ml-auto flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full" style={{ background: "#4ade80" }} />
                        <span className="text-xs" style={{ color: "var(--text-muted)" }}>Online</span>
                    </div>
                </div>

                {/* Messages */}
                <div className="flex-1 overflow-y-auto px-6 py-6 space-y-6">
                    <AnimatePresence>
                        {messages.map((msg) => (
                            <motion.div
                                key={msg.id}
                                initial={{ opacity: 0, y: 16 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ duration: 0.4, ease: "easeOut" }}
                                className={`flex gap-3 ${msg.role === "user" ? "flex-row-reverse" : ""}`}
                            >
                                {/* Avatar */}
                                <div
                                    className="w-8 h-8 rounded-full flex-shrink-0 flex items-center justify-center"
                                    style={{
                                        background:
                                            msg.role === "assistant"
                                                ? "linear-gradient(135deg, #6366f1, #22d3ee)"
                                                : "rgba(255,255,255,0.1)",
                                    }}
                                >
                                    {msg.role === "assistant" ? (
                                        <Sparkles className="w-3.5 h-3.5 text-white" />
                                    ) : (
                                        <User className="w-3.5 h-3.5" style={{ color: "var(--text-secondary)" }} />
                                    )}
                                </div>

                                {/* Bubble */}
                                <div className={`max-w-[75%] ${msg.role === "user" ? "items-end" : "items-start"} flex flex-col gap-2`}>
                                    {/* Agent steps (collapsible) */}
                                    {msg.agentSteps && (
                                        <button
                                            onClick={() => setExpandedSteps(expandedSteps === msg.id ? null : msg.id)}
                                            className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg"
                                            style={{ background: "rgba(99,102,241,0.1)", color: "#818cf8" }}
                                        >
                                            {expandedSteps === msg.id ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                                            Agent reasoning ({msg.agentSteps.length} steps)
                                        </button>
                                    )}

                                    <AnimatePresence>
                                        {msg.agentSteps && expandedSteps === msg.id && (
                                            <motion.div
                                                initial={{ height: 0, opacity: 0 }}
                                                animate={{ height: "auto", opacity: 1 }}
                                                exit={{ height: 0, opacity: 0 }}
                                                className="overflow-hidden w-full"
                                            >
                                                <div
                                                    className="rounded-xl p-3 space-y-2 w-full"
                                                    style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.07)" }}
                                                >
                                                    {msg.agentSteps.map((step, i) => (
                                                        <div key={i} className="flex items-center gap-2">
                                                            <Badge color={step.color} className="flex-shrink-0">{step.agent}</Badge>
                                                            <span className="text-xs" style={{ color: "var(--text-secondary)" }}>{step.action}</span>
                                                        </div>
                                                    ))}
                                                </div>
                                            </motion.div>
                                        )}
                                    </AnimatePresence>

                                    {/* Message content */}
                                    <div
                                        className="rounded-2xl px-4 py-3 text-sm leading-relaxed"
                                        style={{
                                            background:
                                                msg.role === "user"
                                                    ? "linear-gradient(135deg, rgba(99,102,241,0.25), rgba(34,211,238,0.15))"
                                                    : "rgba(255,255,255,0.05)",
                                            border: "1px solid rgba(255,255,255,0.08)",
                                            color: "var(--text-primary)",
                                        }}
                                    >
                                        {msg.role === "assistant" ? (
                                            <div className="prose-dark text-sm">
                                                <ReactMarkdown>{msg.content}</ReactMarkdown>
                                            </div>
                                        ) : (
                                            msg.content
                                        )}
                                    </div>

                                    {/* Citations */}
                                    {msg.citations && (
                                        <div className="flex flex-wrap gap-1.5">
                                            {msg.citations.map((c, i) => (
                                                <span
                                                    key={i}
                                                    className="inline-flex items-center gap-1 text-[10px] px-2 py-0.5 rounded"
                                                    style={{
                                                        background: "rgba(34,211,238,0.08)",
                                                        border: "1px solid rgba(34,211,238,0.15)",
                                                        color: "var(--accent-cyan)",
                                                    }}
                                                >
                                                    <BookOpen className="w-2.5 h-2.5" />
                                                    [{i + 1}] {c}
                                                </span>
                                            ))}
                                        </div>
                                    )}
                                </div>
                            </motion.div>
                        ))}

                        {/* Typing indicator */}
                        {isTyping && (
                            <motion.div
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0 }}
                                className="flex gap-3 items-start"
                            >
                                <div
                                    className="w-8 h-8 rounded-full flex-shrink-0 flex items-center justify-center"
                                    style={{ background: "linear-gradient(135deg, #6366f1, #22d3ee)" }}
                                >
                                    <Sparkles className="w-3.5 h-3.5 text-white" />
                                </div>
                                <div
                                    className="rounded-2xl px-4 py-3 flex items-center gap-1.5"
                                    style={{ background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.08)" }}
                                >
                                    <div className="typing-dot" />
                                    <div className="typing-dot" />
                                    <div className="typing-dot" />
                                </div>
                            </motion.div>
                        )}
                    </AnimatePresence>
                    <div ref={bottomRef} />
                </div>

                {/* Input */}
                <div
                    className="px-6 py-4 border-t flex-shrink-0"
                    style={{ borderColor: "var(--border-card)", background: "var(--bg-secondary)" }}
                >
                    <div
                        className="flex items-end gap-3 rounded-2xl p-3"
                        style={{ background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.1)" }}
                    >
                        <textarea
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyDown={(e) => {
                                if (e.key === "Enter" && !e.shiftKey) {
                                    e.preventDefault();
                                    sendMessage();
                                }
                            }}
                            placeholder="Ask a research question… (Enter to send, Shift+Enter for newline)"
                            rows={1}
                            className="flex-1 bg-transparent resize-none outline-none text-sm leading-relaxed"
                            style={{ color: "var(--text-primary)", maxHeight: "120px" }}
                        />
                        <button
                            onClick={sendMessage}
                            disabled={!input.trim() || isTyping}
                            className="w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0 transition-all duration-200 hover:brightness-110 disabled:opacity-40"
                            style={{ background: "linear-gradient(135deg, #6366f1, #22d3ee)" }}
                        >
                            <Send className="w-4 h-4 text-white" />
                        </button>
                    </div>
                    <p className="text-xs text-center mt-2" style={{ color: "var(--text-muted)" }}>
                        Powered by Agentic RAG · Answers are grounded in indexed documents
                    </p>
                </div>
            </div>

            {/* ── Right Panel: Sources ── */}
            <aside
                className="w-72 xl:w-80 flex-shrink-0 flex flex-col border-l overflow-y-auto hidden lg:flex"
                style={{ borderColor: "var(--border-card)", background: "var(--bg-secondary)" }}
            >
                <div className="p-4 border-b flex-shrink-0" style={{ borderColor: "var(--border-card)" }}>
                    <p className="text-xs font-semibold uppercase tracking-wider" style={{ color: "var(--text-muted)" }}>
                        Retrieved Sources
                    </p>
                </div>

                <div className="p-4 space-y-3 flex-1">
                    <AnimatePresence>
                        {sources.length === 0 ? (
                            <p className="text-xs text-center py-8" style={{ color: "var(--text-muted)" }}>
                                Sources will appear here after a query
                            </p>
                        ) : (
                            sources.map((src, i) => (
                                <motion.div
                                    key={i}
                                    initial={{ opacity: 0, x: 20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    transition={{ delay: i * 0.1 }}
                                    className="glass rounded-xl p-4 space-y-2"
                                >
                                    <div className="flex items-start justify-between gap-2">
                                        <span className="text-xs font-semibold leading-snug" style={{ color: "var(--text-primary)" }}>
                                            {src.title}
                                        </span>
                                        <span className="text-[10px] flex-shrink-0 px-1.5 py-0.5 rounded" style={{ background: "rgba(74,222,128,0.1)", color: "#4ade80" }}>
                                            p.{src.page}
                                        </span>
                                    </div>

                                    {/* Confidence bar */}
                                    <div>
                                        <div className="flex justify-between mb-1">
                                            <span className="text-[10px]" style={{ color: "var(--text-muted)" }}>Relevance</span>
                                            <span className="text-[10px]" style={{ color: "var(--accent-cyan)" }}>{Math.round(src.score * 100)}%</span>
                                        </div>
                                        <div className="h-1 rounded-full overflow-hidden" style={{ background: "rgba(255,255,255,0.06)" }}>
                                            <motion.div
                                                className="h-full rounded-full"
                                                style={{ background: "linear-gradient(90deg, #6366f1, #22d3ee)" }}
                                                initial={{ width: 0 }}
                                                animate={{ width: `${src.score * 100}%` }}
                                                transition={{ duration: 0.8, delay: i * 0.1 }}
                                            />
                                        </div>
                                    </div>

                                    <p className="text-[11px] leading-relaxed line-clamp-3" style={{ color: "var(--text-secondary)" }}>
                                        "{src.excerpt}"
                                    </p>
                                </motion.div>
                            ))
                        )}
                    </AnimatePresence>
                </div>

                {/* Token usage mini chart */}
                <div className="p-4 border-t" style={{ borderColor: "var(--border-card)" }}>
                    <p className="text-[10px] uppercase tracking-wider mb-3" style={{ color: "var(--text-muted)" }}>
                        <BarChart2 className="w-3 h-3 inline mr-1" />
                        Session Stats
                    </p>
                    <div className="space-y-2">
                        {[
                            { label: "Tokens used", value: "2,847" },
                            { label: "Docs retrieved", value: sources.length.toString() },
                            { label: "Messages", value: messages.length.toString() },
                        ].map(({ label, value }) => (
                            <div key={label} className="flex justify-between">
                                <span className="text-xs" style={{ color: "var(--text-muted)" }}>{label}</span>
                                <span className="text-xs font-medium" style={{ color: "var(--text-secondary)" }}>{value}</span>
                            </div>
                        ))}
                    </div>
                </div>
            </aside>
        </div>
    );
}
