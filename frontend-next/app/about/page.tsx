"use client";

import { motion } from "framer-motion";
import {
    BookOpen,
    Layers,
    GitBranch,
    Share2,
    Network,
    ArrowRight,
    ChevronRight,
} from "lucide-react";
import AnimatedSection from "@/components/motion/AnimatedSection";

const RAG_PARADIGMS = [
    {
        id: "naive",
        icon: BookOpen,
        color: "#64748b",
        accentBg: "rgba(100,116,139,0.1)",
        accentBorder: "rgba(100,116,139,0.2)",
        title: "Naïve RAG",
        badge: "Baseline",
        description:
            "The original retrieve-then-read pipeline. Index documents, retrieve top-k chunks via cosine similarity, prepend to LLM prompt. Simple but brittle — prone to context overflow, irrelevant chunk retrieval, and no iterative refinement.",
        strengths: ["Simple to implement", "Low latency", "Works for simple Q&A"],
        weaknesses: ["No query reformulation", "Single-shot retrieval", "No verification"],
        diagram: ["Query", "→", "Embed", "→", "Retrieve k chunks", "→", "Generate"],
    },
    {
        id: "advanced",
        icon: Layers,
        color: "#6366f1",
        accentBg: "rgba(99,102,241,0.1)",
        accentBorder: "rgba(99,102,241,0.2)",
        title: "Advanced RAG",
        badge: "Enhanced",
        description:
            "Adds pre-retrieval and post-retrieval optimizations: query expansion, hypothetical document embeddings (HyDE), re-ranking, chunk compression, and sliding-window chunking. Significantly improves relevance.",
        strengths: ["Query rewriting", "Re-ranking", "Chunk compression"],
        weaknesses: ["Still single-pass", "Limited context awareness"],
        diagram: ["Query Expand", "→", "Retrieve", "→", "Rerank", "→", "Compress", "→", "Generate"],
    },
    {
        id: "modular",
        icon: GitBranch,
        color: "#22d3ee",
        accentBg: "rgba(34,211,238,0.1)",
        accentBorder: "rgba(34,211,238,0.2)",
        title: "Modular RAG",
        badge: "Flexible",
        description:
            "Decomposes the pipeline into swappable modules: Search, Memory, Fusion, Routing, Prediction, Task Adapter. Allows mixing retrieval sources (web, vector DB, SQL) and custom orchestration logic per task type.",
        strengths: ["Highly composable", "Multi-source retrieval", "Task-specific routing"],
        weaknesses: ["Complex orchestration", "Higher engineering overhead"],
        diagram: ["Router", "→", "Retrieval Modules", "→", "Fusion", "→", "Adapter", "→", "Output"],
    },
    {
        id: "graph",
        icon: Share2,
        color: "#f59e0b",
        accentBg: "rgba(245,158,11,0.1)",
        accentBorder: "rgba(245,158,11,0.2)",
        title: "Graph RAG",
        badge: "Structural",
        description:
            "Builds a knowledge graph from the document corpus. Retrieval traverses entity relationships and community summaries, enabling multi-hop reasoning and structured knowledge access beyond flat vector search.",
        strengths: ["Multi-hop reasoning", "Entity relationships", "Global context summaries"],
        weaknesses: ["Graph construction cost", "Requires entity extraction"],
        diagram: ["Query", "→", "Entity Extract", "→", "Graph Traverse", "→", "Community Summary", "→", "Generate"],
    },
    {
        id: "agentic",
        icon: Network,
        color: "#a78bfa",
        accentBg: "rgba(167,139,250,0.1)",
        accentBorder: "rgba(167,139,250,0.2)",
        title: "Agentic RAG",
        badge: "State-of-the-Art",
        description:
            "Autonomous agents orchestrate the entire pipeline: decompose tasks, iteratively retrieve, self-critique answers, use persistent memory, and invoke specialized sub-agents. Ideal for complex, multi-step research workflows requiring high accuracy.",
        strengths: ["Task decomposition", "Iterative self-refinement", "Persistent memory", "Multi-agent collaboration"],
        weaknesses: ["Higher latency per query", "More complex to debug"],
        diagram: ["Planner", "→", "Retriever", "↔", "Memory", "→", "Refiner", "→", "Generator"],
    },
];

export default function AboutPage() {
    return (
        <div
            className="min-h-screen px-6 pt-12 pb-28 relative overflow-hidden"
            style={{ background: "var(--bg-primary)" }}
        >
            {/* Background */}
            <div
                className="absolute top-0 right-0 w-[600px] h-[600px] rounded-full pointer-events-none"
                style={{
                    background: "radial-gradient(circle, rgba(99,102,241,0.07) 0%, transparent 70%)",
                    filter: "blur(50px)",
                }}
            />

            <div className="max-w-4xl mx-auto relative z-10">
                {/* Header */}
                <AnimatedSection className="text-center mb-20">
                    <div
                        className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full mb-5 text-xs font-medium"
                        style={{ background: "rgba(99,102,241,0.1)", border: "1px solid rgba(99,102,241,0.2)", color: "#818cf8" }}
                    >
                        RAG Paradigm Evolution
                    </div>
                    <h1
                        className="text-4xl md:text-5xl font-bold mb-4"
                        style={{ fontFamily: "var(--font-poppins)", color: "var(--text-primary)" }}
                    >
                        Architecture Overview
                    </h1>
                    <p className="text-base max-w-xl mx-auto leading-relaxed" style={{ color: "var(--text-secondary)" }}>
                        From simple retrieve-and-read to fully autonomous multi-agent pipelines — explore the evolution of
                        Retrieval-Augmented Generation.
                    </p>
                </AnimatedSection>

                {/* Timeline */}
                <div className="relative">
                    {/* Vertical line */}
                    <div
                        className="absolute left-6 top-0 bottom-0 w-px hidden md:block"
                        style={{ background: "linear-gradient(to bottom, transparent, rgba(99,102,241,0.4) 10%, rgba(99,102,241,0.4) 90%, transparent)" }}
                    />

                    <div className="space-y-10">
                        {RAG_PARADIGMS.map((paradigm, i) => {
                            const Icon = paradigm.icon;
                            return (
                                <AnimatedSection key={paradigm.id} delay={i * 0.1} direction="left">
                                    <div className="flex gap-6">
                                        {/* Timeline node */}
                                        <div className="flex-shrink-0 hidden md:flex flex-col items-center">
                                            <div
                                                className="w-12 h-12 rounded-xl flex items-center justify-center z-10 relative"
                                                style={{
                                                    background: paradigm.accentBg,
                                                    border: `1px solid ${paradigm.accentBorder}`,
                                                    boxShadow: `0 0 16px ${paradigm.accentBg}`,
                                                }}
                                            >
                                                <Icon className="w-5 h-5" style={{ color: paradigm.color }} />
                                            </div>
                                        </div>

                                        {/* Content Card */}
                                        <div
                                            className="flex-1 glass glass-hover rounded-2xl p-6 md:p-8"
                                            style={{ borderLeft: `3px solid ${paradigm.color}` }}
                                        >
                                            {/* Title row */}
                                            <div className="flex flex-wrap items-center gap-3 mb-4">
                                                <div className="md:hidden">
                                                    <Icon className="w-5 h-5" style={{ color: paradigm.color }} />
                                                </div>
                                                <h2
                                                    className="text-xl font-bold"
                                                    style={{ fontFamily: "var(--font-poppins)", color: "var(--text-primary)" }}
                                                >
                                                    {paradigm.title}
                                                </h2>
                                                <span
                                                    className="text-xs font-semibold px-2.5 py-0.5 rounded-full"
                                                    style={{ background: paradigm.accentBg, color: paradigm.color, border: `1px solid ${paradigm.accentBorder}` }}
                                                >
                                                    {paradigm.badge}
                                                </span>
                                                <span
                                                    className="ml-auto text-sm font-bold"
                                                    style={{ color: "var(--text-muted)" }}
                                                >
                                                    {String(i + 1).padStart(2, "0")}
                                                </span>
                                            </div>

                                            {/* Description */}
                                            <p className="text-sm leading-relaxed mb-5" style={{ color: "var(--text-secondary)" }}>
                                                {paradigm.description}
                                            </p>

                                            {/* Data-flow diagram */}
                                            <div
                                                className="flex flex-wrap items-center gap-1.5 mb-5 p-3 rounded-xl text-xs font-mono"
                                                style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.06)" }}
                                            >
                                                {paradigm.diagram.map((step, j) => (
                                                    <span
                                                        key={j}
                                                        className={step === "→" || step === "↔" ? "" : "px-2.5 py-1 rounded-md"}
                                                        style={
                                                            step === "→" || step === "↔"
                                                                ? { color: "var(--text-muted)" }
                                                                : { background: paradigm.accentBg, color: paradigm.color, border: `1px solid ${paradigm.accentBorder}` }
                                                        }
                                                    >
                                                        {step}
                                                    </span>
                                                ))}
                                            </div>

                                            {/* Strengths / Weaknesses */}
                                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                                                <div>
                                                    <p className="text-xs font-semibold uppercase tracking-wider mb-2" style={{ color: "#4ade80" }}>
                                                        Strengths
                                                    </p>
                                                    <ul className="space-y-1">
                                                        {paradigm.strengths.map((s) => (
                                                            <li key={s} className="flex items-center gap-2 text-xs" style={{ color: "var(--text-secondary)" }}>
                                                                <ChevronRight className="w-3 h-3 flex-shrink-0" style={{ color: "#4ade80" }} />
                                                                {s}
                                                            </li>
                                                        ))}
                                                    </ul>
                                                </div>
                                                <div>
                                                    <p className="text-xs font-semibold uppercase tracking-wider mb-2" style={{ color: "#fb7185" }}>
                                                        Limitations
                                                    </p>
                                                    <ul className="space-y-1">
                                                        {paradigm.weaknesses.map((w) => (
                                                            <li key={w} className="flex items-center gap-2 text-xs" style={{ color: "var(--text-secondary)" }}>
                                                                <ChevronRight className="w-3 h-3 flex-shrink-0" style={{ color: "#fb7185" }} />
                                                                {w}
                                                            </li>
                                                        ))}
                                                    </ul>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </AnimatedSection>
                            );
                        })}
                    </div>
                </div>

                {/* CTA */}
                <AnimatedSection className="mt-20 text-center">
                    <div className="glass rounded-2xl p-10 relative overflow-hidden">
                        <div
                            className="absolute inset-0 pointer-events-none"
                            style={{ background: "radial-gradient(ellipse at 50% 0%, rgba(167,139,250,0.1) 0%, transparent 60%)" }}
                        />
                        <div className="relative z-10">
                            <h2
                                className="text-2xl font-bold mb-3"
                                style={{ fontFamily: "var(--font-poppins)", color: "var(--text-primary)" }}
                            >
                                Experience Agentic RAG in Action
                            </h2>
                            <p className="text-sm mb-6" style={{ color: "var(--text-secondary)" }}>
                                Upload your research papers and watch the multi-agent pipeline work.
                            </p>
                            <a
                                href="/chat"
                                className="inline-flex items-center gap-2 px-7 py-3 rounded-xl text-white text-sm font-semibold"
                                style={{ background: "linear-gradient(135deg, #6366f1, #a78bfa)", boxShadow: "0 0 24px rgba(99,102,241,0.35)" }}
                            >
                                Open Research Chat
                                <ArrowRight className="w-4 h-4" />
                            </a>
                        </div>
                    </div>
                </AnimatedSection>
            </div>
        </div>
    );
}
