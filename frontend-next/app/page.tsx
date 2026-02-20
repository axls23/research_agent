"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import {
  Upload,
  MessageSquare,
  Search,
  Zap,
  Database,
  GitBranch,
  ArrowRight,
  Sparkles,
  BookOpen,
  Brain,
  Network,
} from "lucide-react";
import AnimatedSection from "@/components/motion/AnimatedSection";

const featureCards = [
  {
    icon: Search,
    color: "var(--accent-cyan)",
    glowColor: "rgba(34,211,238,0.15)",
    title: "Retrieval",
    subtitle: "Semantic Search",
    description:
      "Dense vector retrieval with hybrid BM25 + embedding search across your knowledge corpus.",
  },
  {
    icon: Zap,
    color: "#f59e0b",
    glowColor: "rgba(245,158,11,0.15)",
    title: "Augmentation",
    subtitle: "Context Fusion",
    description:
      "Intelligently fuses retrieved context with the question, respecting semantic relevance scores.",
  },
  {
    icon: Brain,
    color: "var(--accent-violet)",
    glowColor: "rgba(167,139,250,0.15)",
    title: "Generation",
    subtitle: "LLM Synthesis",
    description:
      "State-of-the-art language model generates precise, cited answers grounded in your documents.",
  },
  {
    icon: Network,
    color: "#4ade80",
    glowColor: "rgba(74,222,128,0.15)",
    title: "Agentic Workflow",
    subtitle: "Autonomous Pipeline",
    description:
      "Self-directing agents plan, retrieve, refine, and verify — without manual intervention.",
  },
];

const stats = [
  { value: "5×", label: "RAG paradigms supported" },
  { value: "< 2s", label: "Average retrieval latency" },
  { value: "98%", label: "Citation accuracy" },
  { value: "∞", label: "Knowledge sources" },
];

export default function LandingPage() {
  return (
    <div className="animated-gradient min-h-screen">
      {/* ── Hero ── */}
      <section className="relative overflow-hidden pt-28 pb-24 px-6">
        {/* Background orbs */}
        <div
          className="absolute top-[-20%] left-[10%] w-[600px] h-[600px] rounded-full pointer-events-none"
          style={{
            background: "radial-gradient(circle, rgba(99,102,241,0.12) 0%, transparent 70%)",
            filter: "blur(40px)",
          }}
        />
        <div
          className="absolute top-[10%] right-[5%] w-[400px] h-[400px] rounded-full pointer-events-none"
          style={{
            background: "radial-gradient(circle, rgba(34,211,238,0.08) 0%, transparent 70%)",
            filter: "blur(40px)",
          }}
        />

        <div className="max-w-5xl mx-auto text-center relative z-10">
          {/* Pill badge */}
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full mb-8 text-xs font-medium"
            style={{
              background: "rgba(99,102,241,0.12)",
              border: "1px solid rgba(99,102,241,0.25)",
              color: "#818cf8",
            }}
          >
            <Sparkles className="w-3.5 h-3.5" />
            Powered by Agentic RAG · Research-Grade AI
          </motion.div>

          {/* Main Title */}
          <h1
            className="text-5xl md:text-7xl font-bold mb-6 leading-tight tracking-tight"
            style={{ fontFamily: "var(--font-poppins)" }}
          >
            <motion.span
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.7, delay: 0.1 }}
              className="block"
              style={{ color: "var(--text-primary)" }}
            >
              Agentic RAG
            </motion.span>
            <motion.span
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.7, delay: 0.25 }}
              className="block"
              style={{
                background: "linear-gradient(135deg, #6366f1, #22d3ee, #a78bfa)",
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
                backgroundClip: "text",
              }}
            >
              Research Assistant
            </motion.span>
          </h1>

          {/* Subtitle */}
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.4 }}
            className="text-lg md:text-xl max-w-2xl mx-auto mb-10 leading-relaxed"
            style={{ color: "var(--text-secondary)" }}
          >
            Autonomous, context-aware AI for academic research. Upload papers, ask complex queries,
            and watch multi-agent systems retrieve, reason, and synthesize answers with citations.
          </motion.p>

          {/* CTA buttons */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.55 }}
            className="flex flex-col sm:flex-row items-center justify-center gap-4"
          >
            <Link
              href="/upload"
              className="inline-flex items-center gap-2.5 px-7 py-3.5 rounded-xl text-white font-semibold text-sm transition-all duration-200 hover:brightness-110 hover:scale-[1.02] active:scale-[0.98]"
              style={{
                background: "linear-gradient(135deg, #6366f1, #22d3ee)",
                boxShadow: "0 0 32px rgba(99,102,241,0.4)",
              }}
            >
              <Upload className="w-4 h-4" />
              Upload Research Papers
            </Link>
            <Link
              href="/chat"
              className="inline-flex items-center gap-2.5 px-7 py-3.5 rounded-xl font-semibold text-sm transition-all duration-200 hover:bg-white/8 hover:scale-[1.02] active:scale-[0.98]"
              style={{
                border: "1px solid rgba(255,255,255,0.15)",
                color: "var(--text-primary)",
              }}
            >
              <MessageSquare className="w-4 h-4" />
              Ask Research Query
              <ArrowRight className="w-3.5 h-3.5 ml-0.5" />
            </Link>
          </motion.div>
        </div>
      </section>

      {/* ── Stats ── */}
      <AnimatedSection className="px-6 pb-20">
        <div className="max-w-4xl mx-auto grid grid-cols-2 md:grid-cols-4 gap-4">
          {stats.map(({ value, label }, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5, delay: 0.1 * i }}
              className="glass rounded-2xl p-5 text-center"
            >
              <div
                className="text-3xl font-bold mb-1"
                style={{
                  fontFamily: "var(--font-poppins)",
                  background: "linear-gradient(135deg, #6366f1, #22d3ee)",
                  WebkitBackgroundClip: "text",
                  WebkitTextFillColor: "transparent",
                  backgroundClip: "text",
                }}
              >
                {value}
              </div>
              <div className="text-xs" style={{ color: "var(--text-muted)" }}>
                {label}
              </div>
            </motion.div>
          ))}
        </div>
      </AnimatedSection>

      {/* ── Feature Cards ── */}
      <section className="px-6 pb-28">
        <AnimatedSection className="max-w-7xl mx-auto">
          <div className="text-center mb-14">
            <h2
              className="text-3xl md:text-4xl font-bold mb-4"
              style={{ fontFamily: "var(--font-poppins)", color: "var(--text-primary)" }}
            >
              How Agentic RAG Works
            </h2>
            <p className="text-base max-w-xl mx-auto" style={{ color: "var(--text-secondary)" }}>
              Four interconnected pillars that transform raw documents into research-grade answers.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {featureCards.map(({ icon: Icon, color, glowColor, title, subtitle, description }, i) => (
              <AnimatedSection key={i} delay={i * 0.1} direction="up">
                <div
                  className="glass glass-hover rounded-2xl p-6 h-full flex flex-col gap-4 group"
                  style={{ position: "relative", overflow: "hidden" }}
                >
                  {/* Number */}
                  <span
                    className="absolute top-4 right-4 text-5xl font-black opacity-5"
                    style={{ fontFamily: "var(--font-poppins)", color }}
                  >
                    {i + 1}
                  </span>

                  {/* Icon */}
                  <div
                    className="w-11 h-11 rounded-xl flex items-center justify-center transition-all duration-300 group-hover:scale-110"
                    style={{ background: glowColor, boxShadow: `0 0 16px ${glowColor}` }}
                  >
                    <Icon className="w-5 h-5" style={{ color }} />
                  </div>

                  {/* Text */}
                  <div>
                    <div className="text-xs font-medium mb-1" style={{ color: "var(--text-muted)" }}>
                      {subtitle}
                    </div>
                    <h3
                      className="text-lg font-semibold mb-2"
                      style={{ fontFamily: "var(--font-poppins)", color: "var(--text-primary)" }}
                    >
                      {title}
                    </h3>
                    <p className="text-sm leading-relaxed" style={{ color: "var(--text-secondary)" }}>
                      {description}
                    </p>
                  </div>
                </div>
              </AnimatedSection>
            ))}
          </div>
        </AnimatedSection>
      </section>

      {/* ── How to Use ── */}
      <section className="px-6 pb-28">
        <AnimatedSection className="max-w-4xl mx-auto">
          <div className="glass rounded-3xl p-10 md:p-14 text-center relative overflow-hidden">
            <div
              className="absolute inset-0 pointer-events-none"
              style={{
                background:
                  "radial-gradient(ellipse at 50% 0%, rgba(99,102,241,0.12) 0%, transparent 60%)",
              }}
            />
            <div className="relative z-10">
              <div className="flex justify-center gap-10 mb-10 flex-wrap">
                {[
                  { icon: Upload, label: "1. Upload Papers", color: "var(--accent-cyan)" },
                  { icon: Database, label: "2. Index & Embed", color: "var(--accent-violet)" },
                  { icon: MessageSquare, label: "3. Ask Questions", color: "#4ade80" },
                  { icon: BookOpen, label: "4. Get Cited Answers", color: "#f59e0b" },
                ].map(({ icon: Icon, label, color }, i) => (
                  <div key={i} className="flex flex-col items-center gap-3">
                    <div
                      className="w-14 h-14 rounded-2xl flex items-center justify-center"
                      style={{ background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)" }}
                    >
                      <Icon className="w-6 h-6" style={{ color }} />
                    </div>
                    <span className="text-xs font-medium" style={{ color: "var(--text-secondary)" }}>
                      {label}
                    </span>
                  </div>
                ))}
              </div>
              <h2
                className="text-2xl md:text-3xl font-bold mb-4"
                style={{ fontFamily: "var(--font-poppins)", color: "var(--text-primary)" }}
              >
                Ready to accelerate your research?
              </h2>
              <p className="text-base mb-8" style={{ color: "var(--text-secondary)" }}>
                Upload your documents and start asking questions powered by multi-agent AI.
              </p>
              <Link
                href="/upload"
                className="inline-flex items-center gap-2 px-8 py-3.5 rounded-xl text-white font-semibold text-sm"
                style={{
                  background: "linear-gradient(135deg, #6366f1, #22d3ee)",
                  boxShadow: "0 0 28px rgba(99,102,241,0.4)",
                }}
              >
                <GitBranch className="w-4 h-4" />
                Start With Agentic RAG
              </Link>
            </div>
          </div>
        </AnimatedSection>
      </section>
    </div>
  );
}
