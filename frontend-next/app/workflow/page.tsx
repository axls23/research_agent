"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import AnimatedSection from "@/components/motion/AnimatedSection";
import Badge from "@/components/ui/Badge";

interface Node {
    id: string;
    label: string;
    sublabel: string;
    x: number;
    y: number;
    color: string;
    glow: string;
    badge: "cyan" | "violet" | "green" | "amber" | "indigo";
    tooltip: string;
}

interface Edge {
    from: string;
    to: string;
}

const NODES: Node[] = [
    {
        id: "input",
        label: "Research Query",
        sublabel: "User Input",
        x: 320,
        y: 30,
        color: "#6366f1",
        glow: "rgba(99,102,241,0.4)",
        badge: "indigo",
        tooltip: "The user's complex research question, potentially multi-hop and requiring multiple sub-tasks.",
    },
    {
        id: "planner",
        label: "Task Planner",
        sublabel: "Decomposition",
        x: 320,
        y: 160,
        color: "#a78bfa",
        glow: "rgba(167,139,250,0.4)",
        badge: "violet",
        tooltip: "Breaks complex queries into sub-tasks. Decides retrieval strategy, depth, and iteration count.",
    },
    {
        id: "retriever",
        label: "Multi-Retriever",
        sublabel: "Vector + BM25",
        x: 120,
        y: 300,
        color: "#22d3ee",
        glow: "rgba(34,211,238,0.4)",
        badge: "cyan",
        tooltip: "Performs hybrid dense + sparse retrieval across the indexed vector database. Re-ranks by relevance.",
    },
    {
        id: "memory",
        label: "Agent Memory",
        sublabel: "Context Store",
        x: 520,
        y: 300,
        color: "#f59e0b",
        glow: "rgba(245,158,11,0.4)",
        badge: "amber",
        tooltip: "Maintains conversation history, retrieved context, and intermediate reasoning across iterations.",
    },
    {
        id: "refiner",
        label: "Answer Refiner",
        sublabel: "Self-Critique",
        x: 320,
        y: 440,
        color: "#4ade80",
        glow: "rgba(74,222,128,0.4)",
        badge: "green",
        tooltip: "Evaluates draft answers for completeness and accuracy. Triggers another retrieval cycle if needed.",
    },
    {
        id: "generator",
        label: "LLM Generator",
        sublabel: "Grounded Answer",
        x: 320,
        y: 570,
        color: "#6366f1",
        glow: "rgba(99,102,241,0.4)",
        badge: "indigo",
        tooltip: "Generates the final answer with in-line citations, grounded exclusively in retrieved evidence.",
    },
];

const EDGES: Edge[] = [
    { from: "input", to: "planner" },
    { from: "planner", to: "retriever" },
    { from: "planner", to: "memory" },
    { from: "retriever", to: "refiner" },
    { from: "memory", to: "refiner" },
    { from: "refiner", to: "generator" },
    { from: "refiner", to: "retriever" }, // feedback loop
];

function getNodeCenter(node: Node) {
    return { x: node.x + 80, y: node.y + 32 };
}

export default function WorkflowPage() {
    const [hovered, setHovered] = useState<string | null>(null);
    const hoveredNode = NODES.find((n) => n.id === hovered);

    return (
        <div
            className="min-h-screen px-6 pt-12 pb-28 relative overflow-hidden"
            style={{ background: "var(--bg-primary)" }}
        >
            {/* Grid background */}
            <div className="node-bg-grid absolute inset-0 opacity-30 pointer-events-none" />

            {/* Glow */}
            <div
                className="absolute left-1/2 top-0 -translate-x-1/2 w-[800px] h-[400px] pointer-events-none"
                style={{
                    background: "radial-gradient(ellipse, rgba(167,139,250,0.08) 0%, transparent 70%)",
                    filter: "blur(40px)",
                }}
            />

            <div className="max-w-5xl mx-auto relative z-10">
                {/* Header */}
                <AnimatedSection className="text-center mb-16">
                    <div
                        className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full mb-5 text-xs font-medium"
                        style={{ background: "rgba(167,139,250,0.1)", border: "1px solid rgba(167,139,250,0.2)", color: "#a78bfa" }}
                    >
                        Agentic Workflow Visualization
                    </div>
                    <h1
                        className="text-4xl font-bold mb-4"
                        style={{ fontFamily: "var(--font-poppins)", color: "var(--text-primary)" }}
                    >
                        Multi-Agent Pipeline
                    </h1>
                    <p className="text-base max-w-lg mx-auto" style={{ color: "var(--text-secondary)" }}>
                        Hover over any node to understand its role in the autonomous research pipeline.
                    </p>
                </AnimatedSection>

                {/* Diagram + Tooltip */}
                <div className="flex flex-col lg:flex-row gap-10 items-center lg:items-start">
                    {/* SVG Diagram */}
                    <div className="flex-1 w-full max-w-2xl mx-auto">
                        <svg
                            viewBox="0 0 640 660"
                            className="w-full"
                            style={{ overflow: "visible" }}
                        >
                            {/* Edges */}
                            {EDGES.map((edge, i) => {
                                const fromNode = NODES.find((n) => n.id === edge.from)!;
                                const toNode = NODES.find((n) => n.id === edge.to)!;
                                const from = getNodeCenter(fromNode);
                                const to = getNodeCenter(toNode);
                                const mx = (from.x + to.x) / 2;
                                const my = (from.y + to.y) / 2;
                                const isLoop = edge.from === "refiner" && edge.to === "retriever";
                                const d = isLoop
                                    ? `M${from.x},${from.y} C${from.x - 80},${from.y} ${to.x - 80},${to.y} ${to.x},${to.y}`
                                    : `M${from.x},${from.y} Q${mx},${my} ${to.x},${to.y}`;

                                return (
                                    <g key={i}>
                                        <path
                                            d={d}
                                            fill="none"
                                            stroke="rgba(255,255,255,0.08)"
                                            strokeWidth="1.5"
                                        />
                                        <path
                                            d={d}
                                            fill="none"
                                            stroke="url(#edgeGrad)"
                                            strokeWidth="1.5"
                                            strokeDasharray="6 6"
                                            className="animated-dash"
                                            opacity="0.6"
                                        />
                                        {/* Arrow */}
                                        <circle cx={to.x} cy={to.y} r="3" fill="rgba(99,102,241,0.7)" />
                                    </g>
                                );
                            })}

                            {/* Gradient defs */}
                            <defs>
                                <linearGradient id="edgeGrad" x1="0" y1="0" x2="1" y2="1">
                                    <stop offset="0%" stopColor="#6366f1" />
                                    <stop offset="100%" stopColor="#22d3ee" />
                                </linearGradient>
                            </defs>

                            {/* Nodes */}
                            {NODES.map((node) => {
                                const isHov = hovered === node.id;
                                return (
                                    <g
                                        key={node.id}
                                        transform={`translate(${node.x}, ${node.y})`}
                                        onMouseEnter={() => setHovered(node.id)}
                                        onMouseLeave={() => setHovered(null)}
                                        style={{ cursor: "pointer" }}
                                    >
                                        {/* Glow ring on hover */}
                                        {isHov && (
                                            <rect
                                                x="-6" y="-6" width="172" height="76"
                                                rx="18"
                                                fill="transparent"
                                                stroke={node.color}
                                                strokeWidth="2"
                                                opacity="0.6"
                                                filter={`drop-shadow(0 0 12px ${node.glow})`}
                                            />
                                        )}

                                        {/* Card */}
                                        <rect
                                            x="0" y="0" width="160" height="64"
                                            rx="14"
                                            fill={isHov ? `${node.color}22` : "rgba(12,15,30,0.9)"}
                                            stroke={isHov ? node.color : "rgba(255,255,255,0.1)"}
                                            strokeWidth="1.5"
                                        />

                                        {/* Color accent left bar */}
                                        <rect x="0" y="0" width="4" height="64" rx="2" fill={node.color} opacity="0.8" />

                                        {/* Label */}
                                        <text
                                            x="20" y="26"
                                            fontSize="11"
                                            fontWeight="600"
                                            fill={isHov ? node.color : "#e8eaf6"}
                                            fontFamily="Inter, sans-serif"
                                        >
                                            {node.label}
                                        </text>
                                        <text
                                            x="20" y="44"
                                            fontSize="9"
                                            fill="rgba(136,146,176,0.9)"
                                            fontFamily="Inter, sans-serif"
                                        >
                                            {node.sublabel}
                                        </text>
                                    </g>
                                );
                            })}
                        </svg>
                    </div>

                    {/* Tooltip / Info Panel */}
                    <div className="w-full lg:w-64 xl:w-72 flex-shrink-0">
                        <div
                            className="glass rounded-2xl p-5 sticky top-24 transition-all duration-300"
                            style={{ minHeight: "180px" }}
                        >
                            {hoveredNode ? (
                                <motion.div
                                    key={hoveredNode.id}
                                    initial={{ opacity: 0, y: 10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ duration: 0.25 }}
                                >
                                    <div className="flex items-center gap-2 mb-3">
                                        <div
                                            className="w-2.5 h-2.5 rounded-full"
                                            style={{ background: hoveredNode.color, boxShadow: `0 0 8px ${hoveredNode.glow}` }}
                                        />
                                        <Badge color={hoveredNode.badge}>{hoveredNode.sublabel}</Badge>
                                    </div>
                                    <h3
                                        className="text-base font-semibold mb-3"
                                        style={{ fontFamily: "var(--font-poppins)", color: "var(--text-primary)" }}
                                    >
                                        {hoveredNode.label}
                                    </h3>
                                    <p className="text-sm leading-relaxed" style={{ color: "var(--text-secondary)" }}>
                                        {hoveredNode.tooltip}
                                    </p>
                                </motion.div>
                            ) : (
                                <p className="text-sm text-center py-8" style={{ color: "var(--text-muted)" }}>
                                    Hover a node to see its description
                                </p>
                            )}
                        </div>

                        {/* Legend */}
                        <div className="glass rounded-2xl p-4 mt-4 space-y-2.5">
                            <p className="text-xs font-semibold uppercase tracking-wider mb-3" style={{ color: "var(--text-muted)" }}>
                                Legend
                            </p>
                            {[
                                { color: "#6366f1", label: "Orchestration" },
                                { color: "#a78bfa", label: "Planning" },
                                { color: "#22d3ee", label: "Retrieval" },
                                { color: "#f59e0b", label: "Memory" },
                                { color: "#4ade80", label: "Refinement" },
                            ].map(({ color, label }) => (
                                <div key={label} className="flex items-center gap-2">
                                    <div className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ background: color }} />
                                    <span className="text-xs" style={{ color: "var(--text-secondary)" }}>{label}</span>
                                </div>
                            ))}
                            <div className="flex items-center gap-2 pt-1">
                                <svg width="24" height="4"><line x1="0" y1="2" x2="16" y2="2" stroke="#6366f1" strokeWidth="1.5" strokeDasharray="3 3" /><circle cx="20" cy="2" r="2" fill="#6366f1" /></svg>
                                <span className="text-xs" style={{ color: "var(--text-secondary)" }}>Data flow</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
