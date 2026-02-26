"use client";

import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
    Upload,
    FileText,
    CheckCircle2,
    Loader2,
    Database,
    X,
    CloudUpload,
} from "lucide-react";
import AnimatedSection from "@/components/motion/AnimatedSection";
import Badge from "@/components/ui/Badge";

type FileStatus = "idle" | "uploading" | "indexing" | "embedded" | "error";

interface UploadedFile {
    id: string;
    name: string;
    size: number;
    status: FileStatus;
    progress: number;
}

function formatBytes(bytes: number) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

const statusConfig: Record<FileStatus, { label: string; color: "cyan" | "amber" | "green" | "rose" | "indigo" }> = {
    idle: { label: "Queued", color: "indigo" },
    uploading: { label: "Uploading", color: "cyan" },
    indexing: { label: "Indexing", color: "amber" },
    embedded: { label: "Embedded ✓", color: "green" },
    error: { label: "Error", color: "rose" },
};

function simulateUpload(
    file: UploadedFile,
    onUpdate: (id: string, patch: Partial<UploadedFile>) => void
) {
    // Stage 1: Uploading
    onUpdate(file.id, { status: "uploading", progress: 0 });
    let progress = 0;
    const uploadInterval = setInterval(() => {
        progress += Math.random() * 18 + 8;
        if (progress >= 100) {
            clearInterval(uploadInterval);
            onUpdate(file.id, { status: "indexing", progress: 100 });

            // Stage 2: Indexing
            let ndx = 0;
            const indexInterval = setInterval(() => {
                ndx += Math.random() * 20 + 10;
                if (ndx >= 100) {
                    clearInterval(indexInterval);
                    onUpdate(file.id, { status: "embedded", progress: 100 });
                } else {
                    onUpdate(file.id, { progress: ndx });
                }
            }, 200);
        } else {
            onUpdate(file.id, { progress });
        }
    }, 150);
}

export default function UploadPage() {
    const [files, setFiles] = useState<UploadedFile[]>([]);
    const [dragging, setDragging] = useState(false);

    const handleFiles = (incomingFiles: FileList | null) => {
        if (!incomingFiles) return;
        Array.from(incomingFiles).forEach((f) => {
            const newFile: UploadedFile = {
                id: crypto.randomUUID(),
                name: f.name,
                size: f.size,
                status: "idle",
                progress: 0,
            };
            setFiles((prev) => [...prev, newFile]);
            setTimeout(() => simulateUpload(newFile, (id, patch) => {
                setFiles((prev) => prev.map((x) => (x.id === id ? { ...x, ...patch } : x)));
            }), 100);
        });
    };

    const onDrop = useCallback(
        (e: React.DragEvent) => {
            e.preventDefault();
            setDragging(false);
            handleFiles(e.dataTransfer.files);
        },
        []
    );

    const removeFile = (id: string) => setFiles((prev) => prev.filter((f) => f.id !== id));

    return (
        <div className="min-h-screen pt-12 pb-28 px-6 relative overflow-hidden" style={{ background: "var(--bg-primary)" }}>
            {/* Pulsing node grid background */}
            <div className="node-bg-grid absolute inset-0 opacity-40 pointer-events-none" />

            {/* Radial glow */}
            <div
                className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[400px] rounded-full pointer-events-none"
                style={{
                    background: "radial-gradient(ellipse, rgba(99,102,241,0.1) 0%, transparent 70%)",
                    filter: "blur(30px)",
                }}
            />

            <div className="max-w-3xl mx-auto relative z-10">
                {/* Header */}
                <AnimatedSection className="text-center mb-12">
                    <div
                        className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full mb-5 text-xs font-medium"
                        style={{
                            background: "rgba(34,211,238,0.1)",
                            border: "1px solid rgba(34,211,238,0.2)",
                            color: "var(--accent-cyan)",
                        }}
                    >
                        <Database className="w-3.5 h-3.5" />
                        Knowledge Ingestion Pipeline
                    </div>
                    <h1
                        className="text-4xl font-bold mb-3"
                        style={{ fontFamily: "var(--font-poppins)", color: "var(--text-primary)" }}
                    >
                        Upload & Index Research
                    </h1>
                    <p className="text-base max-w-lg mx-auto" style={{ color: "var(--text-secondary)" }}>
                        Upload your PDFs or PPTs. Our pipeline will chunk, embed, and index them into the vector knowledge base.
                    </p>
                </AnimatedSection>

                {/* Dropzone */}
                <AnimatedSection delay={0.1}>
                    <div
                        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
                        onDragLeave={() => setDragging(false)}
                        onDrop={onDrop}
                        className="relative rounded-2xl border-2 border-dashed p-12 text-center transition-all duration-300 cursor-pointer group"
                        style={{
                            borderColor: dragging ? "var(--accent-indigo)" : "rgba(255,255,255,0.12)",
                            background: dragging
                                ? "rgba(99,102,241,0.08)"
                                : "rgba(255,255,255,0.02)",
                            boxShadow: dragging ? "0 0 32px rgba(99,102,241,0.2)" : "none",
                        }}
                        onClick={() => document.getElementById("file-input")?.click()}
                    >
                        <input
                            id="file-input"
                            type="file"
                            className="hidden"
                            multiple
                            accept=".pdf,.ppt,.pptx,.docx"
                            onChange={(e) => handleFiles(e.target.files)}
                        />

                        <motion.div
                            animate={{ y: dragging ? -6 : 0 }}
                            transition={{ type: "spring", stiffness: 300 }}
                            className="flex flex-col items-center gap-4"
                        >
                            <div
                                className="w-20 h-20 rounded-2xl flex items-center justify-center mb-2"
                                style={{
                                    background: "rgba(99,102,241,0.1)",
                                    border: "1px solid rgba(99,102,241,0.2)",
                                    boxShadow: dragging ? "0 0 24px rgba(99,102,241,0.3)" : "none",
                                }}
                            >
                                <CloudUpload
                                    className="w-9 h-9 transition-colors duration-300"
                                    style={{ color: dragging ? "var(--accent-cyan)" : "var(--accent-indigo)" }}
                                />
                            </div>
                            <div>
                                <p
                                    className="text-lg font-semibold mb-1"
                                    style={{ color: "var(--text-primary)" }}
                                >
                                    {dragging ? "Release to upload" : "Drag & drop files here"}
                                </p>
                                <p className="text-sm" style={{ color: "var(--text-muted)" }}>
                                    or click to browse — PDF, PPT, PPTX, DOCX supported
                                </p>
                            </div>
                        </motion.div>
                    </div>
                </AnimatedSection>

                {/* File List */}
                <AnimatePresence>
                    {files.length > 0 && (
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            className="mt-6 space-y-3"
                        >
                            <p className="text-sm font-medium mb-4" style={{ color: "var(--text-secondary)" }}>
                                {files.length} file{files.length > 1 ? "s" : ""} in pipeline
                            </p>

                            {files.map((file) => {
                                const cfg = statusConfig[file.status];
                                const isComplete = file.status === "embedded";
                                const isProcessing = file.status === "uploading" || file.status === "indexing";

                                return (
                                    <motion.div
                                        key={file.id}
                                        layout
                                        initial={{ opacity: 0, x: -20 }}
                                        animate={{ opacity: 1, x: 0 }}
                                        exit={{ opacity: 0, x: 20 }}
                                        className="glass rounded-xl p-4 flex items-center gap-4"
                                    >
                                        {/* File icon */}
                                        <div
                                            className="w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0"
                                            style={{ background: "rgba(255,255,255,0.04)" }}
                                        >
                                            {isComplete ? (
                                                <CheckCircle2 className="w-5 h-5" style={{ color: "#4ade80" }} />
                                            ) : isProcessing ? (
                                                <Loader2 className="w-5 h-5 animate-spin" style={{ color: "var(--accent-cyan)" }} />
                                            ) : (
                                                <FileText className="w-5 h-5" style={{ color: "var(--accent-indigo)" }} />
                                            )}
                                        </div>

                                        {/* Info + progress */}
                                        <div className="flex-1 min-w-0">
                                            <div className="flex items-center justify-between mb-1">
                                                <span
                                                    className="text-sm font-medium truncate"
                                                    style={{ color: "var(--text-primary)" }}
                                                >
                                                    {file.name}
                                                </span>
                                                <Badge color={cfg.color} className="ml-3 flex-shrink-0">
                                                    {cfg.label}
                                                </Badge>
                                            </div>
                                            <div className="flex items-center gap-2">
                                                <div
                                                    className="flex-1 h-1.5 rounded-full overflow-hidden"
                                                    style={{ background: "rgba(255,255,255,0.06)" }}
                                                >
                                                    <motion.div
                                                        className="h-full rounded-full"
                                                        style={{
                                                            background: isComplete
                                                                ? "linear-gradient(90deg, #4ade80, #22d3ee)"
                                                                : "linear-gradient(90deg, #6366f1, #22d3ee)",
                                                        }}
                                                        animate={{ width: `${file.progress}%` }}
                                                        transition={{ duration: 0.3 }}
                                                    />
                                                </div>
                                                <span className="text-xs w-8 text-right flex-shrink-0" style={{ color: "var(--text-muted)" }}>
                                                    {Math.round(file.progress)}%
                                                </span>
                                            </div>
                                            <p className="text-xs mt-0.5" style={{ color: "var(--text-muted)" }}>
                                                {formatBytes(file.size)}
                                            </p>
                                        </div>

                                        {/* Remove */}
                                        {isComplete && (
                                            <button
                                                onClick={() => removeFile(file.id)}
                                                className="p-1.5 rounded-lg transition-colors"
                                                style={{ color: "var(--text-muted)" }}
                                            >
                                                <X className="w-4 h-4" />
                                            </button>
                                        )}
                                    </motion.div>
                                );
                            })}

                            {/* Summary */}
                            {files.some((f) => f.status === "embedded") && (
                                <motion.div
                                    initial={{ opacity: 0 }}
                                    animate={{ opacity: 1 }}
                                    className="glass rounded-xl p-4 flex items-center gap-3"
                                    style={{ border: "1px solid rgba(74,222,128,0.2)", background: "rgba(74,222,128,0.04)" }}
                                >
                                    <CheckCircle2 className="w-5 h-5 flex-shrink-0" style={{ color: "#4ade80" }} />
                                    <div>
                                        <p className="text-sm font-medium" style={{ color: "#4ade80" }}>
                                            {files.filter((f) => f.status === "embedded").length} document
                                            {files.filter((f) => f.status === "embedded").length > 1 ? "s" : ""} embedded into vector DB
                                        </p>
                                        <p className="text-xs" style={{ color: "var(--text-muted)" }}>
                                            Ready for semantic retrieval in the Research Chat
                                        </p>
                                    </div>
                                </motion.div>
                            )}
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* Pipeline steps info */}
                <AnimatedSection delay={0.2} className="mt-10">
                    <div className="glass rounded-2xl p-6">
                        <h3 className="text-sm font-semibold mb-4" style={{ color: "var(--text-secondary)" }}>
                            Ingestion Pipeline
                        </h3>
                        <div className="flex flex-col sm:flex-row gap-4">
                            {[
                                { step: "01", title: "Parse", desc: "Extract text & structure from PDF/PPT" },
                                { step: "02", title: "Chunk", desc: "Semantic chunking with overlap" },
                                { step: "03", title: "Embed", desc: "Dense vector embeddings via model" },
                                { step: "04", title: "Index", desc: "Store in vector DB with metadata" },
                            ].map(({ step, title, desc }) => (
                                <div key={step} className="flex-1 flex items-start gap-3">
                                    <span
                                        className="text-xs font-bold mt-0.5 flex-shrink-0"
                                        style={{ color: "var(--accent-indigo)" }}
                                    >
                                        {step}
                                    </span>
                                    <div>
                                        <div className="text-sm font-medium mb-0.5" style={{ color: "var(--text-primary)" }}>
                                            {title}
                                        </div>
                                        <div className="text-xs leading-snug" style={{ color: "var(--text-muted)" }}>
                                            {desc}
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </AnimatedSection>
            </div>
        </div>
    );
}
