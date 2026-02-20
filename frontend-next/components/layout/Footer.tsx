import Link from "next/link";
import { BrainCircuit, Github, ExternalLink } from "lucide-react";

export default function Footer() {
    return (
        <footer
            className="glass border-t mt-20"
            style={{ borderColor: "var(--border-card)" }}
        >
            <div className="max-w-7xl mx-auto px-6 py-10 flex flex-col md:flex-row items-center justify-between gap-6">
                {/* Brand */}
                <div className="flex items-center gap-2.5">
                    <div
                        className="w-7 h-7 rounded-md flex items-center justify-center"
                        style={{ background: "linear-gradient(135deg, #6366f1, #22d3ee)" }}
                    >
                        <BrainCircuit className="w-3.5 h-3.5 text-white" />
                    </div>
                    <span className="text-sm font-medium" style={{ color: "var(--text-secondary)" }}>
                        Agentic RAG Research Assistant
                    </span>
                </div>

                {/* Links */}
                <nav className="flex items-center gap-6">
                    {[
                        { href: "/", label: "Home" },
                        { href: "/about", label: "Architecture" },
                        { href: "/workflow", label: "Workflow" },
                    ].map(({ href, label }) => (
                        <Link
                            key={href}
                            href={href}
                            className="text-xs transition-colors duration-200 hover:text-white"
                            style={{ color: "var(--text-muted)" }}
                        >
                            {label}
                        </Link>
                    ))}
                    <a
                        href="https://github.com"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex items-center gap-1 text-xs transition-colors duration-200 hover:text-white"
                        style={{ color: "var(--text-muted)" }}
                    >
                        <Github className="w-3.5 h-3.5" />
                        GitHub
                    </a>
                </nav>

                <p className="text-xs" style={{ color: "var(--text-muted)" }}>
                    © 2026 Agentic RAG · Research Lab
                </p>
            </div>
        </footer>
    );
}
