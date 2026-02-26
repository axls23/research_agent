"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { BrainCircuit, Menu, X, Upload, MessageSquare, GitBranch, BookOpen } from "lucide-react";
import { cn } from "@/lib/utils";

const navLinks = [
    { href: "/", label: "Home" },
    { href: "/upload", label: "Upload", icon: Upload },
    { href: "/chat", label: "Research Chat", icon: MessageSquare },
    { href: "/workflow", label: "Workflow", icon: GitBranch },
    { href: "/about", label: "Architecture", icon: BookOpen },
];

export default function Navbar() {
    const pathname = usePathname();
    const [mobileOpen, setMobileOpen] = useState(false);

    return (
        <header
            className="sticky top-0 z-50 glass border-b"
            style={{ borderColor: "var(--border-card)" }}
        >
            <nav className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
                {/* Logo */}
                <Link href="/" className="flex items-center gap-2.5 group">
                    <div
                        className="w-8 h-8 rounded-lg flex items-center justify-center"
                        style={{ background: "linear-gradient(135deg, #6366f1, #22d3ee)" }}
                    >
                        <BrainCircuit className="w-4 h-4 text-white" />
                    </div>
                    <span
                        className="font-semibold text-sm hidden sm:block"
                        style={{ fontFamily: "var(--font-poppins)", color: "var(--text-primary)" }}
                    >
                        AgentRAG
                    </span>
                </Link>

                {/* Desktop Links */}
                <ul className="hidden md:flex items-center gap-1">
                    {navLinks.map(({ href, label }) => {
                        const active = pathname === href;
                        return (
                            <li key={href}>
                                <Link
                                    href={href}
                                    className={cn(
                                        "relative px-4 py-2 rounded-lg text-sm font-medium transition-colors duration-200",
                                        active ? "text-white" : "hover:text-white"
                                    )}
                                    style={{ color: active ? "var(--accent-cyan)" : "var(--text-secondary)" }}
                                >
                                    {active && (
                                        <motion.span
                                            layoutId="pill"
                                            className="absolute inset-0 rounded-lg"
                                            style={{ background: "rgba(99,102,241,0.12)", border: "1px solid rgba(99,102,241,0.25)" }}
                                        />
                                    )}
                                    <span className="relative z-10">{label}</span>
                                </Link>
                            </li>
                        );
                    })}
                </ul>

                {/* CTA + Mobile Toggle */}
                <div className="flex items-center gap-3">
                    <Link
                        href="/chat"
                        className="hidden md:inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium text-white transition-all duration-200"
                        style={{
                            background: "linear-gradient(135deg, #6366f1, #22d3ee)",
                            boxShadow: "0 0 20px rgba(99,102,241,0.3)",
                        }}
                    >
                        <MessageSquare className="w-3.5 h-3.5" />
                        Start Research
                    </Link>
                    <button
                        className="md:hidden p-2 rounded-lg"
                        style={{ color: "var(--text-secondary)" }}
                        onClick={() => setMobileOpen((v) => !v)}
                        aria-label="Toggle menu"
                    >
                        {mobileOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
                    </button>
                </div>
            </nav>

            {/* Mobile Menu */}
            <AnimatePresence>
                {mobileOpen && (
                    <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: "auto" }}
                        exit={{ opacity: 0, height: 0 }}
                        className="md:hidden overflow-hidden glass border-t"
                        style={{ borderColor: "var(--border-card)" }}
                    >
                        <ul className="px-6 py-4 flex flex-col gap-2">
                            {navLinks.map(({ href, label, icon: Icon }) => {
                                const active = pathname === href;
                                return (
                                    <li key={href}>
                                        <Link
                                            href={href}
                                            onClick={() => setMobileOpen(false)}
                                            className="flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium"
                                            style={{
                                                color: active ? "var(--accent-cyan)" : "var(--text-secondary)",
                                                background: active ? "rgba(99,102,241,0.1)" : "transparent",
                                            }}
                                        >
                                            {Icon && <Icon className="w-4 h-4" />}
                                            {label}
                                        </Link>
                                    </li>
                                );
                            })}
                        </ul>
                    </motion.div>
                )}
            </AnimatePresence>
        </header>
    );
}
