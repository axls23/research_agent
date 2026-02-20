"use client";

import { motion } from "framer-motion";

interface TextRevealProps {
    text: string;
    className?: string;
    delay?: number;
    splitBy?: "word" | "char";
}

export default function TextReveal({
    text,
    className,
    delay = 0,
    splitBy = "word",
}: TextRevealProps) {
    const parts = splitBy === "word" ? text.split(" ") : text.split("");

    return (
        <span className={`inline-flex flex-wrap ${splitBy === "word" ? "gap-x-2" : "gap-x-0"}`} style={{ overflow: "hidden" }}>
            {parts.map((part, i) => (
                <motion.span
                    key={i}
                    initial={{ opacity: 0, y: 24 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{
                        duration: 0.5,
                        delay: delay + i * (splitBy === "word" ? 0.07 : 0.03),
                        ease: [0.21, 0.47, 0.32, 0.98],
                    }}
                    className={className}
                    style={{ display: "inline-block" }}
                >
                    {part}
                    {splitBy === "word" ? "" : ""}
                </motion.span>
            ))}
        </span>
    );
}
