"use client";

import { motion, useInView } from "framer-motion";
import { useRef } from "react";

interface AnimatedSectionProps {
    children: React.ReactNode;
    className?: string;
    delay?: number;
    direction?: "up" | "left" | "right" | "none";
}

export default function AnimatedSection({
    children,
    className,
    delay = 0,
    direction = "up",
}: AnimatedSectionProps) {
    const ref = useRef<HTMLDivElement>(null);
    const inView = useInView(ref, { once: true, margin: "-80px" });

    const initialMap = {
        up: { opacity: 0, y: 40 },
        left: { opacity: 0, x: -40 },
        right: { opacity: 0, x: 40 },
        none: { opacity: 0 },
    };

    const animateMap = {
        up: { opacity: 1, y: 0 },
        left: { opacity: 1, x: 0 },
        right: { opacity: 1, x: 0 },
        none: { opacity: 1 },
    };

    return (
        <motion.div
            ref={ref}
            initial={initialMap[direction]}
            animate={inView ? animateMap[direction] : initialMap[direction]}
            transition={{ duration: 0.65, delay, ease: [0.21, 0.47, 0.32, 0.98] }}
            className={className}
        >
            {children}
        </motion.div>
    );
}
