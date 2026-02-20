import { cn } from "@/lib/utils";
import React from "react";

interface CardProps {
    children: React.ReactNode;
    className?: string;
    hover?: boolean;
    style?: React.CSSProperties;
    onClick?: () => void;
}

export default function Card({ children, className, hover = false, style, onClick }: CardProps) {
    return (
        <div
            className={cn(
                "glass rounded-2xl p-6",
                hover && "glass-hover cursor-pointer",
                className
            )}
            style={style}
            onClick={onClick}
        >
            {children}
        </div>
    );
}
