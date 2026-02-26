import { cn } from "@/lib/utils";

type BadgeColor = "cyan" | "violet" | "indigo" | "green" | "amber" | "rose";

interface BadgeProps {
    color?: BadgeColor;
    children: React.ReactNode;
    className?: string;
}

const colorMap: Record<BadgeColor, React.CSSProperties> = {
    cyan: { background: "rgba(34,211,238,0.12)", color: "#22d3ee", border: "1px solid rgba(34,211,238,0.2)" },
    violet: { background: "rgba(167,139,250,0.12)", color: "#a78bfa", border: "1px solid rgba(167,139,250,0.2)" },
    indigo: { background: "rgba(99,102,241,0.12)", color: "#818cf8", border: "1px solid rgba(99,102,241,0.2)" },
    green: { background: "rgba(34,197,94,0.12)", color: "#4ade80", border: "1px solid rgba(34,197,94,0.2)" },
    amber: { background: "rgba(251,191,36,0.12)", color: "#fbbf24", border: "1px solid rgba(251,191,36,0.2)" },
    rose: { background: "rgba(251,113,133,0.12)", color: "#fb7185", border: "1px solid rgba(251,113,133,0.2)" },
};

export default function Badge({ color = "indigo", children, className }: BadgeProps) {
    return (
        <span
            className={cn("inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium", className)}
            style={colorMap[color]}
        >
            {children}
        </span>
    );
}
