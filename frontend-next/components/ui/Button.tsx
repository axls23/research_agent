import { cn } from "@/lib/utils";

type ButtonVariant = "primary" | "secondary" | "ghost" | "outline";
type ButtonSize = "sm" | "md" | "lg";

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
    variant?: ButtonVariant;
    size?: ButtonSize;
    children: React.ReactNode;
    as?: "button";
}

const variantStyles: Record<ButtonVariant, string> = {
    primary:
        "text-white font-medium transition-all duration-200 hover:brightness-110 active:scale-[0.97]",
    secondary:
        "text-white font-medium transition-all duration-200 hover:brightness-110 active:scale-[0.97]",
    ghost:
        "transition-colors duration-200 active:scale-[0.97]",
    outline:
        "font-medium transition-all duration-200 hover:bg-white/5 active:scale-[0.97]",
};

const sizeStyles: Record<ButtonSize, string> = {
    sm: "px-3 py-1.5 text-xs rounded-lg gap-1.5",
    md: "px-5 py-2.5 text-sm rounded-xl gap-2",
    lg: "px-7 py-3.5 text-base rounded-xl gap-2.5",
};

const variantInlineStyles: Record<ButtonVariant, React.CSSProperties> = {
    primary: {
        background: "linear-gradient(135deg, #6366f1, #22d3ee)",
        boxShadow: "0 0 24px rgba(99,102,241,0.35)",
    },
    secondary: {
        background: "linear-gradient(135deg, #7c3aed, #a78bfa)",
        boxShadow: "0 0 24px rgba(124,58,237,0.35)",
    },
    ghost: {
        color: "var(--text-secondary)",
    },
    outline: {
        border: "1px solid rgba(255,255,255,0.15)",
        color: "var(--text-primary)",
    },
};

export default function Button({
    variant = "primary",
    size = "md",
    children,
    className,
    style,
    ...rest
}: ButtonProps) {
    return (
        <button
            className={cn("inline-flex items-center justify-center", variantStyles[variant], sizeStyles[size], className)}
            style={{ ...variantInlineStyles[variant], ...style }}
            {...rest}
        >
            {children}
        </button>
    );
}
