import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "NEXUS Workspace — Isomorphic Mapping Engine",
  description: "Cross-domain semantic bridge graph and agentic discovery pipeline.",
};

export default function WorkspaceLayout({ children }: { children: React.ReactNode }) {
  return (
    <>
      <link
        href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap"
        rel="stylesheet"
      />
      {children}
    </>
  );
}
