"use client";

import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Activity, Bot, Cpu, RefreshCcw, Server } from "lucide-react";

type MonitorResponse = {
  timestamp: string;
  backend: { up: boolean; pid: number; cwd: string };
  ollama: {
    up: boolean;
    base_url: string;
    model_count: number;
    models: string[];
    error?: string;
  };
  subagents: Array<{
    name: string;
    role: string;
    live: boolean;
    last_seen?: string | null;
    last_action?: string | null;
    last_summary?: string | null;
  }>;
  workflow?: {
    available: boolean;
    source_file?: string | null;
    project_id?: string | null;
    project_name?: string | null;
    rigor_level?: string | null;
    completed_at?: string | null;
    step_count: number;
    steps: Array<{
      order: number;
      agent: string;
      agent_raw?: string;
      action: string;
      timestamp?: string | null;
      summary?: string;
    }>;
  };
  stats: {
    log_file: string;
    total_lines_returned: number;
    level_counts: { INFO: number; WARNING: number; ERROR: number };
    last_error?: string | null;
    last_warning?: string | null;
    subagent_config_line?: string | null;
  };
  recent_logs: string[];
};

type SubagentRuntime = MonitorResponse["subagents"][number];
type WorkflowStep = NonNullable<MonitorResponse["workflow"]>["steps"][number];

type PipelineNode = {
  name: string;
  label: string;
  role: string;
  status: "done" | "active" | "pending";
  stepCount: number;
};

const POLL_MS = 2000;

const PIPELINE_ORDER: Array<{ name: string; label: string }> = [
  { name: "deep-reasoner", label: "Deep Reasoner" },
  { name: "literature-search", label: "Literature Search" },
  { name: "data-processing", label: "Data Processing" },
  { name: "knowledge-graph", label: "Knowledge Graph" },
  { name: "analysis", label: "Analysis" },
  { name: "writing", label: "Writing" },
];

function buildPipelineNodes(
  subagents: SubagentRuntime[],
  steps: WorkflowStep[],
  workflowAvailable: boolean,
): PipelineNode[] {
  const runtimeMap = new Map(subagents.map((s) => [s.name, s]));
  const stepCounts = new Map<string, number>();

  for (const step of steps) {
    stepCounts.set(step.agent, (stepCounts.get(step.agent) ?? 0) + 1);
  }

  const doneSet = new Set(stepCounts.keys());
  let activeTaken = false;

  return PIPELINE_ORDER.map((entry, index) => {
    const runtime = runtimeMap.get(entry.name);
    const stepCount = stepCounts.get(entry.name) ?? 0;
    const done = doneSet.has(entry.name);

    let status: PipelineNode["status"] = "pending";
    if (done) {
      status = "done";
    } else if (runtime?.live && !activeTaken) {
      status = "active";
      activeTaken = true;
    } else if (!activeTaken && workflowAvailable) {
      const allPrevDone = PIPELINE_ORDER.slice(0, index).every((prev) => doneSet.has(prev.name));
      if (allPrevDone) {
        status = "active";
        activeTaken = true;
      }
    }

    return {
      name: entry.name,
      label: entry.label,
      role: runtime?.role ?? entry.label,
      status,
      stepCount,
    };
  });
}

export default function BackendMonitorPage() {
  const [data, setData] = useState<MonitorResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastRefresh, setLastRefresh] = useState<string>("");

  const fetchData = async () => {
    try {
      const resp = await fetch("http://localhost:8000/api/backend/monitor?lines=160", {
        cache: "no-store",
      });
      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status}`);
      }
      const payload = (await resp.json()) as MonitorResponse;
      setData(payload);
      setError(null);
      setLastRefresh(new Date().toLocaleTimeString());
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    const timer = setInterval(fetchData, POLL_MS);
    return () => clearInterval(timer);
  }, []);

  const logRows = useMemo(() => {
    if (!data?.recent_logs) return [];
    return data.recent_logs.slice(-60);
  }, [data]);

  const workflowSteps = useMemo(() => {
    return data?.workflow?.steps ?? [];
  }, [data]);

  const pipelineNodes = useMemo(() => {
    return buildPipelineNodes(
      data?.subagents ?? [],
      workflowSteps,
      Boolean(data?.workflow?.available),
    );
  }, [data?.subagents, data?.workflow?.available, workflowSteps]);

  return (
    <div className="animated-gradient min-h-screen px-6 py-10">
      <div className="max-w-7xl mx-auto space-y-6">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h1 className="text-3xl md:text-4xl font-bold" style={{ fontFamily: "var(--font-poppins)" }}>
              Backend Live Monitor
            </h1>
            <p className="text-sm mt-2" style={{ color: "var(--text-secondary)" }}>
              Real-time backend telemetry for presentation demos.
            </p>
          </div>
          <button
            onClick={fetchData}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium"
            style={{
              background: "rgba(99,102,241,0.18)",
              border: "1px solid rgba(99,102,241,0.35)",
              color: "var(--text-primary)",
            }}
          >
            <RefreshCcw className="w-4 h-4" />
            Refresh
          </button>
        </div>

        {loading && (
          <div className="glass rounded-2xl p-6 text-sm" style={{ color: "var(--text-secondary)" }}>
            Loading backend monitor...
          </div>
        )}

        {error && (
          <div className="glass rounded-2xl p-6 text-sm" style={{ color: "#fca5a5" }}>
            Failed to fetch monitor data: {error}. Ensure backend API is running at http://localhost:8000.
          </div>
        )}

        {data && (
          <>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <StatusCard
                title="Backend"
                value={data.backend.up ? "Online" : "Offline"}
                subtitle={`PID ${data.backend.pid}`}
                icon={<Server className="w-4 h-4" />}
                tone={data.backend.up ? "ok" : "bad"}
              />
              <StatusCard
                title="Ollama"
                value={data.ollama.up ? "Connected" : "Disconnected"}
                subtitle={`${data.ollama.model_count} models`}
                icon={<Cpu className="w-4 h-4" />}
                tone={data.ollama.up ? "ok" : "bad"}
              />
              <StatusCard
                title="Subagents"
                value={`${data.subagents.filter((s) => s.live).length}/${data.subagents.length}`}
                subtitle="Live from latest workflow"
                icon={<Bot className="w-4 h-4" />}
                tone="ok"
              />
              <StatusCard
                title="Last Refresh"
                value={lastRefresh || "--:--:--"}
                subtitle={`Poll every ${POLL_MS / 1000}s`}
                icon={<Activity className="w-4 h-4" />}
                tone="neutral"
              />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="glass rounded-2xl p-5 lg:col-span-1"
              >
                <h2 className="text-sm font-semibold mb-4" style={{ color: "var(--text-primary)" }}>
                  Subagent Runtime
                </h2>
                <div className="space-y-3">
                  {data.subagents.map((agent) => (
                    <div
                      key={agent.name}
                      className="rounded-lg p-3"
                      style={{
                        background: "rgba(255,255,255,0.03)",
                        border: "1px solid rgba(255,255,255,0.08)",
                      }}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <div className="text-sm font-medium">{agent.name}</div>
                        <span
                          className="text-xs px-2 py-0.5 rounded"
                          style={{
                            background: agent.live ? "rgba(74,222,128,0.15)" : "rgba(248,113,113,0.15)",
                            color: agent.live ? "#86efac" : "#fca5a5",
                          }}
                        >
                          {agent.live ? "LIVE" : "DOWN"}
                        </span>
                      </div>
                      <div className="text-xs mt-1" style={{ color: "var(--text-secondary)" }}>
                        {agent.role}
                      </div>
                      <div className="text-[11px] mt-1.5" style={{ color: "var(--text-secondary)" }}>
                        Last action: {agent.last_action || "--"}
                      </div>
                      <div className="text-[11px] mt-0.5" style={{ color: "var(--text-secondary)" }}>
                        Last seen: {agent.last_seen ? new Date(agent.last_seen).toLocaleString() : "--"}
                      </div>
                    </div>
                  ))}
                </div>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="glass rounded-2xl p-5 lg:col-span-2"
              >
                <h2 className="text-sm font-semibold mb-4" style={{ color: "var(--text-primary)" }}>
                  Deep-Agent Workflow
                </h2>
                <WorkflowGraph nodes={pipelineNodes} />

                <div
                  className="rounded-xl p-4 mb-4 max-h-[260px] overflow-y-auto"
                  style={{
                    background: "rgba(0,0,0,0.22)",
                    border: "1px solid rgba(255,255,255,0.08)",
                  }}
                >
                  {workflowSteps.length === 0 ? (
                    <div className="text-xs" style={{ color: "var(--text-secondary)" }}>
                      No deep-agent workflow found yet. Run the agentic pipeline to populate this section.
                    </div>
                  ) : (
                    <div className="space-y-2">
                      {workflowSteps.map((step) => (
                        <div
                          key={`${step.order}-${step.agent}-${step.action}`}
                          className="rounded-lg px-3 py-2"
                          style={{
                            background: "rgba(255,255,255,0.03)",
                            border: "1px solid rgba(255,255,255,0.07)",
                          }}
                        >
                          <div className="flex items-center justify-between gap-2">
                            <div className="text-xs font-semibold" style={{ color: "#c7d2fe" }}>
                              {step.order}. {step.agent}
                            </div>
                            <div className="text-[11px]" style={{ color: "var(--text-secondary)" }}>
                              {step.timestamp ? new Date(step.timestamp).toLocaleTimeString() : "--"}
                            </div>
                          </div>
                          <div className="text-xs mt-1" style={{ color: "#93c5fd" }}>
                            {step.action || "(no action)"}
                          </div>
                          <div className="text-[11px] mt-1" style={{ color: "var(--text-secondary)" }}>
                            {step.summary || "No summary available."}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                <h2 className="text-sm font-semibold mb-4" style={{ color: "var(--text-primary)" }}>
                  Live Backend Logs
                </h2>
                <div
                  className="rounded-xl p-4 h-[270px] overflow-y-auto font-mono text-xs whitespace-pre-wrap"
                  style={{
                    background: "rgba(0,0,0,0.35)",
                    border: "1px solid rgba(255,255,255,0.08)",
                    color: "#c7d2fe",
                  }}
                >
                  {logRows.length === 0 ? "No logs available." : null}
                  {logRows.map((line, idx) => (
                    <div
                      key={`${idx}-${line.slice(0, 18)}`}
                      style={{
                        color: line.includes("[ERROR]")
                          ? "#fca5a5"
                          : line.includes("[WARNING]")
                            ? "#fde68a"
                            : "#c7d2fe",
                      }}
                    >
                      {line}
                    </div>
                  ))}
                </div>
              </motion.div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function StatusCard({
  title,
  value,
  subtitle,
  icon,
  tone,
}: {
  title: string;
  value: string;
  subtitle: string;
  icon: React.ReactNode;
  tone: "ok" | "bad" | "neutral";
}) {
  const toneColor =
    tone === "ok" ? "#86efac" : tone === "bad" ? "#fca5a5" : "#93c5fd";

  return (
    <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="glass rounded-2xl p-5">
      <div className="flex items-center justify-between">
        <div className="text-xs uppercase tracking-wider" style={{ color: "var(--text-muted)" }}>
          {title}
        </div>
        <div style={{ color: toneColor }}>{icon}</div>
      </div>
      <div className="text-xl font-semibold mt-3" style={{ color: "var(--text-primary)" }}>
        {value}
      </div>
      <div className="text-xs mt-1" style={{ color: "var(--text-secondary)" }}>
        {subtitle}
      </div>
    </motion.div>
  );
}

function WorkflowGraph({ nodes }: { nodes: PipelineNode[] }) {
  const nodeWidth = 156;
  const nodeHeight = 74;
  const gap = 30;
  const canvasHeight = 152;
  const top = 34;
  const startX = 12;
  const totalWidth = nodes.length * nodeWidth + (nodes.length - 1) * gap + startX * 2;
  const middleY = top + nodeHeight / 2;

  return (
    <div
      className="rounded-xl p-4 mb-4"
      style={{
        background: "rgba(0,0,0,0.22)",
        border: "1px solid rgba(255,255,255,0.08)",
      }}
    >
      <div className="flex items-center justify-between mb-3">
        <div className="text-xs uppercase tracking-wider" style={{ color: "var(--text-muted)" }}>
          Workflow Graph
        </div>
        <div className="text-[11px]" style={{ color: "var(--text-secondary)" }}>
          {nodes.filter((n) => n.status === "done").length}/{nodes.length} completed
        </div>
      </div>

      <div className="overflow-x-auto">
        <div className="relative" style={{ minWidth: `${totalWidth}px`, height: `${canvasHeight}px` }}>
          <svg className="absolute inset-0" width={totalWidth} height={canvasHeight}>
            <defs>
              <marker
                id="pipeline-arrow"
                markerWidth="8"
                markerHeight="8"
                refX="6"
                refY="4"
                orient="auto"
                markerUnits="strokeWidth"
              >
                <path d="M0,0 L0,8 L8,4 z" fill="#94a3b8" />
              </marker>
            </defs>
            {nodes.slice(0, -1).map((node, idx) => {
              const next = nodes[idx + 1];
              const x1 = startX + idx * (nodeWidth + gap) + nodeWidth;
              const x2 = startX + (idx + 1) * (nodeWidth + gap);
              const tone =
                node.status === "done" && next.status === "done"
                  ? "#4ade80"
                  : node.status === "done" && next.status === "active"
                    ? "#60a5fa"
                    : "#64748b";
              const dashed = node.status === "pending" || next.status === "pending";

              return (
                <line
                  key={`${node.name}-${next.name}`}
                  x1={x1}
                  y1={middleY}
                  x2={x2}
                  y2={middleY}
                  stroke={tone}
                  strokeWidth="2.2"
                  strokeDasharray={dashed ? "6 5" : undefined}
                  markerEnd="url(#pipeline-arrow)"
                  opacity={0.92}
                />
              );
            })}
          </svg>

          {nodes.map((node, idx) => {
            const left = startX + idx * (nodeWidth + gap);
            const styleByStatus =
              node.status === "done"
                ? {
                    border: "1px solid rgba(74,222,128,0.55)",
                    background: "rgba(74,222,128,0.13)",
                    accent: "#86efac",
                    pillBg: "rgba(74,222,128,0.2)",
                    pillText: "#bbf7d0",
                  }
                : node.status === "active"
                  ? {
                      border: "1px solid rgba(96,165,250,0.55)",
                      background: "rgba(96,165,250,0.13)",
                      accent: "#93c5fd",
                      pillBg: "rgba(96,165,250,0.2)",
                      pillText: "#bfdbfe",
                    }
                  : {
                      border: "1px solid rgba(148,163,184,0.4)",
                      background: "rgba(148,163,184,0.1)",
                      accent: "#cbd5e1",
                      pillBg: "rgba(148,163,184,0.2)",
                      pillText: "#e2e8f0",
                    };

            return (
              <div
                key={node.name}
                className="absolute rounded-xl px-3 py-2"
                style={{
                  left: `${left}px`,
                  top: `${top}px`,
                  width: `${nodeWidth}px`,
                  height: `${nodeHeight}px`,
                  border: styleByStatus.border,
                  background: styleByStatus.background,
                }}
              >
                <div className="text-[10px] uppercase tracking-wide" style={{ color: styleByStatus.accent }}>
                  {node.name}
                </div>
                <div className="text-sm font-semibold mt-1" style={{ color: "var(--text-primary)" }}>
                  {node.label}
                </div>
                <div className="text-[11px] mt-1" style={{ color: "var(--text-secondary)" }}>
                  {node.stepCount > 0 ? `${node.stepCount} workflow steps` : "No steps yet"}
                </div>
                <span
                  className="absolute top-2 right-2 text-[10px] px-1.5 py-0.5 rounded"
                  style={{
                    background: styleByStatus.pillBg,
                    color: styleByStatus.pillText,
                  }}
                >
                  {node.status.toUpperCase()}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
