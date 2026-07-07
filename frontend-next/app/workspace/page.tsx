"use client";

/* ============================================================================
 * app/workspace/page.tsx — NEXUS Workspace
 * Ported from the Claude Design project "NEXUS Workspace.dc.html".
 * Renders the semantic-bridge graph (lib/nexusEngine.ts) plus the live
 * discovery run wired to the backend at http://localhost:8000.
 * ========================================================================== */

import { useEffect, useRef, useState, useCallback, useMemo } from "react";
import {
  GraphEngine, PIPELINE, RIGORS, SSE_STAGE_MAP,
  buildSiloMap, buildPMap, buildBMap, statusColor, statusText, rgba, shortLabel,
  foldDiscoveryResult, colorForCapability,
  type Bridge, type Silo, type Principle, type Agent, type Tier,
  type HyperedgeMeta, type IsomorphicClusterMeta,
} from "@/lib/nexusEngine";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";
const MONO = "'IBM Plex Mono', monospace";
const SANS = "'IBM Plex Sans', system-ui, sans-serif";

type Conn = "checking" | "live" | "demo";
type SelKind = "mapping" | "principle" | "none";
interface RunState {
  active: boolean; done: boolean; stageIdx: number; query: string;
  rigor: string; output: string; sources: { title: string; id: string }[]; error: string | null;
}
interface LogLine { time: string; tag: string; tagc: string; text: string }
interface GraphData { silos: Silo[]; principles: Principle[]; bridges: Bridge[] }

const clock = (off = 0) => new Date(Date.now() + off * 1000).toTimeString().slice(0, 8);
const fmt = (n: number) => n.toLocaleString("en-US");
const short = (t: string) => (t.length > 22 ? t.slice(0, 21) + "…" : t);

export default function NexusWorkspace() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const engineRef = useRef<GraphEngine | null>(null);
  const queryRef = useRef<HTMLInputElement>(null);

  const [selId, setSelId] = useState<string | null>(null);
  const [selKind, setSelKind] = useState<SelKind>("none");
  const [activeSilo, setActiveSilo] = useState<string | null>(null);
  const [filterKey, setFilterKey] = useState("all");
  const [leftTab, setLeftTab] = useState<"silos" | "agents">("silos");
  const [rigor, setRigor] = useState("prisma");
  const [conn, setConn] = useState<Conn>("checking");
  const [model, setModel] = useState("ollama:qwen2.5:3b");
  const [graphData, setGraphData] = useState<GraphData>({ silos: [], principles: [], bridges: [] });
  const [completedRuns, setCompletedRuns] = useState(0);
  const [agents, setAgents] = useState<Agent[]>([]);
  const [run, setRun] = useState<RunState | null>(null);
  const [log, setLog] = useState<LogLine[]>([]);
  const [realLog, setRealLog] = useState<LogLine[] | null>(null);
  const [runtime, setRuntime] = useState<Record<string, any> | null>(null);

  const pCount = graphData.principles.length;
  const bCount = graphData.bridges.length;
  const siloMap = useMemo(() => buildSiloMap(graphData.silos), [graphData.silos]);
  const pMap = useMemo(() => buildPMap(graphData.principles), [graphData.principles]);
  const bMap = useMemo(() => buildBMap(graphData.bridges), [graphData.bridges]);

  const runRef = useRef<RunState | null>(null);
  runRef.current = run;
  const connRef = useRef<Conn>("checking");
  connRef.current = conn;

  /* ── engine bootstrap ─────────────────────────────────────────────── */
  const onSelect = useCallback((id: string | null, kind: SelKind) => {
    setSelId(id); setSelKind(kind);
  }, []);

  useEffect(() => {
    if (!canvasRef.current) return;
    const eng = new GraphEngine(canvasRef.current, { onSelect });
    engineRef.current = eng;
    return () => eng.destroy();
  }, [onSelect]);

  useEffect(() => { engineRef.current?.setActiveSilo(activeSilo); }, [activeSilo]);
  useEffect(() => { engineRef.current?.setFilter(filterKey); }, [filterKey]);
  useEffect(() => {
    engineRef.current?.setData(graphData.silos, graphData.principles, graphData.bridges);
  }, [graphData]);

  /* ── backend bridge ───────────────────────────────────────────────── */
  const fetchT = async (url: string, opts?: RequestInit, ms = 3000) => {
    const ctl = new AbortController();
    const t = setTimeout(() => ctl.abort(), ms);
    try { return await fetch(url, { ...opts, signal: ctl.signal }); }
    finally { clearTimeout(t); }
  };

  const parseLogLine = (raw: string): LogLine => {
    const s = String(raw);
    const tm = s.match(/\b(\d{2}:\d{2}:\d{2})\b/);
    let tag = "INFO", tagc = "#7c8ce0";
    if (/\[ERROR\]|ERROR/.test(s)) { tag = "ERROR"; tagc = "#e0708f"; }
    else if (/\[WARN|WARNING/.test(s)) { tag = "WARN"; tagc = "#e0a850"; }
    else if (/\[INFO\]|INFO/.test(s)) { tag = "INFO"; tagc = "#5fbf8f"; }
    let text = s.replace(/^\S+\s+\S+\s*/, "").replace(/\[(INFO|WARNING|ERROR|DEBUG)\]\s*/, "").trim();
    if (text.length > 120) text = text.slice(0, 119) + "…";
    return { time: tm ? tm[1] : clock(0), tag, tagc, text: text || s.slice(0, 110) };
  };

  const monTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pollMonitor = useCallback(async function poll() {
    if (connRef.current !== "live") return;
    const base = API_BASE.replace(/\/$/, "");
    try {
      const r = await fetchT(base + "/api/backend/monitor?lines=60", {}, 4000);
      const j = await r.json();
      const lines = (j.recent_logs || []).slice(-24).map((l: string) => parseLogLine(l));
      setRealLog(lines);
      setRuntime(j.subagents || null);
      if (j.backend?.agentic_model) setModel(j.backend.agentic_model);
    } catch { setConn("demo"); return; }
    monTimer.current = setTimeout(poll, 4000);
  }, []);

  const probeBackend = useCallback(async () => {
    const base = API_BASE.replace(/\/$/, "");
    try {
      const r = await fetchT(base + "/health", {}, 2500);
      const j = await r.json();
      if (j && j.status === "ok") { setConn("live"); connRef.current = "live"; pollMonitor(); return; }
    } catch { /* fall through */ }
    setConn("demo");
  }, [pollMonitor]);

  useEffect(() => {
    probeBackend();
    return () => { if (monTimer.current) clearTimeout(monTimer.current); };
  }, [probeBackend]);

  const reconnect = () => { setConn("checking"); connRef.current = "checking"; if (monTimer.current) clearTimeout(monTimer.current); probeBackend(); };

  /* ── agents (capabilities) ────────────────────────────────────────── */
  const fetchCapabilities = useCallback(async () => {
    const base = API_BASE.replace(/\/$/, "");
    try {
      const r = await fetchT(base + "/api/capabilities", {}, 4000);
      const j = await r.json();
      const caps = Array.isArray(j.capabilities) ? j.capabilities : [];
      const mapped: Agent[] = caps.map((c: { name: string; model_tier?: string; tool_names?: string[]; catalog_note?: string; description?: string }, idx: number) => {
        const tier: Tier = c.model_tier === "deep" ? "deep" : "fast";
        return {
          id: c.name, tier, color: colorForCapability(tier, idx),
          tools: Array.isArray(c.tool_names) ? c.tool_names : [],
          note: c.catalog_note || c.description || "",
        };
      });
      setAgents(mapped);
    } catch { setAgents([]); }
  }, []);

  useEffect(() => { fetchCapabilities(); }, [fetchCapabilities]);

  /* ── discovery run ────────────────────────────────────────────────── */
  const advanceStage = (idx: number) =>
    setRun((r) => (r ? { ...r, stageIdx: Math.max(r.stageIdx, idx) } : r));
  const appendOutput = (delta: string) =>
    setRun((r) => (r ? { ...r, output: r.output + delta } : r));
  const finishRun = (patch: Partial<RunState>) =>
    setRun((r) => (r ? { ...r, active: false, done: true, stageIdx: 7, ...patch } : r));
  const closeRun = () => setRun(null);

  const pushRunLog = (stage: string, msg: string) =>
    setLog((l) => [...l.slice(-4), { time: clock(0), tag: (stage || "STAGE").toUpperCase().slice(0, 7), tagc: "#4ec9b0", text: msg || "" }]);

  const streamDiscovery = async (q: string) => {
    const base = API_BASE.replace(/\/$/, "");
    try {
      const resp = await fetch(base + "/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: q, use_harness: true, rigor_level: rigor, mode: "agentic", research_goals: [], allow_auto_override: true }),
      });
      const reader = resp.body!.getReader();
      const dec = new TextDecoder();
      let buf = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const blocks = buf.split("\n\n"); buf = blocks.pop() || "";
        for (const blk of blocks) {
          const ev = (blk.match(/event:\s*(.*)/) || [])[1];
          const dm = blk.match(/data:\s*([\s\S]*)/);
          if (!dm) continue;
          let data: any = {};
          try { data = JSON.parse(dm[1]); } catch { continue; }
          if (ev === "status") {
            const idx = SSE_STAGE_MAP[data.stage];
            if (idx != null) advanceStage(idx);
            pushRunLog(data.stage, data.message);
          } else if (ev === "content") { appendOutput(data.delta || ""); }
          else if (ev === "meta") {
            finishRun({ sources: (data.sources || []).map((s: { title: string; paper_id?: string; year?: string }) => ({ title: s.title, id: s.paper_id || s.year || "" })) });
            const hyperedges: HyperedgeMeta[] = data.hyperedges || [];
            const clusters: IsomorphicClusterMeta[] = data.isomorphicClusters || [];
            if (hyperedges.length || clusters.length) {
              setGraphData((g) => foldDiscoveryResult(g, hyperedges, clusters));
            }
            setCompletedRuns((c) => c + 1);
          }
          else if (ev === "error") { finishRun({ error: data.message || "stream error" }); }
          else if (ev === "done") { finishRun({}); }
        }
      }
      if (runRef.current && runRef.current.active) finishRun({});
    } catch {
      finishRun({ error: "Backend stream failed — no data returned." });
    }
  };

  const runDiscovery = () => {
    if (run && run.active) return;
    const q = (queryRef.current?.value.trim()) || "Map EV battery thermal runaway to biological heat dispersion";
    const next: RunState = { active: true, done: false, stageIdx: 0, query: q, rigor, output: "", sources: [], error: null };
    setRun(next); runRef.current = next;
    if (conn === "live") streamDiscovery(q);
    else finishRun({ error: "Backend is not connected — click the connection pill to retry." });
  };

  /* ── derived ──────────────────────────────────────────────────────── */
  const showDiscovery = !!run;
  const selBridge = selKind === "mapping" && selId ? bMap[selId.replace("b:", "")] : null;
  const selPrinc = selKind === "principle" && selId ? pMap[selId] : null;
  const useReal = conn === "live" && realLog && realLog.length > 0;
  const logSrc = useReal ? realLog! : log;
  const activeAgentId = run && run.active ? (PIPELINE[Math.min(run.stageIdx, PIPELINE.length - 1)] || {}).agent : null;

  const dot = (color: string, size = 9) => ({
    display: "inline-block", width: size, height: size, borderRadius: "50%",
    background: color, flex: "0 0 auto", boxShadow: `0 0 7px ${rgba(color, 0.6)}`,
  });

  /* ── render ───────────────────────────────────────────────────────── */
  return (
    <div style={{ position: "fixed", inset: 0, zIndex: 50, height: "100vh", display: "flex", flexDirection: "column", background: "#0a0c11", color: "#e7eaf0", fontFamily: SANS, fontSize: 14, overflow: "hidden" }}>
      <style>{`
        @keyframes nx-pulse{0%,100%{opacity:.3;transform:scale(1);}50%{opacity:1;transform:scale(1.4);}}
        @keyframes nx-sweep{0%{transform:translateX(-120%);}100%{transform:translateX(220%);}}
        .nx *::-webkit-scrollbar{width:8px;height:8px;}
        .nx *::-webkit-scrollbar-thumb{background:rgba(255,255,255,.1);border-radius:4px;}
      `}</style>

      {/* TOP BAR */}
      <header style={{ height: 54, flex: "0 0 54px", display: "flex", alignItems: "center", justifyContent: "space-between", padding: "0 18px", borderBottom: "1px solid rgba(255,255,255,.07)", background: "#0c0f15" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 18 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <div style={{ width: 22, height: 22, position: "relative" }}>
              <div style={{ position: "absolute", inset: 0, border: "1.5px solid #4ec9b0", transform: "rotate(45deg)", borderRadius: 3 }} />
              <div style={{ position: "absolute", inset: 6, background: "#4ec9b0", transform: "rotate(45deg)", borderRadius: 1 }} />
            </div>
            <div style={{ display: "flex", flexDirection: "column", lineHeight: 1, gap: 3 }}>
              <span style={{ fontFamily: MONO, fontWeight: 600, fontSize: 15, letterSpacing: 4 }}>NEXUS</span>
              <span style={{ fontSize: 9, letterSpacing: 1.5, color: "#5a6270", textTransform: "uppercase" }}>Isomorphic Mapping Engine</span>
            </div>
          </div>
          <div style={{ width: 1, height: 26, background: "rgba(255,255,255,.08)" }} />
          <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "5px 11px", border: "1px solid rgba(255,255,255,.09)", borderRadius: 6, background: "rgba(255,255,255,.02)" }}>
            <span style={{ fontSize: 9, letterSpacing: 1, color: "#5a6270", textTransform: "uppercase" }}>Project</span>
            <span style={{ fontFamily: MONO, fontSize: 12, color: "#c3cad6" }}>Aerospace-Bio Pilot</span>
            <span style={{ color: "#5a6270", fontSize: 10 }}>▾</span>
          </div>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 9, padding: "6px 13px", border: "1px solid rgba(78,201,176,.28)", borderRadius: 20, background: "rgba(78,201,176,.06)" }}>
          <span style={{ width: 7, height: 7, borderRadius: "50%", background: "#4ec9b0", animation: "nx-pulse 1.8s infinite", boxShadow: "0 0 8px #4ec9b0" }} />
          <span style={{ fontFamily: MONO, fontSize: 11, letterSpacing: 1.5, color: "#4ec9b0" }}>AUTONOMOUS SCAN · ACTIVE</span>
          <span style={{ width: 1, height: 14, background: "rgba(78,201,176,.25)" }} />
          <span style={{ fontFamily: MONO, fontSize: 11, color: "#8b93a3" }}>{fmt(pCount)} principles · {fmt(bCount)} bridges</span>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 7, padding: "5px 10px", border: "1px solid rgba(95,191,143,.25)", borderRadius: 6, background: "rgba(95,191,143,.05)" }}>
            <span style={{ fontSize: 11, color: "#5fbf8f" }}>◆</span>
            <span style={{ fontFamily: MONO, fontSize: 10, letterSpacing: .5, color: "#89b7a0" }}>ON-PREM · SOC2</span>
          </div>
        </div>
      </header>

      {/* COMMAND BAR */}
      <div style={{ height: 50, flex: "0 0 50px", display: "flex", alignItems: "center", gap: 12, padding: "0 16px", borderBottom: "1px solid rgba(255,255,255,.07)", background: "#0b0e13" }}>
        <span style={{ fontFamily: MONO, fontSize: 10, letterSpacing: 1.5, color: "#5a6270", whiteSpace: "nowrap" }}>DISCOVERY RUN</span>
        <div style={{ display: "flex", gap: 5 }}>
          {RIGORS.map(([k, label]) => {
            const on = rigor === k;
            return <button key={k} onClick={() => setRigor(k)} style={{ padding: "5px 11px", borderRadius: 6, fontSize: 10.5, fontWeight: 500, cursor: "pointer", fontFamily: MONO, letterSpacing: .3, whiteSpace: "nowrap", background: on ? rgba("#4ec9b0", 0.14) : "transparent", border: `1px solid ${on ? rgba("#4ec9b0", 0.5) : "rgba(255,255,255,.09)"}`, color: on ? "#4ec9b0" : "#8b93a3" }}>{label}</button>;
          })}
        </div>
        <div style={{ flex: 1, display: "flex", alignItems: "center", gap: 9, padding: "0 13px", height: 34, border: "1px solid rgba(255,255,255,.1)", borderRadius: 8, background: "rgba(255,255,255,.02)" }}>
          <span style={{ color: "#4ec9b0", fontFamily: MONO, fontSize: 13 }}>›</span>
          <input ref={queryRef} onKeyDown={(e) => { if (e.key === "Enter") runDiscovery(); }} placeholder="Describe a cross-domain discovery run — e.g. map EV battery thermal runaway to biology" style={{ flex: 1, background: "transparent", border: "none", outline: "none", color: "#e7eaf0", fontFamily: SANS, fontSize: 12.5 }} />
        </div>
        <button onClick={runDiscovery} style={{ padding: "7px 15px", borderRadius: 7, border: "none", cursor: run?.active ? "default" : "pointer", background: run?.active ? rgba("#4ec9b0", 0.3) : "#4ec9b0", color: "#06231d", fontWeight: 600, fontSize: 12, fontFamily: SANS, whiteSpace: "nowrap" }}>{run?.active ? "Running…" : "Run discovery"}</button>
        {(() => {
          const live = conn === "live", checking = conn === "checking";
          const color = live ? "#5fbf8f" : checking ? "#e0a850" : "#5a6270";
          const label = live ? "LIVE · " + API_BASE.replace(/^https?:\/\//, "") : checking ? "CONNECTING…" : "DEMO MODE";
          return (
            <div onClick={reconnect} title="Backend at localhost:8000 — click to reconnect" style={{ display: "flex", alignItems: "center", gap: 8, padding: "6px 11px", borderRadius: 7, cursor: "pointer", border: `1px solid ${rgba(color, 0.3)}`, background: rgba(color, 0.05) }}>
              <span style={{ width: 7, height: 7, borderRadius: "50%", background: color, flex: "0 0 auto", animation: live || checking ? "nx-pulse 1.6s infinite" : "none" }} />
              <span style={{ fontFamily: MONO, fontSize: 10, letterSpacing: .5, color }}>{label}</span>
            </div>
          );
        })()}
      </div>

      {/* MIDDLE */}
      <div className="nx" style={{ flex: 1, minHeight: 0, display: "flex" }}>

        {/* LEFT RAIL */}
        <aside style={{ width: 250, flex: "0 0 250px", borderRight: "1px solid rgba(255,255,255,.07)", background: "#0b0e13", display: "flex", flexDirection: "column", minHeight: 0 }}>
          <div style={{ display: "flex", borderBottom: "1px solid rgba(255,255,255,.06)" }}>
            {(["silos", "agents"] as const).map((t) => {
              const on = leftTab === t;
              return <div key={t} onClick={() => setLeftTab(t)} style={{ flex: 1, padding: "9px 0", textAlign: "center", cursor: "pointer", fontFamily: MONO, fontSize: 10, letterSpacing: 1.5, color: on ? "#e7eaf0" : "#5a6270", background: on ? "rgba(255,255,255,.04)" : "transparent", borderBottom: on ? "2px solid #4ec9b0" : "2px solid transparent" }}>{t.toUpperCase()}</div>;
            })}
          </div>

          {leftTab === "silos" && (
            <div style={{ flex: 1, overflowY: "auto", padding: 8, minHeight: 0 }}>
              <div style={{ padding: "6px 8px 10px", fontSize: 11, color: "#8b93a3" }}>Dark data ingested per domain</div>
              {graphData.silos.map((s) => {
                const active = activeSilo === s.id;
                const count = graphData.principles.filter((p) => p.silo === s.id).length;
                return (
                  <div key={s.id} onClick={() => setActiveSilo(active ? null : s.id)} style={{ padding: "11px 12px", borderRadius: 9, cursor: "pointer", marginBottom: 4, background: active ? rgba(s.color, 0.09) : "transparent", border: active ? `1px solid ${rgba(s.color, 0.4)}` : "1px solid transparent" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
                      <span style={dot(s.color, 10)} />
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ fontSize: 12.5, fontWeight: 500, color: "#dfe4ec" }}>{s.name}</div>
                        <div style={{ fontFamily: MONO, fontSize: 9.5, color: "#5a6270", marginTop: 2 }}>{s.files} · {s.meta}</div>
                      </div>
                      <span style={{ fontFamily: MONO, fontSize: 11, color: "#8b93a3" }}>{count}</span>
                    </div>
                    <div style={{ height: 3, borderRadius: 2, background: "rgba(255,255,255,.06)", marginTop: 9, overflow: "hidden" }}>
                      <div style={{ width: `${Math.round(s.ingest * 100)}%`, height: "100%", background: s.color, borderRadius: 2 }} />
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {leftTab === "agents" && (
            <div style={{ flex: 1, overflowY: "auto", padding: 8, minHeight: 0 }}>
              <div style={{ padding: "6px 8px 10px", fontSize: 11, color: "#8b93a3" }}>LangGraph ReAct subagents · orchestrated per run</div>
              {agents.map((a) => {
                let status = "idle", sc = "#5a6270";
                if (activeAgentId === a.id) { status = "active"; sc = a.color; }
                else if (run && run.done) { status = "done"; sc = "#5fbf8f"; }
                const info = runtime?.[a.id] || runtime?.[a.id.replace(/-/g, "_")];
                if (info && info.status) { status = String(info.status); sc = status === "active" ? a.color : status === "done" ? "#5fbf8f" : "#5a6270"; }
                return (
                  <div key={a.id} style={{ padding: "10px 11px", border: activeAgentId === a.id ? `1px solid ${rgba(a.color, 0.45)}` : "1px solid rgba(255,255,255,.06)", borderRadius: 9, marginBottom: 6, background: activeAgentId === a.id ? rgba(a.color, 0.06) : "rgba(255,255,255,.012)" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
                      <span style={{ width: 9, height: 9, borderRadius: 2, transform: "rotate(45deg)", background: a.color, flex: "0 0 auto", boxShadow: `0 0 7px ${rgba(a.color, 0.6)}` }} />
                      <span style={{ flex: 1, minWidth: 0, fontFamily: MONO, fontSize: 11.5, fontWeight: 500, color: "#dfe4ec" }}>{a.id}</span>
                      <span style={{ fontFamily: MONO, fontSize: 8.5, letterSpacing: .5, color: a.tier === "deep" ? "#a889e0" : "#5b9bd8", padding: "2px 6px", borderRadius: 4, background: a.tier === "deep" ? rgba("#a889e0", 0.12) : rgba("#5b9bd8", 0.12), textTransform: "uppercase" }}>{a.tier}</span>
                      <span style={{ width: 7, height: 7, borderRadius: "50%", background: sc, flex: "0 0 auto", animation: status === "active" ? "nx-pulse 1.2s infinite" : "none" }} />
                    </div>
                    <div style={{ fontSize: 10, color: "#5a6270", margin: "6px 0 7px 18px" }}>{a.note}</div>
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 4, marginLeft: 18 }}>
                      {a.tools.map((t) => <span key={t} style={{ fontFamily: MONO, fontSize: 9, color: "#8b93a3", padding: "2px 6px", borderRadius: 4, background: "rgba(255,255,255,.04)", border: "1px solid rgba(255,255,255,.06)" }}>{t}</span>)}
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          <div style={{ padding: "12px 16px", borderTop: "1px solid rgba(255,255,255,.05)", background: "#090b10" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
              <span style={{ fontFamily: MONO, fontSize: 10, letterSpacing: 1, color: "#5a6270" }}>MODEL</span>
              <span style={{ fontFamily: MONO, fontSize: 11, color: "#c3cad6" }}>{model}</span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 8 }}>
              <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#5fbf8f" }} />
              <span style={{ fontSize: 10, color: "#8b93a3" }}>Local-first · air-gapped (NEXUS_LOCAL_ONLY)</span>
            </div>
          </div>
        </aside>

        {/* CENTER GRAPH */}
        <main style={{ flex: 1, minWidth: 0, position: "relative", background: "radial-gradient(ellipse at 50% 42%,#0d1017 0%,#0a0c11 72%)", overflow: "hidden" }}>
          <canvas ref={canvasRef} style={{ display: "block", width: "100%", height: "100%", cursor: "grab" }} />
          <div style={{ position: "absolute", top: 16, left: 18, pointerEvents: "none" }}>
            <div style={{ fontFamily: MONO, fontSize: 10, letterSpacing: 2, color: "#5a6270" }}>LAYER 2 · SEMANTIC BRIDGE</div>
            <div style={{ fontSize: 17, fontWeight: 600, color: "#eef1f6", marginTop: 3 }}>Cross-Silo Isomorphic Graph</div>
            <div style={{ fontSize: 11, color: "#8b93a3", marginTop: 2 }}>Core principles linked by structural similarity · drag to explore</div>
          </div>
          <div style={{ position: "absolute", top: 20, left: "50%", transform: "translateX(-50%)", display: "flex", gap: 6 }}>
            {[["all", "All"], ["confirmed", "Confirmed"], ["candidate", "Candidates"], ["evaluating", "Evaluating"]].map(([k, label]) => {
              const on = filterKey === k;
              const c = k === "all" ? "#4ec9b0" : statusColor(k as any);
              return <button key={k} onClick={() => setFilterKey(k)} style={{ padding: "6px 13px", borderRadius: 18, fontSize: 11, fontWeight: 500, cursor: "pointer", fontFamily: MONO, letterSpacing: .3, background: on ? rgba(c, 0.14) : "rgba(12,15,21,.8)", border: `1px solid ${on ? rgba(c, 0.5) : "rgba(255,255,255,.08)"}`, color: on ? c : "#8b93a3" }}>{label}</button>;
            })}
          </div>
          <div style={{ position: "absolute", top: 16, right: 18, display: "flex", flexDirection: "column", gap: 6 }}>
            <button onClick={() => engineRef.current?.zoomIn()} style={{ width: 30, height: 30, border: "1px solid rgba(255,255,255,.1)", background: "rgba(12,15,21,.8)", color: "#c3cad6", borderRadius: 7, cursor: "pointer", fontSize: 16 }}>+</button>
            <button onClick={() => engineRef.current?.zoomOut()} style={{ width: 30, height: 30, border: "1px solid rgba(255,255,255,.1)", background: "rgba(12,15,21,.8)", color: "#c3cad6", borderRadius: 7, cursor: "pointer", fontSize: 16 }}>−</button>
            <button onClick={() => engineRef.current?.zoomReset()} style={{ width: 30, height: 30, border: "1px solid rgba(255,255,255,.1)", background: "rgba(12,15,21,.8)", color: "#8b93a3", borderRadius: 7, cursor: "pointer", fontSize: 11, fontFamily: MONO }}>⤢</button>
          </div>
          <div style={{ position: "absolute", bottom: 16, left: 18, display: "flex", gap: 16, padding: "9px 14px", border: "1px solid rgba(255,255,255,.07)", borderRadius: 8, background: "rgba(11,14,19,.82)", backdropFilter: "blur(6px)" }}>
            {[["#5fbf8f", "Confirmed bridge"], ["#e0a850", "Candidate"], ["#6f9bd8", "Evaluating"]].map(([c, l]) => (
              <div key={l} style={{ display: "flex", alignItems: "center", gap: 7 }}><span style={{ width: 9, height: 9, background: c, transform: "rotate(45deg)" }} /><span style={{ fontSize: 10.5, color: "#9aa2b1" }}>{l}</span></div>
            ))}
            <div style={{ width: 1, background: "rgba(255,255,255,.1)" }} />
            <div style={{ display: "flex", alignItems: "center", gap: 7 }}><span style={{ width: 9, height: 9, borderRadius: "50%", background: "#7c8ce0" }} /><span style={{ fontSize: 10.5, color: "#9aa2b1" }}>Core principle</span></div>
          </div>
        </main>

        {/* RIGHT INSPECTOR */}
        <aside style={{ width: 384, flex: "0 0 384px", borderLeft: "1px solid rgba(255,255,255,.07)", background: "#0b0e13", display: "flex", flexDirection: "column", minHeight: 0 }}>
          <div style={{ padding: "14px 18px", borderBottom: "1px solid rgba(255,255,255,.05)", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <div>
              <div style={{ fontFamily: MONO, fontSize: 10, letterSpacing: 2, color: "#5a6270" }}>MAPPING INSPECTOR</div>
              <div style={{ fontSize: 11, color: "#8b93a3", marginTop: 3 }}>Rosetta translation &amp; impact</div>
            </div>
            {(selKind !== "none" || showDiscovery) && (
              <button onClick={() => { engineRef.current?.zoomReset(); }} style={{ fontFamily: MONO, fontSize: 10, color: "#8b93a3", background: "none", border: "1px solid rgba(255,255,255,.1)", borderRadius: 5, padding: "4px 8px", cursor: "pointer" }}>clear</button>
            )}
          </div>

          <div style={{ flex: 1, overflowY: "auto" }}>
            {/* DISCOVERY RUN */}
            {showDiscovery && run && (
              <div style={{ padding: "16px 18px" }}>
                <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 10, marginBottom: 4 }}>
                  <div style={{ fontFamily: MONO, fontSize: 9, letterSpacing: 1.5, color: "#5a6270" }}>AGENTIC PIPELINE · {(RIGORS.find((r) => r[0] === run.rigor) || [])[1] || run.rigor}</div>
                  <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
                    <span style={{ fontFamily: MONO, fontSize: 10, letterSpacing: 1, color: run.error ? "#e0708f" : run.done ? "#5fbf8f" : "#4ec9b0" }}>{run.error ? "FAILED" : run.done ? "COMPLETE" : "RUNNING"}</span>
                    <button onClick={closeRun} style={{ fontFamily: MONO, fontSize: 12, color: "#8b93a3", background: "none", border: "1px solid rgba(255,255,255,.1)", borderRadius: 5, padding: "1px 7px", cursor: "pointer" }}>×</button>
                  </div>
                </div>
                <div style={{ fontSize: 13, color: "#dfe4ec", lineHeight: 1.5, marginBottom: 14 }}>{run.query}</div>

                <div style={{ border: "1px solid rgba(255,255,255,.07)", borderRadius: 10, padding: "14px 15px", background: "rgba(255,255,255,.012)" }}>
                  {PIPELINE.map((st, i) => {
                    let status = i < run.stageIdx ? "done" : i === run.stageIdx ? (run.done ? "done" : "active") : "pending";
                    if (run.done) status = "done";
                    const col = status === "done" ? "#5fbf8f" : status === "active" ? "#4ec9b0" : "#5a6270";
                    return (
                      <div key={st.key} style={{ display: "flex", gap: 11 }}>
                        <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                          <div style={{ width: 20, height: 20, borderRadius: "50%", flex: "0 0 auto", display: "flex", alignItems: "center", justifyContent: "center", fontFamily: MONO, fontSize: 9, fontWeight: 600, color: status === "pending" ? "#5a6270" : "#06231d", background: status === "pending" ? "transparent" : col, border: `1.5px solid ${status === "pending" ? "rgba(255,255,255,.14)" : col}` }}>{status === "done" ? "✓" : i + 1}</div>
                          <div style={{ width: 1.5, flex: 1, minHeight: 8, background: status === "done" ? rgba("#5fbf8f", 0.4) : "rgba(255,255,255,.08)", margin: "2px 0" }} />
                        </div>
                        <div style={{ paddingBottom: 8, flex: 1, minWidth: 0 }}>
                          <div style={{ fontSize: 12, fontWeight: status === "active" ? 600 : 500, color: status === "pending" ? "#6a7280" : "#dfe4ec" }}>{st.label}</div>
                          <div style={{ fontFamily: MONO, fontSize: 9.5, color: status === "active" ? "#4ec9b0" : "#5a6270" }}>{st.agent} · {st.desc}</div>
                        </div>
                      </div>
                    );
                  })}
                </div>

                {run.output && (
                  <div style={{ marginTop: 14 }}>
                    <div style={{ fontFamily: MONO, fontSize: 9, letterSpacing: 1.5, color: "#5a6270", marginBottom: 7 }}>SYNTHESIZED FINDINGS</div>
                    <div style={{ fontSize: 12.5, lineHeight: 1.65, color: "#c3cad6", whiteSpace: "pre-wrap", borderLeft: "2px solid rgba(78,201,176,.4)", paddingLeft: 12 }}>{run.output}</div>
                  </div>
                )}
                {run.error && (
                  <div style={{ marginTop: 12, fontSize: 12, color: "#e0708f", lineHeight: 1.5 }}>{run.error}</div>
                )}

                {run.done && !run.error && (
                  <div style={{ marginTop: 16 }}>
                    <div style={{ fontFamily: MONO, fontSize: 9, letterSpacing: 1.5, color: "#5a6270", marginBottom: 9 }}>CROSS-SILO MAPPING ALERTS</div>
                    {graphData.bridges.filter((x) => x.status === "confirmed").map((b) => {
                      const pa = pMap[b.a], pb = pMap[b.b];
                      return (
                        <div key={b.id} onClick={() => engineRef.current?.selectById("b:" + b.id)} style={{ padding: "10px 12px", border: "1px solid rgba(255,255,255,.07)", borderRadius: 8, background: "rgba(255,255,255,.015)", marginBottom: 7, cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10 }}>
                          <div style={{ minWidth: 0 }}>
                            <div style={{ fontSize: 12.5, fontWeight: 600, color: "#eef1f6" }}>{b.core}</div>
                            <div style={{ fontFamily: MONO, fontSize: 9.5, color: "#8b93a3", marginTop: 2 }}>{short(pa.label)} ⟷ {short(pb.label)}</div>
                          </div>
                          <span style={{ fontFamily: MONO, fontSize: 9.5, color: "#5fbf8f", padding: "2px 7px", borderRadius: 5, background: rgba("#5fbf8f", 0.12) }}>{b.impact}</span>
                        </div>
                      );
                    })}
                  </div>
                )}

                {run.sources.length > 0 && (
                  <div style={{ marginTop: 14 }}>
                    <div style={{ fontFamily: MONO, fontSize: 9, letterSpacing: 1.5, color: "#5a6270", marginBottom: 8 }}>PROVENANCE</div>
                    {run.sources.map((src, i) => (
                      <div key={i} style={{ display: "flex", alignItems: "baseline", gap: 9, padding: "4px 0" }}>
                        <span style={{ fontFamily: MONO, fontSize: 9.5, color: "#5b9bd8", flex: "0 0 auto" }}>{src.id}</span>
                        <span style={{ fontSize: 11.5, color: "#9aa2b1" }}>{src.title}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* EMPTY */}
            {selKind === "none" && !showDiscovery && (
              <div style={{ padding: "16px 18px" }}>
                <div style={{ fontSize: 12, color: "#8b93a3", lineHeight: 1.6, marginBottom: 16 }}>Select a <span style={{ color: "#5fbf8f" }}>◆ bridge</span> in the graph to inspect its isomorphic mapping, or a <span style={{ color: "#7c8ce0" }}>● principle</span> to trace its links.</div>
                <div style={{ fontFamily: MONO, fontSize: 10, letterSpacing: 1.5, color: "#5a6270", marginBottom: 10 }}>TOP DISCOVERIES THIS CYCLE</div>
                {[...graphData.bridges].sort((a, b) => b.conf - a.conf).slice(0, 4).map((b) => {
                  const pa = pMap[b.a], pb = pMap[b.b], sc = statusColor(b.status);
                  return (
                    <div key={b.id} onClick={() => engineRef.current?.selectById("b:" + b.id)} style={{ padding: "12px 13px", border: "1px solid rgba(255,255,255,.07)", borderRadius: 9, background: "rgba(255,255,255,.015)", marginBottom: 9, cursor: "pointer" }}>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 7 }}>
                        <span style={{ fontSize: 13, fontWeight: 600, color: "#eef1f6" }}>{b.core}</span>
                        <span style={{ fontFamily: MONO, fontSize: 10, color: sc, padding: "2px 7px", borderRadius: 5, background: rgba(sc, 0.12) }}>{Math.round(b.conf * 100)}%</span>
                      </div>
                      <div style={{ fontFamily: MONO, fontSize: 10.5, color: "#8b93a3" }}>{short(pa.label)} ⟷ {short(pb.label)}</div>
                    </div>
                  );
                })}
              </div>
            )}

            {/* MAPPING */}
            {selBridge && !showDiscovery && (() => {
              const b = selBridge, pa = pMap[b.a], pb = pMap[b.b];
              const sa = siloMap[pa.silo], sb = siloMap[pb.silo], sc = statusColor(b.status);
              return (
                <div style={{ padding: 18 }}>
                  <div style={{ border: "1px solid rgba(255,255,255,.09)", borderRadius: 12, overflow: "hidden", background: "rgba(255,255,255,.015)" }}>
                    <div style={{ padding: "14px 15px" }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
                        <span style={dot(sa.color, 11)} />
                        <div style={{ flex: 1, minWidth: 0 }}>
                          <div style={{ fontFamily: MONO, fontSize: 9, letterSpacing: 1, color: "#5a6270", textTransform: "uppercase" }}>{sa.name}</div>
                          <div style={{ fontSize: 13, color: "#dfe4ec", fontWeight: 500, marginTop: 2 }}>{pa.label}</div>
                        </div>
                      </div>
                      <div style={{ display: "flex", alignItems: "center", gap: 8, margin: "11px 0 11px 4px" }}>
                        <span style={{ fontFamily: MONO, fontSize: 13, color: "#4ec9b0" }}>⟷</span>
                        <span style={{ fontFamily: MONO, fontSize: 9.5, letterSpacing: 1, color: "#5a6270" }}>ISOMORPHIC MATCH</span>
                        <span style={{ flex: 1, height: 1, background: "rgba(255,255,255,.08)" }} />
                      </div>
                      <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
                        <span style={dot(sb.color, 11)} />
                        <div style={{ flex: 1, minWidth: 0 }}>
                          <div style={{ fontFamily: MONO, fontSize: 9, letterSpacing: 1, color: "#5a6270", textTransform: "uppercase" }}>{sb.name}</div>
                          <div style={{ fontSize: 13, color: "#dfe4ec", fontWeight: 500, marginTop: 2 }}>{pb.label}</div>
                        </div>
                      </div>
                    </div>
                    <div style={{ padding: "13px 15px", borderTop: `1px solid ${rgba("#4ec9b0", 0.18)}`, background: rgba("#4ec9b0", 0.05) }}>
                      <div style={{ fontFamily: MONO, fontSize: 9, letterSpacing: 1.5, color: "#5a6270" }}>ROSETTA CORE PRINCIPLE</div>
                      <div style={{ fontSize: 18, fontWeight: 600, color: "#eef1f6", marginTop: 5, letterSpacing: .2 }}>{b.core}</div>
                    </div>
                  </div>

                  <div style={{ display: "flex", gap: 10, marginTop: 14 }}>
                    <div style={{ flex: 1, border: "1px solid rgba(255,255,255,.07)", borderRadius: 9, padding: "11px 13px" }}>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
                        <span style={{ fontFamily: MONO, fontSize: 9, letterSpacing: 1, color: "#5a6270" }}>GNN CONFIDENCE</span>
                        <span style={{ fontFamily: MONO, fontSize: 15, color: "#eef1f6" }}>{Math.round(b.conf * 100)}%</span>
                      </div>
                      <div style={{ height: 5, borderRadius: 3, background: "rgba(255,255,255,.07)", marginTop: 9, overflow: "hidden" }}>
                        <div style={{ width: `${Math.round(b.conf * 100)}%`, height: "100%", background: sc, borderRadius: 3 }} />
                      </div>
                    </div>
                    <div style={{ width: 118, border: "1px solid rgba(255,255,255,.07)", borderRadius: 9, padding: "11px 13px" }}>
                      <div style={{ fontFamily: MONO, fontSize: 9, letterSpacing: 1, color: "#5a6270" }}>STATUS</div>
                      <div style={{ fontFamily: MONO, fontSize: 12, fontWeight: 600, color: sc, marginTop: 8, letterSpacing: .5 }}>{statusText(b.status)}</div>
                    </div>
                  </div>

                  <div style={{ marginTop: 12, padding: "12px 14px", borderRadius: 9, border: `1px solid ${rgba(sc, 0.25)}`, background: rgba(sc, 0.05) }}>
                    <span style={{ fontFamily: MONO, fontSize: 9, letterSpacing: 1.5, color: "#5a6270" }}>PROJECTED IMPACT</span>
                    <div style={{ fontSize: 20, fontWeight: 600, marginTop: 4, color: "#eef1f6" }}>{b.impact}</div>
                  </div>

                  <div style={{ marginTop: 14 }}>
                    <div style={{ fontFamily: MONO, fontSize: 9, letterSpacing: 1.5, color: "#5a6270", marginBottom: 6 }}>SYNTHESIS</div>
                    <div style={{ fontSize: 12.5, lineHeight: 1.65, color: "#b8c0cd" }}>{b.desc}</div>
                  </div>

                  <div style={{ marginTop: 16 }}>
                    <div style={{ fontFamily: MONO, fontSize: 9, letterSpacing: 1.5, color: "#5a6270", marginBottom: 9 }}>POLYMATHIC TEAM</div>
                    {b.collab.map((c) => {
                      const d = siloMap[c.dom];
                      return (
                        <div key={c.i} style={{ display: "flex", alignItems: "center", gap: 10, padding: "7px 0" }}>
                          <div style={{ width: 30, height: 30, borderRadius: 8, flex: "0 0 auto", display: "flex", alignItems: "center", justifyContent: "center", fontFamily: MONO, fontSize: 11, fontWeight: 600, color: d.color, background: rgba(d.color, 0.12), border: `1px solid ${rgba(d.color, 0.35)}` }}>{c.i}</div>
                          <div style={{ flex: 1, minWidth: 0 }}>
                            <div style={{ fontSize: 12, color: "#dfe4ec" }}>{c.n}</div>
                            <div style={{ fontSize: 10, color: "#5a6270" }}>{c.role}</div>
                          </div>
                          <span style={{ fontFamily: MONO, fontSize: 9, letterSpacing: .5, color: d.color, padding: "3px 7px", borderRadius: 5, background: rgba(d.color, 0.1), textTransform: "uppercase" }}>{d.name}</span>
                        </div>
                      );
                    })}
                  </div>

                  <div style={{ display: "flex", gap: 9, marginTop: 18 }}>
                    <button onClick={() => setLog((l) => [...l.slice(-4), { time: clock(0), tag: "CANVAS", tagc: "#4ec9b0", text: `opened '${b.core}' in Concept Canvas sandbox` }])} style={{ flex: 1, padding: 11, borderRadius: 8, border: "none", background: "#4ec9b0", color: "#06231d", fontWeight: 600, fontSize: 12.5, cursor: "pointer" }}>Open in Concept Canvas ⟶</button>
                    <button onClick={() => setLog((l) => [...l.slice(-4), { time: clock(0), tag: "ASSIGN", tagc: "#e0a850", text: `review requested for '${b.core}'` }])} style={{ padding: "11px 14px", borderRadius: 8, border: "1px solid rgba(255,255,255,.13)", background: "transparent", color: "#c3cad6", fontWeight: 500, fontSize: 12.5, cursor: "pointer" }}>Assign</button>
                  </div>
                </div>
              );
            })()}

            {/* PRINCIPLE */}
            {selPrinc && !showDiscovery && (() => {
              const p = selPrinc, s = siloMap[p.silo];
              const bridges = graphData.bridges.filter((b) => b.a === p.pid || b.b === p.pid);
              return (
                <div style={{ padding: 18 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 11, marginBottom: 6 }}>
                    <span style={dot(s.color, 12)} />
                    <div>
                      <div style={{ fontFamily: MONO, fontSize: 9, letterSpacing: 1, color: "#5a6270", textTransform: "uppercase" }}>{s.name} · CORE PRINCIPLE</div>
                      <div style={{ fontSize: 17, fontWeight: 600, color: "#eef1f6", marginTop: 3 }}>{p.label}</div>
                    </div>
                  </div>
                  <div style={{ fontSize: 12, color: "#8b93a3", lineHeight: 1.6, margin: "12px 0 18px" }}>{p.desc}</div>
                  <div style={{ fontFamily: MONO, fontSize: 9, letterSpacing: 1.5, color: "#5a6270", marginBottom: 10 }}>LINKED BRIDGES · {bridges.length}</div>
                  {bridges.map((b) => {
                    const other = pMap[b.a === p.pid ? b.b : b.a], sc = statusColor(b.status);
                    return (
                      <div key={b.id} onClick={() => engineRef.current?.selectById("b:" + b.id)} style={{ padding: "12px 13px", border: "1px solid rgba(255,255,255,.07)", borderRadius: 9, background: "rgba(255,255,255,.015)", marginBottom: 9, cursor: "pointer" }}>
                        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                          <span style={{ fontSize: 13, fontWeight: 600, color: "#eef1f6" }}>{b.core}</span>
                          <span style={{ fontFamily: MONO, fontSize: 10, color: sc, padding: "2px 7px", borderRadius: 5, background: rgba(sc, 0.12) }}>{Math.round(b.conf * 100)}%</span>
                        </div>
                        <div style={{ fontFamily: MONO, fontSize: 10.5, color: "#8b93a3" }}>⟷ {other.label}</div>
                      </div>
                    );
                  })}
                </div>
              );
            })()}
          </div>
        </aside>
      </div>

      {/* BOTTOM ENGINE LOG */}
      <footer style={{ height: 132, flex: "0 0 132px", borderTop: "1px solid rgba(255,255,255,.07)", background: "#090b10", display: "flex", flexDirection: "column" }}>
        <div style={{ height: 32, flex: "0 0 32px", display: "flex", alignItems: "center", gap: 12, padding: "0 16px", borderBottom: "1px solid rgba(255,255,255,.05)" }}>
          <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#4ec9b0", animation: "nx-pulse 1.6s infinite" }} />
          <span style={{ fontFamily: MONO, fontSize: 10, letterSpacing: 2, color: "#8b93a3" }}>AUTONOMOUS ENGINE</span>
          <span style={{ fontFamily: MONO, fontSize: 9, letterSpacing: 1, color: "#5a6270", padding: "2px 7px", border: "1px solid rgba(255,255,255,.08)", borderRadius: 4 }}>{useReal ? "BACKEND · research_agent.log" : "SIMULATED STREAM"}</span>
          <div style={{ flex: 1, position: "relative", height: 1, overflow: "hidden", margin: "0 8px", background: "rgba(255,255,255,.04)" }}>
            <div style={{ position: "absolute", top: 0, left: 0, width: "40%", height: 1, background: "linear-gradient(90deg,transparent,#4ec9b0,transparent)", animation: "nx-sweep 3s linear infinite" }} />
          </div>
          <span style={{ fontFamily: MONO, fontSize: 10, color: "#5a6270" }}>{completedRuns} discovery run{completedRuns === 1 ? "" : "s"} completed</span>
        </div>
        <div style={{ flex: 1, overflow: "hidden", padding: "8px 16px", display: "flex", flexDirection: "column", gap: 3 }}>
          {logSrc.map((l, i) => (
            <div key={i} style={{ display: "flex", alignItems: "baseline", gap: 12, fontFamily: MONO, fontSize: 11, lineHeight: 1.5 }}>
              <span style={{ color: "#4a525f", flex: "0 0 auto" }}>{l.time}</span>
              <span style={{ color: l.tagc, flex: "0 0 72px", fontWeight: 600, letterSpacing: .5 }}>{l.tag}</span>
              <span style={{ color: "#9aa2b1" }}>{l.text}</span>
            </div>
          ))}
        </div>
      </footer>
    </div>
  );
}
