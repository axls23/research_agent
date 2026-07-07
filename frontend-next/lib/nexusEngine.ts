/* ============================================================================
 * lib/nexusEngine.ts
 * NEXUS Semantic Bridge — framework-agnostic graph engine.
 * Data model mirrors core/orchestrator.py (subagents) + the real discovery-run
 * payloads returned by the backend (hyperedges / isomorphicClusters in the
 * /api/chat/stream `meta` SSE event). Physics + canvas render + interaction.
 * Consumed by app/workspace/page.tsx. No React in here.
 *
 * NOTE: the graph starts EMPTY. There is no standalone "get current graph"
 * endpoint — hypergraph data only comes back tied to a specific discovery run.
 * silos/principles/bridges accumulate at runtime as real discovery runs
 * complete (see GraphEngine.setData, called from page.tsx after each run).
 * ========================================================================== */

export type Tier = "fast" | "deep";
export type BridgeStatus = "confirmed" | "candidate" | "evaluating";

export interface Silo {
  id: string; name: string; color: string; meta: string; files: string; ingest: number;
}
export interface Principle { pid: string; silo: string; label: string; desc: string; }
export interface Collaborator { i: string; n: string; role: string; dom: string; }
export interface Bridge {
  id: string; core: string; a: string; b: string; conf: number;
  status: BridgeStatus; impact: string; desc: string; collab: Collaborator[];
}
export interface Agent {
  id: string; tier: Tier; color: string; tools: string[]; note: string; external?: boolean;
}
export interface PipelineStep { key: string; label: string; agent: string; desc: string; }

interface GNode {
  id: string; type: "p" | "b"; ref: any;
  x: number; y: number; vx: number; vy: number; r: number;
  _fx?: number; _fy?: number;
}
interface Edge { from: string; to: string; ref: Bridge; }

export interface EngineHandlers {
  onSelect: (id: string | null, kind: "mapping" | "principle" | "none") => void;
}

/* ── Palette for silos synthesized from real domain names ────────────────── */
const SILO_PALETTE = ["#e0a34e", "#5fbf8f", "#5b9bd8", "#4fb8c4", "#a889e0", "#e0708f", "#e08560", "#7c8ce0"];
function colorForDomain(existingCount: number): string {
  return SILO_PALETTE[existingCount % SILO_PALETTE.length];
}

/* Real LangGraph ReAct subagents (core/capabilities.py). Populated at runtime
 * from GET /api/capabilities — this is just the fallback shape/palette. */
const AGENT_PALETTE: Record<Tier, string[]> = {
  deep: ["#4ec9b0", "#a889e0", "#7c8ce0", "#5fbf8f"],
  fast: ["#e0a34e", "#5b9bd8", "#e0708f", "#e08560"],
};
export function colorForCapability(tier: Tier, idx: number): string {
  const pal = AGENT_PALETTE[tier] || AGENT_PALETTE.fast;
  return pal[idx % pal.length];
}

export const PIPELINE: PipelineStep[] = [
  { key: "plan",      label: "Plan",         agent: "deep-reasoner",       desc: "Execution plan" },
  { key: "ingest",    label: "Ingest",       agent: "dark-data-ingestion", desc: "Local dark data" },
  { key: "process",   label: "Process",      agent: "data-processing",     desc: "Chunk artifacts" },
  { key: "translate", label: "Translate",    agent: "rosetta-core",        desc: "Jargon → principles" },
  { key: "extract",   label: "Extract",      agent: "knowledge-graph",     desc: "Neo4j graph + vectors" },
  { key: "analyze",   label: "Analyze",      agent: "analysis",            desc: "Cross-silo isomorphic scan" },
  { key: "write",     label: "Write",        agent: "writing",             desc: "Findings + opportunity gaps" },
  { key: "qa",        label: "Reasoning QA", agent: "deep-reasoner",       desc: "Consistency check" },
];

/* Maps SSE stage keys from /api/chat/stream → pipeline index */
export const SSE_STAGE_MAP: Record<string, number> = {
  planner: 0, started: 0, search: 1, process: 4, analysis: 5, writing: 6, done: 7,
};

export const RIGORS: [string, string][] = [
  ["exploratory", "Exploratory"], ["prisma", "PRISMA 2020"], ["cochrane", "Cochrane"],
];

/* ── Lookup builders — call with the current (React-state) arrays ────────── */
export function buildSiloMap(silos: Silo[]): Record<string, Silo> {
  return Object.fromEntries(silos.map((s) => [s.id, s]));
}
export function buildPMap(principles: Principle[]): Record<string, Principle> {
  return Object.fromEntries(principles.map((p) => [p.pid, p]));
}
export function buildBMap(bridges: Bridge[]): Record<string, Bridge> {
  return Object.fromEntries(bridges.map((b) => [b.id, b]));
}

export function statusColor(st: BridgeStatus): string {
  return st === "confirmed" ? "#5fbf8f" : st === "candidate" ? "#e0a850" : "#6f9bd8";
}
export function statusText(st: BridgeStatus): string {
  return st === "confirmed" ? "CONFIRMED" : st === "candidate" ? "CANDIDATE" : "EVALUATING";
}
export function rgba(hex: string, a: number): string {
  const h = hex.replace("#", "");
  const r = parseInt(h.slice(0, 2), 16), g = parseInt(h.slice(2, 4), 16), b = parseInt(h.slice(4, 6), 16);
  return `rgba(${r},${g},${b},${a})`;
}
export function shortLabel(t: string): string {
  return t.length > 22 ? t.slice(0, 21) + "…" : t;
}

/* ── Mapping helpers: real backend payload → graph data ───────────────────
 * `meta` SSE event shape (BE-1 contract):
 *   hyperedges: [{hyperedge_id, principle_name, member_entity_ids, domain_jargon,
 *                 weight, paper_ids, checklist_tags, source_entity_labels}]
 *   isomorphicClusters: [{cluster_id, shared_principle, hyperedge_ids, domains,
 *                          similarity_score, actionable_insight}]
 * These translate loosely onto the old principle/bridge shape so the existing
 * canvas + inspector UI can render them without a redesign. */
export interface HyperedgeMeta {
  hyperedge_id: string; principle_name: string; member_entity_ids?: string[];
  domain_jargon?: string[]; weight?: number; paper_ids?: string[];
  checklist_tags?: string[]; source_entity_labels?: string[];
}
export interface IsomorphicClusterMeta {
  cluster_id: string; shared_principle: string; hyperedge_ids: string[];
  domains?: string[]; similarity_score?: number; actionable_insight?: string;
}

/** Derive (or reuse) a Silo id/record for a domain name pulled off real data. */
function ensureSilo(domain: string, silos: Silo[], siloIdx: Record<string, Silo>): Silo {
  const id = domain.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "") || "domain";
  if (siloIdx[id]) return siloIdx[id];
  const silo: Silo = {
    id, name: domain, color: colorForDomain(silos.length),
    meta: "from discovery run", files: "—", ingest: 1,
  };
  silos.push(silo); siloIdx[id] = silo;
  return silo;
}

/**
 * Fold one discovery run's hyperedges + isomorphic clusters into the
 * accumulated silos/principles/bridges arrays. Mutates and returns new
 * arrays (does not mutate the inputs) so callers can setState directly.
 */
export function foldDiscoveryResult(
  prev: { silos: Silo[]; principles: Principle[]; bridges: Bridge[] },
  hyperedges: HyperedgeMeta[] | undefined,
  clusters: IsomorphicClusterMeta[] | undefined
): { silos: Silo[]; principles: Principle[]; bridges: Bridge[] } {
  const silos = [...prev.silos];
  const siloIdx = buildSiloMap(silos);
  const principles = [...prev.principles];
  const pIdx = buildPMap(principles);
  const bridges = [...prev.bridges];
  const bIdx = buildBMap(bridges);

  for (const he of hyperedges || []) {
    if (!he || !he.hyperedge_id) continue;
    const domain = (he.domain_jargon && he.domain_jargon[0]) || "unlabeled";
    const silo = ensureSilo(domain, silos, siloIdx);
    const pid = he.hyperedge_id;
    if (!pIdx[pid]) {
      const p: Principle = {
        pid, silo: silo.id,
        label: he.principle_name || pid,
        desc: (he.source_entity_labels || []).join(", ") || (he.checklist_tags || []).join(", ") || "",
      };
      principles.push(p); pIdx[pid] = p;
    }
  }

  for (const cl of clusters || []) {
    if (!cl || !cl.cluster_id || !cl.hyperedge_ids || cl.hyperedge_ids.length < 2) continue;
    const [a, b] = cl.hyperedge_ids;
    if (!pIdx[a] || !pIdx[b]) continue; // need both endpoints resolved to hyperedges above
    if (bIdx[cl.cluster_id]) continue;
    const conf = typeof cl.similarity_score === "number" ? Math.max(0, Math.min(1, cl.similarity_score)) : 0.5;
    const status: BridgeStatus = conf >= 0.8 ? "confirmed" : conf >= 0.6 ? "candidate" : "evaluating";
    const bridge: Bridge = {
      id: cl.cluster_id, core: cl.shared_principle || cl.cluster_id, a, b, conf, status,
      impact: cl.actionable_insight ? shortLabel(cl.actionable_insight) : "—",
      desc: cl.actionable_insight || "",
      collab: [],
    };
    bridges.push(bridge); bIdx[bridge.id] = bridge;
  }

  return { silos, principles, bridges };
}

/* ============================================================================
 * GraphEngine — imperative canvas force graph. Instantiated by page.tsx.
 * Owns no static data: silos/principles/bridges are pushed in via setData()
 * whenever page.tsx accumulates a new discovery result. Starts empty.
 * ========================================================================== */
export class GraphEngine {
  canvas: HTMLCanvasElement;
  handlers: EngineHandlers;
  dpr = 1;
  cam = { x: 0, y: 0, s: 1 };
  cssW = 0; cssH = 0;
  alpha = 1;
  nodes: GNode[] = [];
  edges: Edge[] = [];
  nodeMap: Record<string, GNode> = {};
  siloAnchor: Record<string, { x: number; y: number }> = {};
  selectedId: string | null = null;
  activeSilo: string | null = null;
  filterKey: string = "all";
  hoverId: string | null = null;

  silos: Silo[] = [];
  principles: Principle[] = [];
  bridges: Bridge[] = [];

  private raf = 0;
  private ro: ResizeObserver | null = null;
  private camInit = false;
  private userMoved = false;
  private dragNode: GNode | null = null;
  private dragKind: "node" | "pan" | null = null;
  private panStart = { x: 0, y: 0 };
  private downXY = { x: 0, y: 0 };
  private moved = false;
  private resizeRaf = 0;

  constructor(canvas: HTMLCanvasElement, handlers: EngineHandlers) {
    this.canvas = canvas;
    this.handlers = handlers;
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.buildLayout();
    canvas.addEventListener("pointerdown", this.onDown);
    window.addEventListener("pointermove", this.onMove);
    window.addEventListener("pointerup", this.onUp);
    canvas.addEventListener("wheel", this.onWheel, { passive: false });
    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(canvas);
    this.resize();
    this.loop();
  }

  destroy() {
    cancelAnimationFrame(this.raf);
    this.ro?.disconnect();
    this.canvas.removeEventListener("pointerdown", this.onDown);
    window.removeEventListener("pointermove", this.onMove);
    window.removeEventListener("pointerup", this.onUp);
    this.canvas.removeEventListener("wheel", this.onWheel);
  }

  /* external state setters (called from React) */
  setActiveSilo(id: string | null) { this.activeSilo = id; this.kick(0.35); }
  setFilter(k: string) { this.filterKey = k; }
  setSelected(id: string | null, kind: "mapping" | "principle" | "none") {
    this.selectedId = id;
    this.handlers.onSelect(id, kind);
  }

  /** Replace the engine's data with a fresh accumulated snapshot (silos,
   * principles, bridges) and rebuild the layout. Existing node positions are
   * preserved by id where possible so the graph doesn't jump on every update. */
  setData(silos: Silo[], principles: Principle[], bridges: Bridge[]) {
    const prevPos: Record<string, { x: number; y: number }> = {};
    for (const n of this.nodes) prevPos[n.id] = { x: n.x, y: n.y };
    this.silos = silos;
    this.principles = principles;
    this.bridges = bridges;
    this.buildLayout(prevPos);
    this.kick(0.6);
  }

  private buildLayout(prevPos: Record<string, { x: number; y: number }> = {}) {
    const R = 250;
    this.siloAnchor = {};
    this.silos.forEach((s, i) => {
      const ang = (i / Math.max(1, this.silos.length)) * Math.PI * 2 - Math.PI / 2;
      this.siloAnchor[s.id] = { x: Math.cos(ang) * R, y: Math.sin(ang) * R };
    });
    this.nodes = [];
    this.principles.forEach((p) => {
      const a = this.siloAnchor[p.silo] || { x: 0, y: 0 };
      const prior = prevPos[p.pid];
      const jx = (Math.random() - 0.5) * 70, jy = (Math.random() - 0.5) * 70;
      this.nodes.push({ id: p.pid, type: "p", ref: p, x: prior?.x ?? a.x + jx, y: prior?.y ?? a.y + jy, vx: 0, vy: 0, r: 7 });
    });
    this.nodeMap = {}; this.nodes.forEach((n) => (this.nodeMap[n.id] = n));
    this.bridges.forEach((b) => {
      const na = this.nodeMap[b.a], nb = this.nodeMap[b.b];
      if (!na || !nb) return; // sparse/partial data — skip bridges missing an endpoint
      const prior = prevPos["b:" + b.id];
      this.nodes.push({ id: "b:" + b.id, type: "b", ref: b, x: prior?.x ?? (na.x + nb.x) / 2, y: prior?.y ?? (na.y + nb.y) / 2, vx: 0, vy: 0, r: 9 });
    });
    this.nodeMap = {}; this.nodes.forEach((n) => (this.nodeMap[n.id] = n));
    this.edges = [];
    this.bridges.forEach((b) => {
      if (!this.nodeMap[b.a] || !this.nodeMap[b.b]) return;
      this.edges.push({ from: "b:" + b.id, to: b.a, ref: b });
      this.edges.push({ from: "b:" + b.id, to: b.b, ref: b });
    });
  }

  private resize() {
    const c = this.canvas; if (!c) return;
    const r = c.getBoundingClientRect();
    if (r.width < 2 || r.height < 2) {
      cancelAnimationFrame(this.resizeRaf);
      this.resizeRaf = requestAnimationFrame(() => this.resize());
      return;
    }
    this.cssW = r.width; this.cssH = r.height;
    c.width = Math.round(r.width * this.dpr);
    c.height = Math.round(r.height * this.dpr);
    if (!this.camInit) { this.camInit = true; this.fitView(); setTimeout(() => { if (!this.userMoved) this.fitView(); }, 900); }
    else if (!this.userMoved) this.fitView();
    this.kick();
  }

  fitView() {
    if (!this.nodes.length || !this.cssW) return;
    let minX = 1e9, minY = 1e9, maxX = -1e9, maxY = -1e9;
    for (const n of this.nodes) { minX = Math.min(minX, n.x); minY = Math.min(minY, n.y); maxX = Math.max(maxX, n.x); maxY = Math.max(maxY, n.y); }
    const pad = 54;
    const w = (maxX - minX) + pad * 2, h = (maxY - minY) + pad * 2;
    const topInset = 78, botInset = 16;
    const availH = this.cssH - topInset - botInset;
    const s = Math.max(0.32, Math.min(1.5, Math.min(this.cssW / w, availH / h)));
    const midX = (minX + maxX) / 2, midY = (minY + maxY) / 2;
    this.cam.s = s; this.cam.x = this.cssW / 2 - midX * s; this.cam.y = topInset + availH / 2 - midY * s;
  }

  private toScreen(n: GNode) { return { x: n.x * this.cam.s + this.cam.x, y: n.y * this.cam.s + this.cam.y }; }
  private toWorld(px: number, py: number) { return { x: (px - this.cam.x) / this.cam.s, y: (py - this.cam.y) / this.cam.s }; }
  private kick(a = 0.5) { this.alpha = Math.max(this.alpha || 0, a); }

  private loop = () => { this.tick(); this.draw(); this.raf = requestAnimationFrame(this.loop); };

  private tick() {
    const nodes = this.nodes; if (!nodes.length) return;
    const a = Math.max(this.alpha || 0, 0.03);
    this.alpha = (this.alpha || 0) * 0.96;
    for (let i = 0; i < nodes.length; i++) {
      const p = nodes[i]; if (p === this.dragNode) continue;
      let fx = 0, fy = 0;
      for (let j = 0; j < nodes.length; j++) {
        if (i === j) continue;
        const q = nodes[j];
        let dx = p.x - q.x, dy = p.y - q.y;
        let d2 = dx * dx + dy * dy; if (d2 < 0.01) { d2 = 0.01; dx = Math.random() - 0.5; dy = Math.random() - 0.5; }
        const f = 2600 / d2, d = Math.sqrt(d2);
        fx += (dx / d) * f; fy += (dy / d) * f;
      }
      if (p.type === "p") {
        const an = this.siloAnchor[p.ref.silo];
        if (an) { fx += (an.x - p.x) * 0.012; fy += (an.y - p.y) * 0.012; }
      }
      p._fx = fx; p._fy = fy;
    }
    for (const e of this.edges) {
      const A = this.nodeMap[e.from], B = this.nodeMap[e.to];
      if (!A || !B) continue;
      const dx = B.x - A.x, dy = B.y - A.y;
      const d = Math.sqrt(dx * dx + dy * dy) || 0.01;
      const diff = (d - 78) / d * 0.045;
      const fx = dx * diff, fy = dy * diff;
      if (A !== this.dragNode) { A._fx! += fx; A._fy! += fy; }
      if (B !== this.dragNode) { B._fx! -= fx; B._fy! -= fy; }
    }
    for (const b of this.bridges) {
      const bn = this.nodeMap["b:" + b.id]; if (!bn || bn === this.dragNode) continue;
      const na = this.nodeMap[b.a], nb = this.nodeMap[b.b];
      if (!na || !nb) continue;
      const mx = (na.x + nb.x) / 2, my = (na.y + nb.y) / 2;
      bn._fx! += (mx - bn.x) * 0.05; bn._fy! += (my - bn.y) * 0.05;
    }
    for (const p of nodes) {
      if (p === this.dragNode) continue;
      p.vx = (p.vx + p._fx! * a) * 0.82;
      p.vy = (p.vy + p._fy! * a) * 0.82;
      p.x += p.vx; p.y += p.vy;
    }
  }

  private isNeighbor(id: string, sel: string): boolean {
    const selNode = this.nodeMap[sel]; if (!selNode) return false;
    if (selNode.type === "b") return id === selNode.ref.a || id === selNode.ref.b;
    for (const b of this.bridges) {
      if (b.a === sel || b.b === sel) {
        if (id === "b:" + b.id) return true;
        if (id === b.a || id === b.b) return true;
      }
    }
    return false;
  }

  private draw() {
    const ctx = this.canvas.getContext("2d"); if (!ctx) return;
    if (!this.canvas.width || !this.cssW) return;
    const W = this.cssW, H = this.cssH, dpr = this.dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "rgba(255,255,255,0.028)";
    const step = 34;
    const ox = ((this.cam.x % step) + step) % step, oy = ((this.cam.y % step) + step) % step;
    for (let x = ox; x < W; x += step) for (let y = oy; y < H; y += step) { ctx.beginPath(); ctx.arc(x, y, 0.9, 0, 6.28); ctx.fill(); }

    if (!this.nodes.length) return;

    const siloMap = buildSiloMap(this.silos);
    const pMap = buildPMap(this.principles);
    const sel = this.selectedId, activeSilo = this.activeSilo, fk = this.filterKey, hov = this.hoverId;
    const dim = (n: GNode) => {
      let d = 1;
      if (activeSilo) {
        const inSilo = n.type === "p" ? n.ref.silo === activeSilo
          : (pMap[n.ref.a]?.silo === activeSilo || pMap[n.ref.b]?.silo === activeSilo);
        if (!inSilo) d *= 0.16;
      }
      if (fk !== "all" && n.type === "b" && n.ref.status !== fk) d *= 0.16;
      if (sel) {
        const isSel = n.id === sel, isNeighbor = this.isNeighbor(n.id, sel);
        if (!isSel && !isNeighbor) d *= 0.28;
      }
      return d;
    };

    for (const e of this.edges) {
      const A = this.nodeMap[e.from], B = this.nodeMap[e.to];
      if (!A || !B) continue;
      const sa = this.toScreen(A), sb = this.toScreen(B);
      const pnode = A.type === "p" ? A : B;
      const col = siloMap[pnode.ref.silo]?.color || "#7c8ce0";
      const d = Math.min(dim(A), dim(B));
      const bn = A.type === "b" ? A : B;
      ctx.beginPath(); ctx.moveTo(sa.x, sa.y); ctx.lineTo(sb.x, sb.y);
      ctx.strokeStyle = rgba(col, 0.32 * d);
      ctx.lineWidth = bn.ref.status === "confirmed" ? 1.6 : 1.1;
      ctx.setLineDash(bn.ref.status !== "confirmed" ? [4, 4] : []);
      ctx.stroke(); ctx.setLineDash([]);
    }

    for (const n of this.nodes) {
      const s = this.toScreen(n), d = dim(n);
      const isSel = n.id === sel, isHov = n.id === hov;
      if (n.type === "p") {
        const col = siloMap[n.ref.silo]?.color || "#7c8ce0", rr = n.r * this.cam.s;
        if (isSel || isHov) { ctx.beginPath(); ctx.arc(s.x, s.y, rr + 6, 0, 6.28); ctx.fillStyle = rgba(col, 0.14); ctx.fill(); }
        ctx.beginPath(); ctx.arc(s.x, s.y, rr, 0, 6.28);
        ctx.fillStyle = rgba("#0b0e13", d); ctx.fill();
        ctx.lineWidth = isSel ? 2.4 : 1.8; ctx.strokeStyle = rgba(col, d); ctx.stroke();
        ctx.beginPath(); ctx.arc(s.x, s.y, rr * 0.42, 0, 6.28); ctx.fillStyle = rgba(col, d); ctx.fill();
        if (this.cam.s > 0.62 || isSel || isHov) {
          ctx.font = '500 10.5px "IBM Plex Mono", monospace';
          ctx.textAlign = "center"; ctx.textBaseline = "top";
          ctx.fillStyle = rgba("#aeb6c4", 0.85 * d);
          ctx.fillText(shortLabel(n.ref.label), s.x, s.y + rr + 5);
        }
      } else {
        const col = statusColor(n.ref.status), rr = (n.r + (isSel ? 2 : 0)) * this.cam.s;
        ctx.save(); ctx.translate(s.x, s.y); ctx.rotate(Math.PI / 4);
        if (isSel || isHov) { ctx.fillStyle = rgba(col, 0.18); ctx.fillRect(-rr - 6, -rr - 6, (rr + 6) * 2, (rr + 6) * 2); }
        ctx.fillStyle = rgba(col, d); ctx.fillRect(-rr, -rr, rr * 2, rr * 2);
        ctx.strokeStyle = rgba("#0a0c11", d); ctx.lineWidth = 2; ctx.strokeRect(-rr, -rr, rr * 2, rr * 2);
        ctx.restore();
        if (n.ref.status === "confirmed" || isSel || isHov) {
          ctx.font = '600 11px "IBM Plex Sans", sans-serif';
          ctx.textAlign = "center"; ctx.textBaseline = "bottom";
          ctx.fillStyle = rgba("#eef1f6", 0.95 * d);
          ctx.fillText(n.ref.core, s.x, s.y - rr - 7);
        }
      }
    }
  }

  private hitTest(px: number, py: number): GNode | null {
    for (let i = this.nodes.length - 1; i >= 0; i--) {
      const n = this.nodes[i], s = this.toScreen(n);
      const rr = (n.r + 4) * this.cam.s + 3;
      if ((px - s.x) ** 2 + (py - s.y) ** 2 <= rr * rr) return n;
    }
    return null;
  }
  private localPt(e: PointerEvent | WheelEvent) { const r = this.canvas.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; }

  private onDown = (e: PointerEvent) => {
    this.userMoved = true;
    const p = this.localPt(e), n = this.hitTest(p.x, p.y);
    this.downXY = { x: e.clientX, y: e.clientY }; this.moved = false;
    if (n) { this.dragNode = n; n.vx = 0; n.vy = 0; this.dragKind = "node"; }
    else { this.dragKind = "pan"; this.panStart = { x: e.clientX - this.cam.x, y: e.clientY - this.cam.y }; }
    this.canvas.style.cursor = "grabbing";
  };
  private onMove = (e: PointerEvent) => {
    if (this.dragKind === "node" && this.dragNode) {
      const p = this.localPt(e), w = this.toWorld(p.x, p.y);
      this.dragNode.x = w.x; this.dragNode.y = w.y; this.kick(0.4);
      if (Math.abs(e.clientX - this.downXY.x) + Math.abs(e.clientY - this.downXY.y) > 3) this.moved = true;
    } else if (this.dragKind === "pan") {
      this.cam.x = e.clientX - this.panStart.x; this.cam.y = e.clientY - this.panStart.y;
      if (Math.abs(e.clientX - this.downXY.x) + Math.abs(e.clientY - this.downXY.y) > 3) this.moved = true;
    } else if (this.canvas) {
      const p = this.localPt(e), n = this.hitTest(p.x, p.y), nid = n ? n.id : null;
      if (nid !== this.hoverId) { this.hoverId = nid; this.canvas.style.cursor = n ? "pointer" : "grab"; }
    }
  };
  private onUp = () => {
    if (this.dragKind === "node" && this.dragNode && !this.moved) {
      const n = this.dragNode;
      this.setSelected(n.id, n.type === "b" ? "mapping" : "principle");
    }
    if (this.dragNode) { this.dragNode.vx = 0; this.dragNode.vy = 0; }
    this.dragNode = null; this.dragKind = null;
    if (this.canvas) this.canvas.style.cursor = this.hoverId ? "pointer" : "grab";
  };
  private onWheel = (e: WheelEvent) => {
    e.preventDefault(); this.userMoved = true;
    const p = this.localPt(e), w = this.toWorld(p.x, p.y);
    const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
    this.cam.s = Math.max(0.35, Math.min(2.4, this.cam.s * factor));
    this.cam.x = p.x - w.x * this.cam.s; this.cam.y = p.y - w.y * this.cam.s;
  };

  selectById(id: string) {
    const n = this.nodeMap[id]; if (!n) return;
    this.setSelected(id, n.type === "b" ? "mapping" : "principle");
  }
  private zoomBy(f: number) {
    const cx = this.cssW / 2, cy = this.cssH / 2, w = this.toWorld(cx, cy);
    this.cam.s = Math.max(0.35, Math.min(2.4, this.cam.s * f));
    this.cam.x = cx - w.x * this.cam.s; this.cam.y = cy - w.y * this.cam.s;
  }
  zoomIn = () => this.zoomBy(1.2);
  zoomOut = () => this.zoomBy(1 / 1.2);
  zoomReset = () => { this.userMoved = false; this.fitView(); this.setSelected(null, "none"); this.activeSilo = null; this.filterKey = "all"; this.kick(0.6); };
}
