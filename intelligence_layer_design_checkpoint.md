# Design Checkpoint — Intelligence Layer (Research Harness)

**Checkpoint ID:** `CP-INT-01`
**Status:** Proposed — gate not yet passed
**Date:** 2026-07-07
**Deciders:** engineering owner + human research lead / domain expert
**Precedes:** first build sprint on the cognitive core
**Builds on existing harness:** six-stage loop (Input Parser → Think → Route → Gate → Act → Observe → Output), four-tier adaptive RAG + complexity classifier, subagent orchestration, OpenTelemetry tracing, MCP tool layer, zero-trust Docker sandbox

---

## 0. Purpose of this checkpoint

This is a **gate, not a spec dump.** The intelligence layer is where the harness *thinks and decides* — the part that can drift into confident nonsense over a long horizon, bail on the first exception, or manufacture plausible-but-false results faster than anyone can check them. It carries essentially all of the novelty/verifiability risk.

Nothing gets built past this checkpoint until the exit criteria in §8 are met. The purpose is to lock the decisions that are *expensive to reverse once they're load-bearing* — verification topology, provenance, and human gates — before the rest of the system depends on them.

**Scoping note:** "intelligence layer" here means the cognitive/decision core only. The execution layer (simulators, instruments), the knowledge/provenance store, and the governance layer (budgets, kill switch, sandbox) are separate checkpoints, consumed here through contracts (§4). If you intended "intelligence layer" to include the tool/execution layer, this checkpoint is narrower than that and CP-EXEC-01 would cover the rest.

---

## 1. Scope & boundaries

**In scope — the intelligence layer owns cognition and decision:**
task interpretation, knowledge grounding, hypothesis generation, hypothesis triage, experiment-design reasoning, subagent orchestration, result interpretation, and its own long-horizon memory/state.

**Out of scope — owned by adjacent layers, consumed via contract:**

| Layer | Relationship |
|---|---|
| Execution | Intelligence *requests* execution (sim/instrument/code); never actuates directly. |
| Knowledge / Provenance | Intelligence *reads* (grounding) and *writes* (results, lineage). |
| Governance | Intelligence *operates under* budgets, kill switch, sandbox, human gates; cannot override them. |

**Mapping to your existing loop:** the intelligence layer *is* the **Think → Route → Gate** core, plus the memory that persists across turns. **Act** hands off to execution; **Observe** feeds the interpreter; **Input Parser / Output** are the boundary adapters.

---

## 2. Component decomposition

| # | Component | Responsibility | Loop stage | Model tier | Emits provenance |
|---|---|---|---|---|---|
| 1 | Goal Interpreter | Research goal → structured objective + success criteria + constraints | Input Parser | cheap/local | yes |
| 2 | Grounding Subsystem | Dispatch four-tier RAG via complexity classifier; produce a **frontier snapshot** (known / contested / unknown + citations) | Think | tiered | yes |
| 3 | Hypothesis Generator | Produce a **diverse portfolio** against the frontier snapshot; dedup + coverage | Think | strong | yes |
| 4 | Triage Gate | Two-axis (novelty × validity) scoring with ≥1 **external** signal per axis; emit ranked shortlist | Gate | strong + external tools | yes |
| 5 | Experiment-Design Reasoner | Choose experiments that maximally discriminate surviving hypotheses per unit cost; emit execution requests | Route | strong + optimizer | yes |
| 6 | Subagent Orchestrator | Dispatch/aggregate domain subagents; async queue + poll; manage fan-out under budget | Route/Act boundary | cheap coordinator | yes |
| 7 | Interpreter | Execution results → evidence for/against each hypothesis; update ledger + knowledge model | Observe | strong + stats tools | yes |
| 8 | Memory/State Manager | Working memory, hypothesis ledger, provenance links, crash-safe checkpoints | cross-cutting | infra | n/a |

---

## 3. Locked design decisions

ADR-style. Each is a decision that shapes the build; the rejected alternative is the one a naive version would pick.

**D1 — Generator and evaluator are separated.**
The component that *generates* a hypothesis never *scores* it. Different context, different prompt, ideally a different model. Rationale: controlled human studies show LLM self-evaluation is unreliable — the model that produced an idea cannot be trusted to judge it. *Rejected:* single-model self-critique (the "grade your own homework" pattern that quietly passes everything).

**D2 — Triage is two orthogonal axes, each with an external signal.**

| Axis | Question | External signal (required) |
|---|---|---|
| Novelty | New, or already known / obvious / new-only-to-the-model? | Search against knowledge model + live literature |
| Validity | Mechanistically sound, or confident nonsense? | Constraint check / statistical test / simulator pre-check / human |

A hypothesis needs both. Any LLM-internal score is a *prior*, never the verdict. *Rejected:* a single fused LLM "quality" score.

**D3 — Generation optimizes a portfolio for diversity, not a single best.**
Output is a *set* selected for coverage of the hypothesis space, with explicit semantic dedup. Rationale: LLMs plateau on idea diversity when generation is scaled — "generate 1,000" yields mostly near-duplicates after an initial burst. Diversity is an objective, not a side effect of volume. *Rejected:* over-generate-and-rank on the assumption that scale buys variety.

**D4 — Route by cognitive difficulty; reserve the strong model.**
Extend the existing complexity classifier to tier *cognitive* load, not just retrieval load: cheap/local model for parsing, routing, coordination, bookkeeping; strong model only for generation, triage reasoning, and interpretation. This is the cost/step lever. *Also:* if you target Fable 5, its safeguards can route some chemistry/bio queries to a different backend mid-run — treat "the strong model" as an interface that may occasionally return a different model than expected, and don't hard-code assumptions about one consistent backend.

**D5 — Provenance is a first-class output of every cognitive step.**
Each step emits a lineage record (hypothesis ← frontier snapshot ← evidence ← decision) as an OpenTelemetry span linked to the ledger. A conclusion that can't be traced to the exact data, code, and run that produced it does not count as a result. This is likely your actual moat; it is not an afterthought. *Rejected:* logging that captures *what happened* but not *what justified it*.

**D6 — Human gates at three fixed decision points.**
The layer *pauses and requests* rather than proceeding autonomously past: (a) goal confirmation, (b) hypothesis-portfolio approval **before any experiment is committed**, (c) interpretation sign-off. Division by comparative advantage: agents do breadth/volume/bookkeeping; humans do direction and the "is this actually interesting and true" call. *Rejected:* full autonomy with post-hoc human review.

**D7 — Durable hypothesis ledger + crash-safe checkpoint.**
A long run resumes; it does not restart. The Memory/State Manager serializes ledger + working memory + a cursor into the current plan (see §5). *Rejected:* in-context-only state that evaporates on crash or context overflow.

**D8 — Budget and kill-switch awareness.**
The layer tracks a cost/step budget and **degrades or halts** rather than running away; the governance layer can interrupt at any point. Parallel subagents on a long horizon run away in both dollars and wall-clock without this. *Rejected:* unbounded fan-out.

---

## 4. Interface contracts

| Adjacent layer | Intelligence sends | Receives | Invariant |
|---|---|---|---|
| Execution | Execution request (params, dry-run flag, budget) | Structured result + status | Never actuates directly; physical actions require dry-run + human confirm |
| Knowledge / Provenance | Grounding query; result write; lineage span | Frontier snapshot; prior results | Every write carries a provenance record |
| Governance | Current spend/step count; gate-pause request | Budget verdict; kill signal; human decision | Governance verdicts are non-overridable |

---

## 5. Memory & state model

The machine-readable sibling of your `memory_log.md` idiom.

- **Working memory** — current objective, active frontier snapshot, active hypotheses, current plan cursor. Bounded; summarized on overflow.
- **Hypothesis ledger** — append-only. Each entry: `id`, statement, novelty score + evidence links, validity score + evidence links, status (`generated | triaged | testing | supported | refuted | parked`), and lineage span IDs.
- **Provenance graph** — OTel spans linked to ledger entries; reconstructs the full justification chain for any conclusion.
- **Checkpoint** — serialized {ledger + working memory + plan cursor}. **Resume semantics:** on restart, rehydrate ledger and working memory, resume at the cursor; in-flight execution requests are re-checked against results before re-dispatch (idempotency).

---

## 6. Open questions (resolve before or at build)

1. **External validity check, per domain** — constraint solver? simulator pre-check? statistical prior? human? Likely domain-specific; pick one for the first domain.
2. **Evaluator implementation** — separate local model, stronger hosted model, or a debate/tournament ensemble? Trade cost vs. reliability.
3. **Novelty search freshness** — how is the knowledge model kept current so novelty judgments aren't stale?
4. **Hypothesis schema** — canonical representation (variables, predicted effect, testable prediction) that generator, triage, and design reasoner all share.
5. **Subagent parallelism ceiling** — max fan-out vs. budget; static cap or adaptive.

---

## 7. Risks & failure modes → mitigations

| Failure mode | Mitigation (already in design) |
|---|---|
| Self-evaluation passes its own weak hypotheses | D1 (separation) + D2 (external signals) |
| Diversity collapse on scaled generation | D3 (portfolio + dedup as objective) |
| Long-context drift over a multi-hour run | D7 (ledger as source of truth, not context) |
| Hallucinated tool schemas / bad execution requests | Contract validation at the Execution interface (§4) |
| Runaway cost / wall-clock | D8 (budget + kill switch) |
| Over-trust of autonomy on consequential actions | D6 (human gate before experiment commit) |
| Untraceable conclusion | D5 (provenance per step) |

---

## 8. Exit criteria — the gate

All must be true to pass `CP-INT-01` and start building the cognitive core.

- [ ] The four interface contracts (§4) are specified with concrete schemas.
- [ ] Generator↔evaluator separation (D1) exists at least as a stub with distinct prompts/contexts.
- [ ] Triage produces two separate scores (D2), and at least **one external validity check works end-to-end on one domain**.
- [ ] Provenance schema (D5) is defined and a single hypothesis can be traced snapshot → evidence → decision.
- [ ] Hypothesis ledger + checkpoint (D7) round-trips: a run can be killed and resumed without restarting.
- [ ] Budget tracking + kill switch (D8) are wired and demonstrably halt a run.
- [ ] Human-gate mechanism (D6) can pause execution and block on a decision.
- [ ] The eval harness (§9) catches at least one seeded known-bad hypothesis.

---

## 9. Eval plan for the layer itself

Build this *before* trusting the layer end to end. Borrow methodology from IdeaBench / LiveIdeaBench and the Si et al. execute-the-ideas design.

- **Plant test** — seed a known-false and a known-true hypothesis; triage must refute the plant and pass the true one. This is the core "does the gate work" check.
- **Diversity metric** — measure semantic spread of a generated portfolio; regression-test that scaling generation doesn't collapse it.
- **Provenance completeness** — every conclusion in a run resolves to a full lineage chain; fail the run if any doesn't.
- **Ablation** — remove the external verification signal and confirm triage quality measurably degrades (proves the external gate is doing work, not decoration).

---

## 10. Sign-off

**Status:** Proposed → awaiting §8.
**Next checkpoint:** `CP-EXEC-01` (execution/tool layer) or `CP-INT-02` (post-first-build review of the cognitive core).
