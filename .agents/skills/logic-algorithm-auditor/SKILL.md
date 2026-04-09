---
name: logic-algorithm-auditor
description: Scan a codebase for likely logical bugs, algorithmic mistakes, performance traps, and edge-case failures, then propose practical workarounds or fixes. Use this skill whenever the user asks to audit code quality, find hidden logic errors, review algorithm correctness, identify complexity bottlenecks, or request risk-focused code review even if they do not explicitly say "algorithm".
---

# Logic and Algorithm Auditor

Use this skill to run a focused, evidence-based code audit for correctness and algorithmic risk.

## Goals
- Find likely logical and algorithmic defects before runtime failures.
- Prioritize issues by impact and confidence.
- Recommend concrete workarounds and durable fixes.
- Provide tests that reproduce and prevent regressions.

## Inputs to Confirm
Collect these before deep analysis:
- Scope: full repo, folder, or specific files.
- Risk focus: correctness, performance, concurrency, data integrity, or security-adjacent logic.
- Constraints: keep API stable, avoid new deps, target runtime limits.
- Evidence preference: static reasoning only, or include test execution if available.

If missing, proceed with best-effort defaults and state assumptions.

Default assumptions when not specified:
- Scan scope is limited to user-specified folders/files, not full repo.
- Run tests/commands when available to validate high-risk findings.

## Audit Workflow
1. Map execution paths.
- Identify entry points and high-impact modules.
- Trace data flow through transforms, conditionals, loops, caches, and retries.
- Note implicit contracts (invariants, ordering guarantees, idempotency assumptions).

2. Locate high-risk logic patterns.
- Boundary handling: empty inputs, nulls, singletons, max sizes, overflow/underflow.
- State transitions: stale state, race windows, forgotten resets, partial updates.
- Control flow: impossible branches, dead branches, inverted conditions, early-return masking.
- Error handling: swallowed exceptions, retry storms, fallback loops that hide corruption.
- Collection logic: mutation during iteration, unstable sort/group assumptions, key collisions.

3. Validate algorithmic soundness.
- Correctness: does the algorithm satisfy intended invariants for all input classes?
- Complexity: identify hot paths where time or space complexity is likely unacceptable.
- Termination: detect non-terminating or unbounded loops/retries.
- Determinism: check for nondeterministic behavior where reproducibility is required.

4. Produce workaround and fix options.
- Short-term workaround: low-risk patch or guard that reduces immediate impact.
- Long-term fix: structural correction with clearer invariants and reduced complexity risk.
- Tradeoffs: expected performance, maintenance cost, and migration risk.

5. Define verification checks.
- Add failing-first test ideas per issue.
- Include at least one boundary case and one representative normal case.
- If possible, include complexity guard tests for hot paths.

## Decision Logic
Use this branching when reporting findings:
- High severity + high confidence:
  - Recommend immediate patch and targeted regression tests.
- High severity + medium/low confidence:
  - Recommend fast instrumentation, assertive logging, or reproducer before patching.
- Medium severity:
  - Batch with related fixes and add tests before next release.
- Low severity:
  - Document and monitor; fix opportunistically.

If multiple plausible root causes exist, present up to 3 hypotheses ranked by evidence strength.

## Citation Policy by Impact
- High impact findings:
  - Provide precise file, function/symbol, and line-level citation when available.
  - Include the exact condition or path that triggers failure.
- Medium impact findings:
  - Provide file and function/symbol citation at minimum.
  - Add line-level citation if cheap to obtain.
- Low impact findings:
  - Function/module-level citation is acceptable.

## Report Format
Use this exact structure:

# Logic and Algorithm Audit Report
## Scope and assumptions
## Findings (ordered by severity)
For each finding include:
- Location: file and function/symbol
- Risk: correctness, performance, concurrency, or data integrity
- Evidence: concrete code behavior and failure mode
- Workaround: immediate mitigation
- Durable fix: recommended long-term change
- Verification: tests to add

## Open questions
## Residual risk

## Quality Bar
Before finalizing, verify:
- Every finding is tied to concrete code evidence, not generic advice.
- Every high-severity finding has both workaround and durable fix.
- Suggested fixes preserve stated constraints or clearly call out exceptions.
- At least one test recommendation exists per finding.
- Findings are ordered by severity, then confidence.
- Citation depth follows impact tier policy.

## Prompt Examples
- "Audit this repo for algorithmic bugs that could break at scale."
- "Review the scheduler module for logical race conditions and complexity traps."
- "Find hidden correctness bugs in data processing and suggest short-term workarounds."
