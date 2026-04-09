from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .base_agent import ResearchAgent


class ResearcherAgent(ResearchAgent):
    """Agent responsible for conducting research and analysis."""

    MEMORY_PATH = Path("logs/researcher_agent_memory.json")
    MEMORY_LIMIT = 25

    def __init__(self):
        super().__init__(
            name="researcher",
            description="Conducts research, analyzes data, and generates insights",
        )

    def get_required_fields(self) -> List[str]:
        # get the required field from knowledge graph we store to qudrant after performing litrary review over paper

        return [
            "research_topic",
            "research_goals",
        ]

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process research request and return findings."""
        self.logger.info(f"Processing research request: {input_data}")
        await super().process(input_data)

        # Extract research parameters
        topic = input_data["research_topic"]
        goals = input_data["research_goals"]
        rigor_level = input_data.get("rigor_level", "exploratory")
        execution_mode = input_data.get("mode", "agentic")
        execute_pipeline = input_data.get("execute_pipeline", True)

        memory_path = Path(input_data.get("memory_path", self.MEMORY_PATH))
        history = self._load_memory(memory_path)

        # Goal decomposition + memory-informed plan
        plan = await self._decompose_goals(topic, goals, history)

        # Conduct research based on goals
        findings = await self._conduct_research(
            topic,
            goals,
            plan,
            rigor_level=rigor_level,
            mode=execution_mode,
            execute_pipeline=execute_pipeline,
        )

        # Analyze findings
        analysis = await self._analyze_findings(findings, history)

        # Generate insights
        insights = await self._generate_insights(analysis, history)

        self._persist_memory(
            memory_path,
            topic,
            goals,
            plan,
            findings,
            analysis,
            insights,
        )

        return {
            "plan": plan,
            "findings": findings,
            "analysis": analysis,
            "insights": insights,
        }

    async def _conduct_research(
        self,
        topic: str,
        goals: List[str],
        plan: Dict[str, Any],
        *,
        rigor_level: str = "exploratory",
        mode: str = "agentic",
        execute_pipeline: bool = True,
    ) -> Dict[str, Any]:
        """Conduct research on the given topic."""
        self.logger.info(f"Conducting research on {topic} (mode={mode})")

        findings: Dict[str, Any] = {
            "status": "skipped" if not execute_pipeline else "pending",
            "plan": plan,
            "sources": [],
            "data": {},
            "observations": [],
            "workflow": plan.get("workflow"),
        }

        if not execute_pipeline:
            findings["observations"].append(
                "Execution disabled; returning plan-only findings."
            )
            return findings

        try:
            from core.graph import run_research_pipeline

            result = await run_research_pipeline(
                project_name=f"{topic[:32]}-auto",
                research_topic=topic,
                research_goals=goals,
                rigor_level=rigor_level,
                interactive=False,
                mode=mode,
            )
            findings["status"] = "completed"
            findings["data"] = result
            if isinstance(result, dict):
                audit_log = result.get("audit_log") or result.get("audit_trail", [])
                if audit_log:
                    findings["observations"].extend(audit_log)
        except Exception as exc:  # pragma: no cover - defensive fallback
            self.logger.warning(
                "Research pipeline failed; returning structured fallback.", exc_info=exc
            )
            findings["status"] = "partial"
            findings["error"] = str(exc)
            findings["observations"].extend(plan.get("steps", []))

        return findings

    async def _analyze_findings(
        self, findings: Dict[str, Any], history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze research findings."""
        self.logger.info("Analyzing research findings")
        coverage_score, gaps = self._score_coverage(findings)

        prior_topics = {h["topic"] for h in history}
        reuse_signal = any(
            t.lower() in findings.get("plan", {}).get("topic", "").lower()
            for t in prior_topics
        )

        return {
            "status": "completed",
            "coverage_score": coverage_score,
            "gaps": gaps,
            "reuse_signal": reuse_signal,
            "next_actions": self._propose_next_actions(gaps, findings),
        }

    async def _generate_insights(
        self, analysis: Dict[str, Any], history: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate insights from analysis."""
        self.logger.info("Generating insights")
        insights: List[str] = []

        if analysis.get("coverage_score", 0) < 0.6:
            insights.append(
                "Coverage is thin; prioritize targeted retrieval for weak PRISMA labels."
            )

        if analysis.get("reuse_signal"):
            insights.append("Leveraging prior work — reuse embeddings and graph paths.")

        if not insights:
            insights.append("Baseline review ready for synthesis.")

        if history:
            insights.append("Continuous learning: appended run to researcher memory.")

        return insights

    async def _decompose_goals(
        self, topic: str, goals: List[str], history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Break high-level goals into actionable steps using heuristics + memory."""
        normalized_goals = goals or ["literature review"]
        steps = []

        for goal in normalized_goals:
            step = {
                "name": goal,
                "objective": f"Address goal: {goal}",
                "success_criteria": [
                    "Sources gathered",
                    "Evidence extracted",
                    "Findings synthesized",
                ],
            }
            if "analy" in goal.lower() or "experiment" in goal.lower():
                step["success_criteria"].append("Statistical checks completed")
            steps.append(step)

        steps.append(
            {
                "name": "evidence_gap_scan",
                "objective": "Detect missing PRISMA labels and backfill literature",
                "success_criteria": ["No label has fewer than 3 entities"],
            }
        )

        workflow = self._select_workflow(normalized_goals)
        lessons = self._surface_lessons(topic, history)

        return {
            "topic": topic,
            "workflow": workflow,
            "steps": steps,
            "lessons": lessons,
        }

    def _select_workflow(self, goals: List[str]) -> str:
        """Choose a workflow type based on stated goals."""
        goals_text = " ".join(goals).lower()
        if any(key in goals_text for key in ["experiment", "statistical", "analysis"]):
            return "data_analysis"
        return "literature_review"

    def _score_coverage(self, findings: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Estimate how complete the findings are and flag gaps."""
        observations = findings.get("observations", [])
        audit = findings.get("data", {}).get("audit_log", [])
        signals = observations if isinstance(observations, list) else []
        if audit:
            signals.extend(audit)

        completed = findings.get("status") == "completed"
        base_score = 0.75 if completed else 0.4

        if any("needs_more_papers" in str(s) for s in signals):
            base_score -= 0.2

        gaps: List[str] = []
        if base_score < 0.6:
            gaps.append("Insufficient evidence coverage")
        if not signals:
            gaps.append("No audit trail available")
        return max(base_score, 0.0), gaps

    def _propose_next_actions(
        self, gaps: List[str], findings: Dict[str, Any]
    ) -> List[str]:
        """Recommend next actions based on identified gaps."""
        actions = []
        workflow = findings.get("workflow") or "literature_review"
        if "Insufficient evidence coverage" in gaps:
            actions.append(f"Loop back to {workflow} with tighter inclusion criteria.")
        if "No audit trail available" in gaps:
            actions.append("Enable audit logging for downstream runs.")
        if not actions:
            actions.append("Proceed to writing and synthesis.")
        return actions

    def _surface_lessons(self, topic: str, history: List[Dict[str, Any]]) -> List[str]:
        """Return memory-driven lessons relevant to the topic."""
        lessons: List[str] = []
        for entry in history:
            if topic.lower() in entry.get("topic", "").lower():
                lessons.append(entry.get("summary", "Reused prior run insights."))
        return lessons[-3:]

    def _load_memory(self, path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        try:
            return json.loads(path.read_text())
        except Exception:  # pragma: no cover - defensive parsing guard
            return []

    def _persist_memory(
        self,
        path: Path,
        topic: str,
        goals: List[str],
        plan: Dict[str, Any],
        findings: Dict[str, Any],
        analysis: Dict[str, Any],
        insights: List[str],
    ) -> None:
        history = self._load_memory(path)
        history.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "topic": topic,
                "goals": goals,
                "plan": plan,
                "findings_status": findings.get("status"),
                "coverage_score": analysis.get("coverage_score"),
                "insights": insights[:5],
                "summary": insights[0] if insights else "",
            }
        )
        history = history[-self.MEMORY_LIMIT :]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(history, indent=2))
