"""Collaboration Agent – supports team-based research by managing
shared context, task delegation, and progress tracking."""

from core.base_agent import ResearchAgent
from typing import Dict, Any, List


class CollaborationAgent(ResearchAgent):
    """Agent that coordinates team-based research projects.

    Supported actions:
      - ``assign_task``      – delegate a research task to a team member
      - ``track_progress``   – report on project-wide progress
      - ``share_context``    – share research context across team members
    """

    def __init__(self):
        super().__init__(
            name="collaboration",
            description=(
                "Coordinates team research by managing task delegation, "
                "shared context, and progress tracking"
            ),
        )

    def get_required_fields(self) -> List[str]:
        return ["action"]

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        await super().process(input_data)
        action = input_data["action"]

        if action == "assign_task":
            return await self._assign_task(input_data)
        elif action == "track_progress":
            return await self._track_progress(input_data)
        elif action == "share_context":
            return await self._share_context(input_data)
        else:
            raise ValueError(f"Unknown action: {action}")

    async def _assign_task(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Delegate a task to a team member."""
        member = data.get("member", "")
        task = data.get("task", "")
        self.logger.info(f"Assigning task '{task}' to {member}")
        return {"status": "completed", "member": member, "task": task}

    async def _track_progress(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Report on overall project progress."""
        project_id = data.get("project_id", "")
        self.logger.info(f"Tracking progress for project: {project_id}")
        # TODO: aggregate workflow statuses
        return {"status": "completed", "progress": {}}

    async def _share_context(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Share research context across team members."""
        context_data = data.get("context_data", {})
        self.logger.info("Sharing context with team")
        return {"status": "completed", "shared": True}
