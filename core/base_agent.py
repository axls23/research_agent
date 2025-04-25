from google.adk.agents import Agent
from typing import Dict, Any, Optional
import logging

class ResearchAgent(Agent):
    """Base class for all research agents in the system."""
    
    def __init__(self, name: str, description: str):
        super().__init__(name)
        self.description = description
        self.logger = logging.getLogger(f"research_agent.{name}")
        self.context: Optional[Dict[str, Any]] = None
        
    async def initialize(self, context: Dict[str, Any]) -> None:
        """Initialize the agent with context from the orchestrator."""
        self.context = context
        self.logger.info(f"Initialized {self.name} with context: {context}")
        
    async def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data before processing."""
        required_fields = self.get_required_fields()
        missing_fields = [field for field in required_fields if field not in input_data]
        if missing_fields:
            self.logger.error(f"Missing required fields: {missing_fields}")
            return False
        return True
    
    def get_required_fields(self) -> list:
        """Get list of required input fields for the agent."""
        return []
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data and return results."""
        if not await self.validate_input(input_data):
            raise ValueError("Invalid input data")
        return {}
    
    async def cleanup(self) -> None:
        """Clean up any resources used by the agent."""
        self.context = None
        self.logger.info(f"Cleaned up {self.name}") 