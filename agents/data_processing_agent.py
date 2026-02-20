"""Data Processing Agent – ingests, cleans, chunks, and prepares
research documents for downstream analysis."""

from core.base_agent import ResearchAgent
from typing import Dict, Any, List


class DataProcessingAgent(ResearchAgent):
    """Agent responsible for data ingestion, cleaning, and chunking.

    Supported actions:
      - ``prepare_data`` – clean and chunk raw research documents
      - ``extract_text`` – extract text content from PDF documents
    """

    def __init__(self):
        super().__init__(
            name="data_processing",
            description=(
                "Ingests and preprocesses research documents into structured "
                "chunks suitable for analysis and knowledge-graph construction"
            ),
        )

    def get_required_fields(self) -> List[str]:
        return ["action"]

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        await super().process(input_data)
        action = input_data["action"]

        if action == "prepare_data":
            return await self._prepare_data(input_data)
        elif action == "extract_text":
            return await self._extract_text(input_data)
        else:
            raise ValueError(f"Unknown action: {action}")

    async def _prepare_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and chunk documents for further processing."""
        documents = data.get("documents", [])
        chunk_size = data.get("chunk_size", 1000)
        self.logger.info(
            f"Preparing {len(documents)} documents with chunk_size={chunk_size}"
        )
        # TODO: integrate with paperconstructor chunker
        return {"status": "completed", "chunks": [], "document_count": len(documents)}

    async def _extract_text(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract text from PDF documents."""
        file_path = data.get("file_path", "")
        self.logger.info(f"Extracting text from {file_path}")
        # TODO: integrate with PyPDF2 text extraction
        return {"status": "completed", "text": "", "file_path": file_path}
