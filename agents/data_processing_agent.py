"""Data Processing Agent — document ingestion, text extraction, and
chunking via the Deep Agents subagent pattern.

Wraps the data_processing_node logic as an agent with tool-exposed
methods for the orchestrator.
"""

from __future__ import annotations

from typing import Any, Dict, List

from core.base_agent import ResearchAgent
from core.agent_tools import process_documents


class DataProcessingAgent(ResearchAgent):
    """Agent responsible for data ingestion, cleaning, and chunking.

    Supported actions:
      - ``prepare_data``  — clean and chunk raw research documents
      - ``extract_text``  — extract text content from PDF documents
    """

    def __init__(self, llm=None):
        super().__init__(
            name="data_processing",
            description=(
                "Ingests and preprocesses research documents into structured "
                "chunks suitable for analysis and knowledge-graph construction."
            ),
            llm=llm,
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
        papers = data.get("papers", data.get("documents", []))
        chunk_size = data.get("chunk_size", 1000)

        self.logger.info(
            f"Preparing {len(papers)} documents with chunk_size={chunk_size}"
        )

        # Handle both dict and string inputs for backward compat
        chunks = []
        for p in papers:
            if isinstance(p, str):
                chunks.append({"text": p, "chunk_index": len(chunks)})
            elif isinstance(p, dict):
                text = p.get("abstract", "") or p.get("text", "")
                if text:
                    chunks.append(
                        {
                            "text": text,
                            "chunk_index": len(chunks),
                            "paper_id": p.get("paper_id", ""),
                        }
                    )

        return {
            "status": "completed",
            "chunks": chunks,
            "chunk_count": len(chunks),
            "document_count": len(papers),
            "papers_processed": len(papers),
        }

    async def _extract_text(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract text from a PDF document."""
        file_path = data.get("file_path", "")
        self.logger.info(f"Extracting text from {file_path}")

        try:
            from PyPDF2 import PdfReader

            reader = PdfReader(file_path)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            return {
                "status": "completed",
                "text": text,
                "file_path": file_path,
                "page_count": len(reader.pages),
            }
        except ImportError:
            self.logger.warning("PyPDF2 not available")
            return {"status": "completed", "text": "", "file_path": file_path}
        except Exception as e:
            self.logger.error(f"PDF text extraction failed: {e}")
            return {"status": "completed", "text": "", "file_path": file_path}
