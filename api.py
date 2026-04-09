import os
import uuid
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from core.graph import run_research_pipeline

load_dotenv()

app = FastAPI(title="Research Agent API", version="0.1.0")

# Allow requests from Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

class AgentStep(BaseModel):
    agent: str
    action: str
    color: str

class ChatResponse(BaseModel):
    content: str
    citations: Optional[List[str]] = []
    agentSteps: Optional[List[AgentStep]] = []

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    try:
        # Run the agentic pipeline
        # Mode is 'agentic', rigor is exploratory for quick chat, goals are empty
        result_state = await run_research_pipeline(
            project_name=f"ChatSession_{uuid.uuid4().hex[:8]}",
            research_topic=req.message,
            research_goals=[],
            rigor_level="exploratory",
            mode="agentic"
        )
        
        # Extract content
        # Assuming writing_node sets `draft_sections` or there's a conclusion in state
        final_text = ""
        if "draft_sections" in result_state and result_state["draft_sections"]:
            final_text = "\n\n".join(result_state["draft_sections"].values())
        else:
            final_text = "I have completed analyzing the research related to your query."

        # Extract citations
        citations = []
        if "papers" in result_state:
            for p in result_state["papers"][:5]:
                citations.append(p.get("title", "Unknown Source"))
                
        # Extract agent steps from audit_log
        agent_steps = []
        if "audit_log" in result_state:
            for log in result_state["audit_log"]:
                # Map LangGraph agents to colors
                agent_name = log.get("agent", "System")
                color = "cyan"
                if "lit" in agent_name.lower() or "search" in agent_name.lower():
                    color = "violet"
                elif "write" in agent_name.lower():
                    color = "green"
                    
                agent_steps.append({
                    "agent": agent_name,
                    "action": log.get("output_summary", "Processing step"),
                    "color": color
                })

        return ChatResponse(
            content=final_text,
            citations=citations,
            agentSteps=agent_steps
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
