"""
Research Agent - Textual CLI Dashboard
Run with: textual run cli.py
"""

import sys
import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Add project to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.widgets import Header, Footer, Input, Button, Log, Select, Static, Label
from textual.worker import Worker, WorkerState
from textual import work

# Import the core backend logic
from core.graph import run_research_pipeline
from core.llm_provider import create_llm_provider

class ResearchAgentApp(App):
    """A Textual app for running and monitoring the Research AI Agent."""

    CSS = """
    Screen {
        background: $surface;
    }

    #controls {
        dock: top;
        height: auto;
        padding: 1;
        background: $boost;
        border-bottom: solid $primary;
    }

    .form-group {
        height: auto;
        margin-bottom: 1;
    }
    
    .horizontal {
        height: auto;
    }

    #topic_input {
        width: 100%;
    }

    #run_btn {
        margin-top: 1;
        width: 100%;
    }

    #main_log {
        height: 100%;
        border: solid $secondary;
        margin: 1;
        background: $panel;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("d", "toggle_dark", "Toggle dark mode"),
    ]

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        
        with Vertical(id="controls"):
            with Vertical(classes="form-group"):
                yield Label("Research Topic:")
                yield Input(placeholder="e.g. Quantum Machine Learning for molecular dynamics...", id="topic_input")
            
            with Horizontal(classes="horizontal"):
                yield Select(
                    [("Agentic (ReAct)", "agentic"), ("LangGraph Native", "langgraph")],
                    id="mode_select",
                    value="agentic"
                )
                yield Select(
                    [("Exploratory", "exploratory"), ("PRISMA", "prisma"), ("Cochrane", "cochrane")],
                    id="rigor_select",
                    value="prisma"
                )
                yield Select(
                    [("Groq Inference", "groq"), ("Local vLLM", "vllm")],
                    id="provider_select",
                    value="groq"
                )
            
            yield Button("Start Research Pipeline", id="run_btn", variant="success")

        # The scrolling terminal log
        yield Log(id="main_log", highlight=True)
        yield Footer()

    def on_ready(self) -> None:
        """Called when app is ready."""
        log = self.query_one(Log)
        log.write_line("[bold green]NEXUS Engine Ready.[/]")
        log.write_line("Enter a topic and click Start.")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Event handler called when a button is pressed."""
        if event.button.id == "run_btn":
            topic_input = self.query_one("#topic_input", Input)
            topic = topic_input.value
            
            if not topic:
                self.query_one(Log).write_line("[red]Error: Please enter a research topic.[/]")
                return
                
            mode = self.query_one("#mode_select", Select).value
            rigor = self.query_one("#rigor_select", Select).value
            provider = self.query_one("#provider_select", Select).value
            
            self.query_one(Button).disabled = True
            log = self.query_one(Log)
            log.clear()
            log.write_line(f"[cyan]Initializing pipeline for topic:[/] {topic}")
            internal_mode = "agentic" if mode == "agentic" else "deterministic"
            log.write_line(f"[dim]Mode: {mode} | Rigor: {rigor} | Provider: {provider}[/]")
            
            # Start the background async task
            self.run_pipeline_background(topic, mode, rigor, provider)

    @work(exclusive=True, thread=False)
    async def run_pipeline_background(
        self,
        topic: str,
        mode: str,
        rigor: str,
        provider: str,
    ) -> None:
        """Runs the LangGraph pipeline asynchronously without blocking the TUI."""
        log = self.query_one(Log)
        internal_mode = "agentic" if mode == "agentic" else "deterministic"
        
        try:
            # We mock console print outputs to our Log UI component
            # In a real heavy integration, we would create a stream handler for the python logger
            import logging
            class TextualLogHandler(logging.Handler):
                def __init__(self, tui_log):
                    super().__init__()
                    self.tui_log = tui_log
                
                def emit(self, record):
                    msg = self.format(record)
                    # Textual UI updates must be thread-safe; since this is async we just write
                    self.tui_log.write_line(msg)
            
            # Hijack the main logger temporarily
            # Note: This is a hacky way to pipe Langgraph execution logs into Textual
            root_logger = logging.getLogger()
            tui_handler = TextualLogHandler(log)
            tui_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
            root_logger.addHandler(tui_handler)
            
            # Set goals based on topic
            goals = [
                f"Identify core challenges in {topic}",
                f"Extract hyperedges defining {topic}"
            ]

            llm = None
            agentic_model = None
            if provider == "vllm":
                local_model = os.getenv("RLM_PRIMARY_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
                vllm_base = os.getenv("OPENAI_API_BASE", os.getenv("RLM_MODEL_BASE_URL", "http://localhost:8000/v1"))
                os.environ["OPENAI_API_BASE"] = vllm_base
                os.environ.setdefault("OPENAI_API_KEY", "vllm-dummy-key")
                llm = create_llm_provider(provider="openai", model=local_model)
                agentic_model = f"fast_rlm:{local_model}"
                log.write_line(f"[yellow]Using local vLLM at {vllm_base} with model {local_model}[/]")
            else:
                groq_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
                llm = create_llm_provider(provider="groq", model=groq_model)
                agentic_model = os.getenv("AGENTIC_GROQ_MODEL", "groq:llama-3.3-70b-versatile")
                log.write_line(f"[green]Using Groq model {groq_model}[/]")
            
            # Execute the graph
            result_state = await run_research_pipeline(
                project_name="CLI Session",
                research_topic=topic,
                research_goals=goals,
                rigor_level=rigor,
                interactive=False, 
                llm=llm,
                mode=internal_mode,
                agentic_model=agentic_model,
            )
            
            log.write_line("[bold green]\n=== PIPELINE SUCCESS ===[/]")
            if "papers_included" in result_state:
                log.write_line(f"Papers included: {result_state.get('papers_included', 0)}")
                log.write_line(f"Hyperedges extracted: {len(result_state.get('hyperedges', []))}")
                
            if "audit_export_path" in result_state:
                log.write_line(f"Audit exported to: {result_state['audit_export_path']}")
                
        except Exception as e:
            log.write_line(f"[bold red]Pipeline failed: {str(e)}[/]")
        finally:
            # Restore UI state
            if tui_handler in root_logger.handlers:
                root_logger.removeHandler(tui_handler)
            
            # Since thread=False, we are already in the main asyncio loop!
            # We don't need call_from_thread.
            self._enable_button()

    def _enable_button(self):
        self.query_one(Button).disabled = False

if __name__ == "__main__":
    app = ResearchAgentApp()
    app.run()
