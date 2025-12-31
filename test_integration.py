"""
Integration Test Script
Phase 1: Core Infrastructure Implementation
Updated: October 29, 2025

Simple test to verify the integration works.
"""

import asyncio
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from research_agent.utils.context_manager import AdvancedContextManager
from research_agent.core.context import SessionManager
from research_agent.api.integration import ResearchIntegration


async def test_context_manager():
    """Test context manager with loop prevention"""
    print("\n=== Testing Context Manager ===")
    
    cm = AdvancedContextManager(max_tokens=1000, max_iterations=3)
    
    # Add some context
    item1 = cm.add_context(
        content={"query": "climate change effects"},
        context_type="query",
        tokens=50,
        relevance_score=0.9
    )
    print(f"Added context item: {item1}")
    
    # Test loop detection
    for i in range(5):
        is_loop, reason = cm.check_execution_loop(
            agent="test_agent",
            action="search",
            input_data={"query": "same query"}
        )
        
        if is_loop:
            print(f"Loop detected at iteration {i+1}: {reason}")
            break
        else:
            print(f"Iteration {i+1}: No loop detected")
    
    # Get stats
    stats = cm.get_stats()
    print(f"\nContext stats: {stats}")


async def test_session_manager():
    """Test session management"""
    print("\n=== Testing Session Manager ===")
    
    sm = SessionManager()
    
    # Create session
    context = await sm.create_session(
        project_name="Test Project",
        description="Testing integration",
        metadata={"test": True}
    )
    print(f"Created session: {context.project_id}")
    
    # Update session
    updated = await sm.update_session(
        context.project_id,
        {"documents": ["doc1.pdf", "doc2.pdf"]}
    )
    print(f"Updated session: {updated.documents}")
    
    # List sessions
    sessions = await sm.list_sessions()
    print(f"Active sessions: {sessions}")


async def test_integration():
    """Test research integration"""
    print("\n=== Testing Research Integration ===")
    
    integration = ResearchIntegration()
    
    # Test paper search
    papers = await integration.search_papers("machine learning", max_results=3)
    print(f"Found {len(papers)} papers")
    
    if papers:
        print(f"First paper: {papers[0]['title']}")


async def main():
    """Run all tests"""
    print("""
    =========================================
    |    Research Agent Integration Test    |
    =========================================
    """)
    
    try:
        await test_context_manager()
        await test_session_manager()
        await test_integration()
        
        print("\n[SUCCESS] All tests completed successfully!")
        
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
