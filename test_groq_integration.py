#!/usr/bin/env python3
"""
Test Groq Integration
Phase 1: Core Infrastructure Implementation
Updated: October 29, 2025

Test script for Groq open-source models integration with research agent.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from research_agent.utils.groq_models import GroqModelClient, GroqModelRegistry, quick_generate, analyze_topic
from research_agent.agents.literature_review_agent import LiteratureReviewAgent
from research_agent.agents.writing_assistant_agent import WritingAssistantAgent
from research_agent.utils.config import config


async def test_groq_models():
    """Test basic Groq model functionality"""
    print("[TEST] Testing Groq Models Integration")
    print("=" * 50)
    
    # Check if Groq API key is available
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("[ERROR] GROQ_API_KEY environment variable not set")
        print("Please set your Groq API key:")
        print("export GROQ_API_KEY='your-api-key-here'")
        return False
    
    try:
        # Initialize Groq client
        print("[INFO] Initializing Groq client...")
        client = GroqModelClient(api_key=api_key)
        print("[SUCCESS] Groq client initialized successfully")
        
        # Test basic text generation
        print("\n[TEST] Testing basic text generation...")
        response = await client.generate_text(
            "Explain quantum computing in simple terms",
            model="llama-3.1-8b-instant"
        )
        print(f"Response: {response[:200]}...")
        
        # Test research topic analysis
        print("\n[TEST] Testing research topic analysis...")
        analysis = await client.analyze_research_topic(
            "Machine Learning in Healthcare",
            model="meta-llama/llama-4-scout-17b-16e-instruct"
        )
        print(f"Analysis: {analysis['analysis'][:300]}...")
        
        # Test code generation
        print("\n[TEST] Testing code generation...")
        code_result = await client.generate_code(
            "A function to calculate fibonacci numbers",
            language="python",
            model="groq/compound"
        )
        print(f"Generated code:\n{code_result['code'][:300]}...")
        
        # Test document summarization
        print("\n[TEST] Testing document summarization...")
        sample_text = """
        Artificial Intelligence (AI) has revolutionized numerous industries and continues to shape the future of technology. 
        Machine learning algorithms enable computers to learn and make decisions from data without explicit programming. 
        Deep learning, a subset of machine learning, uses neural networks with multiple layers to process complex patterns. 
        Natural language processing allows machines to understand and generate human language. Computer vision enables 
        machines to interpret and analyze visual information. These technologies are being applied across healthcare, 
        finance, transportation, and many other sectors to solve complex problems and improve efficiency.
        """
        
        summary_result = await client.summarize_document(
            sample_text,
            max_length=100,
            model="llama-3.1-8b-instant"
        )
        print(f"Summary: {summary_result['summary']}")
        
        print("\n[SUCCESS] All Groq model tests passed!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Error testing Groq models: {e}")
        return False


async def test_agents_with_groq():
    """Test agents with Groq integration"""
    print("\n[TEST] Testing Agents with Groq Integration")
    print("=" * 50)
    
    try:
        # Test Literature Review Agent
        print("[TEST] Testing Literature Review Agent...")
        lit_agent = LiteratureReviewAgent()
        
        # Test search query formulation
        search_task = {
            "action": "formulate_search_query",
            "topic": "Deep Learning in Medical Imaging",
            "research_goals": ["accuracy", "efficiency", "clinical applications"]
        }
        
        result = await lit_agent.execute(search_task)
        if result["status"] == "completed":
            print("[SUCCESS] Search query formulation successful")
            print(f"Generated queries: {result['result']['queries'][:2]}...")
        else:
            print(f"[ERROR] Search query formulation failed: {result.get('error', 'Unknown error')}")
        
        # Test Writing Assistant Agent
        print("\n[TEST] Testing Writing Assistant Agent...")
        writing_agent = WritingAssistantAgent()
        
        # Test outline generation
        outline_task = {
            "action": "generate_outline",
            "topic": "The Future of Artificial Intelligence",
            "outline_type": "research_paper",
            "requirements": {"sections": 5, "depth": "detailed"}
        }
        
        result = await writing_agent.execute(outline_task)
        if result["status"] == "completed":
            print("[SUCCESS] Outline generation successful")
            print(f"Generated outline: {result['result']['outline'][:200]}...")
        else:
            print(f"[ERROR] Outline generation failed: {result.get('error', 'Unknown error')}")
        
        print("\n[SUCCESS] Agent tests completed!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Error testing agents: {e}")
        return False


async def test_model_registry():
    """Test model registry functionality"""
    print("\n[TEST] Testing Model Registry")
    print("=" * 50)
    
    try:
        registry = GroqModelRegistry()
        
        # List all available models
        print("Available models:")
        for model_name, model_config in registry.MODELS.items():
            print(f"  - {model_name}: {model_config.description}")
        
        # Test model categorization
        print(f"\nLlama models: {registry.list_models_by_category('llama')}")
        print(f"Code generation models: {registry.list_models_by_capability('code_generation')}")
        
        # Test model info retrieval
        model_info = registry.get_model_config("llama-3.1-70b-instruct")
        if model_info:
            print(f"\nModel info for llama-3.1-70b-instruct:")
            print(f"  - Max tokens: {model_info.max_tokens}")
            print(f"  - Context length: {model_info.context_length}")
            print(f"  - Capabilities: {', '.join(model_info.capabilities)}")
        
        print("\n[SUCCESS] Model registry tests passed!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Error testing model registry: {e}")
        return False


async def test_convenience_functions():
    """Test convenience functions"""
    print("\n[TEST] Testing Convenience Functions")
    print("=" * 50)
    
    try:
        # Test quick generation
        print("Testing quick_generate...")
        response = await quick_generate(
            "What are the benefits of renewable energy?",
            model="llama-3.1-8b-instant"
        )
        print(f"Quick generation: {response[:150]}...")
        
        # Test topic analysis
        print("\nTesting analyze_topic...")
        analysis = await analyze_topic(
            "Sustainable Agriculture",
            model="meta-llama/llama-4-scout-17b-16e-instruct"
        )
        print(f"Topic analysis: {analysis['analysis'][:200]}...")
        
        # Test code generation
        print("\nTesting generate_code_quick...")
        from research_agent.utils.groq_models import generate_code_quick
        code = await generate_code_quick(
            "A simple web scraper in Python",
            language="python"
        )
        print(f"Quick code generation: {code[:200]}...")
        
        print("\n[SUCCESS] Convenience function tests passed!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Error testing convenience functions: {e}")
        return False


async def main():
    """Main test function"""
    print("[START] Groq Integration Test Suite")
    print("=" * 60)
    
    # Check environment
    if not os.getenv("GROQ_API_KEY"):
        print("[ERROR] GROQ_API_KEY environment variable not set")
        print("Please set your Groq API key and run again.")
        return
    
    # Run tests
    tests = [
        ("Groq Models", test_groq_models),
        ("Model Registry", test_model_registry),
        ("Convenience Functions", test_convenience_functions),
        ("Agents with Groq", test_agents_with_groq),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"[ERROR] {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("[SUMMARY] Test Results Summary")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "[SUCCESS] PASSED" if result else "[ERROR] FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("[SUCCESS] All tests passed! Groq integration is working correctly.")
    else:
        print("[WARNING] Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    asyncio.run(main())
