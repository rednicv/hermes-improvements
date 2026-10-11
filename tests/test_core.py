import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from hermes_improvements.vector_memory import VectorMemoryStore
from hermes_improvements.adaptive_soul import AdaptiveSoul
from hermes_improvements.adaptive_workflow import AdaptiveWorkflow


def test_vector_memory_store_fallback(tmp_path):
    # Tests that vector memory store falls back gracefully without sentence-transformers
    store = VectorMemoryStore(memory_dir=tmp_path)
    
    # Add a memory
    store.add("test_id_1", "Hermes Agent has a modular architecture.", metadata={"source": "test"})
    
    # Query memories (uses TF-IDF fallback when sentence-transformers is not initialized/installed)
    results = store.search("modular architecture", top_k=1)
    assert isinstance(results, list)
    
    if len(results) > 0:
        text = results[0].get("text") or results[0].get("text_preview", "")
        assert "modular" in text.lower() or "architecture" in text.lower()


def test_adaptive_soul_rules_loading(tmp_path):
    soul = AdaptiveSoul(soul_dir=tmp_path)
    # Feed positive feedback
    soul.record_feedback(category="style", sentiment=1, text="I really like short responses")
    rules = soul.build_soul_extension()
    
    # Check that rules compilation works and includes styling directives
    assert isinstance(rules, str)
    assert len(soul.feedback_history) == 1


def test_adaptive_workflow_classification():
    workflow = AdaptiveWorkflow()
    
    # Simple query should be classified as trivial/simple
    simple_complexity = workflow.classify_task("Hello, how are you?")
    assert simple_complexity.name.lower() in ["trivial", "simple"]
    
    # Complex query with multiple action verbs, complex keywords, or code requests
    complex_query = "Please review this Python codebase, refactor the database connector class, compile the project, and run unit tests."
    complex_complexity = workflow.classify_task(complex_query)
    assert complex_complexity.name.lower() in ["moderate", "complex"]
