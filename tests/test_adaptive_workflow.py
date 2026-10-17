"""Unit tests for AdaptiveWorkflow."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hermes_improvements.adaptive_workflow import AdaptiveWorkflow, TaskComplexity


def test_adaptive_workflow_levels():
    workflow = AdaptiveWorkflow()

    # Trivial
    c_triv = workflow.classify_task("ok")
    assert c_triv == TaskComplexity.TRIVIAL

    # Simple
    c_simp = workflow.classify_task("write a python function to parse json logs")
    assert c_simp in (TaskComplexity.SIMPLE, TaskComplexity.MODERATE)

    # Complex
    c_comp = workflow.classify_task("Please review this entire codebase, refactor the architecture, migrate database schemas and deploy critical production services")
    assert c_comp == TaskComplexity.COMPLEX
