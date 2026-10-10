#!/usr/bin/env python3
"""
Hermes Improvements Package

All components are placed here instead of site-packages to survive
`pip install --upgrade hermes-agent` updates.

Components:
  - vector_memory.py  : VectorMemoryStore (semantic search on memory embeddings)
  - dynamic_memory.py : DynamicMemoryContext + AdaptiveMemoryPrefetch
  - adaptive_soul.py  : AdaptiveSoul + StyleLearner
  - adaptive_workflow.py : AdaptiveWorkflow (task complexity classifier)
  - reasoning_trace.py: ReasoningTracer + SourceAttribution + UncertaintyManager
  - integration.py    : Integration layer + patch_agent hook
"""

from .vector_memory import VectorMemoryStore
from .dynamic_memory import DynamicMemoryContext, AdaptiveMemoryPrefetch
from .adaptive_soul import AdaptiveSoul, StyleLearner
from .adaptive_workflow import AdaptiveWorkflow
from .reasoning_trace import ReasoningTracer, SourceAttribution, UncertaintyManager
from .integration import (
    build_turn_context_block,
    detect_turn_feedback,
    get_agent_stats,
    initialize_hermes_improvements,
    patch_agent_for_improvements,
)

__version__ = "3.2.1"
__all__ = [
    "VectorMemoryStore",
    "DynamicMemoryContext",
    "AdaptiveMemoryPrefetch",
    "AdaptiveSoul",
    "StyleLearner",
    "AdaptiveWorkflow",
    "ReasoningTracer",
    "SourceAttribution",
    "UncertaintyManager",
    "initialize_hermes_improvements",
    "patch_agent_for_improvements",
    "build_turn_context_block",
    "detect_turn_feedback",
    "get_agent_stats",
]
