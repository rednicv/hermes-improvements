"""Unit tests for VectorMemoryStore."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hermes_improvements.vector_memory import VectorMemoryStore, smart_truncate


def test_smart_truncate():
    text = "Propozitie 1. Propozitie 2. Propozitie 3."
    res = smart_truncate(text, max_chars=25)
    assert len(res) <= 25
    assert res.endswith(" […]")


def test_vector_memory_store_lifecycle(tmp_path):
    store = VectorMemoryStore(memory_dir=tmp_path)
    store.add("k1", "Prima memorie vectoriala pentru test", metadata={"tag": "test"})
    
    # Get
    item = store.get("k1")
    assert item is not None
    assert item["text"] == "Prima memorie vectoriala pentru test"
    assert item["metadata"]["tag"] == "test"
    
    # Update
    store.update("k1", "Text actualizat", metadata={"tag": "updated"})
    item2 = store.get("k1")
    assert item2 is not None
    assert item2["text"] == "Text actualizat"
    assert item2["metadata"]["tag"] == "updated"
    
    # Search
    results = store.search("actualizat", top_k=1)
    assert len(results) == 1
    assert results[0]["key"] == "k1"
    
    # Remove
    store.remove("k1")
    assert store.get("k1") is None
