"""Tests for graph-native history tracking."""

import grafeo

from grafeo_memory.history import (
    HAS_HISTORY_EDGE,
    HISTORY_LABEL,
    HistoryEntry,
    get_history,
    record_history,
)


def _make_db():
    return grafeo.GrafeoDB()


def _has_native_cdc(db):
    return hasattr(db, "node_history")


class TestRecordHistory:
    def test_noop_with_native_cdc(self):
        """With native CDC, record_history is a no-op."""
        db = _make_db()
        if not _has_native_cdc(db):
            return  # skip if no CDC
        mem_node = db.create_node(["Memory"], {"text": "alice likes hiking"})
        mem_id = mem_node.id if hasattr(mem_node, "id") else mem_node

        entry = HistoryEntry(event="ADD", new_text="alice likes hiking", timestamp=1000)
        result = record_history(db, mem_id, entry)
        assert result is None  # no-op with CDC

    def test_creates_history_node_legacy(self):
        """Without CDC, record_history creates a History node (legacy)."""
        db = _make_db()
        if _has_native_cdc(db):
            return  # skip legacy test when CDC is available
        mem_node = db.create_node(["Memory"], {"text": "alice likes hiking"})
        mem_id = mem_node.id if hasattr(mem_node, "id") else mem_node

        entry = HistoryEntry(event="ADD", new_text="alice likes hiking", timestamp=1000)
        hist_id = record_history(db, mem_id, entry)

        node = db.get_node(hist_id)
        assert node is not None
        props = node.properties if hasattr(node, "properties") else {}
        if callable(props):
            props = props()
        assert props["event"] == "ADD"
        assert props["new_text"] == "alice likes hiking"

    def test_creates_has_history_edge_legacy(self):
        db = _make_db()
        if _has_native_cdc(db):
            return
        mem_node = db.create_node(["Memory"], {"text": "test"})
        mem_id = mem_node.id if hasattr(mem_node, "id") else mem_node

        entry = HistoryEntry(event="ADD", new_text="test", timestamp=1000)
        record_history(db, mem_id, entry)

        result = db.execute(
            f"MATCH (m)-[:{HAS_HISTORY_EDGE}]->(h:{HISTORY_LABEL}) WHERE id(m) = $mid RETURN h.event",
            {"mid": mem_id},
        )
        events = [next(iter(row.values())) for row in result if isinstance(row, dict)]
        assert "ADD" in events

    def test_actor_fields_stored_legacy(self):
        db = _make_db()
        if _has_native_cdc(db):
            return
        mem_node = db.create_node(["Memory"], {"text": "test"})
        mem_id = mem_node.id if hasattr(mem_node, "id") else mem_node

        entry = HistoryEntry(
            event="UPDATE", old_text="old", new_text="new", timestamp=2000, actor_id="alice", role="user"
        )
        hist_id = record_history(db, mem_id, entry)

        node = db.get_node(hist_id)
        props = node.properties if hasattr(node, "properties") else {}
        if callable(props):
            props = props()
        assert props["actor_id"] == "alice"
        assert props["role"] == "user"


class TestGetHistory:
    def test_retrieves_cdc_events(self):
        """With CDC, get_history returns engine-tracked events."""
        db = _make_db()
        if not _has_native_cdc(db):
            return
        mem_node = db.create_node(["Memory"], {"text": "v1"})
        mem_id = mem_node.id if hasattr(mem_node, "id") else mem_node

        # The CREATE event is automatically tracked by CDC
        entries = get_history(db, mem_id)
        assert len(entries) >= 1
        assert entries[0].event == "ADD"

    def test_cdc_tracks_text_updates(self):
        """CDC tracks text property changes as UPDATE events."""
        db = _make_db()
        if not _has_native_cdc(db):
            return
        mem_node = db.create_node(["Memory"], {"text": "v1"})
        mem_id = mem_node.id if hasattr(mem_node, "id") else mem_node

        db.set_node_property(mem_id, "text", "v2")

        entries = get_history(db, mem_id)
        # Should have ADD (create) + UPDATE (text change)
        events = [e.event for e in entries]
        assert "ADD" in events
        assert "UPDATE" in events

    def test_retrieves_entries_in_order_legacy(self):
        db = _make_db()
        if _has_native_cdc(db):
            return
        mem_node = db.create_node(["Memory"], {"text": "v1"})
        mem_id = mem_node.id if hasattr(mem_node, "id") else mem_node

        record_history(db, mem_id, HistoryEntry(event="ADD", new_text="v1", timestamp=1000))
        record_history(db, mem_id, HistoryEntry(event="UPDATE", old_text="v1", new_text="v2", timestamp=2000))

        entries = get_history(db, mem_id)
        assert len(entries) == 2
        assert entries[0].event == "ADD"
        assert entries[1].event == "UPDATE"
        assert entries[0].timestamp < entries[1].timestamp

    def test_empty_for_nonexistent_node(self):
        db = _make_db()
        entries = get_history(db, 99999)
        assert entries == []
