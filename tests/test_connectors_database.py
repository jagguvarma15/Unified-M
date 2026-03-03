"""Connector regression tests for database adapters."""

from __future__ import annotations

from connectors.database import SQLiteConnector


def test_sqlite_test_connection_returns_true_for_valid_db(tmp_path):
    db_path = tmp_path / "test.db"
    conn = SQLiteConnector(database=str(db_path))
    assert conn.test_connection() is True
