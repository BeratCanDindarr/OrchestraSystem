"""Unit tests for orchestra.service."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from orchestra import service

def test_parse_cli_value():
    assert service._parse_cli_value("true") is True
    assert service._parse_cli_value("false") is False
    assert service._parse_cli_value("123") == 123
    assert service._parse_cli_value("12.3") == 12.3
    assert service._parse_cli_value("[1, 2, 3]") == [1, 2, 3]
    assert service._parse_cli_value("hello") == "hello"

def test_toml_literal():
    assert service._toml_literal(True) == "true"
    assert service._toml_literal(123) == "123"
    assert service._toml_literal("hello") == '"hello"'
    assert service._toml_literal([1, 2]) == "[1, 2]"
    assert service._toml_literal({"a": 1}) == '{"a": 1}'

def test_serialize_rows():
    rows = [{"a": 1}, {"b": 2}]
    res = service._serialize_rows(rows)
    assert res == rows

@patch("orchestra.service._daemon_pid_path")
def test_daemon_status_stopped(mock_pid_path, tmp_path):
    pid_file = tmp_path / "daemon.pid"
    mock_pid_path.return_value = pid_file
    assert service.daemon_status()["running"] is False

def test_list_aliases():
    with patch("orchestra.service.available_aliases", return_value=["✓ a1"]):
        res = service.list_aliases()
        assert "a1" in res["aliases"][0]

def test_config_get_value():
    with patch("orchestra.service._config_tree", return_value={"a": {"b": 1}}):
        res = service.config_get_value("a.b")
        assert res["value"] == 1

def test_get_version():
    assert "version" in service.get_version()

@patch("orchestra.service.get_db")
def test_get_stats(mock_get_db):
    mock_conn = MagicMock()
    mock_get_db.return_value = mock_conn
    mock_conn.execute.return_value.fetchone.side_effect = [[10], [5], [2], [1.5]]
    
    with patch("orchestra.service.backfill"), \
         patch("orchestra.service.list_alias_reputation", return_value=[]):
        stats = service.get_stats()
        assert stats["total_runs"] == 10
        assert stats["completed_runs"] == 5

def test_list_jobs():
    with patch("orchestra.service.list_job_records", return_value=[{"job_id": "j1"}]):
        with patch("orchestra.service.get_db"):
            res = service.list_jobs()
            assert len(res["jobs"]) == 1
            assert res["jobs"][0]["job_id"] == "j1"

def test_get_job():
    with patch("orchestra.service.get_job_record", return_value={"job_id": "j1"}):
        with patch("orchestra.service.get_db"):
            res = service.get_job("j1")
            assert res["job"]["job_id"] == "j1"

def test_list_speculations():
    with patch("orchestra.storage.speculation.list_speculative_plans", return_value=[{"id": 1}]):
        plans = service.list_speculations()
        assert len(plans) == 1
