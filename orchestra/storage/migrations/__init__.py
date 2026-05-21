from __future__ import annotations
"""Migration re-export: migrations.py is the single source of truth.

The migrations/ package re-exports everything from the sibling migrations.py
module. We load it via importlib to avoid circular imports (the package name
shadows the .py file in normal import resolution).
"""

import importlib.util
import sys
from pathlib import Path

_migrations_path = Path(__file__).parent.parent / "migrations.py"
_spec = importlib.util.spec_from_file_location("_migrations_core", _migrations_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load migration spec from {_migrations_path}")
_migrations_mod = importlib.util.module_from_spec(_spec)
sys.modules["_migrations_core"] = _migrations_mod
_spec.loader.exec_module(_migrations_mod)

ensure_schema = _migrations_mod.ensure_schema
MigrationError = _migrations_mod.MigrationError
run_migrations = _migrations_mod.run_migrations
rollback_migration = _migrations_mod.rollback_migration
get_migration_status = _migrations_mod.get_migration_status
_get_migration_modules = _migrations_mod._get_migration_modules
_get_schema_version = _migrations_mod._get_schema_version
_set_schema_version = _migrations_mod._set_schema_version
migrate_v2_to_v3 = _migrations_mod.migrate_v2_to_v3

__all__ = [
    "ensure_schema",
    "MigrationError",
    "run_migrations",
    "rollback_migration",
    "get_migration_status",
    "_get_migration_modules",
    "_get_schema_version",
    "_set_schema_version",
    "migrate_v2_to_v3",
]
