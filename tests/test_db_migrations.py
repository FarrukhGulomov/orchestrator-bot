"""db.run_migrations() — the no-Postgres path only, consistent with this
whole suite's approach (see conftest.py's module docstring): every
Postgres-tier code path in this codebase, including the pre-existing
init_schema()/kv_get()/kv_set(), is exercised in staging/production
against a real database, not unit-tested here. What IS worth covering
without one: that the function is a safe, silent no-op when Postgres
isn't configured, and that the documented convention (migrations/ +
README) actually exists on disk."""

import os

import db


async def test_no_op_without_database_url():
    # No DATABASE_URL anywhere in this test suite's environment (see
    # conftest.py) — db.get_pool() returns None, and run_migrations()
    # must treat that as a normal no-op, not an error.
    assert not db.settings.db_enabled
    assert (await db.run_migrations()) == 0


def test_migrations_directory_exists_with_its_readme():
    assert os.path.isdir(db._MIGRATIONS_DIR)
    assert os.path.isfile(os.path.join(db._MIGRATIONS_DIR, "README.md"))


def test_migrations_dir_resolves_next_to_db_py_regardless_of_cwd():
    # run_migrations() lists os.listdir(_MIGRATIONS_DIR) — if this were a
    # relative path it would silently see nothing (or the wrong directory)
    # whenever the process's cwd isn't the repo root.
    assert os.path.isabs(db._MIGRATIONS_DIR)
    assert os.path.basename(db._MIGRATIONS_DIR) == "migrations"
