# Migrations

`db.py`'s `_SCHEMA_SQL` (`CREATE TABLE IF NOT EXISTS ...`) is safe for creating
tables that don't exist yet, but silently does nothing for a table that
already exists — so it can never apply a change to an already-deployed
database (add a column, change a type, backfill data). That's what this
directory is for.

## Adding a migration

1. Create a new file: `NNNN_short_description.sql`, where `NNNN` is the next
   number after the highest one currently here (zero-padded to 4 digits,
   e.g. `0001_add_tasks_snoozed_at.sql`). Files run in filename order.
2. Write plain SQL. Wrap multi-statement migrations in a transaction
   yourself only if you need finer control than the runner's default (it
   already runs each file inside one transaction — see `db.run_migrations()`
   — so a failure partway through a file rolls back that whole file, not
   just to the last semicolon).
3. Never edit or delete a migration that has already been merged, even to
   fix a typo — a database that already applied it won't re-run it. Ship a
   new migration that corrects the mistake instead.
4. `_SCHEMA_SQL` in `db.py` stays as the baseline for brand-new databases
   (a `CREATE TABLE IF NOT EXISTS` for a table this directory later adds a
   column to would fight with a migration that assumes the column doesn't
   exist yet — so if you add a genuinely new table, add it to `_SCHEMA_SQL`,
   not here; this directory is for changing EXISTING tables).

## How it runs

`db.run_migrations()` is called once at startup, right after
`init_schema()` succeeds (see `bot.py`'s `main()`). It tracks which
filenames have already run in a `schema_migrations` table and only applies
new ones, so it's safe to call on every boot. Missing `DATABASE_URL` (no
Postgres configured) makes it a no-op, same as everything else that needs
Postgres in this codebase.
