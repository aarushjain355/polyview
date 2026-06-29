#!/usr/bin/env python3
"""One-time migration: build a "<env> - summary" spreadsheet for existing environments.

The dashboard's fast path reads a single per-environment summary sheet (one row per case)
instead of one read per case spreadsheet. New runs write that summary automatically via the
reporter (lidar_database_handler.sync). This script back-fills it for environments whose data
predates that change — it reads each case's existing metrics sheet once and writes the summary.

The per-case sheets and their Visualization tabs are left untouched; the summary is additive.

Usage (from polyview_app/):
    python3 tools/build_env_summaries.py            # migrate every environment
    python3 tools/build_env_summaries.py test_env   # migrate one environment
"""
import sys
import time
from pathlib import Path

import toml
from googleapiclient.errors import HttpError

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_ROOT / 'src'))

import database_handler as dh  # noqa: E402

_SHEET = dh.DatabaseHandler._SHEET_MIME
_FOLDER = dh.DatabaseHandler._FOLDER_MIME


def _load_handler() -> dh.DatabaseHandler:
    secrets = toml.load(APP_ROOT / '.streamlit' / 'secrets.toml')
    return dh.DatabaseHandler(
        credentials_info=dict(secrets['google_sheets']),
        root_folder_id=secrets['root_folder_id'],
    )


def _gather_case_sheets(h: dh.DatabaseHandler, env_id: str) -> list[tuple[str, str, str]]:
    """Return (lidar_name, case_path, sheet_id) for every case under the env, covering both
    the flat (sheet directly under lidar) and nested (sheet inside a case folder) layouts."""
    lidars = h._list_children(env_id, _FOLDER)
    lidar_ids = list(lidars.values())
    flat_by = h._list_children_by_parents(lidar_ids, _SHEET)
    folders_by = h._list_children_by_parents(lidar_ids, _FOLDER)

    jobs: list[tuple[str, str, str]] = []
    nested: list[tuple[str, str, str]] = []
    for lidar_name, lidar_id in lidars.items():
        prefix = f'{lidar_name} - '
        for sheet_name, sheet_id in flat_by.get(lidar_id, {}).items():
            case = sheet_name[len(prefix):] if sheet_name.startswith(prefix) else sheet_name
            jobs.append((lidar_name, case, sheet_id))
        for case, case_id in folders_by.get(lidar_id, {}).items():
            nested.append((lidar_name, case, case_id))
    if nested:
        sheets_by_folder = h._list_children_by_parents([c[2] for c in nested], _SHEET)
        for lidar_name, case, case_id in nested:
            sid = sheets_by_folder.get(case_id, {}).get(f'{lidar_name} - {case}')
            if sid is not None:
                jobs.append((lidar_name, case, sid))
    return jobs


def _read_flat(h: dh.DatabaseHandler, sheet_id: str) -> dict | None:
    """Read a case sheet's raw flat metrics ({column: value}) — the un-reshaped form the
    summary sheet stores, matching exactly what the per-case sheet holds. Retries on the
    60/min read-quota limit; returns None if it never succeeds, {} for a genuinely empty sheet."""
    for attempt in range(5):
        try:
            resp = h._sheets_service().spreadsheets().values().get(
                spreadsheetId=sheet_id, range='Sheet1!1:2', valueRenderOption='UNFORMATTED_VALUE',
            ).execute(num_retries=dh.DatabaseHandler._NUM_RETRIES)
            break
        except (HttpError, OSError):
            if attempt == 4:
                return None
            time.sleep(2 ** attempt)
    rows = resp.get('values', [])
    if len(rows) < 2:
        return {}
    headers, values = rows[0], rows[1]
    return {
        str(h_): v for h_, v in zip(headers, values)
        if str(h_) not in ('', 'Case Name') and v != ''
    }


def _write_summary(h: dh.DatabaseHandler, env_id: str, env_name: str,
                   cases: list[tuple[str, str, dict]]) -> tuple[int, int]:
    columns: list[str] = []
    seen: set[str] = set()
    for _, _, fm in cases:
        for key in fm:
            if key not in seen:
                seen.add(key)
                columns.append(key)
    rows = [['Lidar', 'Case Name'] + columns]
    for lidar_name, case, fm in cases:
        rows.append([lidar_name, case] + [fm.get(c, '') for c in columns])

    name = f'{env_name} - summary'
    summary_id = h._list_children(env_id, _SHEET).get(name)
    if summary_id:
        h._sheets_service().spreadsheets().values().clear(
            spreadsheetId=summary_id, range='Sheet1', body={},
        ).execute()
    else:
        created = h._drive_service().files().create(
            body={'name': name, 'mimeType': _SHEET, 'parents': [env_id]},
            fields='id', supportsAllDrives=True,
        ).execute()
        summary_id = created['id']
    h._sheets_service().spreadsheets().values().update(
        spreadsheetId=summary_id, range='Sheet1!A1',
        valueInputOption='USER_ENTERED', body={'values': rows},
    ).execute()
    return len(cases), len(columns)


def migrate_env(h: dh.DatabaseHandler, env_name: str, env_id: str) -> None:
    jobs = _gather_case_sheets(h, env_id)
    if not jobs:
        print(f'  {env_name}: no case sheets found, skipping.')
        return
    # Read sequentially with light pacing: the read quota is 60/min/user, so bursting in
    # parallel just trips 429s. This is the one-time slow step; everyday loads read the
    # single summary sheet instead.
    cases: list[tuple[str, str, dict]] = []
    failed = 0
    for i, (lidar_name, case, sheet_id) in enumerate(jobs):
        flat = _read_flat(h, sheet_id)
        if flat is None:
            failed += 1
        elif flat:
            cases.append((lidar_name, case, flat))
        print(f'    [{i + 1}/{len(jobs)}] {lidar_name} / {case}', end='\r')
        time.sleep(1.05)  # ~57 reads/min, just under the 60/min/user quota
    print()
    if failed:
        print(f'  {env_name}: WARNING — {failed} case sheet(s) could not be read.')
    if not cases:
        print(f'  {env_name}: case sheets had no metrics, skipping.')
        return
    n_cases, n_cols = _write_summary(h, env_id, env_name, cases)
    print(f'  {env_name}: wrote summary with {n_cases} cases × {n_cols} metric columns.')


def main() -> None:
    h = _load_handler()
    if not h.available:
        print('No credentials configured (check .streamlit/secrets.toml).')
        sys.exit(1)
    envs = h._list_children(h._root_folder_id, _FOLDER)
    targets = sys.argv[1:] or list(envs.keys())
    print(f'Migrating {len(targets)} environment(s)...')
    for env_name in targets:
        env_id = envs.get(env_name)
        if env_id is None:
            print(f'  {env_name}: not found on Drive, skipping.')
            continue
        migrate_env(h, env_name, env_id)
    print('Done.')


if __name__ == '__main__':
    main()
