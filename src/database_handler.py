import base64
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


class DatabaseHandler:

    _NUM_RETRIES = 4  # google client auto-retries SSL/socket timeouts this many times
    _MAX_WORKERS = 5  # parallel Sheets reads; ~5/s keeps us under the ~300/min Sheets quota
    _READ_ATTEMPTS = 3  # extra rounds to re-read cases the rate limiter rejected
    _PARENTS_PER_QUERY = 50  # case folders OR'd into a single Drive query (query-length safe)
    _FOLDER_MIME = 'application/vnd.google-apps.folder'
    _SHEET_MIME = 'application/vnd.google-apps.spreadsheet'
    _SCOPES = [
        'https://www.googleapis.com/auth/drive',
        'https://www.googleapis.com/auth/spreadsheets',
    ]

    def __init__(self, credentials_info: dict, root_folder_id: str) -> None:
        self._credentials_info = dict(credentials_info) if credentials_info else {}
        self._root_folder_id = root_folder_id
        self._creds = (
            service_account.Credentials.from_service_account_info(
                self._credentials_info, scopes=self._SCOPES
            )
            if self._credentials_info
            else None
        )
        # Services hold an httplib2.Http that is not thread-safe, so each worker
        # thread gets its own instance via thread-local storage.
        self._thread_local = threading.local()
        self._children_cache: dict[tuple[str, str], dict[str, str]] = {}
        self._cache_lock = threading.Lock()

    @property
    def available(self) -> bool:
        return bool(self._credentials_info) and bool(self._root_folder_id)

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._children_cache.clear()

    def retrieve_environments(self) -> list[str]:
        if not self.available:
            return []
        return list(self._list_children(self._root_folder_id, self._FOLDER_MIME).keys())

    def retrieve_env_data(self, env_name: str) -> dict[str, dict]:
        if not self.available:
            return {}
        env_id = self._list_children(self._root_folder_id, self._FOLDER_MIME).get(env_name)
        if env_id is None:
            return {}
        # Fast path: a single "<env> - summary" sheet holds one row per case, so the whole
        # environment loads in ONE read instead of one per case. Falls through to the
        # per-case scan when no summary exists yet (un-migrated envs).
        summary = self._read_env_summary(env_id, env_name)
        if summary is not None:
            return summary
        lidars = self._list_children(env_id, self._FOLDER_MIME)
        result: dict[str, dict] = {lidar_name: {} for lidar_name in lidars}
        if not lidars:
            return result

        # The reporter writes cases in one of two layouts, which can coexist under a
        # single lidar folder:
        #   - flat   : the metrics sheet "<lidar> - <case>" sits directly under the
        #              lidar folder.
        #   - nested : a case folder named "<case>" under the lidar folder holds the
        #              sheet "<lidar> - <case>".
        # 1. In two batched Drive queries (not one round-trip per lidar), list every
        #    lidar's flat case sheets and nested case folders at once.
        lidar_ids = list(lidars.values())
        flat_by_lidar = self._list_children_by_parents(lidar_ids, self._SHEET_MIME)
        folders_by_lidar = self._list_children_by_parents(lidar_ids, self._FOLDER_MIME)

        # 2. Build (lidar_name, case_path, spreadsheet_id) read jobs. Flat sheets give
        #    the id directly; nested cases need their sheet resolved inside the folder.
        jobs: list[tuple[str, str, str]] = []
        nested: list[tuple[str, str, str]] = []  # (lidar_name, case_path, case_folder_id)
        for lidar_name, lidar_id in lidars.items():
            prefix = f'{lidar_name} - '
            for sheet_name, sheet_id in flat_by_lidar.get(lidar_id, {}).items():
                case_path = sheet_name[len(prefix):] if sheet_name.startswith(prefix) else sheet_name
                jobs.append((lidar_name, case_path, sheet_id))
            for case_path, case_id in folders_by_lidar.get(lidar_id, {}).items():
                nested.append((lidar_name, case_path, case_id))

        # Resolve nested-layout sheets in another batched Drive query rather than one
        # round-trip per case.
        if nested:
            sheets_by_folder = self._list_children_by_parents(
                [case_id for _, _, case_id in nested], self._SHEET_MIME
            )
            for lidar_name, case_path, case_id in nested:
                sheet_id = sheets_by_folder.get(case_id, {}).get(f'{lidar_name} - {case_path}')
                if sheet_id is not None:
                    jobs.append((lidar_name, case_path, sheet_id))

        if not jobs:
            return result

        # 3. Read the metrics sheets in parallel (network-bound). A read rejected by
        #    the API rate limiter is retried in a later round with backoff rather than
        #    dropped, so the overview stays complete. The nested-dict assembly stays on
        #    this thread, so only the I/O is concurrent.
        def _load(job: tuple[str, str, str]):
            _, _, sheet_id = job
            try:
                return job, self._read_case_metrics(sheet_id, propagate_errors=True)
            except (HttpError, OSError):
                return job, None  # None == failed read (distinct from {} == empty case)

        pending = jobs
        for attempt in range(self._READ_ATTEMPTS):
            failed: list[tuple[str, str, str]] = []
            with ThreadPoolExecutor(max_workers=self._MAX_WORKERS) as pool:
                for job, metrics in pool.map(_load, pending):
                    if metrics is None:
                        failed.append(job)
                        continue
                    if not metrics:
                        continue
                    lidar_name, case_path, _ = job
                    segments = case_path.split('/') if case_path else []
                    self._insert_nested(result[lidar_name], segments, metrics)
            if not failed:
                break
            pending = failed
            if attempt < self._READ_ATTEMPTS - 1:
                time.sleep(2 ** attempt)  # let the rate-limit window recover
        if failed:
            print(f'[PolyView] WARNING: {len(failed)} case sheet(s) could not be read after retries.')
        return result

    def retrieve_visualization_data(self, env_name: str, lidar_name: str, case_path: str) -> dict:
        if not self.available:
            return {}
        env_id = self._list_children(self._root_folder_id, self._FOLDER_MIME).get(env_name)
        if env_id is None:
            return {}
        lidar_id = self._list_children(env_id, self._FOLDER_MIME).get(lidar_name)
        if lidar_id is None:
            return {}
        sheet_name = f'{lidar_name} - {case_path}'
        # nested layout: the sheet lives inside a per-case folder...
        spreadsheet_id = None
        case_id = self._list_children(lidar_id, self._FOLDER_MIME).get(case_path)
        if case_id is not None:
            spreadsheet_id = self._list_children(case_id, self._SHEET_MIME).get(sheet_name)
        # ...flat layout: the sheet sits directly under the lidar folder.
        if spreadsheet_id is None:
            spreadsheet_id = self._list_children(lidar_id, self._SHEET_MIME).get(sheet_name)
        if spreadsheet_id is None:
            return {}
        return self._parse_viz_tab(spreadsheet_id)

    def retrieve_bag_download_link(self, env_name: str, lidar_name: str, case_path: str) -> str | None:
        """Returns a direct, click-to-download URL for the case's rosbag zip, or None if no zip
        is on Drive. The zip is shared 'anyone with link' at upload time, so this URL downloads
        without a login (large files show Google's scan-warning page once, then download).
        """
        if not self.available:
            return None
        env_id = self._list_children(self._root_folder_id, self._FOLDER_MIME).get(env_name)
        if env_id is None:
            return None
        lidar_id = self._list_children(env_id, self._FOLDER_MIME).get(lidar_name)
        if lidar_id is None:
            return None
        case_id = self._list_children(lidar_id, self._FOLDER_MIME).get(case_path)
        if case_id is None:
            return None
        for name, file_id in self._list_all_children(case_id).items():
            if name.lower().endswith('.zip'):
                return f'https://drive.google.com/uc?export=download&id={file_id}'
        return None

    def _drive_service(self):
        svc = getattr(self._thread_local, 'drive', None)
        if svc is None:
            svc = build('drive', 'v3', credentials=self._creds, cache_discovery=False)
            self._thread_local.drive = svc
        return svc

    def _sheets_service(self):
        svc = getattr(self._thread_local, 'sheets', None)
        if svc is None:
            svc = build('sheets', 'v4', credentials=self._creds, cache_discovery=False)
            self._thread_local.sheets = svc
        return svc

    def _list_children(self, parent_id: str, mime_type: str) -> dict[str, str]:
        cache_key = (parent_id, mime_type)
        with self._cache_lock:
            cached = self._children_cache.get(cache_key)
        if cached is not None:
            return cached
        children: dict[str, str] = {}
        page_token = None
        while True:
            response = self._drive_service().files().list(
                q=f"'{parent_id}' in parents and mimeType = '{mime_type}' and trashed = false",
                fields='nextPageToken, files(id, name)',
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
                corpora='allDrives',
                pageSize=100,
                pageToken=page_token,
            ).execute(num_retries=self._NUM_RETRIES)
            for item in response.get('files', []):
                children[item['name']] = item['id']
            page_token = response.get('nextPageToken')
            if not page_token:
                break
        with self._cache_lock:
            self._children_cache[cache_key] = children
        return children

    def _list_children_by_parents(self, parent_ids: list[str], mime_type: str) -> dict[str, dict[str, str]]:
        """List the children (of one mime type) of many parent folders in a few batched
        Drive queries instead of one round-trip per parent.

        OR's up to ``_PARENTS_PER_QUERY`` parent ids into a single ``files.list`` call and
        buckets the results by parent. Returns ``{parent_id: {name: id}}`` and seeds the
        per-parent ``_list_children`` cache so later single-parent lookups hit the cache.
        """
        by_parent: dict[str, dict[str, str]] = {pid: {} for pid in parent_ids}
        for start in range(0, len(parent_ids), self._PARENTS_PER_QUERY):
            chunk = parent_ids[start:start + self._PARENTS_PER_QUERY]
            parent_clause = ' or '.join(f"'{pid}' in parents" for pid in chunk)
            page_token = None
            while True:
                response = self._drive_service().files().list(
                    q=f"({parent_clause}) and mimeType = '{mime_type}' and trashed = false",
                    fields='nextPageToken, files(id, name, parents)',
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                    corpora='allDrives',
                    pageSize=1000,
                    pageToken=page_token,
                ).execute(num_retries=self._NUM_RETRIES)
                for item in response.get('files', []):
                    for parent in item.get('parents', []):
                        if parent in by_parent:
                            by_parent[parent][item['name']] = item['id']
                page_token = response.get('nextPageToken')
                if not page_token:
                    break
        with self._cache_lock:
            for pid, names in by_parent.items():
                self._children_cache[(pid, mime_type)] = names
        return by_parent

    def _list_all_children(self, parent_id: str) -> dict[str, str]:
        """Like _list_children but without a mime-type filter — used to find the bag zip,
        whose stored mime type may vary (application/zip vs x-zip-compressed)."""
        cache_key = (parent_id, '*')
        with self._cache_lock:
            cached = self._children_cache.get(cache_key)
        if cached is not None:
            return cached
        children: dict[str, str] = {}
        page_token = None
        while True:
            response = self._drive_service().files().list(
                q=f"'{parent_id}' in parents and trashed = false",
                fields='nextPageToken, files(id, name)',
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
                corpora='allDrives',
                pageSize=100,
                pageToken=page_token,
            ).execute(num_retries=self._NUM_RETRIES)
            for item in response.get('files', []):
                children[item['name']] = item['id']
            page_token = response.get('nextPageToken')
            if not page_token:
                break
        with self._cache_lock:
            self._children_cache[cache_key] = children
        return children

    def _insert_nested(self, root: dict, segments: list[str], leaf: dict) -> None:
        if not segments:
            root.update(leaf)
            return
        cursor = root
        for seg in segments[:-1]:
            key = self._parse_segment_key(seg)
            cursor = cursor.setdefault(key, {})
        last_key = self._parse_segment_key(segments[-1])
        cursor[last_key] = leaf

    def _parse_segment_key(self, name: str) -> Any:
        if '=' in name:
            _, raw = name.split('=', 1)
            return self._cast_value(raw)
        if '_' in name:
            _, raw = name.rsplit('_', 1)
            val = self._cast_value(raw)
            if not isinstance(val, str):
                return val
        return name

    def _cast_value(self, raw: Any) -> Any:
        if not isinstance(raw, str):
            return raw
        if raw.lower() == 'true':
            return True
        if raw.lower() == 'false':
            return False
        try:
            return int(raw)
        except ValueError:
            pass
        try:
            return float(raw)
        except ValueError:
            pass
        return raw

    def _read_case_metrics(self, spreadsheet_id: str, propagate_errors: bool = False) -> dict:
        try:
            response = self._sheets_service().spreadsheets().values().get(
                spreadsheetId=spreadsheet_id,
                range='Sheet1!1:2',
                valueRenderOption='UNFORMATTED_VALUE',
            ).execute(num_retries=self._NUM_RETRIES)
        except (HttpError, OSError):
            # OSError covers socket/TimeoutError. A failed read is NOT an empty
            # case — when reading concurrently the caller retries these so a
            # throttled request doesn't silently drop a case from the overview.
            if propagate_errors:
                raise
            return {}
        rows = response.get('values', [])
        if len(rows) < 2:
            return {}
        return self._reshape_row(rows[0], rows[1])

    def _reshape_row(self, headers: list, values: list) -> dict:
        """Reshape one flat metrics row into the nested `{Metric: {zone_sub: value}}` form
        the app expects. Sheet columns are written zone-first by the reporter
        (`{zone}/{Metric}/{sub}`); identity/empty columns are skipped. Shared by the
        per-case reader and the environment-summary reader."""
        metrics: dict[str, dict] = {}
        for header, value in zip(headers, values):
            if header in ('', 'Case Name', 'Lidar') or value == '':
                continue
            casted = self._cast_value(value)
            parts = str(header).split('/')
            if len(parts) >= 3:
                zone, metric = parts[0], parts[1]
                sub = '_'.join(parts[2:])
                metrics.setdefault(metric, {})[f'{zone}_{sub}'] = casted
            elif len(parts) == 2:
                # Run-global scalars (e.g. `__global__/<sub>`) — keep as-is.
                top, sub = parts
                metrics.setdefault(top, {})[sub] = casted
            else:
                metrics.setdefault('lidar_metadata', {})[str(header)] = casted
        return metrics

    def _read_env_summary(self, env_id: str, env_name: str) -> dict[str, dict] | None:
        """Read the single "<env> - summary" sheet (one row per case) in ONE call and rebuild
        the `{lidar: {case tree}}` structure. Returns None when no summary sheet exists, so the
        caller can fall back to the per-case scan."""
        summary_id = self._list_children(env_id, self._SHEET_MIME).get(f'{env_name} - summary')
        if summary_id is None:
            return None
        try:
            response = self._sheets_service().spreadsheets().values().get(
                spreadsheetId=summary_id,
                range='Sheet1',
                valueRenderOption='UNFORMATTED_VALUE',
            ).execute(num_retries=self._NUM_RETRIES)
        except (HttpError, OSError):
            return None
        rows = response.get('values', [])
        if len(rows) < 2:
            return None
        header = rows[0]
        # Seed every lidar folder so lidars without cases still appear in the selector.
        result: dict[str, dict] = {name: {} for name in self._list_children(env_id, self._FOLDER_MIME)}
        for row in rows[1:]:
            if len(row) < 2:
                continue
            lidar_name, case_path = str(row[0]), str(row[1])
            metrics = self._reshape_row(header[2:], row[2:])
            if not metrics:
                continue
            result.setdefault(lidar_name, {})
            segments = case_path.split('/') if case_path else []
            self._insert_nested(result[lidar_name], segments, metrics)
        return result

    def _parse_viz_tab(self, spreadsheet_id: str) -> dict:
        result: dict = {
            'profile_plane': {},
            'orientation': {},
            'fitted_planes': {},
            'dead_cells': {},
            'worst_points': {},
            'roi_cloud': None,
            'filtered_roi_cloud': None,
        }
        try:
            response = self._sheets_service().spreadsheets().values().get(
                spreadsheetId=spreadsheet_id,
                range='Visualization',
            ).execute(num_retries=self._NUM_RETRIES)
        except (HttpError, OSError):
            return result
        rows = response.get('values', [])

        current_section: str | None = None
        is_cloud: bool = False
        current_cloud_key: str | None = None
        cloud_chunks: list[str] = []

        def _flush_cloud():
            if is_cloud and cloud_chunks and current_cloud_key:
                try:
                    result[current_cloud_key] = self._decode_cloud(''.join(cloud_chunks))
                except Exception as e:
                    print(f'[PolyView] ERROR decoding cloud "{current_cloud_key}": {e}')
                cloud_chunks.clear()

        for row in rows:
            if not row:
                continue
            text = str(row[0])
            if not text:
                continue

            # Bullet "key: value" line — only when we're in a non-cloud section
            if ': ' in text and current_section is not None and not is_cloud:
                key, raw = text.split(': ', 1)
                result[current_section][key] = self._cast_value(raw)
                continue

            # Section heading detection — these strings never appear inside base64 chunks
            # because the writer separates header text with the U+00B7 mid-dot, which is
            # outside the base64 alphabet.
            is_heading = (
                text in ('Orientation', 'ProfilePlane', 'WorstPoints')
                or ' · FittedPlane' in text
                or ' · DeadCells' in text
                or 'base64' in text
            )

            if is_heading:
                _flush_cloud()
                if 'base64' in text:
                    current_cloud_key = text.split(' ·')[0].strip()
                    is_cloud = True
                    current_section = None
                else:
                    is_cloud = False
                    current_cloud_key = None
                    if text == 'ProfilePlane':
                        current_section = 'profile_plane'
                    elif text == 'Orientation':
                        current_section = 'orientation'
                    elif 'FittedPlane' in text:
                        current_section = 'fitted_planes'
                    elif 'DeadCells' in text:
                        current_section = 'dead_cells'
                    elif text == 'WorstPoints':
                        current_section = 'worst_points'
                    else:
                        current_section = None
                continue

            # Anything left, while we're inside a cloud block, is a base64 chunk
            if is_cloud:
                cloud_chunks.append(text)

        _flush_cloud()
        return result

    def _decode_cloud(self, encoded: str) -> np.ndarray:
        cleaned = encoded.strip()
        cleaned += '=' * (-len(cleaned) % 4)
        raw = np.frombuffer(base64.b64decode(cleaned), dtype=np.float32)
        n = len(raw)
        if n % 4 == 0:
            return raw.reshape(-1, 4)
        if n % 3 == 0:
            xyz = raw.reshape(-1, 3)
            return np.hstack([xyz, np.zeros((len(xyz), 1), dtype=np.float32)])
        raise ValueError(f'Point cloud buffer has {n} floats, not divisible by 3 or 4')
