import base64
from typing import Any

import numpy as np
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


class DatabaseHandler:

    _NUM_RETRIES = 4  # google client auto-retries SSL/socket timeouts this many times
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
        self._drive = None
        self._sheets = None
        self._children_cache: dict[tuple[str, str], dict[str, str]] = {}

    @property
    def available(self) -> bool:
        return bool(self._credentials_info) and bool(self._root_folder_id)

    def clear_cache(self) -> None:
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
        result: dict[str, dict] = {}
        for lidar_name, lidar_id in self._list_children(env_id, self._FOLDER_MIME).items():
            result[lidar_name] = self._collect_cases(lidar_id, lidar_name)
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
        spreadsheet_id = self._list_children(lidar_id, self._SHEET_MIME).get(f'{lidar_name} - {case_path}')
        if spreadsheet_id is None:
            return {}
        return self._parse_viz_tab(spreadsheet_id)

    def _drive_service(self):
        if self._drive is None:
            self._drive = build('drive', 'v3', credentials=self._creds, cache_discovery=False)
        return self._drive

    def _sheets_service(self):
        if self._sheets is None:
            self._sheets = build('sheets', 'v4', credentials=self._creds, cache_discovery=False)
        return self._sheets

    def _list_children(self, parent_id: str, mime_type: str) -> dict[str, str]:
        cache_key = (parent_id, mime_type)
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
        self._children_cache[cache_key] = children
        return children

    def _collect_cases(self, lidar_id: str, lidar_name: str) -> dict:
        result: dict = {}
        prefix = f'{lidar_name} - '
        for sheet_name, spreadsheet_id in self._list_children(lidar_id, self._SHEET_MIME).items():
            if not sheet_name.startswith(prefix):
                continue
            case_path = sheet_name[len(prefix):]
            metrics = self._read_case_metrics(spreadsheet_id)
            if not metrics:
                continue
            segments = case_path.split('/') if case_path else []
            self._insert_nested(result, segments, metrics)
        return result

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

    def _read_case_metrics(self, spreadsheet_id: str) -> dict:
        try:
            response = self._sheets_service().spreadsheets().values().get(
                spreadsheetId=spreadsheet_id,
                range='Sheet1!1:2',
                valueRenderOption='UNFORMATTED_VALUE',
            ).execute(num_retries=self._NUM_RETRIES)
        except (HttpError, OSError):
            # OSError covers socket/TimeoutError — skip this case rather than
            # crashing the whole environment load on one stalled request.
            return {}
        rows = response.get('values', [])
        if len(rows) < 2:
            return {}
        headers, values = rows[0], rows[1]
        metrics: dict[str, dict] = {}
        for header, value in zip(headers, values):
            if not header or header == 'Case Name':
                continue
            casted = self._cast_value(value)
            # Sheet columns are written zone-first by the reporter
            # (`{zone}/{Metric}/{sub}`), but every PolyView consumer keys off the
            # metric name with a zone-prefixed sub-key (`{Metric}: {zone}_{sub}`).
            # Reshape on read so the rest of the app sees the structure it expects.
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
                metrics.setdefault('lidar_metadata', {})[header] = casted
        return metrics

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
