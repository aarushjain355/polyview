import base64
import os
from typing import Any

import numpy as np
import requests


class DatabaseHandler:


    def __init__(self, token: str, database_id: str, notion_version: str, base_url: str) -> None:
        self._database_id = database_id
        self._token = token
        self._notion_version = notion_version
        self._base_url = base_url

    @property
    def available(self) -> bool:
        return bool(self._token and self._database_id)

    def retrieve_visualization_data(self) -> dict[str, dict]:
        if not self.available:
            return {}
        pages = self._fetch_pages()
        print(f'[PolyView] all page names: {list(pages.keys())}')
        result = {}
        for name, page_id in pages.items():
            if name.endswith('_VisualizationResults'):
                lidar_name = name[: -len('_VisualizationResults')]
                result[lidar_name] = self._parse_viz_blocks(page_id)
        print(f'[PolyView] visualization keys: {list(result.keys())}')
        return result

    def retrieve_data(self) -> dict[str, dict]:

        if not self.available:
            return {}
        pages = self._fetch_pages()
        return {
            name: self._parse_page_blocks(page_id)
            for name, page_id in pages.items()
            if not name.endswith('_VisualizationResults')
        }

    def _headers(self) -> dict[str, str]:
        return {
            'Authorization': f'Bearer {self._token}',
            'Content-Type': 'application/json',
            'Notion-Version': self._notion_version,
        }

    def _fetch_pages(self) -> dict[str, str]:
        pages: dict[str, str] = {}
        body: dict[str, Any] = {}
        while True:
            resp = requests.post(
                f'{self._base_url}/databases/{self._database_id}/query',
                headers=self._headers(),
                json=body,
            )
            resp.raise_for_status()
            data = resp.json()
            for page in data['results']:
                title_parts = page['properties'].get('Name', {}).get('title', [])
                if title_parts:
                    pages[title_parts[0]['text']['content']] = page['id']
            if not data.get('has_more'):
                break
            body['start_cursor'] = data['next_cursor']
        return pages

    def _fetch_all_blocks(self, page_id: str) -> list:
        blocks: list = []
        params: dict[str, Any] = {}
        while True:
            resp = requests.get(
                f'{self._base_url}/blocks/{page_id}/children',
                headers=self._headers(),
                params=params,
            )
            resp.raise_for_status()
            data = resp.json()
            blocks.extend(data['results'])
            if not data.get('has_more'):
                break
            params['start_cursor'] = data['next_cursor']
        return blocks

    def _parse_viz_blocks(self, page_id: str) -> dict:
        result: dict = {'profile_plane': {}, 'orientation': {}, 'fitted_planes': {}, 'dead_cells': {}, 'roi_cloud': None, 'filtered_roi_cloud': None}
        current_section: str | None = None
        is_cloud: bool = False
        current_cloud_key: str | None = None
        cloud_chunks: list[str] = []

        def _flush_cloud():
            if is_cloud and cloud_chunks and current_cloud_key:
                try:
                    decoded = self._decode_cloud(''.join(cloud_chunks))
                    result[current_cloud_key] = decoded
                    print(f'[PolyView] decoded cloud key="{current_cloud_key}" points={len(decoded)}')
                except Exception as e:
                    print(f'[PolyView] ERROR decoding cloud key="{current_cloud_key}": {e}')
                cloud_chunks.clear()

        for block in self._fetch_all_blocks(page_id):
            btype = block.get('type')
            if btype == 'heading_2':
                _flush_cloud()
                rt = block['heading_2'].get('rich_text', [])
                if not rt:
                    continue
                heading = rt[0]['text']['content']
                print(f'[PolyView] heading="{heading}"')
                if 'base64' in heading:
                    current_cloud_key = heading.split(' ·')[0].strip()
                    is_cloud = True
                    current_section = None
                else:
                    is_cloud = False
                    current_cloud_key = None
                    if heading == 'ProfilePlane':
                        current_section = 'profile_plane'
                    elif heading == 'Orientation':
                        current_section = 'orientation'
                    elif 'FittedPlane' in heading:
                        current_section = 'fitted_planes'
                    elif 'DeadCells' in heading:
                        current_section = 'dead_cells'
                    else:
                        current_section = None
            elif btype == 'bulleted_list_item' and current_section:
                rt = block['bulleted_list_item'].get('rich_text', [])
                if rt and ': ' in rt[0]['text']['content']:
                    key, raw = rt[0]['text']['content'].split(': ', 1)
                    try:
                        val: int | float | str = int(raw)
                    except ValueError:
                        try:
                            val = float(raw)
                        except ValueError:
                            val = raw
                    result[current_section][key] = val
            elif btype == 'paragraph' and is_cloud:
                rt = block['paragraph'].get('rich_text', [])
                if rt:
                    cloud_chunks.append(rt[0]['text']['content'])

        _flush_cloud()
        print(f'[PolyView] viz parse done: roi_cloud={result["roi_cloud"] is not None} filtered_roi_cloud={result["filtered_roi_cloud"] is not None} keys={list(result.keys())}')
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

    def _parse_page_blocks(self, page_id: str) -> dict:
        metrics: dict = {}
        current_category: str | None = None

        for block in self._fetch_all_blocks(page_id):
            btype = block.get('type')
            if btype == 'heading_2':
                rt = block['heading_2'].get('rich_text', [])
                if rt:
                    current_category = rt[0]['text']['content']
                    metrics[current_category] = {}
            elif btype == 'bulleted_list_item' and current_category is not None:
                rt = block['bulleted_list_item'].get('rich_text', [])
                if rt and ': ' in rt[0]['text']['content']:
                    key, raw = rt[0]['text']['content'].split(': ', 1)
                    val: int | float | str
                    try:
                        val = int(raw)
                    except ValueError:
                        try:
                            val = float(raw)
                        except ValueError:
                            val = raw
                    metrics[current_category][key] = val

        return metrics
