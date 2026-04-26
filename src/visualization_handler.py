from collections import defaultdict
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import yaml
from plotly.subplots import make_subplots

_CONFIG_PATH = Path(__file__).parent / 'chart_config.yaml'
_SCHEMAS_PATH = Path(__file__).parent / 'visualization_schemas.yaml'
_CSS_PATH = Path(__file__).parent / 'css' / 'charts.css'


class VisualizationHandler:

    def __init__(self):
        with open(_CONFIG_PATH, 'r') as f:
            self._config = yaml.safe_load(f)
        with open(_SCHEMAS_PATH, 'r') as f:
            self._schemas = yaml.safe_load(f)
        self.glow_css = f'<style>{_CSS_PATH.read_text()}</style>'

    def make_3d_figure(self, viz_data: dict) -> go.Figure:
        layout = self._config['layout']
        orientation = viz_data.get('orientation', {})
        pitch = float(orientation.get('pitch', 0.0))
        roll = float(orientation.get('roll', 0.0))
        yaw = float(orientation.get('yaw', 0.0))
        print(f'[PolyView] orientation: pitch={pitch:.4f} roll={roll:.4f} yaw={yaw:.4f} rad')
        print(f'[PolyView] link origin: x=0.0000, y=0.0000, z=0.0000')
        fig = go.Figure()
        fig.update_layout(
            scene=dict(
                xaxis=dict(title='X (m)', backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(255,255,255,0.08)', color='rgba(255,255,255,0.8)'),
                yaxis=dict(title='Y (m)', backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(255,255,255,0.08)', color='rgba(255,255,255,0.8)'),
                zaxis=dict(title='Z (m)', backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(255,255,255,0.08)', color='rgba(255,255,255,0.8)'),
                bgcolor='rgba(8,10,18,1.0)',
                aspectmode='data',
            ),
            paper_bgcolor=layout['paper_color'],
            height=720,
            legend=dict(
                font=dict(color='white', size=11),
                bgcolor='rgba(10,12,20,0.9)',
                bordercolor='rgba(57,255,20,0.25)',
                borderwidth=1,
                x=0.01, y=0.99,
                xanchor='left', yanchor='top',
            ),
            margin=dict(t=50, b=20, l=20, r=120),
            title=dict(text='<b>3D LiDAR Scene</b>', font=dict(color='white', size=16), x=0.5),
            modebar=dict(bgcolor='rgba(0,0,0,0)', color='rgba(255,255,255,0.2)', activecolor='#39FF14'),
        )
        return fig

    def add_point_cloud(self, fig: go.Figure, viz_data: dict) -> None:
        R, _ = self._parse_orientation(viz_data)
        cloud = viz_data.get('roi_cloud')
        if cloud is None:
            cloud = viz_data.get('filtered_roi_cloud')
        if cloud is None or len(cloud) == 0:
            return
        cloud_xyz = (R @ cloud[:, :3].T).T
        fig.add_trace(go.Scatter3d(
            x=cloud_xyz[:, 0], y=cloud_xyz[:, 1], z=cloud_xyz[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=cloud[:, 3],
                colorscale=[[0, '#00BFFF'], [0.5, '#39FF14'], [1, '#FFD700']],
                opacity=0.85,
                colorbar=dict(title='Intensity', thickness=10, len=0.45, x=1.02),
            ),
            name='ROI Cloud',
            hovertemplate='x: %{x:.3f}  y: %{y:.3f}  z: %{z:.3f}<extra>ROI Cloud</extra>',
        ))

    def add_expected_planes(self, fig: go.Figure, viz_data: dict) -> None:
        fp = viz_data.get('fitted_planes', {})
        if not fp:
            return
        zones = [k.replace('_expected_x', '') for k in fp if k.endswith('_expected_x')]
        if not zones:
            return
        colors = ['#00BFFF', '#39FF14']
        for idx, zone in enumerate(zones):
            x_ref = float(fp.get(f'{zone}_expected_x', 0.0))
            y_min = float(fp.get(f'{zone}_expected_y_min', -1.0))
            y_max = float(fp.get(f'{zone}_expected_y_max', 1.0))
            z_min = float(fp.get(f'{zone}_expected_z_min', 0.0))
            z_max = float(fp.get(f'{zone}_expected_z_max', 2.0))
            color = colors[idx % len(colors)]
            # close the loop: bottom-left → bottom-right → top-right → top-left → bottom-left
            fig.add_trace(go.Scatter3d(
                x=[x_ref, x_ref, x_ref, x_ref, x_ref],
                y=[y_min, y_max, y_max, y_min, y_min],
                z=[z_min, z_min, z_max, z_max, z_min],
                mode='lines',
                line=dict(color=color, width=4),
                name=f'Expected Plane ({zone})',
                hovertemplate=f'Expected plane · {zone}<extra></extra>',
            ))

    def add_fitted_pca_plane(self, fig: go.Figure, viz_data: dict) -> None:
        fp = viz_data.get('fitted_planes', {})
        if not fp:
            return
        zones = [k.replace('_plane_center_x', '') for k in fp if k.endswith('_plane_center_x')]
        colors = ['#FF6B6B', '#FFD700']
        for idx, zone in enumerate(zones):
            cx = float(fp.get(f'{zone}_plane_center_x', 0.0))
            cy = float(fp.get(f'{zone}_plane_center_y', 0.0))
            cz = float(fp.get(f'{zone}_plane_center_z', 0.0))
            nx = float(fp.get(f'{zone}_plane_normal_x', 1.0))
            ny = float(fp.get(f'{zone}_plane_normal_y', 0.0))
            nz = float(fp.get(f'{zone}_plane_normal_z', 0.0))
            y_min = float(fp.get(f'{zone}_plane_bounds_y_min', cy - 0.5))
            y_max = float(fp.get(f'{zone}_plane_bounds_y_max', cy + 0.5))
            z_min = float(fp.get(f'{zone}_plane_bounds_z_min', cz - 0.5))
            z_max = float(fp.get(f'{zone}_plane_bounds_z_max', cz + 0.5))
            corners = np.array([
                [cx - (ny * (y - cy) + nz * (z - cz)) / nx if abs(nx) > 1e-6 else cx, y, z]
                for y, z in [(y_min, z_min), (y_max, z_min), (y_max, z_max), (y_min, z_max)]
            ])
            fig.add_trace(go.Mesh3d(
                x=corners[:, 0].tolist(), y=corners[:, 1].tolist(), z=corners[:, 2].tolist(),
                i=[0, 0], j=[1, 2], k=[2, 3],
                opacity=0.25,
                color=colors[idx % len(colors)],
                name=f'PCA Plane ({zone})',
                hovertemplate=f'Fitted PCA plane · {zone}<extra></extra>',
            ))

    def add_sensor_axes(self, fig: go.Figure, viz_data: dict) -> None:
        R, origin = self._parse_orientation(viz_data)
        for i, (color, label) in enumerate(zip(['#FF6B6B', '#39FF14', '#00BFFF'], ['X fwd', 'Y left', 'Z up'])):
            tip = origin + R[:, i] * 0.4
            fig.add_trace(go.Scatter3d(
                x=[float(origin[0]), float(tip[0])],
                y=[float(origin[1]), float(tip[1])],
                z=[float(origin[2]), float(tip[2])],
                mode='lines',
                line=dict(color=color, width=7),
                name=f'LiDAR {label}',
            ))
        fig.add_trace(go.Scatter3d(
            x=[float(origin[0])], y=[float(origin[1])], z=[float(origin[2])],
            mode='markers',
            marker=dict(size=7, color='white', symbol='circle'),
            name='LiDAR Origin',
        ))

    def add_spatial_dropout_analysis(self, fig: go.Figure, viz_data: dict) -> None:
        dead_cells = viz_data.get('dead_cells', {})
        if not dead_cells:
            return
        fitted_planes = viz_data.get('fitted_planes', {})
        cell_size = float(dead_cells.get('dead_cell_size_m', 0.05))

        zones = {key.rsplit('_dead_cell_', 1)[0] for key in dead_cells if '_dead_cell_' in key}
        palette = self._config['palette']
        for zone_idx, zone in enumerate(sorted(zones)):
            x_ref = float(fitted_planes.get(f'{zone}_expected_x', 0.0))
            y_min = float(fitted_planes.get(f'{zone}_expected_y_min', 0.0))
            y_max = float(fitted_planes.get(f'{zone}_expected_y_max', 1.0))
            z_min = float(fitted_planes.get(f'{zone}_expected_z_min', 0.0))
            z_max = float(fitted_planes.get(f'{zone}_expected_z_max', 1.0))
            n_y = max(1, int(np.ceil((y_max - y_min) / cell_size)))
            n_z = max(1, int(np.ceil((z_max - z_min) / cell_size)))

            dead_set = set()
            for key, val in dead_cells.items():
                if key.startswith(f'{zone}_dead_cell_') and key.endswith('_y_m'):
                    idx_str = key[len(f'{zone}_dead_cell_'):-4]
                    z_key = f'{zone}_dead_cell_{idx_str}_z_m'
                    if z_key in dead_cells:
                        iy = int((float(val) - y_min) / cell_size)
                        iz = int((float(dead_cells[z_key]) - z_min) / cell_size)
                        dead_set.add((iy, iz))

            live_x, live_y, live_z, live_i, live_j, live_k = [], [], [], [], [], []
            dead_x, dead_y, dead_z, dead_i, dead_j, dead_k = [], [], [], [], [], []
            for iy in range(n_y):
                for iz in range(n_z):
                    y0, y1 = y_min + iy * cell_size, y_min + (iy + 1) * cell_size
                    z0, z1 = z_min + iz * cell_size, z_min + (iz + 1) * cell_size
                    corners = [(x_ref, y0, z0), (x_ref, y1, z0), (x_ref, y1, z1), (x_ref, y0, z1)]
                    vx, vy, vz, vi, vj, vk = (dead_x, dead_y, dead_z, dead_i, dead_j, dead_k) if (iy, iz) in dead_set else (live_x, live_y, live_z, live_i, live_j, live_k)
                    base = len(vx)
                    for cx, cy, cz in corners:
                        vx.append(cx); vy.append(cy); vz.append(cz)
                    vi += [base, base]; vj += [base + 1, base + 2]; vk += [base + 2, base + 3]

            if live_x:
                fig.add_trace(go.Mesh3d(x=live_x, y=live_y, z=live_z, i=live_i, j=live_j, k=live_k, color='#FF4444', opacity=0.8, name=f'Live Cells ({zone})', showlegend=True))
            if dead_x:
                fig.add_trace(go.Mesh3d(x=dead_x, y=dead_y, z=dead_z, i=dead_i, j=dead_j, k=dead_k, color='#111111', opacity=0.9, name=f'Dead Cells ({zone})', showlegend=True))

            grid_y_max = y_min + n_y * cell_size
            grid_z_max = z_min + n_z * cell_size
            gx, gy, gz = [], [], []
            for iy in range(n_y + 1):
                y = y_min + iy * cell_size
                gx += [x_ref, x_ref, None]; gy += [y, y, None]; gz += [z_min, grid_z_max, None]
            for iz in range(n_z + 1):
                z = z_min + iz * cell_size
                gx += [x_ref, x_ref, None]; gy += [y_min, grid_y_max, None]; gz += [z, z, None]
            grid_color = palette[zone_idx % len(palette)]
            fig.add_trace(go.Scatter3d(x=gx, y=gy, z=gz, mode='lines', line=dict(color=grid_color, width=3), name=f'Grid ({zone})', showlegend=False, hoverinfo='skip'))

            bx = [x_ref, x_ref, x_ref, x_ref, x_ref]
            by = [y_min, grid_y_max, grid_y_max, y_min, y_min]
            bz = [z_min, z_min, grid_z_max, grid_z_max, z_min]
            fig.add_trace(go.Scatter3d(x=bx, y=by, z=bz, mode='lines', line=dict(color='white', width=5), name=f'Zone Border ({zone})', showlegend=False, hoverinfo='skip'))

            if dead_set:
                dx, dy, dz = [], [], []
                for iy, iz in dead_set:
                    y0 = y_min + iy * cell_size
                    y1 = y_min + (iy + 1) * cell_size
                    z0 = z_min + iz * cell_size
                    z1 = z_min + (iz + 1) * cell_size
                    dx += [x_ref, x_ref, x_ref, x_ref, x_ref, None]
                    dy += [y0, y1, y1, y0, y0, None]
                    dz += [z0, z0, z1, z1, z0, None]
                fig.add_trace(go.Scatter3d(x=dx, y=dy, z=dz, mode='lines', line=dict(color='white', width=2), name=f'Dead Cell Borders ({zone})', showlegend=False, hoverinfo='skip'))

    def _parse_orientation(self, viz_data: dict) -> tuple[np.ndarray, np.ndarray]:
        orientation = viz_data.get('orientation', {})
        pitch = float(orientation.get('pitch', 0.0))
        roll = float(orientation.get('roll', 0.0))
        yaw = float(orientation.get('yaw', 0.0))
        return self._euler_to_rotation(pitch, roll, yaw), np.array([0.0, 0.0, 0.0])

    def _euler_to_rotation(self, pitch: float, roll: float, yaw: float) -> np.ndarray:
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)
        cy, sy = np.cos(yaw), np.sin(yaw)
        return np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp,     cp * sr,                cp * cr               ],
        ])

    def render_single_lidar_metrics(self, lidar_name: str, metrics: dict, thresholds: dict | None = None, secondary_axis_keys: list | None = None, y_padding: float = 0.3, split_by_suffix_categories: list | None = None, split_exclude_suffixes: dict | None = None) -> list[go.Figure]:
        layout = self._config['layout']
        mc = self._config['metric_chart']
        figs = []
        secondary_set = set(secondary_axis_keys or [])
        split_set = set(split_by_suffix_categories or [])
        sorted_categories = sorted(metrics.items(), key=lambda x: ('Intensity' in x[0], 'NoiseRegion' in x[0]))
        for category, values in sorted_categories:
            if not isinstance(values, dict):
                continue
            items = {k: v for k, v in values.items() if isinstance(v, (int, float)) and k != 'visualization'}
            if not items:
                continue
            if category in split_set:
                suffix_groups: dict = defaultdict(dict)
                for k, v in items.items():
                    suffix_groups[k.rsplit('_', 1)[-1]][k] = v
                excluded = set((split_exclude_suffixes or {}).get(category, []))
                subcategory_items = [(f'{category} · {suffix}', grp, False) for suffix, grp in sorted(suffix_groups.items()) if suffix not in excluded]
            else:
                primary_items = {k: v for k, v in items.items() if k not in secondary_set}
                secondary_items = {k: v for k, v in items.items() if k in secondary_set}
                subcategory_items = [
                    (f'{category}', primary_items if primary_items else items, False),
                    (f'{category} (counts)', secondary_items, True),
                ]
            for title, fig_items, is_counts in subcategory_items:
                if not fig_items:
                    continue
                color = self._config['count_bar_color'] if is_counts else self._config['palette'][0]
                display_keys = [
                    k.replace('green_wall_', 'gw_').replace('whiteboard_', 'wb_')
                    for k in fig_items.keys()
                ] if is_counts else list(fig_items.keys())
                vals = list(fig_items.values())
                data_min = min(vals)
                data_max = max(vals)
                data_range = data_max - data_min
                if data_range == 0:
                    data_range = abs(data_max) * 0.1 or 0.001
                pad = data_range * y_padding
                y_range = [data_min - pad, data_max + pad]
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=display_keys,
                    y=vals,
                    marker=dict(color=color, opacity=0.85, line=dict(color=mc['bar_border_color'], width=mc['bar_border_width'])),
                    hovertemplate='<b>%{x}</b><br>%{y:.5g}<extra></extra>',
                    name=title,
                    showlegend=False,
                ))
                for key, val in zip(display_keys, vals):
                    fig.add_annotation(
                        x=key, y=mc['label_y_paper'],
                        xref='x', yref='paper',
                        text=f'{val:.4g}',
                        showarrow=False,
                        yanchor='bottom', xanchor='center',
                        font=dict(size=mc['label_font_size'], color='rgba(255,255,255,0.7)'),
                    )
                if not is_counts:
                    self._apply_threshold_bands(fig, title, category, fig_items, thresholds, y_range)
                fig.update_layout(
                    title=dict(text=f'<b>{title}</b>', font=dict(color='#39FF14', size=13), x=0.0),
                    paper_bgcolor=layout['paper_color'],
                    plot_bgcolor='rgba(8,10,18,0.0)',
                    height=mc['height'],
                    margin=dict(t=mc['margin_top'], b=mc['margin_bottom'], l=mc['margin_left'], r=mc['margin_right']),
                    xaxis=dict(
                        showticklabels=True,
                        tickangle=mc['tick_angle'],
                        tickfont=dict(size=mc['tick_font_size'], color='rgba(255,255,255,0.8)'),
                        showgrid=False,
                        zeroline=False,
                    ),
                    yaxis=dict(
                        tickfont=dict(size=mc['tick_font_size'], color='rgba(255,255,255,0.7)'),
                        gridcolor='rgba(255,255,255,0.04)',
                        zerolinecolor='rgba(255,255,255,0.15)',
                        tickformat='.3g',
                        range=y_range,
                        title=dict(
                            text=self._schemas['category_units'].get(category, ''),
                            font=dict(size=11, color='rgba(255,255,255,0.5)'),
                            standoff=5,
                        ),
                    ),
                    showlegend=True,
                    legend=dict(
                        x=mc['legend_x'], y=mc['legend_y'],
                        xanchor=mc['legend_xanchor'], yanchor=mc['legend_yanchor'],
                        font=dict(color='white', size=mc['legend_font_size']),
                        bgcolor='rgba(10,12,20,0.7)',
                        bordercolor='rgba(255,255,255,0.1)',
                        borderwidth=1,
                    ),
                    modebar=dict(bgcolor='rgba(0,0,0,0)', color='rgba(255,255,255,0.2)', activecolor='#39FF14'),
                )
                fig.update_yaxes(range=y_range)
                figs.append((title, fig))
        return figs

    def _zones_for_key(self, threshold_list: list, key_name: str) -> dict:
        for entry in threshold_list:
            if key_name in entry.get('keys', []):
                return {k: v for k, v in entry.items() if k != 'keys'}
        return {}

    def _apply_threshold_bands(self, fig: go.Figure, title: str, category: str, items: dict, thresholds: dict | None, y_range: list | None = None) -> None:
        if not thresholds:
            return
        threshold_config = thresholds.get(title) or thresholds.get(category)
        if threshold_config is None:
            return
        zone_fill_colors = self._config['zone_fill_colors']
        zone_legend_colors = self._config['zone_legend_colors']
        zone_border_width = self._config['metric_chart']['zone_border_width']
        legend_marker_size = self._config['metric_chart']['legend_marker_size']
        legend_entries: dict[str, str] = {}  # label -> legend color, first occurrence wins
        if isinstance(threshold_config, list):
            for key_idx, key in enumerate(items.keys()):
                zones = self._zones_for_key(threshold_config, key)
                for zone, fill_color in zone_fill_colors.items():
                    zone_data = zones.get(zone, {})
                    if not zone_data.get('enabled'):
                        continue
                    label = zone_data.get('label', '')
                    if label and label not in legend_entries:
                        legend_entries[label] = zone_legend_colors[zone]
                    y0, y1 = zone_data['min'], zone_data['max']
                    if y_range is not None:
                        if y1 <= y_range[0] or y0 >= y_range[1]:
                            continue
                        y0 = max(y0, y_range[0])
                        y1 = min(y1, y_range[1])
                    fig.add_shape(
                        type='rect',
                        x0=key_idx - 0.5, x1=key_idx + 0.5,
                        y0=y0, y1=y1,
                        fillcolor=fill_color, opacity=1.0,
                        line=dict(color=zone_legend_colors[zone], width=zone_border_width),
                        xref='x', yref='y', layer='below',
                    )
        else:
            for zone, fill_color in zone_fill_colors.items():
                zone_data = threshold_config.get(zone, {})
                if not zone_data.get('enabled'):
                    continue
                label = zone_data.get('label', '')
                if label and label not in legend_entries:
                    legend_entries[label] = zone_legend_colors[zone]
                y0, y1 = zone_data['min'], zone_data['max']
                if y_range is not None:
                    if y1 <= y_range[0] or y0 >= y_range[1]:
                        continue
                    y0 = max(y0, y_range[0])
                    y1 = min(y1, y_range[1])
                fig.add_hrect(y0=y0, y1=y1, fillcolor=fill_color, opacity=1.0, line_color=zone_legend_colors[zone], line_width=zone_border_width, layer='below')
        for label, leg_color in legend_entries.items():
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(color=leg_color, size=legend_marker_size, symbol='square'),
                name=label,
                showlegend=True,
            ))

    def render_overview_radar(self, metrics_data: dict) -> go.Figure:
        if not metrics_data:
            return go.Figure()

        layout = self._config['layout']
        lidar_names = list(metrics_data.keys())
        colors = self._colors(len(lidar_names))

        radar_metrics = self._schemas['radar_metrics']
        radar_categories = [
            cat for cat in radar_metrics
            if any(radar_metrics[cat]['key'] in metrics_data[l].get(cat, {}) for l in lidar_names)
        ]
        if not radar_categories:
            return go.Figure()

        raw: dict[str, list[float]] = {cat: [] for cat in radar_categories}
        for lidar in lidar_names:
            for cat in radar_categories:
                key = radar_metrics[cat]['key']
                val = float(metrics_data[lidar].get(cat, {}).get(key, 0) or 0)
                raw[cat].append(val)

        fig = go.Figure()

        for lidar_idx, lidar_name in enumerate(lidar_names):
            scores = []
            for cat in radar_categories:
                key = radar_metrics[cat]['key']
                lower_is_better = radar_metrics[cat]['lower_is_better']
                vals = raw[cat]
                mn, mx = min(vals), max(vals)
                val = float(metrics_data[lidar_name].get(cat, {}).get(key, 0) or 0)
                norm = (val - mn) / (mx - mn) if mx != mn else 0.5
                scores.append(round((1 - norm) if lower_is_better else norm, 3))

            theta = radar_categories + [radar_categories[0]]
            r = scores + [scores[0]]
            color = colors[lidar_idx]

            fig.add_trace(go.Scatterpolar(
                r=r,
                theta=theta,
                fill='toself',
                fillcolor=self._hex_to_rgba(color, 0.18),
                line=dict(color=color, width=3),
                marker=dict(size=7, color=color, symbol='circle', line=dict(color='white', width=1.5)),
                name=lidar_name,
                hovertemplate='<b>%{theta}</b><br>Score: <b>%{r:.3f}</b><extra>' + lidar_name + '</extra>',
            ))

        fig.update_layout(
            polar=dict(
                bgcolor='rgba(10,12,20,0.6)',
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickfont=dict(size=9, color='rgba(255,255,255,0.3)'),
                    gridcolor='rgba(255,255,255,0.06)',
                    linecolor='rgba(255,255,255,0.06)',
                    tickvals=[0.25, 0.5, 0.75, 1.0],
                ),
                angularaxis=dict(
                    tickfont=dict(size=13, color='rgba(255,255,255,0.9)', family='sans-serif'),
                    gridcolor='rgba(255,255,255,0.06)',
                    linecolor='rgba(255,255,255,0.1)',
                    direction='clockwise',
                ),
            ),
            showlegend=True,
            legend=dict(
                font=dict(size=12, color='white'),
                bgcolor='rgba(10,12,20,0.9)',
                bordercolor='rgba(57,255,20,0.25)',
                borderwidth=1,
                x=1.08, y=1.0,
                itemsizing='constant',
                itemclick='toggleothers',
                itemdoubleclick='toggle',
            ),
            paper_bgcolor=layout['paper_color'],
            height=600,
            title=dict(
                text='<b>Overall LiDAR Performance</b>  ·  normalized score per category  ·  1.0 = best in group',
                font=dict(size=14, color='rgba(255,255,255,0.65)', family='sans-serif'),
                x=0.5, xanchor='center',
            ),
            margin=dict(t=70, b=50, l=100, r=220),
            modebar=dict(bgcolor='rgba(0,0,0,0)', color='rgba(255,255,255,0.2)', activecolor='#39FF14'),
        )

        return fig

    def render_metrics_comparison(self, metrics_data: dict) -> go.Figure:
        categories = self._collect_categories(metrics_data)
        if not categories:
            return go.Figure()

        layout = self._config['layout']
        lidar_names = list(metrics_data.keys())
        colors = self._colors(len(lidar_names))
        n_rows = len(categories)

        max_spacing = 1.0 / n_rows if n_rows > 1 else 1.0
        vertical_spacing = min(layout['subplot_vertical_spacing'], max_spacing * 0.85)

        fig = make_subplots(
            rows=n_rows, cols=1,
            subplot_titles=list(categories.keys()),
            vertical_spacing=vertical_spacing,
        )

        for row, (category, metric_names) in enumerate(categories.items(), start=1):
            if category in self._schemas['box_mappings']:
                self._add_box_traces(fig, metrics_data, lidar_names, colors, category, row)
            elif category in self._schemas['error_bar_mappings']:
                self._add_error_bar_traces(fig, metrics_data, lidar_names, colors, category, row)
            elif category in self._schemas['fraction_metrics']:
                self._add_fraction_traces(fig, metrics_data, lidar_names, colors, category, row)
            else:
                self._add_bar_traces(fig, metrics_data, lidar_names, colors, category, metric_names, row)

            self._style_axes(fig, row, layout, category)

        for annotation in fig.layout.annotations:
            annotation.font = dict(size=layout['subplot_title_font_size'], color='#39FF14', family='sans-serif')
            annotation.bgcolor = 'rgba(57,255,20,0.07)'
            annotation.bordercolor = 'rgba(57,255,20,0.2)'
            annotation.borderwidth = 1
            annotation.borderpad = 7
            annotation.y += 0.02

        fig.update_layout(
            height=layout['subplot_height_per_row'] * n_rows,
            barmode='group',
            bargap=0.28,
            bargroupgap=0.06,
            uniformtext=dict(minsize=8, mode='show'),
            legend=dict(
                title=dict(text=f"<b>{layout['legend_title']}</b>", font=dict(size=layout['legend_font_size'] + 2, color='white')),
                font=dict(size=layout['legend_font_size'], color='rgba(255,255,255,0.9)'),
                bgcolor='rgba(10,12,20,0.9)',
                bordercolor='rgba(57,255,20,0.2)',
                borderwidth=1,
                orientation='v',
                x=1.01, xanchor='left',
                y=1.0, yanchor='top',
                itemsizing='constant',
                itemclick='toggleothers',
                itemdoubleclick='toggle',
            ),
            template=layout['template'],
            paper_bgcolor=layout['paper_color'],
            plot_bgcolor='rgba(8,10,18,0.0)',
            margin=dict(t=50, b=80, l=90, r=230),
            hoverlabel=dict(
                bgcolor='#0d0f1a',
                font_size=13,
                font_color='white',
                bordercolor='rgba(57,255,20,0.6)',
                namelength=-1,
            ),
            modebar=dict(bgcolor='rgba(0,0,0,0)', color='rgba(255,255,255,0.2)', activecolor='#39FF14'),
            dragmode='zoom',
        )

        return fig

    def _gradient_marker(self, values: list, color: str) -> dict:
        """Value-driven opacity — brighter bars = higher absolute value."""
        abs_vals = [abs(v) for v in values]
        if not abs_vals:
            return dict(color=color, line=dict(color=color, width=1.5))
        mx = max(abs_vals) or 1
        bar_colors = [self._interpolate_color(color, t / mx) for t in abs_vals]
        return dict(color=bar_colors, line=dict(color=color, width=1.5))

    def _interpolate_color(self, hex_color: str, t: float) -> str:
        h = hex_color.lstrip('#')
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f'rgba({r},{g},{b},{0.4 + 0.6 * t:.3f})'

    def _add_box_traces(self, fig, metrics_data, lidar_names, colors, category, row):
        mapping = self._schemas['box_mappings'][category]
        for lidar_idx, lidar_name in enumerate(lidar_names):
            d = metrics_data[lidar_name].get(category, {})
            color = colors[lidar_idx]
            fig.add_trace(go.Box(
                name=lidar_name,
                x=[lidar_name],
                lowerfence=[d.get(mapping['low'])] if mapping.get('low') and d.get(mapping['low']) is not None else None,
                q1=[d.get(mapping['q1'], 0)],
                median=[d.get(mapping['median'], 0)],
                q3=[d.get(mapping['q3'], 0)],
                upperfence=[d.get(mapping['high'], 0)],
                mean=[d.get(mapping['mean'], 0)],
                marker=dict(color=color, size=10, symbol='diamond', line=dict(color='white', width=1.5)),
                line=dict(color=color, width=2.5),
                fillcolor=self._hex_to_rgba(color, 0.22),
                whiskerwidth=0.6,
                boxmean=True,
                hovertemplate=(
                    f'<b>{lidar_name}</b><br>'
                    'Median: %{median:.5g}<br>'
                    'Q1 / Q3: %{q1:.5g} / %{q3:.5g}'
                    '<extra></extra>'
                ),
                showlegend=(row == 1),
            ), row=row, col=1)

    def _add_error_bar_traces(self, fig, metrics_data, lidar_names, colors, category, row):
        mapping = self._schemas['error_bar_mappings'][category]
        zones = [z['label'] for z in mapping]
        n_lidars = len(lidar_names)
        for lidar_idx, lidar_name in enumerate(lidar_names):
            d = metrics_data[lidar_name].get(category, {})
            means = [d.get(z['mean_key'], 0) for z in mapping]
            stds = [d.get(z['std_key'], 0) if z['std_key'] else 0 for z in mapping]
            color = colors[lidar_idx]
            fig.add_trace(go.Bar(
                name=lidar_name,
                x=zones,
                y=means,
                error_y=dict(
                    type='data', array=stds, visible=True,
                    color=self._hex_to_rgba(color, 0.8),
                    thickness=2, width=6,
                ),
                marker=self._gradient_marker(means, color),
                hovertemplate=f'<b>{lidar_name}</b><br>Zone: %{{x}}<br>Mean: %{{y:.5g}}<extra></extra>',
                showlegend=(row == 1),
            ), row=row, col=1)

            bar_slot_width = 0.72 * 0.94 / n_lidars
            x_offset = (lidar_idx - (n_lidars - 1) / 2) * bar_slot_width
            for zone_idx, (zone, mean, std) in enumerate(zip(zones, means, stds)):
                fig.add_annotation(
                    x=zone_idx + x_offset, y=mean + std,
                    text=f'<b>{mean:.4g}</b>',
                    yshift=10,
                    showarrow=False,
                    font=dict(size=10, color='white', family='monospace'),
                    xanchor='center', yanchor='bottom',
                    row=row, col=1,
                )

    def _add_fraction_traces(self, fig, metrics_data, lidar_names, colors, category, row):
        fraction_keys = self._schemas['fraction_metrics'][category]
        labels = [k.replace('_frac', '').replace('_', ' ') for k in fraction_keys]
        for lidar_idx, lidar_name in enumerate(lidar_names):
            d = metrics_data[lidar_name].get(category, {})
            fracs = [d.get(k, 0) for k in fraction_keys]
            color = colors[lidar_idx]
            fig.add_trace(go.Bar(
                name=lidar_name,
                x=labels,
                y=fracs,
                marker=self._gradient_marker(fracs, color),
                text=[f'{f * 100:.2f}%' for f in fracs],
                textposition='outside',
                textfont=dict(size=11, color='white', family='monospace'),
                hovertemplate=f'<b>{lidar_name}</b><br>%{{x}}: <b>%{{y:.3%}}</b><extra></extra>',
                showlegend=(row == 1),
            ), row=row, col=1)

        fig.add_hline(
            y=0.05,
            line=dict(color='rgba(255,80,80,0.45)', width=1.5, dash='dot'),
            annotation_text='5% threshold',
            annotation_font=dict(color='rgba(255,80,80,0.7)', size=10),
            row=row, col=1,
        )

    def _add_bar_traces(self, fig, metrics_data, lidar_names, colors, category, metric_names, row):
        for lidar_idx, lidar_name in enumerate(lidar_names):
            d = metrics_data[lidar_name].get(category, {})
            values = [d.get(m, 0) for m in metric_names]
            color = colors[lidar_idx]
            fig.add_trace(go.Bar(
                name=lidar_name,
                x=metric_names,
                y=values,
                marker=self._gradient_marker(values, color),
                text=[f'{v:.4g}' for v in values],
                textposition='outside',
                textfont=dict(size=11, color='white', family='monospace'),
                hovertemplate=f'<b>{lidar_name}</b><br>%{{x}}: <b>%{{y:.6g}}</b><extra></extra>',
                showlegend=(row == 1),
            ), row=row, col=1)

        if any(k in category for k in ('Error', 'Offset', 'Residual')):
            fig.add_hline(y=0, line=dict(color='rgba(255,255,255,0.18)', width=1, dash='dot'), row=row, col=1)

    def _style_axes(self, fig, row, layout, category: str = ''):
        units = self._schemas['category_units'].get(category, '')
        y_title = f'Value ({units})' if units else 'Value'
        fig.update_xaxes(
            row=row, col=1,
            tickangle=-40,
            tickfont=dict(size=layout['tick_font_size'] + 1, color='rgba(255,255,255,0.95)', family='sans-serif'),
            showgrid=False,
            linecolor='rgba(255,255,255,0.08)',
            zeroline=False,
            ticks='outside', ticklen=5, tickcolor='rgba(255,255,255,0.15)',
        )
        fig.update_yaxes(
            row=row, col=1,
            tickfont=dict(size=layout['tick_font_size'], color='rgba(255,255,255,0.7)'),
            gridcolor='rgba(255,255,255,0.04)',
            zerolinecolor='rgba(255,255,255,0.2)',
            zerolinewidth=1,
            title_text=f'<b>{y_title}</b>',
            title_font=dict(size=layout['axis_label_font_size'] + 1, color='rgba(255,255,255,0.95)'),
            tickformat='.3g',
            autorange=True,
            ticks='outside', ticklen=5, tickcolor='rgba(255,255,255,0.15)',
        )

    def _collect_categories(self, metrics_data: dict) -> dict[str, list[str]]:
        categories: dict[str, set] = {}
        for lidar_metrics in metrics_data.values():
            for category, metrics in lidar_metrics.items():
                if category not in categories:
                    categories[category] = set()
                categories[category].update(metrics.keys())
        return {cat: sorted(metrics) for cat, metrics in categories.items()}

    def _fraction_color(self, value: float) -> str:
        if value < 0.05:
            return '#39FF14'
        elif value < 0.15:
            return '#FFD700'
        return '#FF6B6B'

    def _hex_to_rgba(self, hex_color: str, alpha: float) -> str:
        h = hex_color.lstrip('#')
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f'rgba({r},{g},{b},{alpha})'

    def _colors(self, n: int) -> list[str]:
        palette = self._config['palette']
        return [palette[i % len(palette)] for i in range(n)]
