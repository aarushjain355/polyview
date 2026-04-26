from pathlib import Path

import streamlit as st
import yaml
import database_handler
import visualization_handler
import time

st.set_page_config(layout='wide', page_title='PolyView LiDAR')

_LOGO_CSS = (Path(__file__).parent / 'css' / 'logo.css').read_text()

st.sidebar.markdown(f'<style>{_LOGO_CSS}</style>', unsafe_allow_html=True)
st.sidebar.markdown(
    '<span class="polyview-logo">PolyView</span>'
    '<span class="polyview-tagline">LiDAR Evaluation Suite</span>'
    '<hr class="polyview-divider">',
    unsafe_allow_html=True,
)

class PolyViewApp:

    def __init__(self):
        self.database_handler = database_handler.DatabaseHandler(st.secrets["notion_token"], st.secrets["notion_database_id"], st.secrets["notion_version"], st.secrets["notion_base_url"])
        self.visualization_handler = visualization_handler.VisualizationHandler()
        descriptions_path = Path(__file__).parent / 'metric_descriptions.yaml'
        with open(descriptions_path, 'r') as f:
            self.metric_descriptions = yaml.safe_load(f)
        settings_path = Path(__file__).parent / 'settings.yaml'
        with open(settings_path, 'r') as f:
            self._settings = yaml.safe_load(f)
        lidar_thresholds_path = Path(__file__).parent / 'lidar_thresholds.yaml'
        with open(lidar_thresholds_path, 'r') as f:
            self._lidar_thresholds = yaml.safe_load(f) or {}
        if 'metrics_data' not in st.session_state:
            st.session_state.metrics_data = {}
        if 'visualization_data' not in st.session_state:
            st.session_state.visualization_data = {}
        if 'thresholds' not in st.session_state:
            st.session_state.thresholds = self._load_thresholds()
        if 'show_settings' not in st.session_state:
            st.session_state.show_settings = False

    def run(self):
        if st.sidebar.button('⚙️ Settings', use_container_width=True):
            st.session_state.show_settings = not st.session_state.show_settings
        if st.session_state.show_settings:
            self.render_settings_page()
        else:
            self.render_lidar_refresh_button()
            self.render_lidar_view_button()


    def render_3d_view(self):
        if not st.session_state.visualization_data:
            st.info('No visualization data loaded. Click "Refresh LiDAR Data from Notion".')
            return
        self.render_3d_button_panel()
        st.markdown(self.visualization_handler.glow_css, unsafe_allow_html=True)
        viz_data = st.session_state.visualization_data.get(self.selected_lidar, {})
        layers = st.session_state.visible_layers
        fig = self.visualization_handler.make_3d_figure(viz_data)
        if 'PointCloud' in layers:
            self.visualization_handler.add_point_cloud(fig, viz_data)
        if 'Expected Planes' in layers:
            self.visualization_handler.add_expected_planes(fig, viz_data)
        if 'Fitted PCA Plane' in layers:
            self.visualization_handler.add_fitted_pca_plane(fig, viz_data)
        if 'Spatial Dropout Analysis' in layers:
            self.visualization_handler.add_spatial_dropout_analysis(fig, viz_data)
        self.visualization_handler.add_sensor_axes(fig, viz_data)
        st.plotly_chart(fig, use_container_width=True, key='3d_scene_chart')


    def render_lidar_view_button(self):
        st.selectbox(
            'View Mode',
            options=['3D Visualization', 'Lidar Metrics Comparison', 'Lidar Metrics Information'],
            key='view_mode'
        )

        if st.session_state.view_mode == '3D Visualization':
            self.render_lidars_markdown()
            self.render_3d_view()
            self.render_lidar_metrics()

        elif st.session_state.view_mode == 'Lidar Metrics Comparison':
            self.render_comparison_graphs()

        elif st.session_state.view_mode == 'Lidar Metrics Information':
            self.render_metrics_page()


    def render_metrics_page(self):
        st.title('LiDAR Metrics Reference')
        st.markdown('_Reference guide for all metrics computed during LiDAR evaluation._')

        schemas = self.visualization_handler._schemas
        radar_metrics = schemas['radar_metrics']
        category_units = schemas['category_units']

        for category in category_units:
            if category not in self.metric_descriptions:
                continue

            unit = category_units[category]
            radar_info = radar_metrics.get(category, {})
            lower_is_better = radar_info.get('lower_is_better', None)
            direction = '↓ lower is better' if lower_is_better is True else '↑ higher is better' if lower_is_better is False else ''

            with st.expander(f'{category}   ·   {unit}   ·   {direction}', expanded=False):
                st.markdown(self.metric_descriptions[category]['description'])

                if st.session_state.metrics_data:
                    st.divider()
                    cols = st.columns(len(st.session_state.metrics_data))
                    for col, (lidar_name, lidar_metrics) in zip(cols, st.session_state.metrics_data.items()):
                        cat_data = lidar_metrics.get(category, {})
                        with col:
                            st.markdown(f'**{lidar_name}**')
                            if cat_data:
                                for key, val in cat_data.items():
                                    display = f'{val:.4g}' if isinstance(val, float) else str(val)
                                    st.markdown(f'`{key}`: {display}')
                            else:
                                st.markdown('_No data_')

    def render_comparison_graphs(self):
        st.markdown(self.visualization_handler.glow_css, unsafe_allow_html=True)
        radar_fig = self.visualization_handler.render_overview_radar(st.session_state.metrics_data)
        st.plotly_chart(radar_fig, use_container_width=True, key='radar_chart')
        st.divider()
        detail_fig = self.visualization_handler.render_metrics_comparison(st.session_state.metrics_data)
        st.plotly_chart(detail_fig, use_container_width=True, key='detail_chart')

    def _resolve_thresholds(self, lidar_name: str) -> dict:
        global_thresholds = st.session_state.get('thresholds', {})
        per_lidar = self._lidar_thresholds.get(lidar_name, {})
        if not per_lidar:
            return global_thresholds
        return {**global_thresholds, **per_lidar}

    def render_lidar_metrics(self):
        metrics = st.session_state.metrics_data.get(self.selected_lidar, {})
        if not metrics:
            return
        st.subheader('LiDAR Metrics')
        figs = self.visualization_handler.render_single_lidar_metrics(
            self.selected_lidar, metrics, self._resolve_thresholds(self.selected_lidar),
            self._settings.get('secondary_axis_keys', []),
            self._settings.get('plot_y_padding', 0.3),
            self._settings.get('split_by_suffix_categories', []),
            self._settings.get('split_exclude_suffixes', {}),
        )
        for category, fig in figs:
            st.plotly_chart(fig, use_container_width=True, key=f'metrics_{self.selected_lidar}_{category}')

    def render_lidar_refresh_button(self):
        if st.button("Refresh LiDAR Data from Notion"):
            with st.spinner("Fetching latest test results..."):
                self.retrieve_notion_data()
                time.sleep(1)  # Simulate loading time
            st.success("Data refreshed!")

    def render_lidars_markdown(self):
        self.selected_lidar = st.selectbox('Select LiDAR', options=list(st.session_state.metrics_data.keys()))

    def retrieve_baseline_metrics(self):
        # function to fetch baseline metrics for comparison, could be from Notion or a local file
        pass

    def _load_thresholds(self) -> dict:
        defaults_path = Path(__file__).parent / 'defaults.yaml'
        defaults = {}
        if defaults_path.exists():
            with open(defaults_path, 'r') as f:
                defaults = (yaml.safe_load(f) or {}).get('thresholds', {})
        user_overrides = self._settings.get('thresholds', {}) or {}
        return {**defaults, **user_overrides}

    def render_settings_page(self):
        st.title('⚙️ Metric Thresholds')
        st.markdown('Configure colored bands on metric graphs to visualize great / ok / bad regions.')
        thresholds = st.session_state.get('thresholds', {})
        thresholdable_metrics = self._settings.get('thresholdable_metrics', [])
        with st.form('thresholds_form'):
            updated: dict = {}
            for metric in thresholdable_metrics:
                st.markdown(f'### {metric}')
                metric_config = thresholds.get(metric, {})
                if isinstance(metric_config, list):
                    updated_list = []
                    for entry_idx, entry in enumerate(metric_config):
                        keys_label = ', '.join(entry.get('keys', []))
                        st.markdown(f'**`{keys_label}`**')
                        updated_entry: dict = {'keys': entry.get('keys', [])}
                        header = st.columns([0.12, 0.55, 1, 1, 2])
                        header[0].markdown('**On**')
                        header[1].markdown('**Zone**')
                        header[2].markdown('**Min**')
                        header[3].markdown('**Max**')
                        header[4].markdown('**Label**')
                        for zone in ('great', 'ok_1', 'ok_2', 'bad_1', 'bad_2'):
                            zone_data = entry.get(zone, {})
                            cols = st.columns([0.12, 0.55, 1, 1, 2])
                            enabled = cols[0].checkbox('', value=bool(zone_data.get('enabled', False)), key=f'{metric}_{entry_idx}_{zone}_enabled')
                            cols[1].markdown(f'**{zone.replace("_", " ").capitalize()}**')
                            min_val = cols[2].number_input('min', value=float(zone_data.get('min', 0.0)), key=f'{metric}_{entry_idx}_{zone}_min', label_visibility='collapsed')
                            max_val = cols[3].number_input('max', value=float(zone_data.get('max', 0.0)), key=f'{metric}_{entry_idx}_{zone}_max', label_visibility='collapsed')
                            label = cols[4].text_input('label', value=str(zone_data.get('label', '')), key=f'{metric}_{entry_idx}_{zone}_label', label_visibility='collapsed')
                            updated_entry[zone] = {'enabled': enabled, 'min': min_val, 'max': max_val, 'label': label}
                        updated_list.append(updated_entry)
                    updated[metric] = updated_list
                else:
                    updated[metric] = {}
                    header = st.columns([0.12, 0.55, 1, 1, 2])
                    header[0].markdown('**On**')
                    header[1].markdown('**Zone**')
                    header[2].markdown('**Min**')
                    header[3].markdown('**Max**')
                    header[4].markdown('**Label**')
                    for zone in ('great', 'ok_1', 'ok_2', 'bad_1', 'bad_2'):
                        zone_data = metric_config.get(zone, {})
                        cols = st.columns([0.12, 0.55, 1, 1, 2])
                        enabled = cols[0].checkbox('', value=bool(zone_data.get('enabled', False)), key=f'{metric}_{zone}_enabled')
                        cols[1].markdown(f'**{zone.replace("_", " ").capitalize()}**')
                        min_val = cols[2].number_input('min', value=float(zone_data.get('min', 0.0)), key=f'{metric}_{zone}_min', label_visibility='collapsed')
                        max_val = cols[3].number_input('max', value=float(zone_data.get('max', 0.0)), key=f'{metric}_{zone}_max', label_visibility='collapsed')
                        label = cols[4].text_input('label', value=str(zone_data.get('label', '')), key=f'{metric}_{zone}_label', label_visibility='collapsed')
                        updated[metric][zone] = {'enabled': enabled, 'min': min_val, 'max': max_val, 'label': label}
                st.divider()
            if st.form_submit_button('💾 Save Thresholds', use_container_width=True):
                st.session_state.thresholds = updated
                self._settings['thresholds'] = updated
                settings_path = Path(__file__).parent / 'settings.yaml'
                with open(settings_path, 'w') as f:
                    yaml.dump(self._settings, f)
                st.success('Thresholds saved!')

    def render_3d_button_panel(self):
        all_layers = ['PointCloud', 'Expected Planes', 'Fitted PCA Plane', "Spatial Dropout Analysis"]
        if 'visible_layers' not in st.session_state:
            st.session_state.visible_layers = all_layers
        st.session_state.visible_layers = st.multiselect(
            'Visible Layers',
            options=all_layers,
            default=st.session_state.visible_layers,
        )

    def retrieve_notion_data(self):
        st.session_state.metrics_data = self.database_handler.retrieve_data()
        st.session_state.visualization_data = self.database_handler.retrieve_visualization_data()



def main():
    app = PolyViewApp()
    app.run()

if __name__ == "__main__":
    main()
