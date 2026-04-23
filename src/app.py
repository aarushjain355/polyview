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
        if 'metrics_data' not in st.session_state:
            st.session_state.metrics_data = {}
        if 'visualization_data' not in st.session_state:
            st.session_state.visualization_data = {}

    def run(self):
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

    def render_lidar_metrics(self):
        metrics = st.session_state.metrics_data.get(self.selected_lidar, {})
        if not metrics:
            return
        st.subheader('LiDAR Metrics')
        figs = self.visualization_handler.render_single_lidar_metrics(self.selected_lidar, metrics)
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

    def render_3d_button_panel(self):
        all_layers = ['PointCloud', 'Expected Planes', 'Fitted PCA Plane']
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
