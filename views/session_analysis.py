"""
BioTDMS Session Analysis View

Streamlit component for selecting sessions, roles, and signatures,
then visualizing the physiological timeseries data.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import List, Dict, Optional

from core.data_loader import (
    UnifiedDataLoader,
    render_session_selector,
    render_role_selector,
    render_signature_selector
)


def render_timeseries_plot(df: pd.DataFrame, 
                           column_metadata: List[Dict],
                           timestamp_col: str = 'timestamp') -> go.Figure:
    """
    Render multi-trace timeseries plot with role-based coloring.
    """
    if df.empty or not column_metadata:
        return go.Figure()
    
    fig = go.Figure()
    
    for meta in column_metadata:
        col = meta['column']
        if col not in df.columns:
            continue
        
        fig.add_trace(go.Scatter(
            x=df[timestamp_col],
            y=df[col],
            mode='lines',
            name=meta['label'],
            line=dict(color=meta['color'], width=1.5),
            hovertemplate=f"{meta['label']}<br>%{{y:.3f}} {meta['unit']}<br>%{{x}}<extra></extra>"
        ))
    
    # Layout
    fig.update_layout(
        height=500,
        margin=dict(l=60, r=20, t=40, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis_title="Time",
        yaxis_title=column_metadata[0]['y_label'] if column_metadata else "Value",
        hovermode='x unified'
    )
    
    return fig


def render_multi_signature_plot(df: pd.DataFrame,
                                 column_metadata: List[Dict],
                                 timestamp_col: str = 'timestamp') -> go.Figure:
    """
    Render subplots grouped by signature type.
    """
    if df.empty or not column_metadata:
        return go.Figure()
    
    # Group by signature
    sig_groups = {}
    for meta in column_metadata:
        sig_id = meta['signature_id']
        if sig_id not in sig_groups:
            sig_groups[sig_id] = []
        sig_groups[sig_id].append(meta)
    
    n_sigs = len(sig_groups)
    if n_sigs == 0:
        return go.Figure()
    
    fig = make_subplots(
        rows=n_sigs,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[list(sig_groups.keys())[i] for i in range(n_sigs)]
    )
    
    for row_idx, (sig_id, metas) in enumerate(sig_groups.items(), start=1):
        for meta in metas:
            col = meta['column']
            if col not in df.columns:
                continue
            
            fig.add_trace(
                go.Scatter(
                    x=df[timestamp_col],
                    y=df[col],
                    mode='lines',
                    name=meta['label'],
                    line=dict(color=meta['color'], width=1.5),
                    legendgroup=sig_id,
                    showlegend=True
                ),
                row=row_idx,
                col=1
            )
        
        # Y-axis label
        fig.update_yaxes(title_text=metas[0]['y_label'], row=row_idx, col=1)
    
    fig.update_layout(
        height=300 * n_sigs,
        margin=dict(l=60, r=20, t=60, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Time", row=n_sigs, col=1)
    
    return fig


class SessionAnalysisView:
    """
    Main view component for session-based analysis.
    """
    
    def __init__(self, project_root: Path):
        self.loader = UnifiedDataLoader(project_root)
    
    def render(self):
        """Render the complete session analysis interface"""
        st.header("📊 Session Physiological Analysis")
        
        # Sidebar controls
        with st.sidebar:
            st.header("Data Selection")
            
            # Session selector
            try:
                session_params = render_session_selector(
                    self.loader.sessions,
                    key_prefix="main"
                )
            except FileNotFoundError as e:
                st.error(str(e))
                st.info("Please run `python core/process_sessions.py /path/to/raw/data` first.")
                return
            
            st.divider()
            
            # Role selector
            selected_roles = render_role_selector(
                self.loader.signatures,
                key_prefix="main"
            )
            
            if not selected_roles:
                st.warning("Select at least one role")
                return
            
            st.divider()
            
            # Signature selector - session physio only for now
            selected_sigs = render_signature_selector(
                self.loader.signatures,
                data_source="session_physio",
                key_prefix="main"
            )
        
        # Main content area
        if not selected_sigs:
            st.info("👈 Select signatures from the sidebar to visualize data")
            
            # Show available signatures summary
            st.subheader("Available Signature Categories")
            categories = self.loader.signatures.get_categories(data_source="session_physio")
            
            cols = st.columns(3)
            for i, cat in enumerate(categories):
                with cols[i % 3]:
                    sigs = self.loader.signatures.get_signatures(
                        data_source="session_physio",
                        category=cat
                    )
                    st.metric(cat, len(sigs))
            
            return
        
        # Load and display data
        with st.spinner("Loading data..."):
            try:
                df, column_metadata = self.loader.load_for_signatures(
                    signature_ids=selected_sigs,
                    roles=selected_roles,
                    session_params=session_params
                )
            except Exception as e:
                st.error(f"Error loading data: {e}")
                return
        
        if df.empty:
            st.warning("No data found for the selected parameters")
            return
        
        # Display info
        st.caption(f"Loaded {len(df):,} timepoints × {len(column_metadata)} signals")
        
        # Plot options
        plot_type = st.radio(
            "Plot layout",
            ["Combined", "Separate by Signature"],
            horizontal=True
        )
        
        if plot_type == "Combined":
            fig = render_timeseries_plot(df, column_metadata)
        else:
            fig = render_multi_signature_plot(df, column_metadata)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Data preview
        with st.expander("📋 Data Preview"):
            st.dataframe(df.head(100), use_container_width=True)
        
        # Column info
        with st.expander("📊 Selected Signals"):
            sig_df = pd.DataFrame(column_metadata)
            st.dataframe(sig_df[['label', 'column', 'role', 'signature_id']], 
                        use_container_width=True)


def main():
    """Standalone entry point for testing"""
    st.set_page_config(
        page_title="BioTDMS Session Analysis",
        layout="wide"
    )
    
    # Assume project root is parent of this file's directory
    project_root = Path(__file__).parent.parent
    
    view = SessionAnalysisView(project_root)
    view.render()


if __name__ == "__main__":
    main()
