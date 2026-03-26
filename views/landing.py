"""
Landing page for BioTDMS Explorer
"""

import streamlit as st


def render_landing_page_styled() -> str | None:
    """Render landing page with workflow options. Returns selected flow or None."""

    st.markdown("""
    Welcome to the **Biological Team Dynamics Monitoring System** explorer.
    Analyze team performance through physiological measurements and evidence-based signatures.
    """)

    st.markdown("---")
    st.markdown("### Choose a Workflow")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### 🎯 Design Measurement Strategy
        Define competencies, select modalities, and get evidence-based recommendations.
        """)
        if st.button("Start Measurement Design →", key="flow_measurement"):
            return "measurement"

    with col2:
        st.markdown("""
        #### 📊 Explore Team Performance
        Select signatures, load real data, and visualize team dynamics over time.
        """)
        if st.button("Explore Performance →", key="flow_performance", type="primary"):
            return "performance"

    return None
