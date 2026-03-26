"""
views/data_management.py — Sidebar data ingestion panel for BioTDMS Explorer

Provides a UI for:
1. Pointing to a raw data directory (DCE structure)
2. Pointing to an entropy/AMI CSV
3. Pointing to a subtask lookup table (Excel)
4. Scanning for new sessions
5. Processing and loading new data
6. Refreshing the session index

Integrates with core.data_ingest for the actual processing.
"""

import streamlit as st
from pathlib import Path
from typing import Optional


def render_data_panel(app_dir: Path):
    """Render the data management section in the sidebar.
    
    Args:
        app_dir: The application root directory (where data/ lives)
    """
    data_dir = app_dir / 'data'
    output_root = data_dir / 'processed_sessions'

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📂 Data Management")

    # Initialize session state for data settings
    if 'data_raw_path' not in st.session_state:
        st.session_state.data_raw_path = str(data_dir / 'raw')
    if 'data_entropy_path' not in st.session_state:
        st.session_state.data_entropy_path = ''
    if 'data_subtask_path' not in st.session_state:
        st.session_state.data_subtask_path = ''
    if 'data_version' not in st.session_state:
        st.session_state.data_version = 0
    if 'last_ingest_report' not in st.session_state:
        st.session_state.last_ingest_report = None

    with st.sidebar.expander("⚙️ Data Settings", expanded=False):
        # Raw data path
        raw_path = st.text_input(
            "Raw data directory",
            value=st.session_state.data_raw_path,
            help="Path to the DCE folder structure containing merged CSVs",
            key="raw_path_input"
        )
        st.session_state.data_raw_path = raw_path

        # Entropy/AMI file
        entropy_path = st.text_input(
            "Entropy/AMI CSV file",
            value=st.session_state.data_entropy_path,
            help="Path to team_entropy_ami.csv (leave empty if already in data/)",
            key="entropy_path_input"
        )
        st.session_state.data_entropy_path = entropy_path

        # Subtask lookup table
        subtask_path = st.text_input(
            "Subtask lookup table (Excel)",
            value=st.session_state.data_subtask_path,
            help="Path to SubTask_LookupTable .xlsx (leave empty if already in data/)",
            key="subtask_path_input"
        )
        st.session_state.data_subtask_path = subtask_path

        # Show current data status
        _show_data_status(output_root, data_dir)

    # Action buttons (outside expander for visibility)
    col1, col2 = st.sidebar.columns(2)

    with col1:
        if st.button("🔍 Scan", use_container_width=True, help="Check for new sessions"):
            _scan_for_new(raw_path, output_root)

    with col2:
        if st.button("🔄 Refresh", use_container_width=True, help="Rebuild session index"):
            _refresh_index(output_root)

    # Process button (full width)
    if st.sidebar.button(
        "⚡ Process New Data",
        use_container_width=True,
        type="primary",
        help="Process new sessions and update supporting data files"
    ):
        _process_new_data(raw_path, entropy_path, subtask_path, output_root, data_dir)

    # Show last report if available
    if st.session_state.last_ingest_report:
        _show_report(st.session_state.last_ingest_report)


def _show_data_status(output_root: Path, data_dir: Path):
    """Show current data availability."""
    # Count existing sessions
    n_parquets = 0
    dces = set()
    if output_root.exists():
        for pq in output_root.rglob('*.parquet'):
            if pq.name != 'sessions_index.parquet':
                n_parquets += 1
                dces.add(pq.parent.name)

    # Check entropy
    has_entropy = (data_dir / 'team_entropy_ami.csv').exists()

    # Check subtask lookup
    has_subtask = any(data_dir.glob('SubTask_LookupTable*.xlsx')) or \
                  any(data_dir.glob('subtask*.xlsx'))

    st.caption(
        f"📊 **{n_parquets}** sessions across **{len(dces)}** DCEs"
    )
    st.caption(
        f"Entropy: {'✅' if has_entropy else '❌'} | "
        f"Subtasks: {'✅' if has_subtask else '❌'}"
    )


def _scan_for_new(raw_path: str, output_root: Path):
    """Scan for new sessions without processing."""
    from core.data_ingest import discover_sessions, find_new_sessions

    raw = Path(raw_path)
    if not raw.exists():
        st.sidebar.error(f"Directory not found: {raw_path}")
        return

    with st.sidebar.status("Scanning...", expanded=True) as status:
        new, existing = find_new_sessions(raw, output_root)
        status.update(label="Scan complete", state="complete")

    if new:
        st.sidebar.success(f"Found **{len(new)}** new sessions to process")
        for sg in new[:10]:
            roles = ', '.join(sg.roles)
            st.sidebar.caption(f"  • {sg.dce}/{sg.team}/{sg.day}/{sg.session} ({roles})")
        if len(new) > 10:
            st.sidebar.caption(f"  ... and {len(new) - 10} more")
    else:
        st.sidebar.info(f"All {len(existing)} sessions already processed")


def _refresh_index(output_root: Path):
    """Rebuild the session index and bump the data version."""
    from core.data_ingest import rebuild_index

    if not output_root.exists():
        st.sidebar.warning("No processed sessions directory found")
        return

    with st.sidebar.status("Rebuilding index...") as status:
        index_df = rebuild_index(output_root)
        st.session_state.data_version += 1
        # Clear cached data so loaders re-read
        st.cache_data.clear()
        status.update(label=f"Index rebuilt: {len(index_df)} sessions", state="complete")


def _process_new_data(raw_path: str, entropy_path: str, subtask_path: str,
                      output_root: Path, data_dir: Path):
    """Run the full ingestion pipeline."""
    from core.data_ingest import ingest_sessions, install_entropy_csv, install_subtask_excel

    raw = Path(raw_path)
    if not raw.exists():
        st.sidebar.error(f"Directory not found: {raw_path}")
        return

    # Process sessions
    progress_bar = st.sidebar.progress(0, text="Starting...")

    def progress_cb(current, total, message):
        if total > 0:
            progress_bar.progress(current / total, text=message)

    report = ingest_sessions(
        raw_root=raw,
        output_root=output_root,
        skip_existing=True,
        progress_callback=progress_cb
    )

    progress_bar.progress(1.0, text="Done!")

    # Handle entropy CSV
    if entropy_path:
        entropy_file = Path(entropy_path)
        if entropy_file.exists() and entropy_file.suffix == '.csv':
            if install_entropy_csv(entropy_file, data_dir):
                report.details.append(f"Entropy CSV installed: {entropy_file.name}")
            else:
                report.errors.append(f"Failed to copy entropy CSV")
        elif entropy_path:  # non-empty but invalid
            report.errors.append(f"Entropy file not found: {entropy_path}")

    # Handle subtask lookup table
    if subtask_path:
        subtask_file = Path(subtask_path)
        if subtask_file.exists() and subtask_file.suffix in ('.xlsx', '.xls'):
            if install_subtask_excel(subtask_file, data_dir):
                report.details.append(f"Subtask table installed: {subtask_file.name}")
            else:
                report.errors.append(f"Failed to copy subtask table")
        elif subtask_path:  # non-empty but invalid
            report.errors.append(f"Subtask file not found: {subtask_path}")

    # Bump version to invalidate caches
    st.session_state.data_version += 1
    st.cache_data.clear()

    # Store report
    st.session_state.last_ingest_report = report

    # Summary toast
    if report.errors:
        st.sidebar.warning(
            f"Processed {report.processed} sessions with {len(report.errors)} errors"
        )
    elif report.processed > 0:
        st.sidebar.success(f"Processed {report.processed} new sessions!")
    else:
        st.sidebar.info("No new sessions to process")


def _show_report(report):
    """Show the last ingestion report in the sidebar."""
    with st.sidebar.expander("📋 Last Ingest Report", expanded=False):
        st.caption(
            f"Discovered: {report.discovered} | "
            f"New: {report.new_sessions} | "
            f"Processed: {report.processed} | "
            f"Skipped: {report.skipped_existing}"
        )
        if report.errors:
            st.markdown("**Errors:**")
            for err in report.errors:
                st.caption(f"❌ {err}")
        if report.details:
            st.markdown("**Details:**")
            for d in report.details[:15]:
                st.caption(f"• {d}")
