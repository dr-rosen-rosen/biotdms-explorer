#!/usr/bin/env python3
"""
Process raw BioTDMS session data into consolidated parquet files.

This script traverses the nested folder structure of DCE data and consolidates
all sensor CSVs for each session into a single wide-format parquet file.

Input structure:
    DCE{N}/Team{N}/Day{N}/Session{N}/{Role}_Subj{NNN}/featurized/{sensor}_*.csv

Output structure:
    processed_sessions/
        sessions_index.parquet
        DCE1/
            Team1_Day1_Session1.parquet
            ...
        DCE2/
            ...
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import re
import logging
from typing import Optional
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress pandas warnings about fragmented DataFrames
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# Constants
ROLES = ['FOA', 'FOM', 'FSO', 'JTAC', 'Lead']
SENSORS = ['eeg', 'ekg', 'gaze', 'pupil', 'resp']

# Columns to exclude from role-prefixing (these are merge keys or metadata)
MERGE_KEYS = ['session_time', 'time_stamp']


def find_dce_folders(data_root: Path) -> list[Path]:
    """Find all DCE folders in the data root."""
    dce_folders = sorted([
        p for p in data_root.iterdir() 
        if p.is_dir() and p.name.startswith('DCE')
    ])
    logger.info(f"Found {len(dce_folders)} DCE folders: {[p.name for p in dce_folders]}")
    return dce_folders


def find_team_folders(dce_path: Path) -> list[Path]:
    """Find all Team folders within a DCE."""
    return sorted([
        p for p in dce_path.iterdir()
        if p.is_dir() and p.name.startswith('Team')
    ])


def find_day_folders(team_path: Path) -> list[Path]:
    """Find all Day folders within a Team."""
    return sorted([
        p for p in team_path.iterdir()
        if p.is_dir() and p.name.startswith('Day')
    ])


def find_session_folders(day_path: Path) -> list[Path]:
    """Find all Session folders within a Day."""
    return sorted([
        p for p in day_path.iterdir()
        if p.is_dir() and p.name.startswith('Session')
    ])


def find_role_folders(session_path: Path) -> list[Path]:
    """Find all role folders (e.g., FOA_Subj001) within a session."""
    role_folders = []
    for role in ROLES:
        matches = list(session_path.glob(f"{role}_Subj*"))
        if matches:
            role_folders.extend(matches)
    return sorted(role_folders)


def extract_role_from_folder(folder_name: str) -> Optional[str]:
    """Extract role name from folder like 'FOA_Subj001'."""
    for role in ROLES:
        if folder_name.startswith(role):
            return role
    return None


def load_scenario_info(session_path: Path) -> Optional[pd.DataFrame]:
    """Load scenario and run information CSV."""
    scenario_files = list(session_path.glob("Scenario_and_Run_Information-data-*.csv"))
    
    if not scenario_files:
        logger.warning(f"No scenario info file found in {session_path}")
        return None
    
    if len(scenario_files) > 1:
        logger.warning(f"Multiple scenario files found in {session_path}, using first")
    
    scenario_df = pd.read_csv(scenario_files[0])
    logger.debug(f"Loaded scenario info with {len(scenario_df)} rows")
    return scenario_df


def load_sensor_file(featurized_path: Path, sensor: str, day: str, session: str, role_subj: str) -> Optional[pd.DataFrame]:
    """Load a specific sensor CSV file (1hz version only)."""
    # Pattern: {sensor}_{day}_{session}_{role_subj}_1hz.csv
    pattern = f"{sensor}_{day}_{session}_{role_subj}_1hz.csv"
    matches = list(featurized_path.glob(pattern))
    
    if not matches:
        # Try case-insensitive match
        pattern_lower = pattern.lower()
        matches = [f for f in featurized_path.iterdir() if f.name.lower() == pattern_lower]
    
    if not matches:
        logger.warning(f"Sensor file not found: {featurized_path / pattern}")
        return None
    
    df = pd.read_csv(matches[0])
    logger.debug(f"Loaded {sensor} data: {len(df)} rows, {len(df.columns)} columns")
    return df


def merge_sensors_for_role(role_folder: Path, day: str, session: str) -> Optional[pd.DataFrame]:
    """Load and merge all sensor CSVs for a single role."""
    featurized_path = role_folder / 'featurized'
    
    if not featurized_path.exists():
        logger.warning(f"No featurized folder found in {role_folder}")
        return None
    
    role_subj = role_folder.name  # e.g., "FOA_Subj001"
    
    sensor_dfs = []
    for sensor in SENSORS:
        df = load_sensor_file(featurized_path, sensor, day, session, role_subj)
        if df is not None:
            sensor_dfs.append(df)
    
    if not sensor_dfs:
        logger.warning(f"No sensor data found for {role_subj}")
        return None
    
    # Merge all sensors on time_stamp (more reliable than session_time per user input)
    merged = sensor_dfs[0]
    for df in sensor_dfs[1:]:
        # Use time_stamp as the primary merge key
        # Keep session_time from first df only to avoid duplicates
        df_to_merge = df.drop(columns=['session_time'], errors='ignore')
        merged = pd.merge(merged, df_to_merge, on='time_stamp', how='outer')
    
    # Sort by timestamp
    merged = merged.sort_values('time_stamp').reset_index(drop=True)
    
    logger.debug(f"Merged {len(sensor_dfs)} sensors for {role_subj}: {len(merged)} rows")
    return merged


def prefix_columns(df: pd.DataFrame, role: str) -> pd.DataFrame:
    """Add role prefix to all columns except merge keys."""
    rename_map = {
        col: f"{role}_{col}" 
        for col in df.columns 
        if col not in MERGE_KEYS
    }
    return df.rename(columns=rename_map)


def merge_roles(role_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge all role DataFrames on time_stamp."""
    if not role_dfs:
        return pd.DataFrame()
    
    # Start with first role
    merged = role_dfs[0]
    
    for df in role_dfs[1:]:
        # Drop session_time from subsequent dfs to avoid duplicates
        df_to_merge = df.drop(columns=['session_time'], errors='ignore')
        merged = pd.merge(merged, df_to_merge, on='time_stamp', how='outer')
    
    # Sort by timestamp
    merged = merged.sort_values('time_stamp').reset_index(drop=True)
    
    return merged


def extract_day_number(day_name: str) -> str:
    """Extract day identifier from folder name like 'Day1' -> 'Day1'."""
    return day_name


def extract_session_number(session_name: str) -> str:
    """Extract session identifier from folder name like 'Session1' -> 'Session1'."""
    return session_name


def process_session(
    session_path: Path,
    dce_name: str,
    team_name: str,
    day_name: str,
    session_name: str
) -> Optional[pd.DataFrame]:
    """Process a single session into a wide-format DataFrame."""
    logger.info(f"Processing: {dce_name}/{team_name}/{day_name}/{session_name}")
    
    # Load scenario metadata
    scenario_df = load_scenario_info(session_path)
    scenario_name = None
    if scenario_df is not None and 'Scenario_Name' in scenario_df.columns:
        # Take first scenario name (assuming one per session)
        scenario_name = scenario_df['Scenario_Name'].iloc[0] if len(scenario_df) > 0 else None
    
    # Find and process each role
    role_folders = find_role_folders(session_path)
    if not role_folders:
        logger.warning(f"No role folders found in {session_path}")
        return None
    
    role_dfs = []
    for role_folder in role_folders:
        role = extract_role_from_folder(role_folder.name)
        if role is None:
            continue
        
        role_df = merge_sensors_for_role(role_folder, day_name, session_name)
        if role_df is not None:
            role_df = prefix_columns(role_df, role)
            role_dfs.append(role_df)
            logger.debug(f"Processed {role}: {len(role_df)} rows, {len(role_df.columns)} columns")
    
    if not role_dfs:
        logger.warning(f"No role data processed for {session_path}")
        return None
    
    # Merge all roles
    merged = merge_roles(role_dfs)
    
    # Add metadata columns at the front
    merged.insert(0, 'dce', dce_name)
    merged.insert(1, 'team', team_name)
    merged.insert(2, 'day', day_name)
    merged.insert(3, 'session', session_name)
    merged.insert(4, 'scenario_name', scenario_name)
    
    logger.info(f"  -> {len(merged)} rows, {len(merged.columns)} columns")
    return merged


def process_all_sessions(data_root: Path, output_root: Path) -> pd.DataFrame:
    """Process all sessions and return an index DataFrame."""
    index_records = []
    
    dce_folders = find_dce_folders(data_root)
    
    for dce_path in dce_folders:
        dce_name = dce_path.name
        dce_output = output_root / dce_name
        dce_output.mkdir(parents=True, exist_ok=True)
        
        for team_path in find_team_folders(dce_path):
            team_name = team_path.name
            
            for day_path in find_day_folders(team_path):
                day_name = day_path.name
                
                for session_path in find_session_folders(day_path):
                    session_name = session_path.name
                    
                    # Process this session
                    session_df = process_session(
                        session_path,
                        dce_name,
                        team_name,
                        day_name,
                        session_name
                    )
                    
                    if session_df is None or len(session_df) == 0:
                        logger.warning(f"Skipping empty session: {dce_name}/{team_name}/{day_name}/{session_name}")
                        continue
                    
                    # Generate output filename
                    output_filename = f"{team_name}_{day_name}_{session_name}.parquet"
                    output_path = dce_output / output_filename
                    
                    # Save to parquet
                    session_df.to_parquet(output_path, index=False)
                    logger.info(f"  Saved: {output_path}")
                    
                    # Add to index
                    index_records.append({
                        'dce': dce_name,
                        'team': team_name,
                        'day': day_name,
                        'session': session_name,
                        'scenario_name': session_df['scenario_name'].iloc[0] if 'scenario_name' in session_df.columns else None,
                        'n_rows': len(session_df),
                        'n_columns': len(session_df.columns),
                        'file_path': str(output_path.relative_to(output_root)),
                        'processed_at': datetime.now().isoformat()
                    })
    
    # Create and save index
    index_df = pd.DataFrame(index_records)
    if len(index_df) > 0:
        index_path = output_root / 'sessions_index.parquet'
        index_df.to_parquet(index_path, index=False)
        logger.info(f"Saved session index: {index_path}")
    
    return index_df


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Process BioTDMS session data into consolidated parquet files.'
    )
    parser.add_argument(
        'data_root',
        type=Path,
        help='Root directory containing DCE folders'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output directory (default: processed_sessions in script directory)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose (debug) logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set default output path
    if args.output is None:
        # Default to data/processed_sessions relative to project root
        # Assumes script is in core/ folder, so go up one level
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        output_root = project_root / 'data' / 'processed_sessions'
    else:
        output_root = args.output
    
    output_root.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Data root: {args.data_root}")
    logger.info(f"Output directory: {output_root}")
    
    # Validate data root
    if not args.data_root.exists():
        logger.error(f"Data root does not exist: {args.data_root}")
        return 1
    
    # Process all sessions
    index_df = process_all_sessions(args.data_root, output_root)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing complete!")
    logger.info(f"Total sessions processed: {len(index_df)}")
    logger.info(f"Output directory: {output_root}")
    
    if len(index_df) > 0:
        print("\nSession Index Summary:")
        print(index_df.to_string(index=False))
    
    return 0


if __name__ == '__main__':
    exit(main())
