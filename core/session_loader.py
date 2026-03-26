#!/usr/bin/env python3
"""
Loader utilities for processed BioTDMS session data.

This module provides functions for loading and querying processed parquet files
from the session processing pipeline.
"""

import pandas as pd
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class SessionDataLoader:
    """
    Loader for processed session parquet files.
    
    Usage:
        # Use default path (data/processed_sessions relative to project root)
        loader = SessionDataLoader.with_default_path()
        
        # Or specify explicitly
        loader = SessionDataLoader('/path/to/processed_sessions')
        
        # Get available sessions
        sessions = loader.list_sessions()
        
        # Load specific columns for a session
        df = loader.load_session(
            dce='DCE1',
            team='Team1', 
            day='Day1',
            session='Session1',
            columns=['FOA_ekg_IBI', 'FOM_ekg_IBI', 'Lead_gaze_entropy']
        )
    """
    
    def __init__(self, processed_root: Path | str):
        """
        Initialize the loader.
        
        Args:
            processed_root: Path to the processed_sessions directory
        """
        self.root = Path(processed_root)
        self._index: Optional[pd.DataFrame] = None
    
    @classmethod
    def with_default_path(cls) -> 'SessionDataLoader':
        """
        Create loader using default path: data/processed_sessions relative to project root.
        
        Assumes this module is in core/ folder.
        """
        module_dir = Path(__file__).parent
        project_root = module_dir.parent
        default_path = project_root / 'data' / 'processed_sessions'
        return cls(default_path)
    
    @property
    def index(self) -> pd.DataFrame:
        """Lazy-load and cache the sessions index."""
        if self._index is None:
            index_path = self.root / 'sessions_index.parquet'
            if not index_path.exists():
                raise FileNotFoundError(
                    f"Sessions index not found at {index_path}. "
                    "Run process_sessions.py first."
                )
            self._index = pd.read_parquet(index_path)
        return self._index
    
    def list_sessions(self, 
                      dce: Optional[str] = None,
                      team: Optional[str] = None) -> pd.DataFrame:
        """
        List available sessions, optionally filtered.
        
        Args:
            dce: Filter by DCE name (e.g., 'DCE1')
            team: Filter by team name (e.g., 'Team1')
            
        Returns:
            DataFrame with session metadata
        """
        df = self.index.copy()
        
        if dce is not None:
            df = df[df['dce'] == dce]
        if team is not None:
            df = df[df['team'] == team]
        
        return df
    
    def get_session_path(self,
                         dce: str,
                         team: str,
                         day: str,
                         session: str) -> Path:
        """Get the file path for a specific session."""
        filename = f"{team}_{day}_{session}.parquet"
        return self.root / dce / filename
    
    def load_session(self,
                     dce: str,
                     team: str,
                     day: str,
                     session: str,
                     columns: Optional[list[str]] = None) -> pd.DataFrame:
        """
        Load a session's data.
        
        Args:
            dce: DCE name (e.g., 'DCE1')
            team: Team name (e.g., 'Team1')
            day: Day name (e.g., 'Day1')
            session: Session name (e.g., 'Session1')
            columns: Optional list of specific columns to load.
                     Metadata columns (dce, team, day, session, scenario_name,
                     session_time, time_stamp) are always included.
                     
        Returns:
            DataFrame with session data
        """
        path = self.get_session_path(dce, team, day, session)
        
        if not path.exists():
            raise FileNotFoundError(f"Session file not found: {path}")
        
        # Always include metadata and time columns
        metadata_cols = ['dce', 'team', 'day', 'session', 'scenario_name', 
                        'session_time', 'time_stamp']
        
        if columns is not None:
            # Combine metadata with requested columns, preserving order
            all_columns = metadata_cols + [c for c in columns if c not in metadata_cols]
            df = pd.read_parquet(path, columns=all_columns)
        else:
            df = pd.read_parquet(path)
        
        return df
    
    def get_available_columns(self,
                              dce: str,
                              team: str,
                              day: str,
                              session: str) -> list[str]:
        """
        Get list of available columns for a session.
        
        Useful for building column selectors in UI.
        """
        path = self.get_session_path(dce, team, day, session)
        
        if not path.exists():
            raise FileNotFoundError(f"Session file not found: {path}")
        
        # Read just the schema (no data)
        pf = pd.read_parquet(path, columns=[])
        return list(pd.read_parquet(path).columns)
    
    def get_signal_columns(self,
                           dce: str,
                           team: str,
                           day: str,
                           session: str,
                           roles: Optional[list[str]] = None,
                           sensors: Optional[list[str]] = None) -> list[str]:
        """
        Get signal columns filtered by role and/or sensor.
        
        Args:
            dce, team, day, session: Session identifiers
            roles: Filter by roles (e.g., ['FOA', 'FOM'])
            sensors: Filter by sensors (e.g., ['ekg', 'gaze'])
            
        Returns:
            List of matching column names
        """
        all_cols = self.get_available_columns(dce, team, day, session)
        
        # Metadata columns to exclude
        metadata = {'dce', 'team', 'day', 'session', 'scenario_name', 
                   'session_time', 'time_stamp'}
        
        signal_cols = [c for c in all_cols if c not in metadata]
        
        if roles is not None:
            signal_cols = [c for c in signal_cols 
                          if any(c.startswith(f"{role}_") for role in roles)]
        
        if sensors is not None:
            # Sensor appears after role prefix: FOA_ekg_IBI
            signal_cols = [c for c in signal_cols
                          if any(f"_{sensor}_" in c or c.endswith(f"_{sensor}") 
                                for sensor in sensors)]
        
        return signal_cols


def load_session_quick(processed_root: Path | str,
                       dce: str,
                       team: str,
                       day: str,
                       session: str,
                       columns: Optional[list[str]] = None) -> pd.DataFrame:
    """
    Convenience function for one-off session loading.
    
    For repeated access, use SessionDataLoader class for caching benefits.
    """
    loader = SessionDataLoader(processed_root)
    return loader.load_session(dce, team, day, session, columns)


# Example usage and testing
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test session data loader')
    parser.add_argument('processed_root', type=Path, 
                       help='Path to processed_sessions directory')
    args = parser.parse_args()
    
    loader = SessionDataLoader(args.processed_root)
    
    print("Available sessions:")
    print(loader.list_sessions())
    
    # If sessions exist, show sample data
    sessions = loader.list_sessions()
    if len(sessions) > 0:
        first = sessions.iloc[0]
        print(f"\nSample columns from {first['file_path']}:")
        cols = loader.get_available_columns(
            first['dce'], first['team'], first['day'], first['session']
        )
        print(f"  Total columns: {len(cols)}")
        print(f"  First 20: {cols[:20]}")
