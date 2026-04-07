# BioTDMS CLI — Integration Guide for Notebook Team

## Overview

`cli.py` processes raw physiological sensor data into the format the BioTDMS
Explorer app expects. Call it from notebooks via `os.system()` or `subprocess`
to stand up the app's data layer without launching the Streamlit UI.

## Prerequisites

- Python 3.12+
- Dependencies: `pandas`, `pyarrow` (for parquet), `openpyxl` (for subtask Excel)
- The `cli.py` script and `core/data_ingest.py` must be in the BioTDMS project directory

## File Naming Conventions

| File type          | Naming pattern                          | Example                              |
|--------------------|-----------------------------------------|--------------------------------------|
| Entropy/AMI CSV    | `team_entropy_ami_{DCE}.csv`            | `team_entropy_ami_DCE3.csv`          |
| Subtask lookup     | `SubTask_LookupTable_{DCE}.xlsx`        | `SubTask_LookupTable_DCE3.xlsx`      |
| Merged sensor CSV  | `merged_{Day}_{Session}_{Role}_{Subject}_1hz.csv` | `merged_Day3_Session2_FOA_Subj023_1hz.csv` |

**Important**: Entropy and subtask files must have unique names per DCE so
multiple files can coexist in the app's data directory.

## Basic Usage

```python
import os

# From a notebook, process DCE3 data:
os.system(
    "python /path/to/biotdms/cli.py "
    "--raw-dir /path/to/DCE3_data "
    "--output-dir /path/to/biotdms/data/processed_sessions "
    "--data-dir /path/to/biotdms/data "
    "--entropy /path/to/team_entropy_ami_DCE3.csv "
    "--subtask /path/to/SubTask_LookupTable_DCE3.xlsx"
)
```

## Full Argument Reference

| Argument              | Required | Description |
|-----------------------|----------|-------------|
| `--raw-dir DIR`       | Yes      | Path to raw data (DCE{N}/Team{N}/... structure) |
| `--output-dir DIR`    | Yes      | Where processed session parquets are written |
| `--data-dir DIR`      | No       | Where entropy CSVs and subtask tables go (default: `data/`) |
| `--entropy FILE`      | No       | Entropy/AMI CSV to install (repeatable for multiple DCEs) |
| `--subtask FILE`      | No       | Subtask Excel to install (repeatable for multiple DCEs) |
| `--force`             | No       | Reprocess sessions even if parquets already exist |
| `--quiet`             | No       | Print summary only, suppress per-session output |
| `--rebuild-index-only`| No       | Just rebuild the session index from existing parquets |

## Common Patterns

### Process new data incrementally (skip existing sessions)
```bash
python cli.py --raw-dir /data/DCE3 --output-dir data/processed_sessions
```

### Process everything from scratch
```bash
python cli.py --raw-dir /data/DCE3 --output-dir data/processed_sessions --force
```

### Multiple entropy files in one call
```bash
python cli.py --raw-dir /data/raw \
    --output-dir data/processed_sessions \
    --entropy /data/team_entropy_ami_DCE2.csv \
    --entropy /data/team_entropy_ami_DCE3.csv
```

### Quiet mode for automated pipelines
```bash
python cli.py --raw-dir /data/DCE3 --output-dir data/processed_sessions --quiet
```

### Just rebuild the session index (no reprocessing)
```bash
python cli.py --raw-dir /data/DCE3 --output-dir data/processed_sessions --rebuild-index-only
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0    | Success — all sessions processed without errors |
| 1    | Partial success — some sessions failed (check output for details) |
| 2    | Fatal error — bad arguments, missing directories, invalid files |

## What Gets Created

```
data/
├── processed_sessions/
│   ├── DCE2/
│   │   ├── Team1_Day1_Session1.parquet
│   │   └── ...
│   ├── DCE3/
│   │   ├── Team1_Day1_Session1.parquet
│   │   └── ...
│   └── sessions_index.parquet        # auto-rebuilt after each run
├── team_entropy_ami_DCE2.csv         # installed via --entropy
├── team_entropy_ami_DCE3.csv
└── SubTask_LookupTable_DCE3.xlsx     # installed via --subtask
```

Each session parquet contains:
- Metadata columns: `dce`, `team`, `day`, `session`
- `time_stamp` (unix epoch, 1hz)
- `trial_running` (boolean)
- Role-prefixed signals: `{Role}_{measure}` (e.g., `FOA_IBI`, `Lead_alpha_Fp1`, `JTAC_hbo`)
